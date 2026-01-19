#!/usr/bin/env python3
"""
Validation Framework for Low-Cost Star Tracker.

This module provides comprehensive validation tests for the star tracker
algorithms using synthetic data. It generates quantitative metrics for:
- Positional accuracy (centroid detection)
- SNR improvement from stacking
- Processing performance benchmarks
- Motion compensation effectiveness
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import json
from pathlib import Path


@dataclass
class ValidationResult:
    """Container for validation test results."""
    test_name: str
    passed: bool
    metrics: Dict
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    notes: str = ""


@dataclass
class Star:
    """Represents a star with known position and magnitude."""
    x: float  # True x position
    y: float  # True y position
    magnitude: float
    flux: float = 0.0

    def __post_init__(self):
        if self.flux == 0.0:
            # Convert magnitude to flux (arbitrary zero point)
            self.flux = 10 ** (-0.4 * (self.magnitude - 10))


class SyntheticStarField:
    """Generate synthetic star fields for validation."""

    def __init__(self, width: int = 1920, height: int = 1080, seed: int = None):
        self.width = width
        self.height = height
        if seed is not None:
            np.random.seed(seed)

    def generate_stars(self, num_stars: int = 100,
                       min_mag: float = 2.0, max_mag: float = 8.0) -> List[Star]:
        """Generate random star positions with realistic magnitude distribution."""
        stars = []
        for _ in range(num_stars):
            x = np.random.uniform(50, self.width - 50)
            y = np.random.uniform(50, self.height - 50)
            # Magnitude follows approximate stellar luminosity function
            mag = min_mag + (max_mag - min_mag) * np.random.power(2.5)
            stars.append(Star(x=x, y=y, magnitude=mag))
        return stars

    def render_image(self, stars: List[Star],
                     psf_fwhm: float = 2.5,
                     background: float = 100.0,
                     read_noise: float = 5.0,
                     add_noise: bool = True) -> np.ndarray:
        """Render star field to image with realistic noise model."""
        image = np.ones((self.height, self.width), dtype=np.float64) * background

        # Add stars as 2D Gaussians
        sigma = psf_fwhm / 2.355
        stamp_size = int(6 * sigma) + 1

        for star in stars:
            x_int, y_int = int(round(star.x)), int(round(star.y))

            # Create stamp bounds
            x_min = max(0, x_int - stamp_size)
            x_max = min(self.width, x_int + stamp_size + 1)
            y_min = max(0, y_int - stamp_size)
            y_max = min(self.height, y_int + stamp_size + 1)

            if x_max <= x_min or y_max <= y_min:
                continue

            # Generate Gaussian stamp
            y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
            gauss = np.exp(-((x_grid - star.x)**2 + (y_grid - star.y)**2) / (2 * sigma**2))
            gauss = gauss / gauss.sum() * star.flux * 10000  # Scale flux

            image[y_min:y_max, x_min:x_max] += gauss

        if add_noise:
            # Poisson noise (photon noise)
            image = np.random.poisson(np.maximum(image, 0).astype(np.int64)).astype(np.float64)
            # Read noise (Gaussian)
            image += np.random.normal(0, read_noise, image.shape)

        return image


class CentroidDetector:
    """Star detection and centroid measurement."""

    def __init__(self, threshold_sigma: float = 3.0, min_area: int = 5, max_area: int = 500):
        self.threshold_sigma = threshold_sigma
        self.min_area = min_area
        self.max_area = max_area

    def detect_stars(self, image: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Detect stars and measure centroids.

        Returns: List of (x, y, flux) tuples
        """
        # Estimate background
        background = np.median(image)
        noise = 1.4826 * np.median(np.abs(image - background))

        # Threshold
        threshold = background + self.threshold_sigma * noise
        binary = image > threshold

        # Find connected components (simple approach)
        from scipy.ndimage import label, find_objects
        labeled, num_features = label(binary)

        detections = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            area = np.sum(mask)

            if area < self.min_area or area > self.max_area:
                continue

            # Measure centroid
            y_coords, x_coords = np.where(mask)

            # Expand region for accurate centroiding
            y_min, y_max = y_coords.min() - 2, y_coords.max() + 3
            x_min, x_max = x_coords.min() - 2, x_coords.max() + 3
            y_min, x_min = max(0, y_min), max(0, x_min)
            y_max, x_max = min(image.shape[0], y_max), min(image.shape[1], x_max)

            stamp = image[y_min:y_max, x_min:x_max] - background
            stamp = np.maximum(stamp, 0)

            if stamp.sum() <= 0:
                continue

            # Intensity-weighted centroid
            y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
            x_centroid = np.sum(x_grid * stamp) / np.sum(stamp)
            y_centroid = np.sum(y_grid * stamp) / np.sum(stamp)
            flux = np.sum(stamp)

            detections.append((x_centroid, y_centroid, flux))

        return detections

    def match_detections(self, detections: List[Tuple],
                         true_stars: List[Star],
                         max_distance: float = 3.0) -> List[Tuple[Star, Tuple, float]]:
        """Match detections to true star positions."""
        matches = []
        used_detections = set()

        for star in true_stars:
            best_match = None
            best_distance = max_distance

            for i, det in enumerate(detections):
                if i in used_detections:
                    continue

                dist = np.sqrt((det[0] - star.x)**2 + (det[1] - star.y)**2)
                if dist < best_distance:
                    best_distance = dist
                    best_match = (i, det)

            if best_match is not None:
                used_detections.add(best_match[0])
                matches.append((star, best_match[1], best_distance))

        return matches


def validate_centroid_accuracy(num_trials: int = 10,
                                num_stars: int = 50,
                                seed: int = 42) -> ValidationResult:
    """
    Test centroid detection accuracy against known star positions.

    Measures:
    - Mean positional error
    - RMS positional error
    - Detection completeness
    - False positive rate
    """
    print("Running centroid accuracy validation...")

    np.random.seed(seed)

    all_errors = []
    detection_rates = []
    false_positive_rates = []

    for trial in range(num_trials):
        # Generate star field
        field = SyntheticStarField(1920, 1080, seed=seed + trial)
        stars = field.generate_stars(num_stars, min_mag=3.0, max_mag=7.0)
        image = field.render_image(stars, psf_fwhm=2.5, background=100, read_noise=5)

        # Detect stars
        detector = CentroidDetector(threshold_sigma=3.0)
        detections = detector.detect_stars(image)

        # Match to true positions
        matches = detector.match_detections(detections, stars, max_distance=3.0)

        # Compute errors
        errors = [m[2] for m in matches]
        all_errors.extend(errors)

        detection_rates.append(len(matches) / len(stars))
        false_positive_rates.append((len(detections) - len(matches)) / max(len(detections), 1))

    all_errors = np.array(all_errors)

    metrics = {
        "mean_error_pixels": float(np.mean(all_errors)),
        "rms_error_pixels": float(np.sqrt(np.mean(all_errors**2))),
        "median_error_pixels": float(np.median(all_errors)),
        "max_error_pixels": float(np.max(all_errors)),
        "detection_rate": float(np.mean(detection_rates)),
        "false_positive_rate": float(np.mean(false_positive_rates)),
        "num_trials": num_trials,
        "stars_per_trial": num_stars,
        "total_measurements": len(all_errors)
    }

    # Pass criteria
    passed = (metrics["rms_error_pixels"] < 0.5 and
              metrics["detection_rate"] > 0.90 and
              metrics["false_positive_rate"] < 0.10)

    print(f"  RMS Error: {metrics['rms_error_pixels']:.3f} pixels")
    print(f"  Detection Rate: {metrics['detection_rate']*100:.1f}%")
    print(f"  False Positive Rate: {metrics['false_positive_rate']*100:.1f}%")

    return ValidationResult(
        test_name="centroid_accuracy",
        passed=passed,
        metrics=metrics,
        notes=f"Target: RMS < 0.5 px, Detection > 90%, FP < 10%"
    )


def validate_snr_scaling(max_frames: int = 36, seed: int = 42) -> ValidationResult:
    """
    Verify that SNR improves as sqrt(N) with frame stacking.

    Measures SNR improvement for different numbers of stacked frames
    and fits to theoretical sqrt(N) model.
    """
    print("Running SNR scaling validation...")

    np.random.seed(seed)

    field = SyntheticStarField(512, 512, seed=seed)
    stars = field.generate_stars(20, min_mag=5.0, max_mag=6.0)

    # Reference (clean) image for SNR calculation
    clean_image = field.render_image(stars, add_noise=False)

    frame_counts = [1, 4, 9, 16, 25, 36]
    frame_counts = [n for n in frame_counts if n <= max_frames]

    snr_values = []
    theoretical_snr = []

    for n_frames in frame_counts:
        # Stack n frames
        stacked = np.zeros((512, 512), dtype=np.float64)
        for _ in range(n_frames):
            frame = field.render_image(stars, add_noise=True, background=100, read_noise=5)
            stacked += frame
        stacked /= n_frames

        # Measure SNR (signal / noise in background region)
        signal_region = stacked[200:300, 200:300]
        noise_region = stacked[0:50, 0:50]  # Corner without stars

        signal = np.max(signal_region) - np.median(noise_region)
        noise = np.std(noise_region)
        snr = signal / noise if noise > 0 else 0

        snr_values.append(snr)
        theoretical_snr.append(snr_values[0] * np.sqrt(n_frames))

    # Fit to sqrt(N) model
    def sqrt_model(n, a):
        return a * np.sqrt(n)

    try:
        popt, _ = curve_fit(sqrt_model, frame_counts, snr_values, p0=[snr_values[0]])
        fit_quality = np.corrcoef(snr_values, sqrt_model(np.array(frame_counts), *popt))[0, 1]
    except:
        fit_quality = 0.0
        popt = [snr_values[0]]

    metrics = {
        "frame_counts": frame_counts,
        "measured_snr": [float(s) for s in snr_values],
        "theoretical_snr": [float(s) for s in theoretical_snr],
        "sqrt_fit_coefficient": float(popt[0]),
        "fit_correlation": float(fit_quality),
        "snr_improvement_at_25_frames": float(snr_values[-1] / snr_values[0]) if len(snr_values) > 1 else 1.0,
        "theoretical_improvement_at_25_frames": np.sqrt(frame_counts[-1])
    }

    # Pass if correlation with sqrt model > 0.95
    passed = fit_quality > 0.95

    print(f"  SNR improvement at {frame_counts[-1]} frames: {metrics['snr_improvement_at_25_frames']:.2f}x")
    print(f"  Theoretical improvement: {metrics['theoretical_improvement_at_25_frames']:.2f}x")
    print(f"  Fit correlation: {fit_quality:.4f}")

    return ValidationResult(
        test_name="snr_scaling",
        passed=passed,
        metrics=metrics,
        notes=f"Verifies SNR ∝ sqrt(N). Target: correlation > 0.95"
    )


def validate_processing_performance(image_sizes: List[Tuple[int, int]] = None,
                                     num_iterations: int = 5) -> ValidationResult:
    """
    Benchmark processing performance for key operations.

    Measures time for:
    - Star detection
    - Centroid measurement
    - Frame stacking
    """
    print("Running processing performance benchmark...")

    if image_sizes is None:
        image_sizes = [(640, 480), (1280, 720), (1920, 1080)]

    results = {}

    for width, height in image_sizes:
        size_key = f"{width}x{height}"
        results[size_key] = {}

        # Generate test image
        field = SyntheticStarField(width, height, seed=42)
        stars = field.generate_stars(100)
        image = field.render_image(stars)

        # Benchmark star detection
        detector = CentroidDetector()
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = detector.detect_stars(image)
            times.append(time.perf_counter() - start)
        results[size_key]["detection_time_ms"] = float(np.mean(times) * 1000)

        # Benchmark stacking (10 frames)
        frames = [field.render_image(stars) for _ in range(10)]
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            stacked = np.mean(frames, axis=0)
            times.append(time.perf_counter() - start)
        results[size_key]["stacking_10_frames_ms"] = float(np.mean(times) * 1000)

        print(f"  {size_key}: detection={results[size_key]['detection_time_ms']:.1f}ms, "
              f"stacking={results[size_key]['stacking_10_frames_ms']:.1f}ms")

    # Calculate throughput
    hd_time = results.get("1920x1080", {}).get("detection_time_ms", 1000)
    throughput_fps = 1000 / hd_time if hd_time > 0 else 0

    metrics = {
        "benchmarks": results,
        "hd_detection_fps": float(throughput_fps),
        "num_iterations": num_iterations,
        "platform": "Python/NumPy (single-threaded)"
    }

    # Pass if HD detection < 500ms (2+ fps possible)
    passed = hd_time < 500

    return ValidationResult(
        test_name="processing_performance",
        passed=passed,
        metrics=metrics,
        notes=f"Target: HD detection < 500ms for real-time feasibility"
    )


def validate_motion_compensation(seed: int = 42) -> ValidationResult:
    """
    Validate motion compensation effectiveness.

    Simulates camera motion and measures improvement after compensation.
    """
    print("Running motion compensation validation...")

    np.random.seed(seed)

    field = SyntheticStarField(640, 480, seed=seed)
    stars = field.generate_stars(30, min_mag=4.0, max_mag=6.0)

    # Generate reference (stationary) image
    reference = field.render_image(stars, add_noise=True)

    # Simulate motion blur (shift stars during exposure)
    motion_pixels = 15  # pixels of motion
    num_subframes = 20

    blurred = np.zeros_like(reference)
    for i in range(num_subframes):
        # Shift stars progressively
        shift = motion_pixels * i / num_subframes
        shifted_stars = [Star(s.x + shift, s.y + shift * 0.3, s.magnitude) for s in stars]
        frame = field.render_image(shifted_stars, add_noise=True, background=100, read_noise=3)
        blurred += frame
    blurred /= num_subframes

    # Simulate compensation (shift back to reference)
    compensated = np.zeros_like(reference)
    for i in range(num_subframes):
        shift = motion_pixels * i / num_subframes
        shifted_stars = [Star(s.x + shift, s.y + shift * 0.3, s.magnitude) for s in stars]
        frame = field.render_image(shifted_stars, add_noise=True, background=100, read_noise=3)

        # Apply inverse shift (simulating gyro-based compensation)
        from scipy.ndimage import shift as ndshift
        compensated_frame = ndshift(frame, (-shift * 0.3, -shift), mode='constant', cval=100)
        compensated += compensated_frame
    compensated /= num_subframes

    # Measure star FWHM in each image
    def measure_fwhm(image, stars):
        fwhms = []
        for star in stars[:10]:  # Sample 10 stars
            x, y = int(star.x), int(star.y)
            if 10 < x < image.shape[1]-10 and 10 < y < image.shape[0]-10:
                stamp = image[y-5:y+6, x-5:x+6]
                profile = stamp[5, :]  # Horizontal profile
                half_max = (profile.max() + profile.min()) / 2
                above_half = profile > half_max
                if np.sum(above_half) > 0:
                    fwhm = np.sum(above_half)
                    fwhms.append(fwhm)
        return np.mean(fwhms) if fwhms else 10.0

    fwhm_reference = measure_fwhm(reference, stars)
    fwhm_blurred = measure_fwhm(blurred, stars)
    fwhm_compensated = measure_fwhm(compensated, stars)

    # Measure detection rates
    detector = CentroidDetector(threshold_sigma=3.0)
    det_reference = len(detector.detect_stars(reference))
    det_blurred = len(detector.detect_stars(blurred))
    det_compensated = len(detector.detect_stars(compensated))

    metrics = {
        "motion_pixels": motion_pixels,
        "fwhm_reference": float(fwhm_reference),
        "fwhm_blurred": float(fwhm_blurred),
        "fwhm_compensated": float(fwhm_compensated),
        "fwhm_improvement_ratio": float(fwhm_blurred / fwhm_compensated) if fwhm_compensated > 0 else 1.0,
        "detections_reference": det_reference,
        "detections_blurred": det_blurred,
        "detections_compensated": det_compensated,
        "detection_recovery_rate": float(det_compensated / det_reference) if det_reference > 0 else 0.0
    }

    print(f"  FWHM - Reference: {fwhm_reference:.2f}, Blurred: {fwhm_blurred:.2f}, Compensated: {fwhm_compensated:.2f}")
    print(f"  Detections - Reference: {det_reference}, Blurred: {det_blurred}, Compensated: {det_compensated}")

    # Pass if compensation recovers >80% of detections
    passed = metrics["detection_recovery_rate"] > 0.80

    return ValidationResult(
        test_name="motion_compensation",
        passed=passed,
        metrics=metrics,
        notes=f"Target: >80% detection recovery after compensation"
    )


def run_all_validations(output_dir: str = "validation/results") -> Dict:
    """Run all validation tests and save results."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LOW-COST STAR TRACKER VALIDATION SUITE")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(validate_centroid_accuracy())
    results.append(validate_snr_scaling())
    results.append(validate_processing_performance())
    results.append(validate_motion_compensation())

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)

    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"  {r.test_name}: {status}")

    print(f"\nOverall: {passed_count}/{total_count} tests passed")

    # Save results
    output_data = {
        "summary": {
            "passed": passed_count,
            "total": total_count,
            "pass_rate": float(passed_count / total_count),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "tests": [
            {
                "name": r.test_name,
                "passed": bool(r.passed),
                "metrics": r.metrics,
                "notes": r.notes,
                "timestamp": r.timestamp
            }
            for r in results
        ]
    }

    with open(output_path / "validation_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path / 'validation_results.json'}")

    return output_data


if __name__ == "__main__":
    run_all_validations()

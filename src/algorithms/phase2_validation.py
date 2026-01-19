#!/usr/bin/env python3
"""
Phase 2 Algorithm Validation Suite.

Validates the three Phase 2 algorithm enhancements:
1. Gyroscope drift compensation
2. Optical calibration
3. Triangle matching robustness

Generates validation plots and metrics for documentation.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.algorithms.drift_compensation import (
    DriftCompensator, GyroParameters, StarMatch as DriftStarMatch
)
from src.algorithms.optical_calibration import (
    OpticalCalibrator, CalibrationConfig
)
from src.algorithms.triangle_matching import (
    FalseStarFilter, DetectedStar, ConfidenceMetrics
)


def validate_drift_compensation() -> dict:
    """
    Validate gyroscope drift compensation algorithm.

    Tests:
    - Bias estimation accuracy
    - Attitude error reduction
    - Long-term stability
    """
    print("\n" + "=" * 60)
    print("Validating Drift Compensation")
    print("=" * 60)

    results = {
        "test": "drift_compensation",
        "passed": False,
        "metrics": {}
    }

    # Initialize compensator
    compensator = DriftCompensator(
        gyro_params=GyroParameters(
            arw=0.5,
            bias_instability=5.0,
            initial_bias=np.zeros(3)
        ),
        update_interval=1.0
    )

    # Simulation parameters
    duration = 120.0  # seconds
    dt = 0.01  # 100 Hz
    star_update_interval = 5.0

    # True bias (unknown to filter)
    true_bias = np.array([0.001, -0.002, 0.0005])  # rad/s

    # Run simulation
    t = 0.0
    last_star_time = 0.0
    true_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    errors_without_comp = []
    errors_with_comp = []
    uncorrected_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    while t < duration:
        # True rotation rate
        true_omega = np.array([0.01, 0.005, -0.002])
        noise = np.random.normal(0, 0.001, 3)
        gyro_measurement = true_omega + true_bias + noise

        # Update true quaternion
        omega_quat = np.array([0, true_omega[0], true_omega[1], true_omega[2]])
        q_dot = 0.5 * compensator._quaternion_multiply(true_quaternion, omega_quat)
        true_quaternion = true_quaternion + q_dot * dt
        true_quaternion = true_quaternion / np.linalg.norm(true_quaternion)

        # Update uncorrected estimate (no bias compensation)
        omega_quat_biased = np.array([0, gyro_measurement[0], gyro_measurement[1], gyro_measurement[2]])
        q_dot_biased = 0.5 * compensator._quaternion_multiply(uncorrected_quaternion, omega_quat_biased)
        uncorrected_quaternion = uncorrected_quaternion + q_dot_biased * dt
        uncorrected_quaternion = uncorrected_quaternion / np.linalg.norm(uncorrected_quaternion)

        # Predict with compensator
        compensator.predict(gyro_measurement, dt)

        # Periodic star updates
        if t - last_star_time >= star_update_interval:
            n_stars = np.random.randint(5, 15)
            star_matches = []

            from scipy.spatial.transform import Rotation

            for _ in range(n_stars):
                catalog_dir = np.random.randn(3)
                catalog_dir = catalog_dir / np.linalg.norm(catalog_dir)

                R_true = Rotation.from_quat([
                    true_quaternion[1], true_quaternion[2],
                    true_quaternion[3], true_quaternion[0]
                ]).as_matrix()

                cam_dir = R_true @ catalog_dir
                if cam_dir[2] > 0.1:
                    pixel = compensator.K @ cam_dir
                    pixel = pixel[:2] / pixel[2]
                    pixel += np.random.normal(0, 0.5, 2)

                    star_matches.append(DriftStarMatch(
                        detected_x=pixel[0],
                        detected_y=pixel[1],
                        catalog_direction=catalog_dir,
                        timestamp=t,
                        confidence=1.0
                    ))

            compensator.update_with_stars(star_matches, t)
            last_star_time = t

        # Compute errors
        # Uncorrected error
        q_err_uncorr = compensator._quaternion_multiply(
            true_quaternion,
            compensator._quaternion_conjugate(uncorrected_quaternion)
        )
        err_uncorr = 2.0 * np.arcsin(np.linalg.norm(q_err_uncorr[1:4]))
        errors_without_comp.append(np.rad2deg(err_uncorr))

        # Corrected error
        q_err_corr = compensator._quaternion_multiply(
            true_quaternion,
            compensator._quaternion_conjugate(compensator.quaternion)
        )
        err_corr = 2.0 * np.arcsin(np.linalg.norm(q_err_corr[1:4]))
        errors_with_comp.append(np.rad2deg(err_corr))

        t += dt

    # Analyze results
    stats = compensator.get_drift_statistics()

    final_error_uncorrected = errors_without_comp[-1]
    final_error_corrected = errors_with_comp[-1]

    bias_error = np.linalg.norm(
        compensator.gyro_bias - true_bias
    )

    improvement_factor = final_error_uncorrected / max(final_error_corrected, 0.001)

    results["metrics"] = {
        "duration_s": duration,
        "n_corrections": stats['n_corrections'],
        "final_error_uncorrected_deg": float(final_error_uncorrected),
        "final_error_corrected_deg": float(final_error_corrected),
        "mean_error_corrected_deg": float(np.mean(errors_with_comp)),
        "max_error_corrected_deg": float(max(errors_with_comp)),
        "bias_estimation_error_rad_s": float(bias_error),
        "true_bias_deg_s": np.rad2deg(true_bias).tolist(),
        "estimated_bias_deg_s": np.rad2deg(compensator.gyro_bias).tolist(),
        "improvement_factor": float(improvement_factor),
        "errors_with_comp": errors_with_comp[::100],  # Subsample for storage
        "errors_without_comp": errors_without_comp[::100]
    }

    # Pass criteria (1.5° threshold realistic for MEMS gyroscopes)
    passed = (
        final_error_corrected < 1.5 and  # Less than 1.5 degree final error
        improvement_factor > 5.0  # At least 5x improvement
    )
    results["passed"] = passed

    print(f"\nResults:")
    print(f"  Final error (uncorrected): {final_error_uncorrected:.2f}°")
    print(f"  Final error (corrected): {final_error_corrected:.4f}°")
    print(f"  Improvement factor: {improvement_factor:.1f}x")
    print(f"  Bias estimation error: {np.rad2deg(bias_error)*3600:.2f} arcsec/s")
    print(f"  Test passed: {passed}")

    return results


def validate_optical_calibration() -> dict:
    """
    Validate optical calibration pipeline.

    Tests:
    - Dark frame subtraction effectiveness
    - Flat field correction accuracy
    - Hot pixel identification
    """
    print("\n" + "=" * 60)
    print("Validating Optical Calibration")
    print("=" * 60)

    results = {
        "test": "optical_calibration",
        "passed": False,
        "metrics": {}
    }

    # Image parameters
    height, width = 540, 960  # Reduced for faster testing

    # Generate synthetic dark frames
    print("Generating synthetic calibration frames...")

    # True hot pixel locations
    n_hot = 100
    hot_y = np.random.randint(0, height, n_hot)
    hot_x = np.random.randint(0, width, n_hot)
    hot_values = np.random.uniform(500, 2000, n_hot)

    dark_frames = []
    for _ in range(10):
        dark = np.random.poisson(100, (height, width)).astype(np.float64)
        dark[hot_y, hot_x] = hot_values
        dark += np.random.normal(0, 5, dark.shape)
        dark_frames.append(dark)

    # Generate synthetic flat frames with vignetting
    cy, cx = height / 2, width / 2
    y, x = np.ogrid[:height, :width]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_max = np.sqrt(cx**2 + cy**2)
    true_vignette = 1.0 - 0.3 * (r / r_max)**2

    flat_frames = []
    for _ in range(10):
        flat = np.random.poisson(30000, (height, width)).astype(np.float64)
        flat = flat * true_vignette
        flat += np.random.normal(0, 50, flat.shape)
        flat_frames.append(flat)

    # Create calibrator
    calibrator = OpticalCalibrator(CalibrationConfig(hot_pixel_threshold=5.0))
    calibrator.calibrate_from_frames(dark_frames, flat_frames)

    # Generate test image
    print("Testing calibration...")
    test_image = np.random.poisson(500, (height, width)).astype(np.float64)

    # Add stars
    n_stars = 50
    star_positions = []
    for _ in range(n_stars):
        star_y = np.random.randint(50, height - 50)
        star_x = np.random.randint(50, width - 50)
        flux = np.random.uniform(5000, 50000)
        star_positions.append((star_y, star_x, flux))

        sigma = 2.5
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if 0 <= star_y + dy < height and 0 <= star_x + dx < width:
                    val = flux * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                    test_image[star_y + dy, star_x + dx] += val

    # Apply vignetting and dark
    test_image = test_image * true_vignette
    test_image += dark_frames[0]

    # Apply calibration
    calibrated = calibrator.apply_calibration(test_image)

    # Measure vignetting correction
    center_before = np.mean(test_image[height//2-25:height//2+25, width//2-25:width//2+25])
    corner_before = np.mean(test_image[25:75, 25:75])
    ratio_before = center_before / corner_before

    center_after = np.mean(calibrated[height//2-25:height//2+25, width//2-25:width//2+25])
    corner_after = np.mean(calibrated[25:75, 25:75])
    ratio_after = center_after / max(corner_after, 1)

    # Measure hot pixel correction
    hot_pixel_mask = calibrator.calibration_data.hot_pixel_map
    hot_detected = np.sum(hot_pixel_mask) if hot_pixel_mask is not None else 0

    hot_before_mean = np.mean(test_image[hot_y, hot_x]) if n_hot > 0 else 0
    hot_after_mean = np.mean(calibrated[hot_y, hot_x]) if n_hot > 0 else 0

    results["metrics"] = {
        "image_size": [height, width],
        "n_dark_frames": 10,
        "n_flat_frames": 10,
        "hot_pixels_true": n_hot,
        "hot_pixels_detected": int(hot_detected),
        "hot_pixel_detection_rate": float(hot_detected / n_hot) if n_hot > 0 else 0,
        "vignetting_ratio_before": float(ratio_before),
        "vignetting_ratio_after": float(ratio_after),
        "vignetting_correction_accuracy": float(1 - abs(ratio_after - 1)),
        "hot_pixel_mean_before": float(hot_before_mean),
        "hot_pixel_mean_after": float(hot_after_mean),
        "hot_pixel_reduction": float(1 - hot_after_mean / max(hot_before_mean, 1))
    }

    # Pass criteria
    passed = (
        abs(ratio_after - 1.0) < 0.1 and  # Within 10% uniformity
        hot_detected >= n_hot * 0.8  # Detect at least 80% of hot pixels
    )
    results["passed"] = passed

    print(f"\nResults:")
    print(f"  Vignetting ratio before: {ratio_before:.3f}")
    print(f"  Vignetting ratio after: {ratio_after:.3f}")
    print(f"  Hot pixels detected: {hot_detected}/{n_hot} ({100*hot_detected/n_hot:.1f}%)")
    print(f"  Hot pixel mean before: {hot_before_mean:.1f}")
    print(f"  Hot pixel mean after: {hot_after_mean:.1f}")
    print(f"  Test passed: {passed}")

    return results


def validate_false_star_rejection() -> dict:
    """
    Validate false star rejection capabilities.

    Tests:
    - Cosmic ray rejection
    - Hot pixel filtering
    - Satellite/trail detection
    - SNR filtering
    """
    print("\n" + "=" * 60)
    print("Validating False Star Rejection")
    print("=" * 60)

    results = {
        "test": "false_star_rejection",
        "passed": False,
        "metrics": {}
    }

    # Create mix of real and false detections
    np.random.seed(42)

    detections = []
    labels = []  # True = real star, False = false detection

    # Add real stars
    n_real = 30
    for i in range(n_real):
        det = DetectedStar(
            x=np.random.uniform(100, 1820),
            y=np.random.uniform(100, 980),
            flux=np.random.uniform(5000, 50000),
            snr=np.random.uniform(20, 100),
            fwhm=np.random.uniform(2.5, 4.0),
            elongation=np.random.uniform(1.0, 1.3),
            peak_value=np.random.uniform(1000, 10000)
        )
        detections.append(det)
        labels.append(True)

    # Add cosmic rays (sharp, high peak)
    n_cosmic = 10
    for i in range(n_cosmic):
        det = DetectedStar(
            x=np.random.uniform(100, 1820),
            y=np.random.uniform(100, 980),
            flux=np.random.uniform(1000, 5000),
            snr=np.random.uniform(10, 50),
            fwhm=np.random.uniform(0.5, 1.2),  # Too sharp
            elongation=np.random.uniform(1.0, 3.0),
            peak_value=np.random.uniform(5000, 20000)
        )
        detections.append(det)
        labels.append(False)

    # Add noise spikes (low SNR)
    n_noise = 8
    for i in range(n_noise):
        det = DetectedStar(
            x=np.random.uniform(100, 1820),
            y=np.random.uniform(100, 980),
            flux=np.random.uniform(100, 500),
            snr=np.random.uniform(2, 4),  # Too low SNR
            fwhm=np.random.uniform(2, 5),
            elongation=np.random.uniform(1.0, 1.5),
            peak_value=np.random.uniform(50, 200)
        )
        detections.append(det)
        labels.append(False)

    # Add satellite trails (elongated)
    n_satellite = 5
    for i in range(n_satellite):
        det = DetectedStar(
            x=np.random.uniform(100, 1820),
            y=np.random.uniform(100, 980),
            flux=np.random.uniform(2000, 10000),
            snr=np.random.uniform(15, 40),
            fwhm=np.random.uniform(3, 6),
            elongation=np.random.uniform(2.5, 5.0),  # Too elongated
            peak_value=np.random.uniform(500, 2000)
        )
        detections.append(det)
        labels.append(False)

    # Initialize filter
    false_filter = FalseStarFilter(
        min_snr=5.0,
        max_elongation=2.0,
        min_fwhm=1.5,
        max_fwhm=10.0
    )

    # Apply filter
    valid, rejected = false_filter.filter_detections(detections, (1080, 1920))

    # Compute metrics
    n_total = len(detections)
    n_false = n_cosmic + n_noise + n_satellite

    # Count correct rejections and false negatives
    valid_indices = set()
    for v in valid:
        for i, d in enumerate(detections):
            if v.x == d.x and v.y == d.y:
                valid_indices.add(i)
                break

    true_positives = sum(1 for i in valid_indices if labels[i])  # Real stars kept
    false_positives = sum(1 for i in valid_indices if not labels[i])  # False stars kept
    true_negatives = sum(1 for i in range(n_total) if i not in valid_indices and not labels[i])  # False stars rejected
    false_negatives = sum(1 for i in range(n_total) if i not in valid_indices and labels[i])  # Real stars rejected

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 0.001)

    false_rejection_rate = false_positives / max(n_false, 1)

    results["metrics"] = {
        "n_total_detections": n_total,
        "n_real_stars": n_real,
        "n_false_detections": n_false,
        "n_cosmic_rays": n_cosmic,
        "n_noise_spikes": n_noise,
        "n_satellites": n_satellite,
        "n_valid_output": len(valid),
        "n_rejected_output": len(rejected),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "false_rejection_rate": float(false_rejection_rate)
    }

    # Pass criteria
    passed = (
        precision > 0.85 and  # At least 85% of kept stars are real
        recall > 0.90 and  # Keep at least 90% of real stars
        true_negatives >= n_false * 0.7  # Reject at least 70% of false stars
    )
    results["passed"] = passed

    print(f"\nResults:")
    print(f"  Input: {n_real} real stars + {n_false} false detections")
    print(f"  Output: {len(valid)} valid, {len(rejected)} rejected")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1_score:.3f}")
    print(f"  False detections rejected: {true_negatives}/{n_false}")
    print(f"  Test passed: {passed}")

    return results


def validate_confidence_metrics() -> dict:
    """
    Validate confidence metric calculations.
    """
    print("\n" + "=" * 60)
    print("Validating Confidence Metrics")
    print("=" * 60)

    results = {
        "test": "confidence_metrics",
        "passed": False,
        "metrics": {}
    }

    # Test photometric consistency
    from src.algorithms.triangle_matching import StarMatch, CatalogStar

    # Create consistent matches (brighter flux = lower magnitude)
    consistent_matches = []
    for i in range(10):
        mag = 3.0 + i * 0.5  # Increasing magnitude (dimmer)
        flux = 10000 / (i + 1)  # Decreasing flux

        consistent_matches.append(StarMatch(
            detected=DetectedStar(
                x=100 + i * 50, y=100 + i * 30,
                flux=flux, snr=50, fwhm=3.0, elongation=1.0
            ),
            catalog=CatalogStar(
                hip_id=i, ra=0, dec=0, magnitude=mag,
                unit_vector=np.array([0, 0, 1])
            ),
            confidence=0.9,
            residual=0.5
        ))

    # Create inconsistent matches (random flux vs magnitude)
    inconsistent_matches = []
    np.random.seed(42)
    for i in range(10):
        mag = 3.0 + np.random.uniform(0, 4)
        flux = np.random.uniform(1000, 10000)

        inconsistent_matches.append(StarMatch(
            detected=DetectedStar(
                x=100 + i * 50, y=100 + i * 30,
                flux=flux, snr=50, fwhm=3.0, elongation=1.0
            ),
            catalog=CatalogStar(
                hip_id=i, ra=0, dec=0, magnitude=mag,
                unit_vector=np.array([0, 0, 1])
            ),
            confidence=0.9,
            residual=0.5
        ))

    # Calculate photometric consistency
    consistent_score = ConfidenceMetrics.photometric_consistency(consistent_matches)
    inconsistent_score = ConfidenceMetrics.photometric_consistency(inconsistent_matches)

    # Test coverage score
    spread_matches = []
    for i in range(10):
        spread_matches.append(StarMatch(
            detected=DetectedStar(
                x=100 + i * 170,  # Spread across image
                y=100 + (i % 5) * 200,
                flux=5000, snr=50, fwhm=3.0, elongation=1.0
            ),
            catalog=CatalogStar(
                hip_id=i, ra=0, dec=0, magnitude=5.0,
                unit_vector=np.array([0, 0, 1])
            ),
            confidence=0.9,
            residual=0.5
        ))

    clustered_matches = []
    for i in range(10):
        clustered_matches.append(StarMatch(
            detected=DetectedStar(
                x=960 + np.random.uniform(-50, 50),  # Clustered at center
                y=540 + np.random.uniform(-50, 50),
                flux=5000, snr=50, fwhm=3.0, elongation=1.0
            ),
            catalog=CatalogStar(
                hip_id=i, ra=0, dec=0, magnitude=5.0,
                unit_vector=np.array([0, 0, 1])
            ),
            confidence=0.9,
            residual=0.5
        ))

    spread_coverage = ConfidenceMetrics.coverage_score(spread_matches, (1080, 1920))
    clustered_coverage = ConfidenceMetrics.coverage_score(clustered_matches, (1080, 1920))

    results["metrics"] = {
        "photometric_consistent_score": float(consistent_score),
        "photometric_inconsistent_score": float(inconsistent_score),
        "photometric_discrimination": float(consistent_score - inconsistent_score),
        "coverage_spread_score": float(spread_coverage),
        "coverage_clustered_score": float(clustered_coverage),
        "coverage_discrimination": float(spread_coverage - clustered_coverage)
    }

    # Pass criteria
    passed = (
        consistent_score > inconsistent_score and  # Photometric discrimination works
        spread_coverage > clustered_coverage  # Coverage discrimination works
    )
    results["passed"] = passed

    print(f"\nResults:")
    print(f"  Photometric score (consistent): {consistent_score:.3f}")
    print(f"  Photometric score (inconsistent): {inconsistent_score:.3f}")
    print(f"  Coverage score (spread): {spread_coverage:.3f}")
    print(f"  Coverage score (clustered): {clustered_coverage:.3f}")
    print(f"  Test passed: {passed}")

    return results


def run_all_validations(output_dir: str = "validation/results") -> dict:
    """Run all Phase 2 validations and save results."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 2 Algorithm Validation Suite")
    print("=" * 60)
    print(f"Output directory: {output_path}")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "phase": 2,
        "tests": []
    }

    # Run all validations
    tests = [
        validate_drift_compensation,
        validate_optical_calibration,
        validate_false_star_rejection,
        validate_confidence_metrics
    ]

    for test_func in tests:
        try:
            result = test_func()
            all_results["tests"].append(result)
        except Exception as e:
            print(f"Error in {test_func.__name__}: {e}")
            all_results["tests"].append({
                "test": test_func.__name__,
                "passed": False,
                "error": str(e)
            })

    # Summary
    passed_count = sum(1 for t in all_results["tests"] if t.get("passed", False))
    total_count = len(all_results["tests"])

    all_results["summary"] = {
        "total_tests": total_count,
        "passed": passed_count,
        "failed": total_count - passed_count,
        "pass_rate": passed_count / total_count if total_count > 0 else 0
    }

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for test in all_results["tests"]:
        status = "PASS" if test.get("passed", False) else "FAIL"
        print(f"  {test['test']}: {status}")

    print(f"\nOverall: {passed_count}/{total_count} tests passed")

    # Save results
    results_file = output_path / "phase2_validation_results.json"
    with open(results_file, 'w') as f:
        # Handle numpy types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [deep_convert(i) for i in obj]
            return convert(obj)

        json.dump(deep_convert(all_results), f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    run_all_validations()

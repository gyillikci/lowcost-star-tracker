#!/usr/bin/env python3
"""
Optical Calibration Module for Low-Cost Star Tracker.

This module provides calibration routines for consumer camera optics:
- Dark frame subtraction (hot pixel removal, thermal noise)
- Flat field correction (vignetting, sensor non-uniformity)
- Bad pixel mapping and interpolation
- Temperature-dependent calibration

These corrections improve photometric accuracy and star detection reliability.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from scipy.ndimage import median_filter, gaussian_filter
from pathlib import Path
import json


@dataclass
class CalibrationConfig:
    """Configuration for optical calibration."""
    # Dark frame settings
    dark_frame_count: int = 10  # Number of dark frames to average
    hot_pixel_threshold: float = 5.0  # Sigma threshold for hot pixels

    # Flat field settings
    flat_smoothing_sigma: float = 50.0  # Gaussian smoothing for flat
    vignetting_model: str = "radial"  # 'radial', 'polynomial', or 'measured'

    # Bad pixel settings
    bad_pixel_max_neighbors: int = 2  # Max bad neighbors for interpolation

    # Temperature compensation
    temp_coefficient: float = 0.1  # Dark current increase per degree C


@dataclass
class CalibrationData:
    """Container for calibration data."""
    master_dark: Optional[np.ndarray] = None
    master_flat: Optional[np.ndarray] = None
    bad_pixel_mask: Optional[np.ndarray] = None
    hot_pixel_map: Optional[np.ndarray] = None
    vignetting_model: Optional[np.ndarray] = None
    reference_temperature: float = 20.0  # °C
    metadata: Dict = field(default_factory=dict)


class DarkFrameCalibrator:
    """
    Creates and applies dark frame calibration.

    Dark frames capture:
    - Hot pixels (defective pixels with high dark current)
    - Thermal noise pattern
    - Bias/offset pattern
    """

    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()
        self.master_dark = None
        self.dark_std = None
        self.hot_pixel_mask = None

    def create_master_dark(self, dark_frames: List[np.ndarray],
                            temperature: float = 20.0) -> np.ndarray:
        """
        Create master dark frame from multiple dark exposures.

        Args:
            dark_frames: List of dark frame images
            temperature: Sensor temperature during capture (°C)

        Returns:
            Master dark frame (median combined)
        """
        if len(dark_frames) < 3:
            raise ValueError("Need at least 3 dark frames for reliable master")

        # Stack frames
        stack = np.stack(dark_frames, axis=0).astype(np.float64)

        # Median combine (robust to cosmic rays)
        self.master_dark = np.median(stack, axis=0)

        # Compute noise level
        self.dark_std = np.std(stack, axis=0)

        # Identify hot pixels
        median_level = np.median(self.master_dark)
        mad = np.median(np.abs(self.master_dark - median_level))
        sigma = 1.4826 * mad  # Robust sigma estimate

        threshold = median_level + self.config.hot_pixel_threshold * sigma
        self.hot_pixel_mask = self.master_dark > threshold

        hot_count = np.sum(self.hot_pixel_mask)
        total_pixels = self.master_dark.size
        print(f"Master dark created: {hot_count} hot pixels "
              f"({100*hot_count/total_pixels:.3f}%)")

        return self.master_dark

    def apply_dark_correction(self, image: np.ndarray,
                               temperature: float = None) -> np.ndarray:
        """
        Apply dark frame correction to an image.

        Args:
            image: Input image
            temperature: Current sensor temperature (for scaling)

        Returns:
            Dark-subtracted image
        """
        if self.master_dark is None:
            raise RuntimeError("Master dark not created. Call create_master_dark first.")

        # Scale dark for temperature if provided
        dark = self.master_dark.copy()
        if temperature is not None:
            temp_factor = 1.0 + self.config.temp_coefficient * (temperature - 20.0)
            dark = dark * temp_factor

        # Subtract dark
        corrected = image.astype(np.float64) - dark

        return corrected

    def get_hot_pixel_mask(self) -> np.ndarray:
        """Get mask of hot pixels (True = hot)."""
        return self.hot_pixel_mask


class FlatFieldCalibrator:
    """
    Creates and applies flat field calibration.

    Flat fields correct for:
    - Vignetting (brightness falloff toward edges)
    - Dust/debris on sensor or optics
    - Pixel-to-pixel sensitivity variations
    """

    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()
        self.master_flat = None
        self.normalized_flat = None

    def create_master_flat(self, flat_frames: List[np.ndarray],
                            dark_frame: np.ndarray = None) -> np.ndarray:
        """
        Create master flat field from multiple flat exposures.

        Args:
            flat_frames: List of flat field images
            dark_frame: Master dark to subtract (optional)

        Returns:
            Master flat field (normalized)
        """
        if len(flat_frames) < 3:
            raise ValueError("Need at least 3 flat frames for reliable master")

        # Stack and median combine
        stack = np.stack(flat_frames, axis=0).astype(np.float64)

        # Subtract dark if provided
        if dark_frame is not None:
            stack = stack - dark_frame

        # Median combine
        self.master_flat = np.median(stack, axis=0)

        # Normalize to mean = 1.0
        mean_level = np.mean(self.master_flat)
        self.normalized_flat = self.master_flat / mean_level

        # Prevent division by zero
        self.normalized_flat = np.maximum(self.normalized_flat, 0.1)

        vignetting = 1.0 - np.min(self.normalized_flat)
        print(f"Master flat created: {vignetting*100:.1f}% vignetting")

        return self.normalized_flat

    def create_synthetic_flat(self, shape: Tuple[int, int],
                               vignetting_strength: float = 0.3) -> np.ndarray:
        """
        Create synthetic flat field model for vignetting correction.

        Useful when actual flat frames are not available.

        Args:
            shape: Image shape (height, width)
            vignetting_strength: Relative brightness drop at corners (0-1)

        Returns:
            Synthetic flat field
        """
        height, width = shape
        cy, cx = height / 2, width / 2

        # Create radial distance map
        y, x = np.ogrid[:height, :width]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)

        # Normalize radius to corner distance
        r_max = np.sqrt(cx**2 + cy**2)
        r_norm = r / r_max

        # Vignetting model: cos^4 law approximation
        # V(r) = 1 - strength * r^2
        self.normalized_flat = 1.0 - vignetting_strength * r_norm**2

        print(f"Synthetic flat created: {vignetting_strength*100:.0f}% corner vignetting")

        return self.normalized_flat

    def apply_flat_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply flat field correction to an image.

        Args:
            image: Input image (should be dark-subtracted)

        Returns:
            Flat-corrected image
        """
        if self.normalized_flat is None:
            raise RuntimeError("Flat field not created. Call create_master_flat first.")

        # Divide by normalized flat
        corrected = image.astype(np.float64) / self.normalized_flat

        return corrected


class BadPixelCorrector:
    """
    Identifies and corrects bad pixels (hot, dead, stuck).
    """

    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()
        self.bad_pixel_mask = None

    def create_bad_pixel_mask(self,
                               hot_pixel_mask: np.ndarray = None,
                               dead_pixel_mask: np.ndarray = None,
                               custom_mask: np.ndarray = None) -> np.ndarray:
        """
        Create combined bad pixel mask.

        Args:
            hot_pixel_mask: Mask of hot pixels
            dead_pixel_mask: Mask of dead (unresponsive) pixels
            custom_mask: Additional custom mask

        Returns:
            Combined bad pixel mask
        """
        shape = None

        if hot_pixel_mask is not None:
            shape = hot_pixel_mask.shape
            self.bad_pixel_mask = hot_pixel_mask.copy()
        else:
            self.bad_pixel_mask = None

        if dead_pixel_mask is not None:
            if self.bad_pixel_mask is None:
                self.bad_pixel_mask = dead_pixel_mask.copy()
                shape = dead_pixel_mask.shape
            else:
                self.bad_pixel_mask = self.bad_pixel_mask | dead_pixel_mask

        if custom_mask is not None:
            if self.bad_pixel_mask is None:
                self.bad_pixel_mask = custom_mask.copy()
                shape = custom_mask.shape
            else:
                self.bad_pixel_mask = self.bad_pixel_mask | custom_mask

        if self.bad_pixel_mask is None:
            raise ValueError("At least one mask must be provided")

        bad_count = np.sum(self.bad_pixel_mask)
        print(f"Bad pixel mask: {bad_count} pixels "
              f"({100*bad_count/self.bad_pixel_mask.size:.3f}%)")

        return self.bad_pixel_mask

    def correct_bad_pixels(self, image: np.ndarray,
                            method: str = "median") -> np.ndarray:
        """
        Correct bad pixels by interpolation from neighbors.

        Args:
            image: Input image
            method: Interpolation method ('median', 'mean', 'bilinear')

        Returns:
            Corrected image
        """
        if self.bad_pixel_mask is None:
            return image

        corrected = image.copy().astype(np.float64)

        if method == "median":
            # Replace bad pixels with local median
            filtered = median_filter(image, size=3)
            corrected[self.bad_pixel_mask] = filtered[self.bad_pixel_mask]

        elif method == "mean":
            # Replace with local mean (excluding bad pixels)
            kernel_size = 3
            pad = kernel_size // 2

            # Pad image
            padded = np.pad(corrected, pad, mode='reflect')
            padded_mask = np.pad(self.bad_pixel_mask, pad, mode='constant', constant_values=True)

            bad_coords = np.where(self.bad_pixel_mask)

            for y, x in zip(bad_coords[0], bad_coords[1]):
                # Extract neighborhood
                neighborhood = padded[y:y+kernel_size, x:x+kernel_size]
                mask_neighborhood = padded_mask[y:y+kernel_size, x:x+kernel_size]

                # Mean of good neighbors
                good_values = neighborhood[~mask_neighborhood]
                if len(good_values) > 0:
                    corrected[y, x] = np.mean(good_values)

        elif method == "bilinear":
            # Bilinear interpolation from 4 nearest good neighbors
            # Simplified: use scipy's approach
            from scipy.ndimage import binary_dilation

            # Dilate bad pixel mask to find boundary pixels
            dilated = binary_dilation(self.bad_pixel_mask, iterations=1)
            boundary = dilated & ~self.bad_pixel_mask

            # Use boundary values to interpolate
            # This is a simplified approach
            filtered = gaussian_filter(image.astype(np.float64), sigma=1)
            corrected[self.bad_pixel_mask] = filtered[self.bad_pixel_mask]

        return corrected


class OpticalCalibrator:
    """
    Complete optical calibration pipeline.

    Combines dark, flat, and bad pixel corrections.
    """

    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()

        self.dark_calibrator = DarkFrameCalibrator(config)
        self.flat_calibrator = FlatFieldCalibrator(config)
        self.bad_pixel_corrector = BadPixelCorrector(config)

        self.calibration_data = CalibrationData()

    def calibrate_from_frames(self,
                               dark_frames: List[np.ndarray],
                               flat_frames: List[np.ndarray] = None,
                               temperature: float = 20.0):
        """
        Create calibration data from captured frames.

        Args:
            dark_frames: List of dark frame images
            flat_frames: List of flat field images (optional)
            temperature: Reference temperature (°C)
        """
        # Create master dark
        self.calibration_data.master_dark = self.dark_calibrator.create_master_dark(
            dark_frames, temperature
        )
        self.calibration_data.hot_pixel_map = self.dark_calibrator.get_hot_pixel_mask()
        self.calibration_data.reference_temperature = temperature

        # Create master flat
        if flat_frames is not None:
            self.calibration_data.master_flat = self.flat_calibrator.create_master_flat(
                flat_frames, self.calibration_data.master_dark
            )
        else:
            # Create synthetic flat based on typical vignetting
            shape = self.calibration_data.master_dark.shape
            self.calibration_data.master_flat = self.flat_calibrator.create_synthetic_flat(
                shape, vignetting_strength=0.25
            )

        # Create bad pixel mask
        self.calibration_data.bad_pixel_mask = self.bad_pixel_corrector.create_bad_pixel_mask(
            hot_pixel_mask=self.calibration_data.hot_pixel_map
        )

        self.calibration_data.metadata = {
            "n_dark_frames": len(dark_frames),
            "n_flat_frames": len(flat_frames) if flat_frames else 0,
            "reference_temp": temperature,
            "hot_pixel_count": int(np.sum(self.calibration_data.hot_pixel_map)),
            "bad_pixel_count": int(np.sum(self.calibration_data.bad_pixel_mask))
        }

    def apply_calibration(self, image: np.ndarray,
                           temperature: float = None) -> np.ndarray:
        """
        Apply full calibration pipeline to an image.

        Args:
            image: Raw input image
            temperature: Current sensor temperature (optional)

        Returns:
            Calibrated image
        """
        # 1. Dark subtraction
        calibrated = self.dark_calibrator.apply_dark_correction(image, temperature)

        # 2. Flat field correction
        calibrated = self.flat_calibrator.apply_flat_correction(calibrated)

        # 3. Bad pixel correction
        calibrated = self.bad_pixel_corrector.correct_bad_pixels(calibrated)

        # 4. Clip negative values
        calibrated = np.maximum(calibrated, 0)

        return calibrated

    def save_calibration(self, filepath: str):
        """Save calibration data to file."""
        filepath = Path(filepath)

        # Save arrays as .npz
        np.savez(
            filepath.with_suffix('.npz'),
            master_dark=self.calibration_data.master_dark,
            master_flat=self.calibration_data.master_flat,
            bad_pixel_mask=self.calibration_data.bad_pixel_mask,
            hot_pixel_map=self.calibration_data.hot_pixel_map
        )

        # Save metadata as JSON
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(self.calibration_data.metadata, f, indent=2)

        print(f"Calibration saved to {filepath}")

    def load_calibration(self, filepath: str):
        """Load calibration data from file."""
        filepath = Path(filepath)

        # Load arrays
        data = np.load(filepath.with_suffix('.npz'))
        self.calibration_data.master_dark = data['master_dark']
        self.calibration_data.master_flat = data['master_flat']
        self.calibration_data.bad_pixel_mask = data['bad_pixel_mask']
        self.calibration_data.hot_pixel_map = data['hot_pixel_map']

        # Update calibrators
        self.dark_calibrator.master_dark = self.calibration_data.master_dark
        self.dark_calibrator.hot_pixel_mask = self.calibration_data.hot_pixel_map
        self.flat_calibrator.normalized_flat = self.calibration_data.master_flat
        self.bad_pixel_corrector.bad_pixel_mask = self.calibration_data.bad_pixel_mask

        # Load metadata
        with open(filepath.with_suffix('.json'), 'r') as f:
            self.calibration_data.metadata = json.load(f)

        print(f"Calibration loaded from {filepath}")


def demonstrate_calibration():
    """Demonstrate optical calibration with synthetic data."""
    print("=" * 60)
    print("Optical Calibration Demonstration")
    print("=" * 60)

    # Image parameters
    height, width = 1080, 1920

    # Generate synthetic dark frames
    print("\nGenerating synthetic dark frames...")
    dark_frames = []
    for i in range(10):
        # Base dark level
        dark = np.random.poisson(100, (height, width)).astype(np.float64)

        # Add hot pixels
        n_hot = 500
        hot_y = np.random.randint(0, height, n_hot)
        hot_x = np.random.randint(0, width, n_hot)
        dark[hot_y, hot_x] = np.random.uniform(500, 2000, n_hot)

        # Add read noise
        dark += np.random.normal(0, 5, dark.shape)

        dark_frames.append(dark)

    # Generate synthetic flat frames
    print("Generating synthetic flat frames...")
    flat_frames = []

    # Vignetting pattern
    cy, cx = height / 2, width / 2
    y, x = np.ogrid[:height, :width]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_max = np.sqrt(cx**2 + cy**2)
    vignette = 1.0 - 0.3 * (r / r_max)**2

    for i in range(10):
        flat = np.random.poisson(30000, (height, width)).astype(np.float64)
        flat = flat * vignette  # Apply vignetting
        flat += np.random.normal(0, 50, flat.shape)  # Read noise
        flat_frames.append(flat)

    # Create calibrator and run calibration
    calibrator = OpticalCalibrator()
    calibrator.calibrate_from_frames(dark_frames, flat_frames)

    # Generate test image
    print("\nGenerating test image...")
    test_image = np.random.poisson(500, (height, width)).astype(np.float64)

    # Add stars
    n_stars = 100
    for _ in range(n_stars):
        star_y = np.random.randint(100, height - 100)
        star_x = np.random.randint(100, width - 100)
        flux = np.random.uniform(5000, 50000)

        # Gaussian star
        sigma = 2.5
        for dy in range(-10, 11):
            for dx in range(-10, 11):
                val = flux * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                test_image[star_y + dy, star_x + dx] += val

    # Apply vignetting to simulate real capture
    test_image = test_image * vignette

    # Add dark current and hot pixels (same pattern as calibration)
    test_image += dark_frames[0]

    # Apply calibration
    print("\nApplying calibration...")
    calibrated = calibrator.apply_calibration(test_image)

    # Compare statistics
    print("\n" + "=" * 60)
    print("Calibration Results")
    print("=" * 60)

    # Center vs corner comparison
    center_region = test_image[height//2-50:height//2+50, width//2-50:width//2+50]
    corner_region = test_image[50:150, 50:150]

    center_cal = calibrated[height//2-50:height//2+50, width//2-50:width//2+50]
    corner_cal = calibrated[50:150, 50:150]

    print("\nBefore calibration:")
    print(f"  Center mean: {np.mean(center_region):.1f}")
    print(f"  Corner mean: {np.mean(corner_region):.1f}")
    print(f"  Ratio: {np.mean(center_region)/np.mean(corner_region):.3f}")

    print("\nAfter calibration:")
    print(f"  Center mean: {np.mean(center_cal):.1f}")
    print(f"  Corner mean: {np.mean(corner_cal):.1f}")
    print(f"  Ratio: {np.mean(center_cal)/np.mean(corner_cal):.3f}")

    # Hot pixel correction
    hot_mask = calibrator.calibration_data.hot_pixel_map
    if hot_mask is not None:
        hot_before = np.mean(test_image[hot_mask])
        hot_after = np.mean(calibrated[hot_mask])
        print(f"\nHot pixel regions:")
        print(f"  Before: {hot_before:.1f} (anomalously high)")
        print(f"  After: {hot_after:.1f} (corrected)")

    return calibrator, test_image, calibrated


if __name__ == "__main__":
    demonstrate_calibration()

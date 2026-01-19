#!/usr/bin/env python3
"""
Motion Deblur Algorithm with Overlap Handling.

Implements motion blur compensation for star field images using
IMU attitude data. Handles the corner case where star trails
overlap when frame shifts significantly during exposure.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.fft import fft2, ifft2, fftshift
from dataclasses import dataclass
import cv2

from synthetic_data import IMUData, SyntheticStarField
from psf_generator import MotionPSFGenerator, OverlapAwarePSFGenerator, PSFParams


@dataclass
class DeblurParams:
    """Parameters for motion deblur algorithm."""
    method: str = 'richardson_lucy'  # 'richardson_lucy', 'wiener', or 'spatially_varying'
    iterations: int = 30  # For Richardson-Lucy
    wiener_k: float = 0.01  # Wiener filter regularization
    clip_negative: bool = True  # Clip negative values
    handle_overlap: bool = True  # Enable overlap handling
    overlap_blend_width: int = 50  # Width of overlap blending region
    psf_kernel_size: int = 51  # PSF kernel size


class MotionDeblur:
    """
    Motion blur compensation using IMU attitude data.

    Implements multiple deconvolution methods and handles the
    challenging case of star trail overlap in wide-field images.
    """

    def __init__(self,
                 image_width: int = 1920,
                 image_height: int = 1080,
                 focal_length_px: float = 1000.0):
        """
        Initialize deblur processor.

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            focal_length_px: Focal length in pixels
        """
        self.width = image_width
        self.height = image_height
        self.focal_length = focal_length_px

        self.psf_generator = MotionPSFGenerator(
            image_width, image_height, focal_length_px
        )
        self.overlap_generator = OverlapAwarePSFGenerator(
            image_width, image_height, focal_length_px
        )

    def richardson_lucy(self,
                         image: np.ndarray,
                         psf: np.ndarray,
                         iterations: int = 30,
                         clip: bool = True) -> np.ndarray:
        """
        Richardson-Lucy deconvolution algorithm.

        Args:
            image: Blurred input image
            psf: Point spread function
            iterations: Number of iterations
            clip: Clip negative values

        Returns:
            Deblurred image
        """
        # Ensure image is float64 and positive
        image = np.maximum(image.astype(np.float64), 1e-10)

        # Initialize estimate
        estimate = image.copy()

        # Flip PSF for correlation
        psf_flipped = psf[::-1, ::-1]

        for i in range(iterations):
            # Compute convolution of estimate with PSF
            convolved = convolve2d(estimate, psf, mode='same', boundary='wrap')
            convolved = np.maximum(convolved, 1e-10)

            # Compute ratio
            ratio = image / convolved

            # Update estimate
            correction = convolve2d(ratio, psf_flipped, mode='same', boundary='wrap')
            estimate = estimate * correction

            if clip:
                estimate = np.maximum(estimate, 0)

        return estimate

    def wiener_filter(self,
                       image: np.ndarray,
                       psf: np.ndarray,
                       k: float = 0.01) -> np.ndarray:
        """
        Wiener deconvolution in frequency domain.

        Args:
            image: Blurred input image
            psf: Point spread function
            k: Regularization parameter (noise-to-signal ratio estimate)

        Returns:
            Deblurred image
        """
        # Pad PSF to image size
        psf_padded = np.zeros_like(image)
        kh, kw = psf.shape
        cy, cx = psf_padded.shape[0] // 2, psf_padded.shape[1] // 2
        y_start = cy - kh // 2
        x_start = cx - kw // 2
        psf_padded[y_start:y_start+kh, x_start:x_start+kw] = psf

        # Shift so PSF center is at origin (for FFT)
        psf_shifted = np.roll(psf_padded, (-cy, -cx), axis=(0, 1))

        # FFTs
        img_fft = fft2(image.astype(np.float64))
        psf_fft = fft2(psf_shifted)

        # Wiener filter
        psf_conj = np.conj(psf_fft)
        psf_abs_sq = np.abs(psf_fft) ** 2

        # H* / (|H|^2 + K)
        wiener = psf_conj / (psf_abs_sq + k)

        # Apply filter
        result_fft = img_fft * wiener
        result = np.real(ifft2(result_fft))

        return result

    def deblur_with_overlap_handling(self,
                                      image: np.ndarray,
                                      imu_data: IMUData,
                                      params: DeblurParams = None) -> Tuple[np.ndarray, dict]:
        """
        Deblur image with special handling for star trail overlap.

        When the frame shifts significantly during exposure, star trails
        from stars exiting one side of the frame can overlap with trails
        from stars entering from the other side. This method handles
        this by:

        1. Detecting overlap regions based on frame shift
        2. Separating the image into regions by entry/exit time
        3. Processing each region with appropriate time-windowed PSF
        4. Blending the results

        Args:
            image: Blurred input image
            imu_data: IMU attitude data
            params: Deblur parameters

        Returns:
            Tuple of (deblurred_image, metadata)
        """
        if params is None:
            params = DeblurParams()

        metadata = {
            'method': 'overlap_aware',
            'frame_shift': None,
            'overlap_detected': False,
            'regions_processed': 0
        }

        # Detect overlap
        overlap_info = self.overlap_generator.detect_overlap_regions(imu_data)
        frame_shift = overlap_info['frame_shift']
        metadata['frame_shift'] = frame_shift

        has_overlap = (overlap_info['has_horizontal_overlap'] or
                       overlap_info['has_vertical_overlap'])

        if not has_overlap or not params.handle_overlap:
            # No significant overlap - use standard deblur
            psf_params = PSFParams(kernel_size=params.psf_kernel_size)
            psf = self.psf_generator.generate_uniform_psf(imu_data, psf_params)

            if params.method == 'richardson_lucy':
                result = self.richardson_lucy(image, psf, params.iterations, params.clip_negative)
            else:
                result = self.wiener_filter(image, psf, params.wiener_k)

            metadata['regions_processed'] = 1
            return result, metadata

        # Overlap detected - process with time-windowed approach
        metadata['overlap_detected'] = True

        # Compute which regions have overlapping trails
        overlap_regions = self._compute_overlap_mask(imu_data, params)

        # Split processing into overlapping and non-overlapping regions
        result = np.zeros_like(image, dtype=np.float64)
        weight_map = np.zeros_like(image, dtype=np.float64)

        # Process main (non-overlap) region with full PSF
        main_mask = ~overlap_regions['combined_mask']
        if np.any(main_mask):
            psf_params = PSFParams(kernel_size=params.psf_kernel_size)
            full_psf = self.psf_generator.generate_uniform_psf(imu_data, psf_params)

            if params.method == 'richardson_lucy':
                main_result = self.richardson_lucy(image, full_psf, params.iterations)
            else:
                main_result = self.wiener_filter(image, full_psf, params.wiener_k)

            # Create soft mask for blending
            main_weight = gaussian_filter(main_mask.astype(np.float64), params.overlap_blend_width / 4)
            result += main_result * main_weight
            weight_map += main_weight
            metadata['regions_processed'] += 1

        # Process overlap regions with time-windowed PSFs
        for region_name, region_info in overlap_regions.items():
            if region_name == 'combined_mask':
                continue
            if not np.any(region_info['mask']):
                continue

            # Generate time-windowed PSF for this region
            time_start = region_info['time_start']
            time_end = region_info['time_end']

            windowed_imu = self._window_imu_data(imu_data, time_start, time_end)
            if windowed_imu.num_samples < 2:
                continue

            psf_params = PSFParams(kernel_size=params.psf_kernel_size)
            windowed_psf = self.psf_generator.generate_uniform_psf(windowed_imu, psf_params)

            # Deblur this region
            if params.method == 'richardson_lucy':
                region_result = self.richardson_lucy(image, windowed_psf, params.iterations)
            else:
                region_result = self.wiener_filter(image, windowed_psf, params.wiener_k)

            # Create soft mask for blending
            region_weight = gaussian_filter(
                region_info['mask'].astype(np.float64),
                params.overlap_blend_width / 4
            )
            result += region_result * region_weight
            weight_map += region_weight
            metadata['regions_processed'] += 1

        # Normalize by weights
        weight_map = np.maximum(weight_map, 1e-10)
        result /= weight_map

        if params.clip_negative:
            result = np.maximum(result, 0)

        return result, metadata

    def _compute_overlap_mask(self,
                               imu_data: IMUData,
                               params: DeblurParams) -> dict:
        """
        Compute masks for overlap regions.

        Identifies regions where star trails from different parts of
        the image can overlap due to large frame shifts.
        """
        frame_shift = self.overlap_generator.compute_frame_shift(imu_data)
        dx, dy = frame_shift

        overlap_regions = {}

        # Horizontal overlap (frame shifts left/right)
        if abs(dx) > 0.3 * self.width:
            # Left edge region (stars that exit left / enter from left)
            left_mask = np.zeros((self.height, self.width), dtype=bool)
            left_width = int(min(abs(dx), self.width * 0.3))
            left_mask[:, :left_width] = True

            # Right edge region
            right_mask = np.zeros((self.height, self.width), dtype=bool)
            right_mask[:, -left_width:] = True

            if dx > 0:
                # Frame shifts right: early exposure is on left, late on right
                overlap_regions['left_early'] = {
                    'mask': left_mask,
                    'time_start': 0.0,
                    'time_end': 0.5  # First half of exposure
                }
                overlap_regions['right_late'] = {
                    'mask': right_mask,
                    'time_start': 0.5,
                    'time_end': 1.0
                }
            else:
                # Frame shifts left
                overlap_regions['right_early'] = {
                    'mask': right_mask,
                    'time_start': 0.0,
                    'time_end': 0.5
                }
                overlap_regions['left_late'] = {
                    'mask': left_mask,
                    'time_start': 0.5,
                    'time_end': 1.0
                }

        # Vertical overlap
        if abs(dy) > 0.3 * self.height:
            top_mask = np.zeros((self.height, self.width), dtype=bool)
            bottom_mask = np.zeros((self.height, self.width), dtype=bool)
            edge_height = int(min(abs(dy), self.height * 0.3))
            top_mask[:edge_height, :] = True
            bottom_mask[-edge_height:, :] = True

            if dy > 0:
                overlap_regions['top_early'] = {
                    'mask': top_mask,
                    'time_start': 0.0,
                    'time_end': 0.5
                }
                overlap_regions['bottom_late'] = {
                    'mask': bottom_mask,
                    'time_start': 0.5,
                    'time_end': 1.0
                }
            else:
                overlap_regions['bottom_early'] = {
                    'mask': bottom_mask,
                    'time_start': 0.0,
                    'time_end': 0.5
                }
                overlap_regions['top_late'] = {
                    'mask': top_mask,
                    'time_start': 0.5,
                    'time_end': 1.0
                }

        # Combined mask of all overlap regions
        combined = np.zeros((self.height, self.width), dtype=bool)
        for region_name, region_info in overlap_regions.items():
            combined |= region_info['mask']

        overlap_regions['combined_mask'] = combined

        return overlap_regions

    def _window_imu_data(self,
                          imu_data: IMUData,
                          time_start: float,
                          time_end: float) -> IMUData:
        """
        Extract a time window from IMU data.

        Args:
            imu_data: Full IMU data
            time_start: Start time as fraction (0-1)
            time_end: End time as fraction (0-1)

        Returns:
            Windowed IMU data
        """
        duration = imu_data.duration
        t_start = time_start * duration
        t_end = time_end * duration

        # Find indices in this time range
        mask = (imu_data.timestamps >= t_start) & (imu_data.timestamps <= t_end)
        indices = np.where(mask)[0]

        if len(indices) < 2:
            # Return minimal valid data
            return IMUData(
                timestamps=np.array([0.0, 0.01]),
                quaternions=imu_data.quaternions[:2],
                angular_velocity=imu_data.angular_velocity[:2] if imu_data.angular_velocity is not None else None
            )

        return IMUData(
            timestamps=imu_data.timestamps[indices] - imu_data.timestamps[indices[0]],
            quaternions=imu_data.quaternions[indices],
            angular_velocity=imu_data.angular_velocity[indices] if imu_data.angular_velocity is not None else None
        )

    def deblur(self,
               image: np.ndarray,
               imu_data: IMUData,
               params: DeblurParams = None) -> Tuple[np.ndarray, dict]:
        """
        Main deblur method - automatically selects approach.

        Args:
            image: Blurred input image
            imu_data: IMU attitude data
            params: Deblur parameters

        Returns:
            Tuple of (deblurred_image, metadata)
        """
        if params is None:
            params = DeblurParams()

        if params.method == 'spatially_varying':
            return self._spatially_varying_deblur(image, imu_data, params)

        if params.handle_overlap:
            return self.deblur_with_overlap_handling(image, imu_data, params)

        # Standard uniform PSF approach
        psf_params = PSFParams(kernel_size=params.psf_kernel_size)
        psf = self.psf_generator.generate_uniform_psf(imu_data, psf_params)

        metadata = {'method': params.method, 'psf_shape': psf.shape}

        if params.method == 'richardson_lucy':
            result = self.richardson_lucy(image, psf, params.iterations, params.clip_negative)
        else:
            result = self.wiener_filter(image, psf, params.wiener_k)

        return result, metadata

    def _spatially_varying_deblur(self,
                                   image: np.ndarray,
                                   imu_data: IMUData,
                                   params: DeblurParams) -> Tuple[np.ndarray, dict]:
        """
        Spatially-varying deconvolution.

        Uses a grid of PSFs across the image and interpolates between them.
        This is more accurate for wide-field cameras but slower.
        """
        grid_size = (3, 3)
        psf_params = PSFParams(kernel_size=params.psf_kernel_size)

        # Generate PSF grid
        psfs, x_positions, y_positions = self.psf_generator.generate_psf_grid(
            imu_data, grid_size, psf_params
        )

        result = np.zeros_like(image, dtype=np.float64)
        weight_map = np.zeros_like(image, dtype=np.float64)

        # Process tiles
        rows, cols = grid_size
        tile_height = self.height // rows
        tile_width = self.width // cols

        for i in range(rows):
            for j in range(cols):
                # Tile boundaries with overlap
                overlap = 50
                y_start = max(0, i * tile_height - overlap)
                y_end = min(self.height, (i + 1) * tile_height + overlap)
                x_start = max(0, j * tile_width - overlap)
                x_end = min(self.width, (j + 1) * tile_width + overlap)

                # Extract tile
                tile = image[y_start:y_end, x_start:x_end]
                psf = psfs[i, j]

                # Deblur tile
                if params.method == 'richardson_lucy':
                    deblurred_tile = self.richardson_lucy(tile, psf, params.iterations)
                else:
                    deblurred_tile = self.wiener_filter(tile, psf, params.wiener_k)

                # Create weight mask (soft edges)
                weight = np.ones_like(deblurred_tile)
                fade_pixels = overlap // 2 if overlap > 0 else 10
                for k in range(fade_pixels):
                    alpha = k / fade_pixels
                    if i > 0 and y_start > 0:
                        weight[k, :] *= alpha
                    if i < rows - 1 and y_end < self.height:
                        weight[-(k+1), :] *= alpha
                    if j > 0 and x_start > 0:
                        weight[:, k] *= alpha
                    if j < cols - 1 and x_end < self.width:
                        weight[:, -(k+1)] *= alpha

                # Add to result
                result[y_start:y_end, x_start:x_end] += deblurred_tile * weight
                weight_map[y_start:y_end, x_start:x_end] += weight

        # Normalize
        weight_map = np.maximum(weight_map, 1e-10)
        result /= weight_map

        if params.clip_negative:
            result = np.maximum(result, 0)

        metadata = {
            'method': 'spatially_varying',
            'grid_size': grid_size,
            'tiles_processed': rows * cols
        }

        return result, metadata


def compute_quality_metrics(original: np.ndarray,
                             blurred: np.ndarray,
                             deblurred: np.ndarray) -> dict:
    """
    Compute quality metrics comparing original, blurred, and deblurred images.

    Args:
        original: Original sharp image
        blurred: Motion-blurred image
        deblurred: Deblurred result

    Returns:
        Dictionary with quality metrics
    """
    def psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_val = max(img1.max(), img2.max())
        return 20 * np.log10(max_val / np.sqrt(mse))

    def ssim(img1, img2):
        """Simplified SSIM computation."""
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        mu1 = gaussian_filter(img1, 1.5)
        mu2 = gaussian_filter(img2, 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = gaussian_filter(img1 ** 2, 1.5) - mu1_sq
        sigma2_sq = gaussian_filter(img2 ** 2, 1.5) - mu2_sq
        sigma12 = gaussian_filter(img1 * img2, 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        return np.mean(ssim_map)

    # Normalize images to common range
    scale = max(original.max(), blurred.max(), deblurred.max())
    if scale > 0:
        orig_norm = original / scale * 255
        blur_norm = blurred / scale * 255
        deblur_norm = deblurred / scale * 255
    else:
        orig_norm = original
        blur_norm = blurred
        deblur_norm = deblurred

    return {
        'psnr_blurred': psnr(orig_norm, blur_norm),
        'psnr_deblurred': psnr(orig_norm, deblur_norm),
        'psnr_improvement': psnr(orig_norm, deblur_norm) - psnr(orig_norm, blur_norm),
        'ssim_blurred': ssim(orig_norm, blur_norm),
        'ssim_deblurred': ssim(orig_norm, deblur_norm),
        'ssim_improvement': ssim(orig_norm, deblur_norm) - ssim(orig_norm, blur_norm)
    }


if __name__ == '__main__':
    from synthetic_data import SyntheticStarField, IMUMotionSimulator, MotionBlurRenderer

    print("Testing motion deblur with overlap handling...")

    # Create star field
    star_field = SyntheticStarField(
        width=1920, height=1080,
        num_stars=100,
        seed=42
    )

    # Generate motion with significant drift (will cause overlap)
    print("\nGenerating motion scenario with large drift...")
    motion_sim = IMUMotionSimulator(duration=8.0, sample_rate=200)
    imu_data = motion_sim.generate_combined_motion(
        drift_rate=1.0,  # 1 deg/s drift
        vib_frequencies=[5.0, 12.0],
        vib_amplitudes=[0.05, 0.02]
    )

    # Render sharp and blurred images
    sharp_image = star_field.render_sharp_image()
    sharp_noisy = star_field.add_noise(sharp_image)

    renderer = MotionBlurRenderer(
        width=star_field.width,
        height=star_field.height,
        focal_length_px=1200
    )

    blurred_image, blur_meta = renderer.render_blurred_image(
        star_field, imu_data,
        num_subframes=int(8.0 * 30)
    )
    blurred_noisy = star_field.add_noise(blurred_image)

    print(f"Max motion: {blur_meta['max_motion_pixels']:.1f} pixels")
    print(f"Overlap events: {blur_meta['overlap_events']}")

    # Test deblurring
    deblur = MotionDeblur(1920, 1080, 1200)

    # Test Richardson-Lucy with overlap handling
    print("\nDeblurring with Richardson-Lucy (overlap-aware)...")
    params_rl = DeblurParams(
        method='richardson_lucy',
        iterations=20,
        handle_overlap=True
    )
    result_rl, meta_rl = deblur.deblur(blurred_noisy, imu_data, params_rl)
    print(f"  Regions processed: {meta_rl.get('regions_processed', 1)}")
    print(f"  Overlap detected: {meta_rl.get('overlap_detected', False)}")

    # Test Wiener deconvolution
    print("\nDeblurring with Wiener filter...")
    params_wiener = DeblurParams(
        method='wiener',
        wiener_k=0.001,
        handle_overlap=True
    )
    result_wiener, meta_wiener = deblur.deblur(blurred_noisy, imu_data, params_wiener)

    # Compute quality metrics
    print("\nQuality metrics:")
    metrics_rl = compute_quality_metrics(sharp_noisy, blurred_noisy, result_rl)
    metrics_wiener = compute_quality_metrics(sharp_noisy, blurred_noisy, result_wiener)

    print(f"  Richardson-Lucy:")
    print(f"    PSNR improvement: {metrics_rl['psnr_improvement']:.2f} dB")
    print(f"    SSIM improvement: {metrics_rl['ssim_improvement']:.4f}")

    print(f"  Wiener:")
    print(f"    PSNR improvement: {metrics_wiener['psnr_improvement']:.2f} dB")
    print(f"    SSIM improvement: {metrics_wiener['ssim_improvement']:.4f}")

    print("\nMotion deblur test complete!")

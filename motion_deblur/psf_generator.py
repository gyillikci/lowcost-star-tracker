#!/usr/bin/env python3
"""
PSF Generator from IMU Quaternion Trajectory.

Generates Point Spread Functions (PSFs) representing motion blur
from IMU attitude data. Handles spatially-varying PSFs for wide-field
cameras where motion blur differs across the image.
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass

from synthetic_data import IMUData


@dataclass
class PSFParams:
    """Parameters for PSF generation."""
    kernel_size: int = 51  # Size of PSF kernel (odd number)
    star_fwhm: float = 2.5  # Star FWHM in pixels
    normalize: bool = True  # Normalize PSF to sum to 1
    subsampling: int = 3  # Subpixel sampling factor


class MotionPSFGenerator:
    """
    Generate motion blur PSFs from IMU quaternion trajectory.

    This class computes spatially-varying PSFs that account for
    how motion blur changes across the image field.
    """

    def __init__(self,
                 image_width: int = 1920,
                 image_height: int = 1080,
                 focal_length_px: float = 1000.0):
        """
        Initialize PSF generator.

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            focal_length_px: Focal length in pixels
        """
        self.width = image_width
        self.height = image_height
        self.focal_length = focal_length_px

        # Camera intrinsic matrix
        self.K = np.array([
            [focal_length_px, 0, image_width / 2],
            [0, focal_length_px, image_height / 2],
            [0, 0, 1]
        ])
        self.K_inv = np.linalg.inv(self.K)

    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to rotation matrix."""
        w, x, y, z = q

        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])

    def compute_motion_trajectory(self,
                                   pixel_pos: Tuple[float, float],
                                   imu_data: IMUData,
                                   num_samples: int = 100) -> np.ndarray:
        """
        Compute the trajectory of a point across the image during exposure.

        Args:
            pixel_pos: (x, y) position of the point
            imu_data: IMU orientation data
            num_samples: Number of points in trajectory

        Returns:
            Array of shape (N, 2) with trajectory points relative to start
        """
        # Back-project pixel to 3D direction
        point_2d = np.array([pixel_pos[0], pixel_pos[1], 1.0])
        direction = self.K_inv @ point_2d
        direction = direction / np.linalg.norm(direction)

        # Reference orientation (first frame)
        R_ref = self.quaternion_to_rotation_matrix(imu_data.quaternions[0])

        # Direction in world frame
        dir_world = R_ref.T @ direction

        # Sample trajectory
        indices = np.linspace(0, len(imu_data.quaternions) - 1, num_samples).astype(int)
        trajectory = []

        start_pos = None

        for idx in indices:
            R = self.quaternion_to_rotation_matrix(imu_data.quaternions[idx])

            # Project to current camera frame
            dir_cam = R @ dir_world

            if dir_cam[2] > 0:
                projected = self.K @ dir_cam
                x = projected[0] / projected[2]
                y = projected[1] / projected[2]

                if start_pos is None:
                    start_pos = np.array([x, y])

                # Store relative position
                trajectory.append([x - start_pos[0], y - start_pos[1]])

        return np.array(trajectory) if trajectory else np.array([[0.0, 0.0]])

    def generate_psf_at_position(self,
                                  pixel_pos: Tuple[float, float],
                                  imu_data: IMUData,
                                  params: PSFParams = None) -> np.ndarray:
        """
        Generate motion blur PSF for a specific image position.

        Args:
            pixel_pos: (x, y) position in the image
            imu_data: IMU orientation data
            params: PSF parameters

        Returns:
            2D PSF kernel
        """
        if params is None:
            params = PSFParams()

        # Compute motion trajectory
        trajectory = self.compute_motion_trajectory(
            pixel_pos, imu_data,
            num_samples=max(100, int(imu_data.duration * 50))
        )

        # Create PSF kernel with subsampling
        kernel_size = params.kernel_size
        sub = params.subsampling

        # High-resolution kernel for anti-aliasing
        hi_res_size = kernel_size * sub
        kernel_hires = np.zeros((hi_res_size, hi_res_size), dtype=np.float64)

        center = hi_res_size // 2
        sigma = params.star_fwhm / 2.355 * sub  # Star sigma at high res

        # Plot trajectory points onto kernel
        weight = 1.0 / len(trajectory)

        for dx, dy in trajectory:
            # Convert to kernel coordinates (with subsampling)
            kx = center + dx * sub
            ky = center + dy * sub

            if 0 <= kx < hi_res_size and 0 <= ky < hi_res_size:
                # Add Gaussian at this position
                x_int, y_int = int(round(kx)), int(round(ky))

                # Small stamp around the point
                stamp_size = int(4 * sigma) + 1
                x_min = max(0, x_int - stamp_size)
                x_max = min(hi_res_size, x_int + stamp_size + 1)
                y_min = max(0, y_int - stamp_size)
                y_max = min(hi_res_size, y_int + stamp_size + 1)

                if x_max > x_min and y_max > y_min:
                    y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
                    gauss = np.exp(-((x_grid - kx)**2 + (y_grid - ky)**2) / (2 * sigma**2))
                    kernel_hires[y_min:y_max, x_min:x_max] += gauss * weight

        # Downsample to final resolution
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = kernel_hires[i*sub:(i+1)*sub, j*sub:(j+1)*sub].sum()

        # Normalize
        if params.normalize and kernel.sum() > 0:
            kernel /= kernel.sum()

        return kernel

    def generate_uniform_psf(self,
                              imu_data: IMUData,
                              params: PSFParams = None) -> np.ndarray:
        """
        Generate a single PSF at the image center (assumes spatially uniform blur).

        Args:
            imu_data: IMU orientation data
            params: PSF parameters

        Returns:
            2D PSF kernel
        """
        center_pos = (self.width / 2, self.height / 2)
        return self.generate_psf_at_position(center_pos, imu_data, params)

    def generate_psf_grid(self,
                           imu_data: IMUData,
                           grid_size: Tuple[int, int] = (3, 3),
                           params: PSFParams = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a grid of spatially-varying PSFs.

        For wide-field cameras, motion blur varies across the image.
        This generates PSFs at a grid of positions for spatially-varying
        deconvolution.

        Args:
            imu_data: IMU orientation data
            grid_size: (rows, cols) of the PSF grid
            params: PSF parameters

        Returns:
            Tuple of:
            - psfs: Array of shape (rows, cols, kernel_size, kernel_size)
            - x_positions: X coordinates of PSF centers
            - y_positions: Y coordinates of PSF centers
        """
        if params is None:
            params = PSFParams()

        rows, cols = grid_size

        # Compute grid positions (excluding edges)
        x_margin = self.width * 0.1
        y_margin = self.height * 0.1

        x_positions = np.linspace(x_margin, self.width - x_margin, cols)
        y_positions = np.linspace(y_margin, self.height - y_margin, rows)

        # Generate PSFs
        psfs = np.zeros((rows, cols, params.kernel_size, params.kernel_size))

        for i, y in enumerate(y_positions):
            for j, x in enumerate(x_positions):
                psfs[i, j] = self.generate_psf_at_position((x, y), imu_data, params)

        return psfs, x_positions, y_positions


class OverlapAwarePSFGenerator(MotionPSFGenerator):
    """
    PSF generator that handles star trail overlap scenarios.

    When the frame shifts significantly during exposure, star trails
    from one side of the image can overlap with trails from the other side.
    This class generates PSFs that account for this wrap-around effect.
    """

    def __init__(self,
                 image_width: int = 1920,
                 image_height: int = 1080,
                 focal_length_px: float = 1000.0):
        super().__init__(image_width, image_height, focal_length_px)

    def compute_frame_shift(self, imu_data: IMUData) -> Tuple[float, float]:
        """
        Compute total frame shift during exposure.

        Args:
            imu_data: IMU orientation data

        Returns:
            (dx, dy) total shift in pixels
        """
        trajectory = self.compute_motion_trajectory(
            (self.width / 2, self.height / 2),
            imu_data,
            num_samples=len(imu_data.quaternions)
        )

        if len(trajectory) > 1:
            return trajectory[-1][0], trajectory[-1][1]
        return 0.0, 0.0

    def detect_overlap_regions(self,
                                imu_data: IMUData,
                                margin: float = 50.0) -> dict:
        """
        Detect regions where star trail overlap can occur.

        Args:
            imu_data: IMU orientation data
            margin: Margin from image edges to check (pixels)

        Returns:
            Dictionary with overlap information
        """
        dx, dy = self.compute_frame_shift(imu_data)

        overlap_info = {
            'frame_shift': (dx, dy),
            'has_horizontal_overlap': False,
            'has_vertical_overlap': False,
            'overlap_regions': []
        }

        # Check for horizontal overlap (frame shifts right/left)
        if abs(dx) > self.width - 2 * margin:
            overlap_info['has_horizontal_overlap'] = True
            if dx > 0:
                # Frame shifted right: left edge stars appear on right
                overlap_info['overlap_regions'].append({
                    'type': 'horizontal',
                    'direction': 'right',
                    'source_region': (0, margin),  # Left edge
                    'target_region': (self.width - margin, self.width)  # Right edge
                })
            else:
                # Frame shifted left
                overlap_info['overlap_regions'].append({
                    'type': 'horizontal',
                    'direction': 'left',
                    'source_region': (self.width - margin, self.width),
                    'target_region': (0, margin)
                })

        # Check for vertical overlap
        if abs(dy) > self.height - 2 * margin:
            overlap_info['has_vertical_overlap'] = True
            if dy > 0:
                overlap_info['overlap_regions'].append({
                    'type': 'vertical',
                    'direction': 'down',
                    'source_region': (0, margin),
                    'target_region': (self.height - margin, self.height)
                })
            else:
                overlap_info['overlap_regions'].append({
                    'type': 'vertical',
                    'direction': 'up',
                    'source_region': (self.height - margin, self.height),
                    'target_region': (0, margin)
                })

        return overlap_info

    def generate_overlap_aware_psf(self,
                                    pixel_pos: Tuple[float, float],
                                    imu_data: IMUData,
                                    params: PSFParams = None) -> Tuple[np.ndarray, dict]:
        """
        Generate PSF with overlap information for a specific position.

        For positions near edges where overlap can occur, this returns
        additional information about the overlap regions.

        Args:
            pixel_pos: (x, y) position
            imu_data: IMU orientation data
            params: PSF parameters

        Returns:
            Tuple of (psf_kernel, overlap_info)
        """
        if params is None:
            params = PSFParams()

        # Get base PSF
        psf = self.generate_psf_at_position(pixel_pos, imu_data, params)

        # Compute full trajectory
        trajectory = self.compute_motion_trajectory(
            pixel_pos, imu_data,
            num_samples=max(100, int(imu_data.duration * 50))
        )

        # Analyze trajectory for overlap
        overlap_info = {
            'position': pixel_pos,
            'trajectory_length': len(trajectory),
            'exits_frame': False,
            'enters_frame': False,
            'wrap_around': False,
            'exit_edge': None,
            'entry_edge': None,
            'exit_time_fraction': None,
            'entry_time_fraction': None
        }

        # Check each point in trajectory
        for i, (dx, dy) in enumerate(trajectory):
            new_x = pixel_pos[0] + dx
            new_y = pixel_pos[1] + dy

            # Check if trajectory exits frame
            if not overlap_info['exits_frame']:
                if new_x < 0:
                    overlap_info['exits_frame'] = True
                    overlap_info['exit_edge'] = 'left'
                    overlap_info['exit_time_fraction'] = i / len(trajectory)
                elif new_x >= self.width:
                    overlap_info['exits_frame'] = True
                    overlap_info['exit_edge'] = 'right'
                    overlap_info['exit_time_fraction'] = i / len(trajectory)
                elif new_y < 0:
                    overlap_info['exits_frame'] = True
                    overlap_info['exit_edge'] = 'top'
                    overlap_info['exit_time_fraction'] = i / len(trajectory)
                elif new_y >= self.height:
                    overlap_info['exits_frame'] = True
                    overlap_info['exit_edge'] = 'bottom'
                    overlap_info['exit_time_fraction'] = i / len(trajectory)

        # Check for wrap-around (trajectory exits one side and could
        # conceptually re-enter from the other)
        frame_shift = self.compute_frame_shift(imu_data)
        if abs(frame_shift[0]) > self.width * 0.5 or abs(frame_shift[1]) > self.height * 0.5:
            overlap_info['wrap_around'] = True

        return psf, overlap_info


def estimate_psf_from_imu_file(imu_file: str,
                                image_size: Tuple[int, int] = (1920, 1080),
                                focal_length_px: float = 1000.0) -> np.ndarray:
    """
    Convenience function to estimate PSF from an IMU data file.

    Args:
        imu_file: Path to .npz file with IMU data
        image_size: (width, height) of image
        focal_length_px: Focal length in pixels

    Returns:
        PSF kernel
    """
    # Load IMU data
    data = np.load(imu_file)
    imu_data = IMUData(
        timestamps=data['timestamps'],
        quaternions=data['quaternions'],
        angular_velocity=data.get('angular_velocity', None)
    )

    # Generate PSF
    generator = MotionPSFGenerator(
        image_width=image_size[0],
        image_height=image_size[1],
        focal_length_px=focal_length_px
    )

    return generator.generate_uniform_psf(imu_data)


def visualize_psf(psf: np.ndarray, title: str = "Motion PSF") -> None:
    """
    Visualize a PSF kernel.

    Args:
        psf: 2D PSF kernel
        title: Plot title
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Linear scale
    axes[0].imshow(psf, cmap='hot', interpolation='nearest')
    axes[0].set_title(f"{title} (Linear)")
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")

    # Log scale
    psf_log = np.log10(psf + 1e-10)
    axes[1].imshow(psf_log, cmap='hot', interpolation='nearest')
    axes[1].set_title(f"{title} (Log)")
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from synthetic_data import IMUMotionSimulator

    print("Testing PSF generation from IMU data...")

    # Generate synthetic IMU data
    motion_sim = IMUMotionSimulator(duration=5.0, sample_rate=200)
    imu_data = motion_sim.generate_combined_motion(
        drift_rate=0.5,
        vib_frequencies=[5.0, 12.0],
        vib_amplitudes=[0.1, 0.05]
    )

    # Generate PSF
    generator = MotionPSFGenerator(
        image_width=1920,
        image_height=1080,
        focal_length_px=1200
    )

    params = PSFParams(kernel_size=101, star_fwhm=2.5)

    # Center PSF
    center_psf = generator.generate_uniform_psf(imu_data, params)
    print(f"Center PSF shape: {center_psf.shape}")
    print(f"PSF sum: {center_psf.sum():.4f}")
    print(f"PSF max: {center_psf.max():.4f}")

    # Corner PSFs
    print("\nGenerating corner PSFs...")
    corners = [(100, 100), (1820, 100), (100, 980), (1820, 980)]
    for pos in corners:
        psf = generator.generate_psf_at_position(pos, imu_data, params)
        trajectory = generator.compute_motion_trajectory(pos, imu_data)
        motion = np.linalg.norm(trajectory[-1]) if len(trajectory) > 1 else 0
        print(f"  Position {pos}: motion = {motion:.1f} px")

    # Test overlap detection
    print("\nTesting overlap-aware PSF generation...")
    overlap_gen = OverlapAwarePSFGenerator(1920, 1080, 1200)

    # Large motion scenario
    large_motion = IMUMotionSimulator(duration=30.0, sample_rate=200)
    large_imu = large_motion.generate_drift_motion(drift_rate=2.0)

    overlap_info = overlap_gen.detect_overlap_regions(large_imu)
    print(f"Frame shift: {overlap_info['frame_shift']}")
    print(f"Horizontal overlap: {overlap_info['has_horizontal_overlap']}")
    print(f"Vertical overlap: {overlap_info['has_vertical_overlap']}")

    # Generate overlap-aware PSF at edge
    edge_psf, edge_info = overlap_gen.generate_overlap_aware_psf((100, 540), large_imu, params)
    print(f"Edge PSF exits frame: {edge_info['exits_frame']}")
    if edge_info['exits_frame']:
        print(f"  Exit edge: {edge_info['exit_edge']}")
        print(f"  Exit time fraction: {edge_info['exit_time_fraction']:.2f}")

    print("\nPSF generation test complete!")

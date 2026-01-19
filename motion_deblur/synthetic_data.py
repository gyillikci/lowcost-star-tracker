#!/usr/bin/env python3
"""
Synthetic Star Field Generator with IMU-based Motion Blur.

Generates realistic star fields and applies motion blur based on
simulated or real IMU attitude data from Orange Cube.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation, Slerp
import cv2


@dataclass
class Star:
    """Represents a star in the field."""
    x: float  # X position (pixels)
    y: float  # Y position (pixels)
    magnitude: float  # Visual magnitude
    flux: float  # Integrated flux (ADU)


@dataclass
class IMUData:
    """IMU data from Orange Cube or synthetic source."""
    timestamps: np.ndarray  # Time in seconds from exposure start
    quaternions: np.ndarray  # Orientation quaternions [w, x, y, z]
    angular_velocity: Optional[np.ndarray] = None  # rad/s [wx, wy, wz]

    @property
    def num_samples(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        return self.timestamps[-1] - self.timestamps[0]

    @property
    def sample_rate(self) -> float:
        return self.num_samples / self.duration


class SyntheticStarField:
    """Generate synthetic star fields for testing."""

    def __init__(self,
                 width: int = 1920,
                 height: int = 1080,
                 num_stars: int = 200,
                 min_magnitude: float = 1.0,
                 max_magnitude: float = 8.0,
                 star_fwhm: float = 2.5,
                 background_level: float = 100.0,
                 read_noise: float = 5.0,
                 seed: Optional[int] = None):
        """
        Initialize star field generator.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            num_stars: Number of stars to generate
            min_magnitude: Brightest star magnitude
            max_magnitude: Faintest star magnitude
            star_fwhm: Star FWHM in pixels
            background_level: Sky background in ADU
            read_noise: Read noise standard deviation
            seed: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.num_stars = num_stars
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
        self.star_fwhm = star_fwhm
        self.background_level = background_level
        self.read_noise = read_noise

        if seed is not None:
            np.random.seed(seed)

        self.stars = self._generate_stars()

    def _generate_stars(self) -> List[Star]:
        """Generate random star positions and magnitudes."""
        stars = []

        # Generate positions with some margin
        margin = 50
        x_positions = np.random.uniform(margin, self.width - margin, self.num_stars)
        y_positions = np.random.uniform(margin, self.height - margin, self.num_stars)

        # Generate magnitudes (more faint stars than bright)
        # Use exponential distribution to favor fainter stars
        magnitudes = np.random.exponential(2.0, self.num_stars) + self.min_magnitude
        magnitudes = np.clip(magnitudes, self.min_magnitude, self.max_magnitude)

        # Convert magnitude to flux (arbitrary scale)
        # flux = 10^((zero_point - mag) / 2.5)
        zero_point = 15.0  # Arbitrary zero point
        fluxes = 10 ** ((zero_point - magnitudes) / 2.5)

        for i in range(self.num_stars):
            stars.append(Star(
                x=x_positions[i],
                y=y_positions[i],
                magnitude=magnitudes[i],
                flux=fluxes[i]
            ))

        return stars

    def render_sharp_image(self) -> np.ndarray:
        """Render a sharp (unblurred) star field image."""
        image = np.ones((self.height, self.width), dtype=np.float64) * self.background_level

        sigma = self.star_fwhm / 2.355  # Convert FWHM to sigma

        for star in self.stars:
            # Create star PSF (2D Gaussian)
            x_int, y_int = int(round(star.x)), int(round(star.y))

            # Define region around star
            size = int(6 * sigma) + 1
            x_min = max(0, x_int - size)
            x_max = min(self.width, x_int + size + 1)
            y_min = max(0, y_int - size)
            y_max = min(self.height, y_int + size + 1)

            # Create coordinate grids
            y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]

            # 2D Gaussian
            dx = x_grid - star.x
            dy = y_grid - star.y
            gaussian = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            gaussian *= star.flux / (2 * np.pi * sigma**2)  # Normalize

            image[y_min:y_max, x_min:x_max] += gaussian

        return image

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add realistic noise to image."""
        # Poisson noise (shot noise)
        noisy = np.random.poisson(np.maximum(image, 0))

        # Read noise (Gaussian)
        noisy = noisy + np.random.normal(0, self.read_noise, image.shape)

        return noisy.astype(np.float64)


class IMUMotionSimulator:
    """Simulate realistic IMU motion patterns."""

    def __init__(self, duration: float = 10.0, sample_rate: float = 200.0):
        """
        Initialize IMU motion simulator.

        Args:
            duration: Exposure duration in seconds
            sample_rate: IMU sample rate in Hz
        """
        self.duration = duration
        self.sample_rate = sample_rate
        self.num_samples = int(duration * sample_rate)
        self.timestamps = np.linspace(0, duration, self.num_samples)

    def generate_drift_motion(self,
                               drift_rate: float = 0.5,
                               drift_axis: np.ndarray = None) -> IMUData:
        """
        Generate smooth drift motion (like untracked mount).

        Args:
            drift_rate: Drift rate in degrees/second
            drift_axis: Axis of rotation [x, y, z], default is around Y

        Returns:
            IMUData with quaternions representing the motion
        """
        if drift_axis is None:
            drift_axis = np.array([0.0, 1.0, 0.0])  # Drift around Y axis

        drift_axis = drift_axis / np.linalg.norm(drift_axis)

        # Angles over time
        angles = np.deg2rad(drift_rate) * self.timestamps

        # Create quaternions
        quaternions = np.zeros((self.num_samples, 4))
        for i, angle in enumerate(angles):
            r = Rotation.from_rotvec(angle * drift_axis)
            quaternions[i] = r.as_quat()  # [x, y, z, w] scipy format

        # Convert to [w, x, y, z] format
        quaternions = quaternions[:, [3, 0, 1, 2]]

        # Compute angular velocity
        angular_velocity = np.zeros((self.num_samples, 3))
        angular_velocity[:] = np.deg2rad(drift_rate) * drift_axis

        return IMUData(
            timestamps=self.timestamps.copy(),
            quaternions=quaternions,
            angular_velocity=angular_velocity
        )

    def generate_vibration_motion(self,
                                   frequencies: List[float] = [5.0, 12.0, 25.0],
                                   amplitudes: List[float] = [0.1, 0.05, 0.02]) -> IMUData:
        """
        Generate oscillatory vibration motion.

        Args:
            frequencies: Vibration frequencies in Hz
            amplitudes: Vibration amplitudes in degrees

        Returns:
            IMUData with quaternions representing vibration
        """
        # Generate angle variations for each axis
        angle_x = np.zeros(self.num_samples)
        angle_y = np.zeros(self.num_samples)
        angle_z = np.zeros(self.num_samples)

        for freq, amp in zip(frequencies, amplitudes):
            phase = np.random.uniform(0, 2 * np.pi)
            angle_x += np.deg2rad(amp) * np.sin(2 * np.pi * freq * self.timestamps + phase)

            phase = np.random.uniform(0, 2 * np.pi)
            angle_y += np.deg2rad(amp) * np.sin(2 * np.pi * freq * self.timestamps + phase)

            phase = np.random.uniform(0, 2 * np.pi)
            angle_z += np.deg2rad(amp * 0.5) * np.sin(2 * np.pi * freq * self.timestamps + phase)

        # Create quaternions
        quaternions = np.zeros((self.num_samples, 4))
        for i in range(self.num_samples):
            r = Rotation.from_euler('xyz', [angle_x[i], angle_y[i], angle_z[i]])
            q = r.as_quat()  # [x, y, z, w]
            quaternions[i] = [q[3], q[0], q[1], q[2]]  # [w, x, y, z]

        # Compute angular velocity (numerical derivative)
        angular_velocity = np.zeros((self.num_samples, 3))
        dt = 1.0 / self.sample_rate
        angular_velocity[:, 0] = np.gradient(angle_x, dt)
        angular_velocity[:, 1] = np.gradient(angle_y, dt)
        angular_velocity[:, 2] = np.gradient(angle_z, dt)

        return IMUData(
            timestamps=self.timestamps.copy(),
            quaternions=quaternions,
            angular_velocity=angular_velocity
        )

    def generate_combined_motion(self,
                                  drift_rate: float = 0.3,
                                  drift_axis: np.ndarray = None,
                                  vib_frequencies: List[float] = [8.0, 15.0],
                                  vib_amplitudes: List[float] = [0.05, 0.02]) -> IMUData:
        """
        Generate combined drift + vibration motion.

        This is the most realistic scenario for handheld or
        poorly tracked astrophotography.
        """
        drift_data = self.generate_drift_motion(drift_rate, drift_axis)
        vib_data = self.generate_vibration_motion(vib_frequencies, vib_amplitudes)

        # Combine quaternions by multiplication
        combined_quats = np.zeros_like(drift_data.quaternions)

        for i in range(self.num_samples):
            # Convert to scipy format [x, y, z, w]
            q1 = drift_data.quaternions[i, [1, 2, 3, 0]]
            q2 = vib_data.quaternions[i, [1, 2, 3, 0]]

            r1 = Rotation.from_quat(q1)
            r2 = Rotation.from_quat(q2)

            combined = r1 * r2
            q_combined = combined.as_quat()
            combined_quats[i] = [q_combined[3], q_combined[0], q_combined[1], q_combined[2]]

        # Combine angular velocities
        combined_omega = drift_data.angular_velocity + vib_data.angular_velocity

        return IMUData(
            timestamps=self.timestamps.copy(),
            quaternions=combined_quats,
            angular_velocity=combined_omega
        )


class MotionBlurRenderer:
    """
    Render motion-blurred images from star field and IMU data.

    Handles the important corner case of star trail overlap when
    the frame shifts significantly during exposure.
    """

    def __init__(self,
                 width: int = 1920,
                 height: int = 1080,
                 focal_length_px: float = 1000.0):
        """
        Initialize motion blur renderer.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            focal_length_px: Focal length in pixels (for projection)
        """
        self.width = width
        self.height = height
        self.focal_length = focal_length_px

        # Camera intrinsic matrix
        self.K = np.array([
            [focal_length_px, 0, width / 2],
            [0, focal_length_px, height / 2],
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

    def project_point(self, point_3d: np.ndarray, R: np.ndarray) -> Tuple[float, float]:
        """Project 3D point to image plane given rotation."""
        # Rotate point
        rotated = R @ point_3d

        # Project to image plane
        if rotated[2] <= 0:
            return None, None

        projected = self.K @ rotated
        x = projected[0] / projected[2]
        y = projected[1] / projected[2]

        return x, y

    def star_to_3d_direction(self, star: Star) -> np.ndarray:
        """Convert star image position to 3D direction vector."""
        # Back-project from image to 3D ray
        point_2d = np.array([star.x, star.y, 1.0])
        direction = self.K_inv @ point_2d
        direction = direction / np.linalg.norm(direction)
        return direction

    def compute_star_trajectory(self,
                                 star: Star,
                                 imu_data: IMUData,
                                 num_points: int = 100) -> np.ndarray:
        """
        Compute the trajectory of a star across the image during exposure.

        Args:
            star: Star object with initial position
            imu_data: IMU orientation data
            num_points: Number of points in trajectory

        Returns:
            Array of (x, y) positions along the trajectory
        """
        # Get star direction in initial camera frame
        star_dir = self.star_to_3d_direction(star)

        # Reference orientation (first frame)
        R_ref = self.quaternion_to_rotation_matrix(imu_data.quaternions[0])

        # Star direction in world frame
        star_world = R_ref.T @ star_dir

        # Sample trajectory at num_points
        indices = np.linspace(0, len(imu_data.quaternions) - 1, num_points).astype(int)
        trajectory = []

        for idx in indices:
            R = self.quaternion_to_rotation_matrix(imu_data.quaternions[idx])

            # Project star to current camera frame
            star_cam = R @ star_world

            if star_cam[2] > 0:
                x, y = self.project_point(star_world, R)
                if x is not None:
                    trajectory.append([x, y])

        return np.array(trajectory) if trajectory else np.array([])

    def render_blurred_image(self,
                              star_field: SyntheticStarField,
                              imu_data: IMUData,
                              num_subframes: int = 50,
                              handle_wraparound: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Render motion-blurred star field image.

        This method handles the corner case where star trails can overlap
        when the frame shifts significantly during exposure.

        Args:
            star_field: Star field to render
            imu_data: IMU motion data
            num_subframes: Number of sub-exposures to simulate
            handle_wraparound: Enable wrap-around handling for large motions

        Returns:
            Tuple of (blurred_image, metadata_dict)
        """
        image = np.zeros((self.height, self.width), dtype=np.float64)
        image += star_field.background_level

        # Track which stars are visible at which times
        star_visibility = {}  # star_idx -> list of (time_idx, x, y)

        sigma = star_field.star_fwhm / 2.355

        # Reference orientation
        R_ref = self.quaternion_to_rotation_matrix(imu_data.quaternions[0])

        # Pre-compute star directions in world frame
        star_world_dirs = []
        for star in star_field.stars:
            star_dir = self.star_to_3d_direction(star)
            star_world = R_ref.T @ star_dir
            star_world_dirs.append(star_world)

        # Sample subframes evenly across exposure
        subframe_indices = np.linspace(0, len(imu_data.quaternions) - 1, num_subframes).astype(int)
        flux_per_subframe = 1.0 / num_subframes

        # Track stars entering/leaving frame for overlap detection
        stars_in_frame = set()
        overlap_regions = []

        for sub_idx, time_idx in enumerate(subframe_indices):
            R = self.quaternion_to_rotation_matrix(imu_data.quaternions[time_idx])

            current_stars_in_frame = set()

            for star_idx, (star, star_world) in enumerate(zip(star_field.stars, star_world_dirs)):
                # Project star to current camera frame
                star_cam = R @ star_world

                if star_cam[2] <= 0:
                    continue

                x, y = self.project_point(star_world, R)

                if x is None:
                    continue

                # Check if in frame (with margin for PSF)
                margin = 3 * sigma
                in_frame = (-margin <= x < self.width + margin and
                           -margin <= y < self.height + margin)

                if in_frame:
                    current_stars_in_frame.add(star_idx)

                    # Track visibility
                    if star_idx not in star_visibility:
                        star_visibility[star_idx] = []
                    star_visibility[star_idx].append((time_idx, x, y))

                    # Render star at this position
                    x_int, y_int = int(round(x)), int(round(y))

                    size = int(4 * sigma) + 1
                    x_min = max(0, x_int - size)
                    x_max = min(self.width, x_int + size + 1)
                    y_min = max(0, y_int - size)
                    y_max = min(self.height, y_int + size + 1)

                    if x_max > x_min and y_max > y_min:
                        y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
                        dx = x_grid - x
                        dy = y_grid - y
                        gaussian = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                        gaussian *= star.flux * flux_per_subframe / (2 * np.pi * sigma**2)

                        image[y_min:y_max, x_min:x_max] += gaussian

            # Detect stars entering/leaving frame (for overlap handling)
            if handle_wraparound:
                entering = current_stars_in_frame - stars_in_frame
                leaving = stars_in_frame - current_stars_in_frame

                if entering or leaving:
                    overlap_regions.append({
                        'time_idx': time_idx,
                        'entering': list(entering),
                        'leaving': list(leaving)
                    })

            stars_in_frame = current_stars_in_frame

        # Compute motion statistics
        max_motion = 0
        motion_vectors = []

        for star_idx, positions in star_visibility.items():
            if len(positions) > 1:
                start = np.array(positions[0][1:])
                end = np.array(positions[-1][1:])
                motion = np.linalg.norm(end - start)
                max_motion = max(max_motion, motion)
                motion_vectors.append(end - start)

        avg_motion = np.mean([np.linalg.norm(v) for v in motion_vectors]) if motion_vectors else 0

        metadata = {
            'max_motion_pixels': max_motion,
            'avg_motion_pixels': avg_motion,
            'num_visible_stars': len(star_visibility),
            'overlap_events': len(overlap_regions),
            'overlap_regions': overlap_regions,
            'star_visibility': star_visibility
        }

        return image, metadata


def generate_test_dataset(output_dir: str = None,
                          num_samples: int = 5,
                          exposure_times: List[float] = [1.0, 5.0, 10.0]):
    """
    Generate a test dataset with various motion scenarios.

    Args:
        output_dir: Directory to save images (None for no saving)
        num_samples: Number of samples per scenario
        exposure_times: List of exposure durations to test
    """
    import os

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    scenarios = []

    for exp_time in exposure_times:
        for scenario_name in ['drift', 'vibration', 'combined']:
            print(f"Generating {scenario_name} motion, {exp_time}s exposure...")

            # Create star field
            star_field = SyntheticStarField(
                width=1920, height=1080,
                num_stars=150,
                seed=42
            )

            # Generate sharp reference
            sharp_image = star_field.render_sharp_image()

            # Create motion simulator
            motion_sim = IMUMotionSimulator(duration=exp_time, sample_rate=200)

            # Generate motion based on scenario
            if scenario_name == 'drift':
                imu_data = motion_sim.generate_drift_motion(drift_rate=0.5)
            elif scenario_name == 'vibration':
                imu_data = motion_sim.generate_vibration_motion()
            else:
                imu_data = motion_sim.generate_combined_motion(
                    drift_rate=0.3,
                    vib_frequencies=[5.0, 12.0],
                    vib_amplitudes=[0.08, 0.03]
                )

            # Render blurred image
            renderer = MotionBlurRenderer(
                width=star_field.width,
                height=star_field.height,
                focal_length_px=1200
            )

            blurred_image, metadata = renderer.render_blurred_image(
                star_field, imu_data,
                num_subframes=int(exp_time * 30)  # 30 subframes per second
            )

            # Add noise
            blurred_noisy = star_field.add_noise(blurred_image)
            sharp_noisy = star_field.add_noise(sharp_image)

            scenario = {
                'name': f"{scenario_name}_{exp_time}s",
                'sharp_image': sharp_noisy,
                'blurred_image': blurred_noisy,
                'imu_data': imu_data,
                'star_field': star_field,
                'metadata': metadata
            }
            scenarios.append(scenario)

            print(f"  Max motion: {metadata['max_motion_pixels']:.1f} px")
            print(f"  Overlap events: {metadata['overlap_events']}")

            if output_dir:
                # Save images
                cv2.imwrite(
                    os.path.join(output_dir, f"{scenario['name']}_sharp.png"),
                    np.clip(sharp_noisy, 0, 65535).astype(np.uint16)
                )
                cv2.imwrite(
                    os.path.join(output_dir, f"{scenario['name']}_blurred.png"),
                    np.clip(blurred_noisy, 0, 65535).astype(np.uint16)
                )

                # Save IMU data
                np.savez(
                    os.path.join(output_dir, f"{scenario['name']}_imu.npz"),
                    timestamps=imu_data.timestamps,
                    quaternions=imu_data.quaternions,
                    angular_velocity=imu_data.angular_velocity
                )

    return scenarios


if __name__ == '__main__':
    # Demo: Generate test dataset
    print("Generating synthetic test dataset...")
    scenarios = generate_test_dataset(
        output_dir='motion_deblur/test_data',
        num_samples=1,
        exposure_times=[2.0, 5.0, 10.0]
    )
    print(f"\nGenerated {len(scenarios)} test scenarios")

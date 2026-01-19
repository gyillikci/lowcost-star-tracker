#!/usr/bin/env python3
"""
Gyroscope Drift Compensation with Star-Aided Correction.

This module addresses the MEMS gyroscope drift limitation by using
detected star positions to correct accumulated attitude errors.

Key features:
- Online bias estimation during observation
- Star-aided drift correction using matched star positions
- Temperature-aware compensation model
- Adaptive Kalman filtering for optimal fusion
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import time


@dataclass
class GyroParameters:
    """Gyroscope noise and drift parameters."""
    # Angle Random Walk (ARW) in deg/sqrt(hr)
    arw: float = 0.42  # Typical MEMS: 0.3-1.0
    # Bias Instability in deg/hr
    bias_instability: float = 3.0  # Typical MEMS: 1-10
    # Initial bias estimate (deg/s)
    initial_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Bias random walk in deg/hr/sqrt(hr)
    bias_random_walk: float = 0.1
    # Temperature coefficient (deg/s/°C)
    temp_coefficient: float = 0.003


@dataclass
class StarMatch:
    """A matched star observation."""
    # Detected position in image (pixels)
    detected_x: float
    detected_y: float
    # Catalog position (unit vector in camera frame)
    catalog_direction: np.ndarray
    # Detection timestamp
    timestamp: float
    # Match confidence (0-1)
    confidence: float = 1.0


class DriftCompensator:
    """
    Compensates for gyroscope drift using star observations.

    The algorithm:
    1. Integrates gyro measurements to track attitude
    2. When stars are detected, computes attitude from star positions
    3. Fuses gyro and star-based attitudes using Kalman filter
    4. Estimates and compensates for gyro bias in real-time
    """

    def __init__(self,
                 gyro_params: GyroParameters = None,
                 camera_matrix: np.ndarray = None,
                 update_interval: float = 1.0):
        """
        Initialize drift compensator.

        Args:
            gyro_params: Gyroscope noise parameters
            camera_matrix: 3x3 camera intrinsic matrix
            update_interval: Minimum time between star updates (seconds)
        """
        self.params = gyro_params or GyroParameters()

        # Default camera matrix (can be overridden)
        if camera_matrix is None:
            # Assume 1920x1080, ~90° FOV
            fx = fy = 1000.0
            cx, cy = 960.0, 540.0
            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        self.K = camera_matrix
        self.K_inv = np.linalg.inv(camera_matrix)

        self.update_interval = update_interval

        # State: quaternion [w, x, y, z] and gyro bias [bx, by, bz]
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.gyro_bias = self.params.initial_bias.copy()

        # Covariance matrix (7x7: 4 quaternion + 3 bias)
        # Using reduced 6x6 for attitude error + bias
        self.P = np.eye(6) * 0.01

        # Process noise
        self._compute_process_noise()

        # Tracking
        self.last_update_time = None
        self.last_star_update_time = None
        self.correction_history = []

    def _compute_process_noise(self):
        """Compute process noise covariance from gyro parameters."""
        # Convert units: deg to rad, hr to s
        arw_rad = np.deg2rad(self.params.arw) / 60.0  # rad/sqrt(s)
        bias_rw_rad = np.deg2rad(self.params.bias_random_walk) / 3600.0  # rad/s/sqrt(s)

        # Process noise for attitude (ARW^2 * dt)
        self.sigma_attitude = arw_rad
        # Process noise for bias (bias RW^2 * dt)
        self.sigma_bias = bias_rw_rad

    def predict(self, gyro_measurement: np.ndarray, dt: float):
        """
        Predict step: integrate gyroscope measurement.

        Args:
            gyro_measurement: Angular velocity [wx, wy, wz] in rad/s
            dt: Time step in seconds
        """
        # Subtract estimated bias
        omega = gyro_measurement - self.gyro_bias

        # Quaternion integration (first-order)
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * self._quaternion_multiply(self.quaternion, omega_quat)

        # Update quaternion
        self.quaternion = self.quaternion + q_dot * dt
        self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)

        # Update covariance
        # Simplified: add process noise
        Q = np.eye(6)
        Q[:3, :3] *= (self.sigma_attitude * np.sqrt(dt))**2
        Q[3:, 3:] *= (self.sigma_bias * np.sqrt(dt))**2

        self.P = self.P + Q

    def update_with_stars(self,
                          star_matches: List[StarMatch],
                          current_time: float) -> Dict:
        """
        Update step: correct attitude using star observations.

        Args:
            star_matches: List of matched star observations
            current_time: Current timestamp

        Returns:
            Dictionary with correction metrics
        """
        if len(star_matches) < 2:
            return {"status": "insufficient_stars", "correction": 0.0}

        # Check update interval
        if (self.last_star_update_time is not None and
            current_time - self.last_star_update_time < self.update_interval):
            return {"status": "too_soon", "correction": 0.0}

        # Estimate attitude from star observations
        star_attitude = self._estimate_attitude_from_stars(star_matches)

        if star_attitude is None:
            return {"status": "estimation_failed", "correction": 0.0}

        # Compute attitude error (small angle approximation)
        q_error = self._quaternion_multiply(
            star_attitude,
            self._quaternion_conjugate(self.quaternion)
        )

        # Convert to rotation vector (small angle: 2 * [qx, qy, qz])
        if q_error[0] < 0:
            q_error = -q_error
        attitude_error = 2.0 * q_error[1:4]

        # Measurement noise (depends on centroid accuracy and number of stars)
        n_stars = len(star_matches)
        sigma_star = 0.001 / np.sqrt(n_stars)  # rad, improves with more stars
        R = np.eye(3) * sigma_star**2

        # Kalman gain
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)  # Attitude measurement

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        innovation = attitude_error
        dx = K @ innovation

        # Apply attitude correction
        delta_q = np.array([1.0, dx[0]/2, dx[1]/2, dx[2]/2])
        delta_q = delta_q / np.linalg.norm(delta_q)
        self.quaternion = self._quaternion_multiply(delta_q, self.quaternion)
        self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)

        # Apply bias correction
        self.gyro_bias = self.gyro_bias + dx[3:6]

        # Update covariance
        self.P = (np.eye(6) - K @ H) @ self.P

        # Record correction
        correction_magnitude = np.linalg.norm(attitude_error)
        self.correction_history.append({
            "time": current_time,
            "correction_rad": correction_magnitude,
            "correction_deg": np.rad2deg(correction_magnitude),
            "bias_estimate": self.gyro_bias.copy(),
            "n_stars": n_stars
        })

        self.last_star_update_time = current_time

        return {
            "status": "success",
            "correction_deg": np.rad2deg(correction_magnitude),
            "bias_estimate_deg_s": np.rad2deg(self.gyro_bias),
            "n_stars": n_stars
        }

    def _estimate_attitude_from_stars(self,
                                       star_matches: List[StarMatch]) -> Optional[np.ndarray]:
        """
        Estimate attitude quaternion from star observations using QUEST algorithm.

        Args:
            star_matches: List of matched stars

        Returns:
            Attitude quaternion or None if estimation fails
        """
        if len(star_matches) < 2:
            return None

        # Build observation and reference vectors
        obs_vectors = []  # Detected directions in camera frame
        ref_vectors = []  # Catalog directions in reference frame
        weights = []

        for match in star_matches:
            # Convert pixel to unit vector
            pixel = np.array([match.detected_x, match.detected_y, 1.0])
            direction = self.K_inv @ pixel
            direction = direction / np.linalg.norm(direction)

            obs_vectors.append(direction)
            ref_vectors.append(match.catalog_direction)
            weights.append(match.confidence)

        obs_vectors = np.array(obs_vectors)
        ref_vectors = np.array(ref_vectors)
        weights = np.array(weights)
        weights = weights / weights.sum()

        # QUEST algorithm (simplified)
        # Build B matrix
        B = np.zeros((3, 3))
        for i in range(len(weights)):
            B += weights[i] * np.outer(obs_vectors[i], ref_vectors[i])

        # SVD to find rotation
        U, S, Vt = np.linalg.svd(B)

        # Ensure proper rotation (det = +1)
        det = np.linalg.det(U @ Vt)
        if det < 0:
            U[:, -1] *= -1

        R = U @ Vt

        # Convert to quaternion
        return self._rotation_matrix_to_quaternion(R)

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def _quaternion_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Conjugate of quaternion [w, x, y, z]."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
        rot = Rotation.from_matrix(R)
        q = rot.as_quat()  # Returns [x, y, z, w]
        return np.array([q[3], q[0], q[1], q[2]])  # Convert to [w, x, y, z]

    def get_rotation_matrix(self) -> np.ndarray:
        """Get current attitude as rotation matrix."""
        w, x, y, z = self.quaternion
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])

    def get_drift_statistics(self) -> Dict:
        """Get statistics about drift corrections."""
        if not self.correction_history:
            return {"status": "no_corrections"}

        corrections = [c["correction_deg"] for c in self.correction_history]

        return {
            "n_corrections": len(corrections),
            "total_correction_deg": sum(corrections),
            "mean_correction_deg": np.mean(corrections),
            "max_correction_deg": max(corrections),
            "final_bias_estimate_deg_s": np.rad2deg(self.gyro_bias).tolist(),
            "correction_history": self.correction_history
        }


class AdaptiveBiasEstimator:
    """
    Estimates gyroscope bias adaptively during observation.

    Uses periods of detected stars to estimate and track bias drift.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize bias estimator.

        Args:
            window_size: Number of samples for running average
        """
        self.window_size = window_size
        self.bias_samples = []
        self.current_estimate = np.zeros(3)
        self.uncertainty = np.ones(3) * 0.01  # rad/s

    def add_sample(self,
                   gyro_measurement: np.ndarray,
                   star_rate: np.ndarray,
                   confidence: float = 1.0):
        """
        Add a bias sample from gyro vs star-derived rate comparison.

        Args:
            gyro_measurement: Raw gyro measurement (rad/s)
            star_rate: Angular rate derived from star motion (rad/s)
            confidence: Confidence in star rate measurement
        """
        bias_sample = gyro_measurement - star_rate

        self.bias_samples.append({
            "bias": bias_sample,
            "confidence": confidence,
            "timestamp": time.time()
        })

        # Keep only recent samples
        if len(self.bias_samples) > self.window_size:
            self.bias_samples.pop(0)

        # Update estimate
        self._update_estimate()

    def _update_estimate(self):
        """Update bias estimate from samples."""
        if not self.bias_samples:
            return

        # Weighted average
        biases = np.array([s["bias"] for s in self.bias_samples])
        weights = np.array([s["confidence"] for s in self.bias_samples])
        weights = weights / weights.sum()

        self.current_estimate = np.average(biases, axis=0, weights=weights)

        # Estimate uncertainty from variance
        if len(biases) > 1:
            self.uncertainty = np.std(biases, axis=0)

    def get_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current bias estimate and uncertainty.

        Returns:
            Tuple of (bias_estimate, uncertainty) in rad/s
        """
        return self.current_estimate.copy(), self.uncertainty.copy()


def simulate_drift_compensation():
    """Demonstrate drift compensation with synthetic data."""
    print("=" * 60)
    print("Gyroscope Drift Compensation Demonstration")
    print("=" * 60)

    # Initialize compensator
    compensator = DriftCompensator(
        gyro_params=GyroParameters(
            arw=0.5,
            bias_instability=5.0,
            initial_bias=np.zeros(3)
        ),
        update_interval=1.0
    )

    # Simulate observation
    duration = 60.0  # seconds
    dt = 0.01  # 100 Hz
    star_update_interval = 5.0  # Update from stars every 5 seconds

    # True bias (unknown to the filter)
    true_bias = np.array([0.001, -0.002, 0.0005])  # rad/s (~0.06, -0.11, 0.03 deg/s)

    # Simulate
    t = 0.0
    last_star_time = 0.0
    true_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    errors = []

    while t < duration:
        # Generate gyro measurement with true bias and noise
        true_omega = np.array([0.01, 0.005, -0.002])  # True rotation rate
        noise = np.random.normal(0, 0.001, 3)  # Gyro noise
        gyro_measurement = true_omega + true_bias + noise

        # Update true quaternion
        omega_quat = np.array([0, true_omega[0], true_omega[1], true_omega[2]])
        q_dot = 0.5 * compensator._quaternion_multiply(true_quaternion, omega_quat)
        true_quaternion = true_quaternion + q_dot * dt
        true_quaternion = true_quaternion / np.linalg.norm(true_quaternion)

        # Predict with gyro
        compensator.predict(gyro_measurement, dt)

        # Periodic star updates
        if t - last_star_time >= star_update_interval:
            # Simulate star observations (from true attitude with small noise)
            n_stars = np.random.randint(5, 15)
            star_matches = []

            for _ in range(n_stars):
                # Random catalog direction
                catalog_dir = np.random.randn(3)
                catalog_dir = catalog_dir / np.linalg.norm(catalog_dir)

                # Project to image using true attitude
                R_true = Rotation.from_quat([
                    true_quaternion[1], true_quaternion[2],
                    true_quaternion[3], true_quaternion[0]
                ]).as_matrix()

                cam_dir = R_true @ catalog_dir
                if cam_dir[2] > 0.1:  # In front of camera
                    # Project to pixels with noise
                    pixel = compensator.K @ cam_dir
                    pixel = pixel[:2] / pixel[2]
                    pixel += np.random.normal(0, 0.5, 2)  # Centroid noise

                    star_matches.append(StarMatch(
                        detected_x=pixel[0],
                        detected_y=pixel[1],
                        catalog_direction=catalog_dir,
                        timestamp=t,
                        confidence=1.0
                    ))

            # Apply star update
            result = compensator.update_with_stars(star_matches, t)
            if result["status"] == "success":
                print(f"  t={t:.1f}s: Correction {result['correction_deg']:.4f}°, "
                      f"Bias est: {result['bias_estimate_deg_s']} deg/s")

            last_star_time = t

        # Compute attitude error
        q_error = compensator._quaternion_multiply(
            true_quaternion,
            compensator._quaternion_conjugate(compensator.quaternion)
        )
        error_rad = 2.0 * np.arcsin(np.linalg.norm(q_error[1:4]))
        errors.append(np.rad2deg(error_rad))

        t += dt

    # Final statistics
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    stats = compensator.get_drift_statistics()
    print(f"Number of corrections: {stats['n_corrections']}")
    print(f"Total correction applied: {stats['total_correction_deg']:.4f}°")
    print(f"Final bias estimate: {stats['final_bias_estimate_deg_s']} deg/s")
    print(f"True bias: {np.rad2deg(true_bias).tolist()} deg/s")

    print(f"\nAttitude error statistics:")
    print(f"  Final error: {errors[-1]:.4f}°")
    print(f"  Mean error: {np.mean(errors):.4f}°")
    print(f"  Max error: {max(errors):.4f}°")

    # Without compensation, error would be ~bias * duration
    uncorrected_error = np.linalg.norm(true_bias) * duration
    print(f"\nWithout compensation, error would be: {np.rad2deg(uncorrected_error):.2f}°")
    print(f"Improvement factor: {np.rad2deg(uncorrected_error) / max(errors):.1f}x")

    return compensator, errors


if __name__ == "__main__":
    simulate_drift_compensation()

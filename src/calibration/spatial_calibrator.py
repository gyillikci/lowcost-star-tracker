#!/usr/bin/env python3
"""
Spatial Calibrator Module.

Estimates the rotation matrix that aligns the IMU coordinate frame
with the camera coordinate frame. This is essential for correctly
applying gyroscope measurements to camera motion compensation.

The extrinsic rotation R_cam_imu transforms vectors from IMU frame
to camera frame: v_cam = R_cam_imu @ v_imu

Methods:
- Hand-eye calibration (AX=XB problem)
- Procrustes alignment
- Continuous optimization
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from scipy.linalg import svd


@dataclass
class SpatialCalibrationResult:
    """Result of spatial calibration."""
    # Rotation from IMU frame to camera frame
    R_cam_imu: np.ndarray  # 3x3 rotation matrix

    # Euler angles for interpretability
    euler_angles_deg: np.ndarray  # roll, pitch, yaw in degrees

    # Calibration quality metrics
    residual_error: float  # Mean rotation error in radians
    confidence: float  # Calibration confidence (0-1)

    # Method used
    method: str


class SpatialCalibrator:
    """
    Spatial calibration between camera and IMU.

    Determines the rotation matrix that transforms vectors
    from the IMU coordinate frame to the camera coordinate frame.
    """

    def __init__(self):
        """Initialize spatial calibrator."""
        self.result: Optional[SpatialCalibrationResult] = None

        # Collected rotation pairs for calibration
        self.rotations_camera: List[np.ndarray] = []  # Camera rotations
        self.rotations_imu: List[np.ndarray] = []  # IMU rotations

    def add_rotation_pair(self,
                          R_camera: np.ndarray,
                          R_imu: np.ndarray):
        """
        Add a pair of corresponding rotations for calibration.

        Args:
            R_camera: Rotation observed by camera (3x3)
            R_imu: Rotation measured by IMU (3x3)
        """
        self.rotations_camera.append(R_camera.copy())
        self.rotations_imu.append(R_imu.copy())

    def clear_rotations(self):
        """Clear collected rotation pairs."""
        self.rotations_camera = []
        self.rotations_imu = []

    def calibrate_procrustes(self) -> SpatialCalibrationResult:
        """
        Calibrate using Procrustes analysis.

        Finds R_cam_imu that minimizes:
            sum_i || R_camera_i - R_cam_imu @ R_imu_i @ R_cam_imu^T ||^2

        Returns:
            SpatialCalibrationResult with optimal rotation
        """
        if len(self.rotations_camera) < 3:
            raise ValueError("Need at least 3 rotation pairs for calibration")

        n = len(self.rotations_camera)

        # Convert to axis-angle representation
        axes_camera = []
        axes_imu = []

        for R_cam, R_imu in zip(self.rotations_camera, self.rotations_imu):
            # Get rotation vectors
            r_cam = Rotation.from_matrix(R_cam).as_rotvec()
            r_imu = Rotation.from_matrix(R_imu).as_rotvec()

            # Normalize to get axes (if angle > 0)
            norm_cam = np.linalg.norm(r_cam)
            norm_imu = np.linalg.norm(r_imu)

            if norm_cam > 0.01 and norm_imu > 0.01:
                axes_camera.append(r_cam / norm_cam)
                axes_imu.append(r_imu / norm_imu)

        if len(axes_camera) < 3:
            raise ValueError("Need at least 3 significant rotations")

        axes_camera = np.array(axes_camera)
        axes_imu = np.array(axes_imu)

        # Procrustes: find R that minimizes ||axes_camera - axes_imu @ R^T||
        # Solution: R = V @ U^T where USV^T = axes_camera^T @ axes_imu

        H = axes_camera.T @ axes_imu
        U, S, Vt = svd(H)

        R_cam_imu = U @ Vt

        # Ensure proper rotation (det = 1)
        if np.linalg.det(R_cam_imu) < 0:
            U[:, -1] *= -1
            R_cam_imu = U @ Vt

        # Compute residual error
        residual = self._compute_residual_error(R_cam_imu)

        # Convert to Euler angles
        euler = Rotation.from_matrix(R_cam_imu).as_euler('xyz', degrees=True)

        # Confidence based on residual
        confidence = np.exp(-residual * 10)

        self.result = SpatialCalibrationResult(
            R_cam_imu=R_cam_imu,
            euler_angles_deg=euler,
            residual_error=residual,
            confidence=confidence,
            method="procrustes"
        )

        return self.result

    def calibrate_hand_eye(self) -> SpatialCalibrationResult:
        """
        Calibrate using hand-eye calibration (AX=XB formulation).

        Classic hand-eye calibration problem:
            R_camera @ X = X @ R_imu

        Uses the method of Park and Martin (1994).

        Returns:
            SpatialCalibrationResult with optimal rotation
        """
        if len(self.rotations_camera) < 2:
            raise ValueError("Need at least 2 rotation pairs")

        n = len(self.rotations_camera)

        # Build the system of equations
        # For each pair: log(R_cam) = X @ log(R_imu) @ X^T
        # In axis-angle: alpha_cam = X @ alpha_imu

        M = np.zeros((3, 3))

        for R_cam, R_imu in zip(self.rotations_camera, self.rotations_imu):
            # Get rotation vectors (axis-angle)
            alpha_cam = Rotation.from_matrix(R_cam).as_rotvec()
            alpha_imu = Rotation.from_matrix(R_imu).as_rotvec()

            # Build correlation matrix
            M += np.outer(alpha_imu, alpha_cam)

        # SVD solution
        U, S, Vt = svd(M)
        R_cam_imu = Vt.T @ U.T

        # Ensure proper rotation
        if np.linalg.det(R_cam_imu) < 0:
            Vt[-1, :] *= -1
            R_cam_imu = Vt.T @ U.T

        # Compute residual
        residual = self._compute_residual_error(R_cam_imu)

        # Euler angles
        euler = Rotation.from_matrix(R_cam_imu).as_euler('xyz', degrees=True)

        # Confidence
        confidence = np.exp(-residual * 10)

        self.result = SpatialCalibrationResult(
            R_cam_imu=R_cam_imu,
            euler_angles_deg=euler,
            residual_error=residual,
            confidence=confidence,
            method="hand_eye"
        )

        return self.result

    def calibrate_optimization(self,
                               initial_guess: np.ndarray = None) -> SpatialCalibrationResult:
        """
        Calibrate using continuous optimization.

        Directly optimizes Euler angles to minimize rotation error.

        Args:
            initial_guess: Initial Euler angles in degrees (optional)

        Returns:
            SpatialCalibrationResult with optimal rotation
        """
        if len(self.rotations_camera) < 2:
            raise ValueError("Need at least 2 rotation pairs")

        # Initial guess
        if initial_guess is None:
            # Try Procrustes for initial guess
            try:
                procrustes_result = self.calibrate_procrustes()
                initial_guess = procrustes_result.euler_angles_deg
            except Exception:
                initial_guess = np.zeros(3)

        # Objective function
        def objective(euler_deg):
            R = Rotation.from_euler('xyz', euler_deg, degrees=True).as_matrix()
            return self._compute_residual_error(R)

        # Optimize
        result = minimize(
            objective,
            initial_guess,
            method='Powell',
            options={'maxiter': 1000}
        )

        optimal_euler = result.x
        R_cam_imu = Rotation.from_euler('xyz', optimal_euler, degrees=True).as_matrix()
        residual = result.fun

        # Confidence
        confidence = np.exp(-residual * 10)

        self.result = SpatialCalibrationResult(
            R_cam_imu=R_cam_imu,
            euler_angles_deg=optimal_euler,
            residual_error=residual,
            confidence=confidence,
            method="optimization"
        )

        return self.result

    def calibrate_from_streams(self,
                               gyro_stream,  # GyroStream
                               video_stream,  # VideoStream
                               time_offset: float = 0.0) -> SpatialCalibrationResult:
        """
        Calibrate from gyro and video streams.

        Extracts rotation pairs from synchronized streams and
        performs calibration.

        Args:
            gyro_stream: GyroStream with IMU data
            video_stream: VideoStream with tracked features
            time_offset: Time offset (video_time = gyro_time + offset)

        Returns:
            SpatialCalibrationResult with optimal rotation
        """
        self.clear_rotations()

        # For each frame pair, extract rotations
        frames = list(video_stream.iterate_frames())

        for i in range(1, len(frames)):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]

            # Get camera rotation
            rvec = video_stream.compute_frame_rotation(i)
            if rvec is None:
                continue

            R_camera = Rotation.from_rotvec(rvec).as_matrix()

            # Get IMU rotation (integrate gyro)
            t_start = prev_frame.timestamp - time_offset
            t_end = curr_frame.timestamp - time_offset

            try:
                times, quats = gyro_stream.integrate_to_quaternion(
                    t_start=t_start, t_end=t_end
                )

                if len(quats) < 2:
                    continue

                # Rotation from first to last quaternion
                q_start = quats[0]
                q_end = quats[-1]

                # Relative rotation
                R_start = Rotation.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]]).as_matrix()
                R_end = Rotation.from_quat([q_end[1], q_end[2], q_end[3], q_end[0]]).as_matrix()
                R_imu = R_end @ R_start.T

                self.add_rotation_pair(R_camera, R_imu)

            except Exception:
                continue

        if len(self.rotations_camera) < 3:
            raise ValueError("Could not extract enough rotation pairs")

        # Calibrate using optimization (most robust)
        return self.calibrate_optimization()

    def _compute_residual_error(self, R_cam_imu: np.ndarray) -> float:
        """
        Compute mean rotation error for given R_cam_imu.

        Error is the geodesic distance between predicted and
        observed camera rotations.
        """
        errors = []

        for R_cam, R_imu in zip(self.rotations_camera, self.rotations_imu):
            # Predicted camera rotation
            R_cam_pred = R_cam_imu @ R_imu @ R_cam_imu.T

            # Error rotation
            R_error = R_cam_pred.T @ R_cam

            # Geodesic distance (rotation angle)
            trace = np.trace(R_error)
            trace = np.clip(trace, -1, 3)
            angle = np.arccos((trace - 1) / 2)

            errors.append(angle)

        return np.mean(errors)

    def refine_with_gyro_bias(self,
                              gyro_stream,
                              video_stream,
                              time_offset: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Jointly refine R_cam_imu and gyro bias.

        Args:
            gyro_stream: GyroStream
            video_stream: VideoStream
            time_offset: Time offset

        Returns:
            Tuple of (R_cam_imu, gyro_bias)
        """
        if self.result is None:
            self.calibrate_from_streams(gyro_stream, video_stream, time_offset)

        # Initial values
        initial_euler = self.result.euler_angles_deg
        initial_bias = gyro_stream.bias_estimate.copy()

        # Joint optimization
        def objective(params):
            euler_deg = params[:3]
            bias = params[3:]

            # Update gyro stream bias temporarily
            old_bias = gyro_stream.bias_estimate.copy()
            gyro_stream.bias_estimate = bias

            # Collect rotation pairs with new bias
            self.clear_rotations()

            frames = list(video_stream.iterate_frames())
            for i in range(1, len(frames)):
                rvec = video_stream.compute_frame_rotation(i)
                if rvec is None:
                    continue

                R_camera = Rotation.from_rotvec(rvec).as_matrix()

                t_start = frames[i-1].timestamp - time_offset
                t_end = frames[i].timestamp - time_offset

                try:
                    times, quats = gyro_stream.integrate_to_quaternion(
                        t_start=t_start, t_end=t_end
                    )
                    if len(quats) >= 2:
                        q_start, q_end = quats[0], quats[-1]
                        R_start = Rotation.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]]).as_matrix()
                        R_end = Rotation.from_quat([q_end[1], q_end[2], q_end[3], q_end[0]]).as_matrix()
                        R_imu = R_end @ R_start.T
                        self.add_rotation_pair(R_camera, R_imu)
                except Exception:
                    pass

            # Restore bias
            gyro_stream.bias_estimate = old_bias

            if len(self.rotations_camera) < 3:
                return 1e6

            R_cam_imu = Rotation.from_euler('xyz', euler_deg, degrees=True).as_matrix()
            return self._compute_residual_error(R_cam_imu)

        # Optimize
        initial_params = np.concatenate([initial_euler, initial_bias])

        result = minimize(
            objective,
            initial_params,
            method='Powell',
            options={'maxiter': 500}
        )

        optimal_euler = result.x[:3]
        optimal_bias = result.x[3:]

        R_cam_imu = Rotation.from_euler('xyz', optimal_euler, degrees=True).as_matrix()

        # Update result
        self.result = SpatialCalibrationResult(
            R_cam_imu=R_cam_imu,
            euler_angles_deg=optimal_euler,
            residual_error=result.fun,
            confidence=np.exp(-result.fun * 10),
            method="joint_optimization"
        )

        return R_cam_imu, optimal_bias


def demonstrate_spatial_calibration():
    """Demonstrate spatial calibration with synthetic data."""
    print("=" * 60)
    print("Spatial Calibration Demonstration")
    print("=" * 60)

    np.random.seed(42)

    # True R_cam_imu (rotation from IMU to camera frame)
    true_euler_deg = np.array([5.0, -3.0, 2.0])  # roll, pitch, yaw
    true_R = Rotation.from_euler('xyz', true_euler_deg, degrees=True).as_matrix()

    print(f"\nTrue R_cam_imu Euler angles: {true_euler_deg} deg")

    # Generate synthetic rotation pairs
    calibrator = SpatialCalibrator()
    n_pairs = 20

    for _ in range(n_pairs):
        # Random IMU rotation
        random_axis = np.random.randn(3)
        random_axis /= np.linalg.norm(random_axis)
        random_angle = np.random.uniform(0.05, 0.3)

        R_imu = Rotation.from_rotvec(random_angle * random_axis).as_matrix()

        # Camera rotation = R_cam_imu @ R_imu @ R_cam_imu^T
        R_camera = true_R @ R_imu @ true_R.T

        # Add noise
        noise_angle = np.random.uniform(0, 0.02)
        noise_axis = np.random.randn(3)
        noise_axis /= np.linalg.norm(noise_axis)
        R_noise = Rotation.from_rotvec(noise_angle * noise_axis).as_matrix()
        R_camera = R_noise @ R_camera

        calibrator.add_rotation_pair(R_camera, R_imu)

    print(f"\nCollected {len(calibrator.rotations_camera)} rotation pairs")

    # Test different methods
    print("\n1. Procrustes method:")
    result1 = calibrator.calibrate_procrustes()
    error1 = np.linalg.norm(result1.euler_angles_deg - true_euler_deg)
    print(f"   Estimated: {result1.euler_angles_deg}")
    print(f"   Error: {error1:.3f} deg")
    print(f"   Residual: {np.rad2deg(result1.residual_error):.4f} deg")

    print("\n2. Hand-eye method:")
    result2 = calibrator.calibrate_hand_eye()
    error2 = np.linalg.norm(result2.euler_angles_deg - true_euler_deg)
    print(f"   Estimated: {result2.euler_angles_deg}")
    print(f"   Error: {error2:.3f} deg")
    print(f"   Residual: {np.rad2deg(result2.residual_error):.4f} deg")

    print("\n3. Optimization method:")
    result3 = calibrator.calibrate_optimization()
    error3 = np.linalg.norm(result3.euler_angles_deg - true_euler_deg)
    print(f"   Estimated: {result3.euler_angles_deg}")
    print(f"   Error: {error3:.3f} deg")
    print(f"   Residual: {np.rad2deg(result3.residual_error):.4f} deg")

    return calibrator


if __name__ == "__main__":
    demonstrate_spatial_calibration()

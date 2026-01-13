#!/usr/bin/env python3
"""
Auto Calibrator Module.

Provides automatic calibration of camera-IMU systems for star tracker
applications. Orchestrates temporal synchronization, spatial alignment,
and parameter estimation.

Inspired by CRISP but tailored for star tracker use cases.

Usage:
    calibrator = AutoCalibrator(camera_model)
    calibrator.load_data(video_path, imu_data_path)
    calibrator.initialize()
    results = calibrator.calibrate()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

from .gyro_stream import GyroStream, GyroParameters
from .video_stream import VideoStream
from .camera_model import CameraModel, CameraIntrinsics
from .temporal_calibrator import TemporalCalibrator, TemporalCalibrationResult
from .spatial_calibrator import SpatialCalibrator, SpatialCalibrationResult


@dataclass
class CalibrationResults:
    """Complete calibration results."""
    # Temporal calibration
    time_offset: float  # seconds (video_time = imu_time + offset)
    time_offset_confidence: float

    # Spatial calibration
    R_cam_imu: np.ndarray  # 3x3 rotation matrix
    euler_angles_deg: np.ndarray  # roll, pitch, yaw
    spatial_confidence: float

    # Gyroscope parameters
    gyro_bias: np.ndarray  # rad/s
    gyro_sample_rate: float  # Hz

    # Camera parameters
    camera_matrix: np.ndarray  # 3x3
    readout_time: float  # Rolling shutter time (seconds)

    # Quality metrics
    overall_confidence: float
    temporal_residual: float
    spatial_residual: float

    # Metadata
    n_frames_used: int
    calibration_duration: float
    method: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "time_offset": float(self.time_offset),
            "time_offset_confidence": float(self.time_offset_confidence),
            "R_cam_imu": self.R_cam_imu.tolist(),
            "euler_angles_deg": self.euler_angles_deg.tolist(),
            "spatial_confidence": float(self.spatial_confidence),
            "gyro_bias": self.gyro_bias.tolist(),
            "gyro_sample_rate": float(self.gyro_sample_rate),
            "camera_matrix": self.camera_matrix.tolist(),
            "readout_time": float(self.readout_time),
            "overall_confidence": float(self.overall_confidence),
            "temporal_residual": float(self.temporal_residual),
            "spatial_residual": float(self.spatial_residual),
            "n_frames_used": self.n_frames_used,
            "calibration_duration": float(self.calibration_duration),
            "method": self.method
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CalibrationResults':
        """Create from dictionary."""
        return cls(
            time_offset=data["time_offset"],
            time_offset_confidence=data["time_offset_confidence"],
            R_cam_imu=np.array(data["R_cam_imu"]),
            euler_angles_deg=np.array(data["euler_angles_deg"]),
            spatial_confidence=data["spatial_confidence"],
            gyro_bias=np.array(data["gyro_bias"]),
            gyro_sample_rate=data["gyro_sample_rate"],
            camera_matrix=np.array(data["camera_matrix"]),
            readout_time=data["readout_time"],
            overall_confidence=data["overall_confidence"],
            temporal_residual=data["temporal_residual"],
            spatial_residual=data["spatial_residual"],
            n_frames_used=data["n_frames_used"],
            calibration_duration=data["calibration_duration"],
            method=data["method"]
        )


class AutoCalibrator:
    """
    Automatic camera-IMU calibration pipeline.

    Performs:
    1. Feature detection and tracking in video
    2. Gyroscope bias estimation
    3. Temporal synchronization (time offset)
    4. Spatial alignment (rotation matrix)
    5. Joint refinement

    Designed for star tracker applications with GoPro or similar cameras.
    """

    def __init__(self, camera_model: CameraModel = None):
        """
        Initialize auto calibrator.

        Args:
            camera_model: Camera projection model (optional, can be set later)
        """
        self.camera_model = camera_model

        # Data streams
        self.gyro_stream: Optional[GyroStream] = None
        self.video_stream: Optional[VideoStream] = None

        # Sub-calibrators
        self.temporal_calibrator = TemporalCalibrator()
        self.spatial_calibrator = SpatialCalibrator()

        # Results
        self.results: Optional[CalibrationResults] = None

        # State
        self._initialized = False

    def load_data(self,
                  video_path: str = None,
                  imu_data_path: str = None,
                  imu_format: str = "csv",
                  video_start: float = 0.0,
                  video_duration: float = None,
                  max_frames: int = 500):
        """
        Load video and IMU data for calibration.

        Args:
            video_path: Path to video file
            imu_data_path: Path to IMU data file
            imu_format: IMU data format ("csv", "gopro", "json")
            video_start: Start time in video (seconds)
            video_duration: Duration to use (None = all)
            max_frames: Maximum frames to load
        """
        # Load gyro data
        if imu_data_path:
            self.gyro_stream = GyroStream()

            if imu_format == "gopro":
                self.gyro_stream.load_from_gopro(imu_data_path)
            elif imu_format == "csv":
                # Assume format: timestamp, gx, gy, gz (rad/s)
                self.gyro_stream.load_from_csv(
                    imu_data_path,
                    time_col=0, gyro_cols=(1, 2, 3),
                    time_scale=1.0, gyro_scale=1.0
                )
            elif imu_format == "json":
                with open(imu_data_path, 'r') as f:
                    data = json.load(f)
                timestamps = np.array(data['timestamps'])
                gyro = np.array(data['gyro'])
                self.gyro_stream.load_from_array(timestamps, gyro)

            print(f"Loaded IMU data: {self.gyro_stream}")

        # Load video
        if video_path:
            self.video_stream = VideoStream(self.camera_model)

            end_time = None
            if video_duration:
                end_time = video_start + video_duration

            self.video_stream.load_video(
                video_path,
                start_time=video_start,
                end_time=end_time,
                max_frames=max_frames
            )

            print(f"Loaded video: {self.video_stream}")

    def load_synthetic_data(self,
                            duration: float = 10.0,
                            gyro_rate: float = 200.0,
                            video_rate: float = 30.0,
                            true_time_offset: float = 0.05,
                            true_euler_deg: np.ndarray = None):
        """
        Load synthetic data for testing.

        Args:
            duration: Simulation duration in seconds
            gyro_rate: Gyroscope sample rate
            video_rate: Video frame rate
            true_time_offset: True time offset (video = imu + offset)
            true_euler_deg: True R_cam_imu Euler angles
        """
        import cv2

        np.random.seed(42)

        if true_euler_deg is None:
            true_euler_deg = np.array([5.0, -3.0, 2.0])

        # Generate gyro data
        gyro_times = np.arange(0, duration, 1.0 / gyro_rate)
        gyro_data = np.column_stack([
            0.3 * np.sin(2 * np.pi * 0.5 * gyro_times),
            0.2 * np.cos(2 * np.pi * 0.7 * gyro_times),
            0.1 * np.sin(2 * np.pi * 1.0 * gyro_times)
        ])
        gyro_data += np.random.normal(0, 0.01, gyro_data.shape)

        self.gyro_stream = GyroStream()
        self.gyro_stream.load_from_array(gyro_times, gyro_data)

        # Generate video frames with features
        width, height = 640, 480
        video_times = np.arange(0, duration, 1.0 / video_rate)

        n_features = 100
        base_positions = np.random.uniform(
            [50, 50], [width - 50, height - 50], (n_features, 2)
        )

        images = []

        from scipy.interpolate import interp1d
        from scipy.spatial.transform import Rotation

        # Interpolate gyro to video times (with offset)
        gyro_interp = interp1d(gyro_times, gyro_data, axis=0,
                               bounds_error=False, fill_value='extrapolate')

        # True R_cam_imu
        R_cam_imu = Rotation.from_euler('xyz', true_euler_deg, degrees=True).as_matrix()

        cumulative_rotation = np.eye(3)

        for i, t in enumerate(video_times):
            # Get gyro at shifted time
            gyro_t = t - true_time_offset
            omega = gyro_interp(gyro_t)

            # Transform to camera frame
            omega_cam = R_cam_imu @ omega

            # Integrate rotation
            dt = 1.0 / video_rate
            angle = np.linalg.norm(omega_cam) * dt
            if angle > 1e-6:
                axis = omega_cam / np.linalg.norm(omega_cam)
                dR = Rotation.from_rotvec(angle * axis).as_matrix()
                cumulative_rotation = dR @ cumulative_rotation

            # Project features
            center = np.array([width / 2, height / 2])
            R_2d = cumulative_rotation[:2, :2]
            positions = (base_positions - center) @ R_2d.T + center
            positions += np.random.normal(0, 0.5, positions.shape)

            # Create image
            img = np.zeros((height, width), dtype=np.uint8) + 20
            for pos in positions:
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(img, (x, y), 3, 255, -1)

            img = img.astype(np.float32) + np.random.normal(0, 5, img.shape)
            img = np.clip(img, 0, 255).astype(np.uint8)
            images.append(img)

        self.video_stream = VideoStream(self.camera_model)
        self.video_stream.load_frames_from_array(images, video_times)

        print(f"Generated synthetic data:")
        print(f"  Gyro: {self.gyro_stream}")
        print(f"  Video: {self.video_stream}")
        print(f"  True time offset: {true_time_offset*1000:.1f} ms")
        print(f"  True R_cam_imu: {true_euler_deg} deg")

        return true_time_offset, true_euler_deg

    def initialize(self):
        """
        Initialize calibration (detect features, estimate initial params).

        This step prepares data for calibration but doesn't compute
        final parameters.
        """
        if self.gyro_stream is None or self.video_stream is None:
            raise RuntimeError("Load data first with load_data()")

        print("\nInitializing calibration...")

        # 1. Detect and track features in video
        print("  Detecting features...")
        self.video_stream.detect_features(max_features=300)

        print("  Tracking features...")
        self.video_stream.track_features()

        # 2. Estimate gyro bias
        print("  Estimating gyro bias...")
        self.gyro_stream.estimate_bias_static()
        print(f"    Bias: {self.gyro_stream.bias_estimate * 1000} mrad/s")

        # 3. Resample gyro to uniform rate if needed
        if self.gyro_stream.sample_rate > 0:
            target_rate = min(500, self.gyro_stream.sample_rate)
            self.gyro_stream = self.gyro_stream.resample(target_rate)
            print(f"    Resampled to {target_rate} Hz")

        self._initialized = True
        print("  Initialization complete")

    def calibrate(self,
                  method: str = "full",
                  refine_bias: bool = True) -> CalibrationResults:
        """
        Perform full calibration.

        Args:
            method: Calibration method ("full", "temporal_only", "spatial_only")
            refine_bias: Whether to jointly refine gyro bias

        Returns:
            CalibrationResults with all parameters
        """
        if not self._initialized:
            self.initialize()

        print("\nRunning calibration...")
        import time
        start_time = time.time()

        # Get video-derived angular velocity
        video_times, video_omega = self.video_stream.get_angular_velocity_from_video()

        if len(video_omega) < 5:
            raise ValueError("Insufficient video data for calibration")

        # Default values
        time_offset = 0.0
        time_confidence = 0.0
        temporal_residual = 0.0

        R_cam_imu = np.eye(3)
        euler_deg = np.zeros(3)
        spatial_confidence = 0.0
        spatial_residual = 0.0

        # Step 1: Temporal calibration
        if method in ["full", "temporal_only"]:
            print("  Step 1: Temporal calibration...")
            temporal_result = self.temporal_calibrator.calibrate_cross_correlation(
                self.gyro_stream.timestamps,
                self.gyro_stream.gyro_data - self.gyro_stream.bias_estimate,
                video_times,
                video_omega
            )
            time_offset = temporal_result.time_offset
            time_confidence = temporal_result.confidence
            temporal_residual = 1.0 - temporal_result.correlation

            print(f"    Time offset: {time_offset*1000:.2f} ms")
            print(f"    Confidence: {time_confidence:.3f}")

        # Step 2: Spatial calibration
        if method in ["full", "spatial_only"]:
            print("  Step 2: Spatial calibration...")
            try:
                spatial_result = self.spatial_calibrator.calibrate_from_streams(
                    self.gyro_stream,
                    self.video_stream,
                    time_offset
                )
                R_cam_imu = spatial_result.R_cam_imu
                euler_deg = spatial_result.euler_angles_deg
                spatial_confidence = spatial_result.confidence
                spatial_residual = spatial_result.residual_error

                print(f"    R_cam_imu Euler: {euler_deg} deg")
                print(f"    Confidence: {spatial_confidence:.3f}")
            except Exception as e:
                print(f"    Spatial calibration failed: {e}")

        # Step 3: Joint refinement
        if method == "full" and refine_bias:
            print("  Step 3: Joint refinement...")
            try:
                R_cam_imu, refined_bias = self.spatial_calibrator.refine_with_gyro_bias(
                    self.gyro_stream,
                    self.video_stream,
                    time_offset
                )
                self.gyro_stream.bias_estimate = refined_bias
                euler_deg = self.spatial_calibrator.result.euler_angles_deg
                spatial_residual = self.spatial_calibrator.result.residual_error

                print(f"    Refined bias: {refined_bias * 1000} mrad/s")
                print(f"    Refined R_cam_imu: {euler_deg} deg")
            except Exception as e:
                print(f"    Refinement failed: {e}")

        # Compute overall confidence
        overall_confidence = np.sqrt(time_confidence * spatial_confidence)

        elapsed = time.time() - start_time

        # Get camera matrix
        if self.camera_model is not None:
            camera_matrix = self.camera_model.intrinsics.K
            readout_time = self.camera_model.intrinsics.readout_time
        else:
            camera_matrix = np.eye(3)
            readout_time = 0.0

        self.results = CalibrationResults(
            time_offset=time_offset,
            time_offset_confidence=time_confidence,
            R_cam_imu=R_cam_imu,
            euler_angles_deg=euler_deg,
            spatial_confidence=spatial_confidence,
            gyro_bias=self.gyro_stream.bias_estimate,
            gyro_sample_rate=self.gyro_stream.sample_rate,
            camera_matrix=camera_matrix,
            readout_time=readout_time,
            overall_confidence=overall_confidence,
            temporal_residual=temporal_residual,
            spatial_residual=spatial_residual,
            n_frames_used=len(self.video_stream),
            calibration_duration=elapsed,
            method=method
        )

        print(f"\nCalibration complete ({elapsed:.2f}s)")
        print(f"  Overall confidence: {overall_confidence:.3f}")

        return self.results

    def save_results(self, filepath: str):
        """Save calibration results to JSON file."""
        if self.results is None:
            raise RuntimeError("No results to save. Run calibrate() first.")

        with open(filepath, 'w') as f:
            json.dump(self.results.to_dict(), f, indent=2)

        print(f"Results saved to {filepath}")

    def load_results(self, filepath: str) -> CalibrationResults:
        """Load calibration results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.results = CalibrationResults.from_dict(data)
        return self.results

    def get_corrected_gyro_at(self, timestamp: float) -> np.ndarray:
        """
        Get calibrated gyroscope reading at a given time.

        Applies bias correction and transforms to camera frame.

        Args:
            timestamp: Time in video frame

        Returns:
            Angular velocity in camera frame [rad/s]
        """
        if self.results is None:
            raise RuntimeError("Calibration not complete")

        # Shift timestamp by time offset
        imu_time = timestamp - self.results.time_offset

        # Get bias-corrected gyro
        gyro_imu = self.gyro_stream.get_corrected_gyro(imu_time)

        # Transform to camera frame
        gyro_cam = self.results.R_cam_imu @ gyro_imu

        return gyro_cam

    def print_summary(self):
        """Print calibration summary."""
        if self.results is None:
            print("No calibration results available")
            return

        r = self.results

        print("\n" + "=" * 60)
        print("CALIBRATION SUMMARY")
        print("=" * 60)

        print("\nTemporal Calibration:")
        print(f"  Time offset: {r.time_offset*1000:.3f} ms")
        print(f"  Confidence: {r.time_offset_confidence:.3f}")

        print("\nSpatial Calibration (R_cam_imu):")
        print(f"  Roll:  {r.euler_angles_deg[0]:+.3f}°")
        print(f"  Pitch: {r.euler_angles_deg[1]:+.3f}°")
        print(f"  Yaw:   {r.euler_angles_deg[2]:+.3f}°")
        print(f"  Confidence: {r.spatial_confidence:.3f}")

        print("\nGyroscope:")
        print(f"  Bias: {r.gyro_bias*1000} mrad/s")
        print(f"  Sample rate: {r.gyro_sample_rate:.1f} Hz")

        print("\nQuality:")
        print(f"  Overall confidence: {r.overall_confidence:.3f}")
        print(f"  Temporal residual: {r.temporal_residual:.4f}")
        print(f"  Spatial residual: {np.rad2deg(r.spatial_residual):.4f}°")
        print(f"  Frames used: {r.n_frames_used}")

        print("=" * 60)


def demonstrate_auto_calibration():
    """Demonstrate full auto calibration with synthetic data."""
    print("=" * 60)
    print("Auto Calibration Demonstration")
    print("=" * 60)

    # Create calibrator
    calibrator = AutoCalibrator()

    # Generate synthetic data
    true_offset, true_euler = calibrator.load_synthetic_data(
        duration=10.0,
        true_time_offset=0.05,
        true_euler_deg=np.array([5.0, -3.0, 2.0])
    )

    # Run calibration
    results = calibrator.calibrate(method="full", refine_bias=True)

    # Print summary
    calibrator.print_summary()

    # Evaluate accuracy
    print("\nAccuracy Evaluation:")
    print(f"  Time offset error: {abs(results.time_offset - true_offset)*1000:.2f} ms")
    print(f"  Euler angle error: {np.linalg.norm(results.euler_angles_deg - true_euler):.3f}°")

    return calibrator


if __name__ == "__main__":
    demonstrate_auto_calibration()

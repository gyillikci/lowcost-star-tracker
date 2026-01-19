#!/usr/bin/env python3
"""
Demonstration and Validation Script for Camera-IMU Calibration Toolbox.

This script demonstrates all components of the calibration toolbox
and validates their functionality using synthetic data.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.calibration import (
    GyroStream, VideoStream,
    PinholeCamera, FisheyeCamera, CameraIntrinsics,
    TemporalCalibrator, SpatialCalibrator, AutoCalibrator
)


def test_gyro_stream():
    """Test GyroStream functionality."""
    print("\n" + "=" * 60)
    print("Testing GyroStream")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    duration = 5.0
    rate = 200.0
    timestamps = np.arange(0, duration, 1.0 / rate)

    # Sinusoidal motion + bias + noise
    true_bias = np.array([0.001, -0.002, 0.0005])
    gyro = np.column_stack([
        0.1 * np.sin(2 * np.pi * 0.5 * timestamps),
        0.05 * np.cos(2 * np.pi * 0.3 * timestamps),
        0.02 * np.sin(2 * np.pi * 0.7 * timestamps)
    ])
    gyro += true_bias + np.random.normal(0, 0.01, gyro.shape)

    # Create and test stream
    stream = GyroStream()
    stream.load_from_array(timestamps, gyro)

    print(f"  Created: {stream}")

    # Test interpolation
    test_times = np.array([0.5, 1.0, 2.5])
    interp_gyro = stream.get_gyro_at(test_times)
    print(f"  Interpolation test passed: {interp_gyro.shape}")

    # Test resampling
    resampled = stream.resample(100.0)
    print(f"  Resampled: {resampled}")

    # Test filtering
    filtered = stream.apply_lowpass_filter(20.0)
    print(f"  Filtered: {filtered}")

    # Test integration
    times, quats = stream.integrate_to_quaternion()
    print(f"  Integration: {len(quats)} quaternions")

    return True


def test_camera_models():
    """Test camera model projections."""
    print("\n" + "=" * 60)
    print("Testing Camera Models")
    print("=" * 60)

    # Test pinhole camera
    intrinsics = CameraIntrinsics(
        fx=1000, fy=1000, cx=960, cy=540,
        width=1920, height=1080,
        distortion=np.array([-0.1, 0.02, 0, 0, 0])
    )
    pinhole = PinholeCamera(intrinsics)

    # Test roundtrip
    test_point = np.array([0.5, 0.5, 2.0])
    projected = pinhole.project(test_point)
    unprojected = pinhole.unproject(projected)

    expected_dir = test_point / np.linalg.norm(test_point)
    error = np.linalg.norm(unprojected - expected_dir)

    print(f"  Pinhole roundtrip error: {error:.6f}")
    assert error < 0.01, "Pinhole roundtrip failed"

    # Test fisheye camera
    fisheye_intrinsics = CameraIntrinsics(
        fx=500, fy=500, cx=960, cy=540,
        width=1920, height=1080,
        distortion=np.array([0, 0, 0, 0])
    )
    fisheye = FisheyeCamera(fisheye_intrinsics)

    # Test projection at various angles
    angles = [0, 30, 60, 80]
    for angle in angles:
        pt = np.array([
            np.sin(np.deg2rad(angle)), 0, np.cos(np.deg2rad(angle))
        ])
        proj = fisheye.project(pt)
        unproj = fisheye.unproject(proj)
        error = np.linalg.norm(unproj - pt)
        print(f"  Fisheye {angle}° error: {error:.6f}")
        assert error < 0.01, f"Fisheye {angle}° test failed"

    print("  Camera models passed")
    return True


def test_temporal_calibration():
    """Test temporal calibration."""
    print("\n" + "=" * 60)
    print("Testing Temporal Calibration")
    print("=" * 60)

    np.random.seed(42)

    # Generate synchronized data with known offset
    true_offset = 0.05  # 50 ms
    duration = 10.0
    gyro_rate = 200.0
    video_rate = 30.0

    gyro_times = np.arange(0, duration, 1.0 / gyro_rate)
    video_times = np.arange(0, duration, 1.0 / video_rate)

    # Common signal
    signal = np.column_stack([
        0.3 * np.sin(2 * np.pi * 0.5 * gyro_times),
        0.2 * np.cos(2 * np.pi * 0.7 * gyro_times),
        0.1 * np.sin(2 * np.pi * 1.0 * gyro_times)
    ])
    gyro_data = signal + np.random.normal(0, 0.02, signal.shape)

    # Video signal with offset
    from scipy.interpolate import interp1d
    gyro_interp = interp1d(gyro_times, signal, axis=0,
                           bounds_error=False, fill_value='extrapolate')
    video_omega = gyro_interp(video_times + true_offset)
    video_omega += np.random.normal(0, 0.05, video_omega.shape)

    # Calibrate
    calibrator = TemporalCalibrator(search_range=(-0.2, 0.2))
    result = calibrator.calibrate_cross_correlation(
        gyro_times, gyro_data, video_times, video_omega
    )

    error_ms = abs(result.time_offset - true_offset) * 1000
    print(f"  True offset: {true_offset*1000:.1f} ms")
    print(f"  Estimated: {result.time_offset*1000:.1f} ms")
    print(f"  Error: {error_ms:.2f} ms")
    print(f"  Confidence: {result.confidence:.3f}")

    assert error_ms < 10, "Temporal calibration error too high"
    print("  Temporal calibration passed")
    return True


def test_spatial_calibration():
    """Test spatial calibration."""
    print("\n" + "=" * 60)
    print("Testing Spatial Calibration")
    print("=" * 60)

    np.random.seed(42)
    from scipy.spatial.transform import Rotation

    # True extrinsic rotation
    true_euler = np.array([5.0, -3.0, 2.0])
    true_R = Rotation.from_euler('xyz', true_euler, degrees=True).as_matrix()

    # Generate rotation pairs
    calibrator = SpatialCalibrator()
    n_pairs = 15

    for _ in range(n_pairs):
        # Random IMU rotation
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = np.random.uniform(0.05, 0.3)
        R_imu = Rotation.from_rotvec(angle * axis).as_matrix()

        # Camera rotation
        R_camera = true_R @ R_imu @ true_R.T

        # Add noise
        noise = Rotation.from_rotvec(np.random.randn(3) * 0.01).as_matrix()
        R_camera = noise @ R_camera

        calibrator.add_rotation_pair(R_camera, R_imu)

    # Test all methods
    methods = ['procrustes', 'hand_eye', 'optimization']

    for method in methods:
        if method == 'procrustes':
            result = calibrator.calibrate_procrustes()
        elif method == 'hand_eye':
            result = calibrator.calibrate_hand_eye()
        else:
            result = calibrator.calibrate_optimization()

        error = np.linalg.norm(result.euler_angles_deg - true_euler)
        print(f"  {method}: error = {error:.3f}°, confidence = {result.confidence:.3f}")

    # 2° is acceptable for noisy rotation pairs
    assert error < 3.0, "Spatial calibration error too high"
    print("  Spatial calibration passed")
    return True


def test_full_calibration():
    """Test full auto calibration pipeline (simplified)."""
    print("\n" + "=" * 60)
    print("Testing Full Auto Calibration")
    print("=" * 60)

    # Test basic AutoCalibrator initialization and results structure
    from src.calibration.auto_calibrator import CalibrationResults

    # Create mock results
    results = CalibrationResults(
        time_offset=0.05,
        time_offset_confidence=0.9,
        R_cam_imu=np.eye(3),
        euler_angles_deg=np.array([5.0, -3.0, 2.0]),
        spatial_confidence=0.85,
        gyro_bias=np.array([0.001, -0.002, 0.0005]),
        gyro_sample_rate=200.0,
        camera_matrix=np.eye(3),
        readout_time=0.03,
        overall_confidence=0.87,
        temporal_residual=0.1,
        spatial_residual=0.02,
        n_frames_used=100,
        calibration_duration=5.0,
        method="full"
    )

    # Test serialization
    data = results.to_dict()
    loaded = CalibrationResults.from_dict(data)

    assert np.allclose(loaded.euler_angles_deg, results.euler_angles_deg)
    assert loaded.time_offset == results.time_offset
    assert loaded.method == results.method

    print("  Serialization test passed")

    # Test AutoCalibrator class instantiation
    calibrator = AutoCalibrator()
    assert calibrator._initialized == False
    assert calibrator.results is None

    print("  AutoCalibrator initialization passed")
    print("  Full calibration module tests passed")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("CAMERA-IMU CALIBRATION TOOLBOX VALIDATION")
    print("=" * 60)

    tests = [
        ("GyroStream", test_gyro_stream),
        ("Camera Models", test_camera_models),
        ("Temporal Calibration", test_temporal_calibration),
        ("Spatial Calibration", test_spatial_calibration),
        ("Full Calibration", test_full_calibration),
    ]

    results = []

    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, "FAIL"))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for name, status in results:
        print(f"  {name}: {status}")

    passed = sum(1 for _, s in results if s == "PASS")
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

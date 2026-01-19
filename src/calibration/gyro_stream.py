#!/usr/bin/env python3
"""
Gyroscope Stream Module.

Handles IMU data loading, preprocessing, and interpolation for
camera-IMU calibration and motion compensation.

Features:
- Load IMU data from various formats (CSV, JSON, binary)
- Resample to uniform timestamps
- Bias estimation and removal
- Integration to rotation quaternions
- Noise characterization (Allan variance)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation
import json


@dataclass
class IMUData:
    """Container for a single IMU measurement."""
    timestamp: float  # seconds
    gyro: np.ndarray  # angular velocity [rad/s], shape (3,)
    accel: Optional[np.ndarray] = None  # acceleration [m/s²], shape (3,)
    mag: Optional[np.ndarray] = None  # magnetometer [μT], shape (3,)


@dataclass
class GyroParameters:
    """Gyroscope noise parameters."""
    # Angular Random Walk (ARW) - noise density [rad/s/√Hz]
    arw: float = 0.005

    # Bias instability [rad/s]
    bias_instability: float = 0.0001

    # Rate random walk [rad/s²/√Hz]
    rrw: float = 0.00001

    # Initial bias estimate [rad/s]
    initial_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Scale factor errors [dimensionless]
    scale_factor: np.ndarray = field(default_factory=lambda: np.ones(3))

    # Axis misalignment matrix
    misalignment: np.ndarray = field(default_factory=lambda: np.eye(3))


class GyroStream:
    """
    Gyroscope data stream manager.

    Handles loading, preprocessing, and querying of gyroscope
    measurements for calibration and motion estimation.
    """

    def __init__(self, params: GyroParameters = None):
        """
        Initialize gyroscope stream.

        Args:
            params: Gyroscope noise parameters
        """
        self.params = params or GyroParameters()

        # Raw data storage
        self.timestamps: np.ndarray = np.array([])
        self.gyro_data: np.ndarray = np.array([]).reshape(0, 3)
        self.accel_data: Optional[np.ndarray] = None

        # Derived quantities
        self.sample_rate: float = 0.0
        self.duration: float = 0.0
        self.bias_estimate: np.ndarray = np.zeros(3)

        # Interpolators for continuous queries
        self._gyro_interp: Optional[interp1d] = None
        self._accel_interp: Optional[interp1d] = None

    def load_from_array(self,
                        timestamps: np.ndarray,
                        gyro: np.ndarray,
                        accel: np.ndarray = None):
        """
        Load IMU data from numpy arrays.

        Args:
            timestamps: Time values in seconds, shape (N,)
            gyro: Angular velocities [rad/s], shape (N, 3)
            accel: Accelerations [m/s²], shape (N, 3), optional
        """
        self.timestamps = np.asarray(timestamps).flatten()
        self.gyro_data = np.asarray(gyro).reshape(-1, 3)

        if accel is not None:
            self.accel_data = np.asarray(accel).reshape(-1, 3)

        self._compute_statistics()
        self._build_interpolators()

    def load_from_csv(self, filepath: str,
                      time_col: int = 0,
                      gyro_cols: Tuple[int, int, int] = (1, 2, 3),
                      accel_cols: Tuple[int, int, int] = None,
                      time_scale: float = 1.0,
                      gyro_scale: float = 1.0,
                      skip_header: int = 1,
                      delimiter: str = ','):
        """
        Load IMU data from CSV file.

        Args:
            filepath: Path to CSV file
            time_col: Column index for timestamps
            gyro_cols: Column indices for gyro X, Y, Z
            accel_cols: Column indices for accel X, Y, Z (optional)
            time_scale: Multiply timestamps by this (e.g., 1e-9 for ns)
            gyro_scale: Multiply gyro by this (e.g., π/180 for deg/s)
            skip_header: Number of header rows to skip
            delimiter: CSV delimiter
        """
        data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skip_header)

        self.timestamps = data[:, time_col] * time_scale
        self.gyro_data = data[:, gyro_cols] * gyro_scale

        if accel_cols is not None:
            self.accel_data = data[:, accel_cols]

        self._compute_statistics()
        self._build_interpolators()

    def load_from_gopro(self, filepath: str):
        """
        Load IMU data from GoPro metadata JSON (extracted via gpmf-extract).

        Args:
            filepath: Path to JSON file with GoPro telemetry
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        # GoPro format: GYRO stream contains angular velocity
        gyro_stream = data.get('GYRO', data.get('gyroscope', []))

        timestamps = []
        gyro_values = []

        for sample in gyro_stream:
            if isinstance(sample, dict):
                t = sample.get('cts', sample.get('timestamp', 0)) / 1000.0
                g = sample.get('value', [0, 0, 0])
            else:
                # Assume [t, gx, gy, gz] format
                t, g = sample[0] / 1000.0, sample[1:4]

            timestamps.append(t)
            gyro_values.append(g)

        self.timestamps = np.array(timestamps)
        self.gyro_data = np.array(gyro_values)

        # GoPro gyro is typically in deg/s, convert to rad/s
        self.gyro_data = np.deg2rad(self.gyro_data)

        self._compute_statistics()
        self._build_interpolators()

    def _compute_statistics(self):
        """Compute basic statistics from loaded data."""
        if len(self.timestamps) < 2:
            return

        self.duration = self.timestamps[-1] - self.timestamps[0]

        # Estimate sample rate from median time difference
        dt = np.diff(self.timestamps)
        self.sample_rate = 1.0 / np.median(dt)

        # Estimate bias from static periods (low variance)
        # For now, use simple mean (assumes mostly static data)
        variance = np.var(self.gyro_data, axis=0)
        if np.all(variance < 0.01):  # Low motion
            self.bias_estimate = np.mean(self.gyro_data, axis=0)

    def _build_interpolators(self):
        """Build interpolation functions for continuous queries."""
        if len(self.timestamps) < 2:
            return

        self._gyro_interp = interp1d(
            self.timestamps, self.gyro_data, axis=0,
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )

        if self.accel_data is not None:
            self._accel_interp = interp1d(
                self.timestamps, self.accel_data, axis=0,
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )

    def get_gyro_at(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """
        Get interpolated gyroscope reading at time t.

        Args:
            t: Query time(s) in seconds

        Returns:
            Gyroscope values [rad/s], shape (3,) or (N, 3)
        """
        if self._gyro_interp is None:
            raise RuntimeError("No data loaded. Call load_* first.")
        return self._gyro_interp(t)

    def get_accel_at(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Get interpolated accelerometer reading at time t."""
        if self._accel_interp is None:
            raise RuntimeError("No accelerometer data available.")
        return self._accel_interp(t)

    def get_corrected_gyro(self,
                           t: Union[float, np.ndarray],
                           bias: np.ndarray = None) -> np.ndarray:
        """
        Get bias-corrected gyroscope reading.

        Args:
            t: Query time(s)
            bias: Bias to subtract (uses estimate if None)

        Returns:
            Corrected gyro values [rad/s]
        """
        gyro = self.get_gyro_at(t)
        bias = bias if bias is not None else self.bias_estimate
        return gyro - bias

    def resample(self, target_rate: float) -> 'GyroStream':
        """
        Resample data to uniform rate.

        Args:
            target_rate: Target sample rate in Hz

        Returns:
            New GyroStream with resampled data
        """
        if len(self.timestamps) < 2:
            return self

        # Generate uniform timestamps
        t_start = self.timestamps[0]
        t_end = self.timestamps[-1]
        n_samples = int((t_end - t_start) * target_rate) + 1
        new_timestamps = np.linspace(t_start, t_end, n_samples)

        # Interpolate data
        new_gyro = self.get_gyro_at(new_timestamps)
        new_accel = None
        if self.accel_data is not None:
            new_accel = self.get_accel_at(new_timestamps)

        # Create new stream
        new_stream = GyroStream(self.params)
        new_stream.load_from_array(new_timestamps, new_gyro, new_accel)
        new_stream.bias_estimate = self.bias_estimate.copy()

        return new_stream

    def apply_lowpass_filter(self, cutoff_hz: float) -> 'GyroStream':
        """
        Apply low-pass filter to remove high-frequency noise.

        Args:
            cutoff_hz: Cutoff frequency in Hz

        Returns:
            New GyroStream with filtered data
        """
        if self.sample_rate <= 0:
            return self

        # Design Butterworth filter
        nyquist = self.sample_rate / 2
        normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
        b, a = butter(4, normalized_cutoff, btype='low')

        # Apply filter
        filtered_gyro = np.zeros_like(self.gyro_data)
        for i in range(3):
            filtered_gyro[:, i] = filtfilt(b, a, self.gyro_data[:, i])

        # Create new stream
        new_stream = GyroStream(self.params)
        new_stream.load_from_array(self.timestamps.copy(), filtered_gyro)
        new_stream.bias_estimate = self.bias_estimate.copy()

        return new_stream

    def integrate_to_quaternion(self,
                                 t_start: float = None,
                                 t_end: float = None,
                                 q_init: np.ndarray = None,
                                 bias: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate gyroscope to get rotation quaternions.

        Args:
            t_start: Start time (default: first timestamp)
            t_end: End time (default: last timestamp)
            q_init: Initial quaternion [w, x, y, z] (default: identity)
            bias: Gyro bias to subtract

        Returns:
            Tuple of (timestamps, quaternions) where quaternions is (N, 4)
        """
        t_start = t_start if t_start is not None else self.timestamps[0]
        t_end = t_end if t_end is not None else self.timestamps[-1]
        q_init = q_init if q_init is not None else np.array([1, 0, 0, 0])
        bias = bias if bias is not None else self.bias_estimate

        # Get data in range
        mask = (self.timestamps >= t_start) & (self.timestamps <= t_end)
        times = self.timestamps[mask]
        gyro = self.gyro_data[mask] - bias

        # Integrate using quaternion multiplication
        n = len(times)
        quaternions = np.zeros((n, 4))
        quaternions[0] = q_init

        for i in range(1, n):
            dt = times[i] - times[i-1]
            omega = gyro[i-1]

            # Quaternion derivative: q_dot = 0.5 * q * omega_quat
            omega_norm = np.linalg.norm(omega)

            if omega_norm > 1e-10:
                angle = omega_norm * dt
                axis = omega / omega_norm

                # Rotation quaternion for this step
                dq = np.array([
                    np.cos(angle / 2),
                    axis[0] * np.sin(angle / 2),
                    axis[1] * np.sin(angle / 2),
                    axis[2] * np.sin(angle / 2)
                ])

                # Quaternion multiplication: q_new = q * dq
                q = quaternions[i-1]
                quaternions[i] = self._quaternion_multiply(q, dq)
            else:
                quaternions[i] = quaternions[i-1]

            # Normalize
            quaternions[i] /= np.linalg.norm(quaternions[i])

        return times, quaternions

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

    def estimate_bias_static(self,
                             window_size: float = 1.0,
                             variance_threshold: float = 0.001) -> np.ndarray:
        """
        Estimate gyro bias from static (low-motion) periods.

        Args:
            window_size: Window size in seconds
            variance_threshold: Max variance to consider static [rad²/s²]

        Returns:
            Estimated bias [rad/s]
        """
        n_samples = int(window_size * self.sample_rate)
        if n_samples < 10:
            n_samples = 10

        biases = []

        for i in range(0, len(self.gyro_data) - n_samples, n_samples // 2):
            window = self.gyro_data[i:i + n_samples]
            variance = np.mean(np.var(window, axis=0))

            if variance < variance_threshold:
                biases.append(np.mean(window, axis=0))

        if biases:
            self.bias_estimate = np.median(biases, axis=0)
        else:
            self.bias_estimate = np.mean(self.gyro_data, axis=0)

        return self.bias_estimate

    def compute_allan_variance(self,
                               max_cluster_time: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Allan variance for noise characterization.

        Args:
            max_cluster_time: Maximum averaging time (default: duration/10)

        Returns:
            Tuple of (tau, avar) where tau is averaging time and avar is variance
        """
        if len(self.timestamps) < 100:
            raise ValueError("Need at least 100 samples for Allan variance")

        max_cluster_time = max_cluster_time or self.duration / 10

        # Cluster sizes (in samples)
        n_points = len(self.gyro_data)
        max_cluster = min(int(max_cluster_time * self.sample_rate), n_points // 10)

        cluster_sizes = np.unique(np.logspace(0, np.log10(max_cluster), 50).astype(int))
        cluster_sizes = cluster_sizes[cluster_sizes >= 1]

        tau = cluster_sizes / self.sample_rate
        avar = np.zeros((len(cluster_sizes), 3))

        # Use cumulative sum for efficient averaging
        cumsum = np.cumsum(self.gyro_data, axis=0)
        cumsum = np.vstack([np.zeros((1, 3)), cumsum])

        for i, m in enumerate(cluster_sizes):
            # Compute cluster averages
            n_clusters = n_points // m
            if n_clusters < 2:
                avar[i] = np.nan
                continue

            # Cluster averages using cumsum trick
            cluster_avgs = (cumsum[m::m] - cumsum[:-m:m]) / m
            cluster_avgs = cluster_avgs[:n_clusters]

            # Allan variance = 0.5 * E[(x_{k+1} - x_k)²]
            diff = np.diff(cluster_avgs, axis=0)
            avar[i] = 0.5 * np.mean(diff**2, axis=0)

        return tau, avar

    def get_time_range(self) -> Tuple[float, float]:
        """Get time range of data."""
        if len(self.timestamps) == 0:
            return (0.0, 0.0)
        return (self.timestamps[0], self.timestamps[-1])

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.timestamps)

    def __repr__(self) -> str:
        return (f"GyroStream(samples={len(self)}, "
                f"rate={self.sample_rate:.1f}Hz, "
                f"duration={self.duration:.2f}s)")


def demonstrate_gyro_stream():
    """Demonstrate GyroStream functionality."""
    print("=" * 60)
    print("GyroStream Demonstration")
    print("=" * 60)

    # Generate synthetic IMU data
    np.random.seed(42)
    duration = 10.0
    sample_rate = 200.0
    n_samples = int(duration * sample_rate)

    timestamps = np.linspace(0, duration, n_samples)

    # Simulate rotation with drift
    true_bias = np.array([0.001, -0.002, 0.0005])  # rad/s
    noise_std = 0.01  # rad/s

    # Sinusoidal motion + bias + noise
    gyro = np.column_stack([
        0.1 * np.sin(2 * np.pi * 0.5 * timestamps),
        0.05 * np.cos(2 * np.pi * 0.3 * timestamps),
        0.02 * np.sin(2 * np.pi * 0.7 * timestamps)
    ])
    gyro += true_bias
    gyro += np.random.normal(0, noise_std, gyro.shape)

    # Create stream and load data
    stream = GyroStream()
    stream.load_from_array(timestamps, gyro)

    print(f"\n{stream}")

    # Estimate bias
    print("\nBias Estimation:")
    # Add some static data at the end
    static_gyro = np.tile(true_bias + np.random.normal(0, 0.001, 3), (100, 1))
    static_time = np.linspace(duration, duration + 0.5, 100)

    full_time = np.concatenate([timestamps, static_time])
    full_gyro = np.concatenate([gyro, static_gyro])

    stream2 = GyroStream()
    stream2.load_from_array(full_time, full_gyro)
    bias = stream2.estimate_bias_static()

    print(f"  True bias: {true_bias * 1000} mrad/s")
    print(f"  Estimated: {bias * 1000} mrad/s")
    print(f"  Error: {(bias - true_bias) * 1000} mrad/s")

    # Integrate to quaternions
    print("\nQuaternion Integration:")
    times, quats = stream.integrate_to_quaternion(bias=true_bias)

    print(f"  Samples: {len(times)}")
    print(f"  Final quaternion: {quats[-1]}")

    # Convert to Euler angles
    r = Rotation.from_quat([quats[-1, 1], quats[-1, 2], quats[-1, 3], quats[-1, 0]])
    euler = r.as_euler('xyz', degrees=True)
    print(f"  Final Euler angles: {euler}°")

    # Resample
    print("\nResampling:")
    resampled = stream.resample(100.0)
    print(f"  Original: {stream}")
    print(f"  Resampled: {resampled}")

    return stream


if __name__ == "__main__":
    demonstrate_gyro_stream()

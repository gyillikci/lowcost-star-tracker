"""
Gyroscope data extraction from GoPro video files.

This module handles extraction and processing of GPMF (GoPro Metadata Format)
gyroscope telemetry from MP4 video files.
"""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d


@dataclass
class GyroData:
    """Container for processed gyroscope data."""
    
    timestamps: np.ndarray  # Time in seconds from video start
    angular_velocity: np.ndarray  # (N, 3) array of [wx, wy, wz] in rad/s
    orientations: np.ndarray  # (N, 4) array of quaternions [w, x, y, z]
    sample_rate: float
    
    @property
    def duration(self) -> float:
        return self.timestamps[-1] - self.timestamps[0]
    
    @property
    def num_samples(self) -> int:
        return len(self.timestamps)


class GyroExtractor:
    """Extract and process gyroscope data from GoPro video files."""
    
    def __init__(
        self,
        sample_rate_hz: float = 200.0,
        bias_estimation: bool = True,
        bias_window_seconds: float = 2.0,
        low_pass_cutoff_hz: float = 50.0,
        integration_method: str = "rk4",
    ):
        self.sample_rate_hz = sample_rate_hz
        self.bias_estimation = bias_estimation
        self.bias_window_seconds = bias_window_seconds
        self.low_pass_cutoff_hz = low_pass_cutoff_hz
        self.integration_method = integration_method
    
    def extract(self, video_path: Path) -> GyroData:
        """
        Extract gyroscope data from a GoPro video file.
        
        Args:
            video_path: Path to the GoPro MP4 file
            
        Returns:
            GyroData containing timestamps, angular velocities, and orientations
        """
        # Extract raw GPMF data
        raw_timestamps, raw_gyro = self._extract_gpmf(video_path)
        
        # Estimate and remove bias
        if self.bias_estimation:
            bias = self._estimate_bias(raw_gyro, raw_timestamps)
            raw_gyro = raw_gyro - bias
        
        # Apply low-pass filter
        filtered_gyro = self._lowpass_filter(raw_gyro)
        
        # Integrate to get orientations
        orientations = self._integrate_orientations(raw_timestamps, filtered_gyro)
        
        return GyroData(
            timestamps=raw_timestamps,
            angular_velocity=filtered_gyro,
            orientations=orientations,
            sample_rate=self.sample_rate_hz,
        )
    
    def _extract_gpmf(self, video_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract raw GPMF gyroscope data from video file.
        
        This is a placeholder implementation. In production, this would use
        the gpmf-parser library or similar to extract actual telemetry.
        """
        # TODO: Implement actual GPMF parsing
        # For now, return simulated data for development
        
        # Try to use ffprobe to get video duration
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                capture_output=True, text=True
            )
            duration = float(result.stdout.strip())
        except Exception:
            duration = 30.0  # Default fallback
        
        num_samples = int(duration * self.sample_rate_hz)
        timestamps = np.linspace(0, duration, num_samples)
        
        # Simulated gyro data with small random motion
        # In real implementation, this comes from GPMF stream
        np.random.seed(42)
        angular_velocity = np.random.randn(num_samples, 3) * 0.01
        
        return timestamps, angular_velocity
    
    def _estimate_bias(
        self, 
        gyro_data: np.ndarray, 
        timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Estimate gyroscope bias assuming camera is mostly stationary.
        
        Uses samples from the beginning and end of the recording where
        the camera is likely on a tripod.
        """
        window_samples = int(self.bias_window_seconds * self.sample_rate_hz)
        
        # Use samples from start and end
        start_samples = gyro_data[:window_samples]
        end_samples = gyro_data[-window_samples:]
        
        # Compute robust mean (median) of both windows
        start_bias = np.median(start_samples, axis=0)
        end_bias = np.median(end_samples, axis=0)
        
        # Average the two estimates
        bias = (start_bias + end_bias) / 2
        
        return bias
    
    def _lowpass_filter(self, gyro_data: np.ndarray) -> np.ndarray:
        """Apply Butterworth low-pass filter to gyro data."""
        nyquist = self.sample_rate_hz / 2
        normalized_cutoff = self.low_pass_cutoff_hz / nyquist
        
        # Design filter
        b, a = butter(4, normalized_cutoff, btype='low')
        
        # Apply filter to each axis
        filtered = np.zeros_like(gyro_data)
        for i in range(3):
            filtered[:, i] = filtfilt(b, a, gyro_data[:, i])
        
        return filtered
    
    def _integrate_orientations(
        self, 
        timestamps: np.ndarray, 
        angular_velocity: np.ndarray
    ) -> np.ndarray:
        """
        Integrate angular velocity to compute orientation quaternions.
        
        Uses quaternion integration: dq/dt = 0.5 * q ⊗ [0, ω]
        """
        n_samples = len(timestamps)
        quaternions = np.zeros((n_samples, 4))
        quaternions[0] = [1, 0, 0, 0]  # Initial orientation (identity)
        
        for i in range(1, n_samples):
            dt = timestamps[i] - timestamps[i-1]
            omega = angular_velocity[i-1]
            
            if self.integration_method == "rk4":
                q_new = self._rk4_step(quaternions[i-1], omega, dt)
            else:  # euler
                q_new = self._euler_step(quaternions[i-1], omega, dt)
            
            # Normalize quaternion
            quaternions[i] = q_new / np.linalg.norm(q_new)
        
        return quaternions
    
    def _euler_step(
        self, 
        q: np.ndarray, 
        omega: np.ndarray, 
        dt: float
    ) -> np.ndarray:
        """Euler integration step for quaternion."""
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * self._quaternion_multiply(q, omega_quat)
        return q + q_dot * dt
    
    def _rk4_step(
        self, 
        q: np.ndarray, 
        omega: np.ndarray, 
        dt: float
    ) -> np.ndarray:
        """Fourth-order Runge-Kutta integration step."""
        def derivative(q_curr):
            omega_quat = np.array([0, omega[0], omega[1], omega[2]])
            return 0.5 * self._quaternion_multiply(q_curr, omega_quat)
        
        k1 = derivative(q)
        k2 = derivative(q + 0.5 * dt * k1)
        k3 = derivative(q + 0.5 * dt * k2)
        k4 = derivative(q + dt * k3)
        
        return q + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
    
    def get_orientation_at_time(
        self, 
        gyro_data: GyroData, 
        time: float
    ) -> np.ndarray:
        """
        Get interpolated orientation quaternion at a specific time.
        
        Args:
            gyro_data: GyroData object
            time: Time in seconds
            
        Returns:
            Quaternion [w, x, y, z] at the specified time
        """
        # Find surrounding samples
        idx = np.searchsorted(gyro_data.timestamps, time)
        
        if idx == 0:
            return gyro_data.orientations[0]
        if idx >= len(gyro_data.timestamps):
            return gyro_data.orientations[-1]
        
        # Spherical linear interpolation (SLERP)
        t0, t1 = gyro_data.timestamps[idx-1], gyro_data.timestamps[idx]
        q0, q1 = gyro_data.orientations[idx-1], gyro_data.orientations[idx]
        
        alpha = (time - t0) / (t1 - t0)
        
        return self._slerp(q0, q1, alpha)
    
    @staticmethod
    def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between quaternions."""
        # Ensure shortest path
        dot = np.dot(q0, q1)
        if dot < 0:
            q1 = -q1
            dot = -dot
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)
        
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        
        q2 = q1 - q0 * dot
        q2 = q2 / np.linalg.norm(q2)
        
        return q0 * np.cos(theta) + q2 * np.sin(theta)

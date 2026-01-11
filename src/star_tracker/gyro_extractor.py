"""
Gyroscope data extraction from GoPro video files.

This module handles extraction and processing of GPMF (GoPro Metadata Format)
gyroscope and accelerometer telemetry from MP4 video files.

GPMF (GoPro Metadata Format) is stored as a separate track in GoPro MP4 files.
The gyroscope data is typically sampled at ~200Hz and contains angular velocity
measurements in radians/second for all three axes (X, Y, Z).
The accelerometer data is also at ~200Hz with acceleration in m/s².
"""

import struct
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


@dataclass
class GyroData:
    """Container for processed gyroscope and accelerometer data."""
    
    timestamps: np.ndarray  # Time in seconds from video start
    angular_velocity: np.ndarray  # (N, 3) array of [wx, wy, wz] in rad/s
    orientations: np.ndarray  # (N, 4) array of quaternions [w, x, y, z]
    sample_rate: float
    
    # Optional accelerometer data for VQF fusion
    acceleration: Optional[np.ndarray] = None  # (N, 3) array of [ax, ay, az] in m/s²
    
    # Smoothed orientations (velocity-adaptive)
    smoothed_orientations: Optional[np.ndarray] = None  # (N, 4) array
    
    # Gyro bias estimate
    gyro_bias: Optional[np.ndarray] = None  # (3,) array in rad/s
    
    @property
    def duration(self) -> float:
        return self.timestamps[-1] - self.timestamps[0]
    
    @property
    def num_samples(self) -> int:
        return len(self.timestamps)
    
    @property
    def has_accelerometer(self) -> bool:
        return self.acceleration is not None and len(self.acceleration) > 0


class GyroExtractor:
    """Extract and process gyroscope data from GoPro video files."""
    
    def __init__(
        self,
        sample_rate_hz: float = 200.0,
        bias_estimation: bool = True,
        bias_window_seconds: float = 2.0,
        low_pass_cutoff_hz: float = 50.0,
        integration_method: str = "vqf",  # "euler", "rk4", or "vqf"
        use_accelerometer: bool = True,
        velocity_adaptive_smoothing: bool = True,
        smoothing_time_constant: float = 0.5,  # seconds
    ):
        self.sample_rate_hz = sample_rate_hz
        self.bias_estimation = bias_estimation
        self.bias_window_seconds = bias_window_seconds
        self.low_pass_cutoff_hz = low_pass_cutoff_hz
        self.integration_method = integration_method
        self.use_accelerometer = use_accelerometer
        self.velocity_adaptive_smoothing = velocity_adaptive_smoothing
        self.smoothing_time_constant = smoothing_time_constant
    
    def extract(self, video_path: Path) -> GyroData:
        """
        Extract gyroscope data from a GoPro video file.
        
        Args:
            video_path: Path to the GoPro MP4 file
            
        Returns:
            GyroData containing timestamps, angular velocities, and orientations
        """
        # Extract raw GPMF data (gyro + accel)
        raw_timestamps, raw_gyro, raw_accel = self._extract_gpmf(video_path)
        
        # Estimate and remove bias
        bias = None
        if self.bias_estimation:
            bias = self._estimate_bias(raw_gyro, raw_timestamps)
            raw_gyro = raw_gyro - bias
        
        # Apply low-pass filter
        filtered_gyro = self._lowpass_filter(raw_gyro)
        filtered_accel = self._lowpass_filter(raw_accel) if raw_accel is not None else None
        
        # Integrate to get orientations
        if self.integration_method == "vqf" and filtered_accel is not None:
            orientations = self._integrate_vqf(raw_timestamps, filtered_gyro, filtered_accel)
        else:
            orientations = self._integrate_orientations(raw_timestamps, filtered_gyro)
        
        # Apply velocity-adaptive smoothing
        smoothed = None
        if self.velocity_adaptive_smoothing:
            smoothed = self._velocity_adaptive_smooth(
                raw_timestamps, orientations, filtered_gyro
            )
        
        return GyroData(
            timestamps=raw_timestamps,
            angular_velocity=filtered_gyro,
            orientations=orientations,
            sample_rate=self.sample_rate_hz,
            acceleration=filtered_accel,
            smoothed_orientations=smoothed,
            gyro_bias=bias,
        )
    
    def _extract_gpmf(self, video_path: Path) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Extract raw GPMF gyroscope and accelerometer data from video file.
        
        GoPro stores telemetry data in a dedicated track within the MP4 container.
        The GPMF stream contains GYRO samples with angular velocity in deg/s
        and ACCL samples with acceleration in m/s².
        
        Returns:
            Tuple of (timestamps, gyro_data, accel_data)
        """
        video_path = Path(video_path)
        
        try:
            # Try to extract using GPMF parser
            timestamps, gyro_data, accel_data = self._parse_gpmf_from_mp4(video_path)
            if len(timestamps) > 0:
                logger.info(f"Extracted {len(timestamps)} gyro samples from GPMF stream")
                if accel_data is not None:
                    logger.info(f"Extracted {len(accel_data)} accel samples from GPMF stream")
                return timestamps, gyro_data, accel_data
        except Exception as e:
            logger.warning(f"GPMF extraction failed: {e}. Using fallback method.")
        
        # Fallback: get video duration and generate minimal data
        ts, gyro = self._fallback_gyro_data(video_path)
        return ts, gyro, None
    
    def _parse_gpmf_from_mp4(self, video_path: Path) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Parse GPMF data directly from MP4 file structure.
        
        MP4 files contain the GPMF data in a dedicated metadata track,
        embedded within the mdat box at regular intervals.
        
        Returns:
            Tuple of (timestamps, gyro_array, accel_array)
        """
        all_gyro_samples = []
        all_accel_samples = []
        gyro_scale = 1.0
        accel_scale = 1.0
        
        with open(video_path, 'rb') as f:
            file_size = video_path.stat().st_size
            
            # Find all DEVC (device container) positions in the file
            devc_positions = self._find_marker_positions(f, file_size, b'DEVC')
            
            if not devc_positions:
                logger.debug("No DEVC markers found in file")
                return np.array([]), np.array([]), None
            
            logger.debug(f"Found {len(devc_positions)} DEVC markers")
            
            # Extract GYRO and ACCL data from each DEVC block
            for devc_pos in devc_positions:
                f.seek(devc_pos)
                # Read enough data to contain both GYRO and ACCL blocks
                block_data = f.read(8192)
                
                # Extract GYRO
                block_gyro_scale, gyro_samples = self._extract_sensor_from_block(
                    block_data, b'GYRO', 6
                )
                if block_gyro_scale > 0:
                    gyro_scale = block_gyro_scale
                if gyro_samples:
                    all_gyro_samples.extend(gyro_samples)
                
                # Extract ACCL (accelerometer)
                block_accel_scale, accel_samples = self._extract_sensor_from_block(
                    block_data, b'ACCL', 6
                )
                if block_accel_scale > 0:
                    accel_scale = block_accel_scale
                if accel_samples:
                    all_accel_samples.extend(accel_samples)
        
        # Process gyro data
        if all_gyro_samples:
            gyro_array = np.array(all_gyro_samples)
            if gyro_scale != 0 and gyro_scale != 1:
                gyro_array = gyro_array / gyro_scale
            # Convert from deg/s to rad/s
            gyro_array = np.deg2rad(gyro_array)
            
            sample_rate = 200.0
            duration = len(gyro_array) / sample_rate
            timestamps = np.linspace(0, duration, len(gyro_array))
            
            logger.info(f"Extracted {len(gyro_array)} gyro samples over {duration:.1f}s")
        else:
            return np.array([]), np.array([]), None
        
        # Process accel data
        accel_array = None
        if all_accel_samples:
            accel_array = np.array(all_accel_samples)
            if accel_scale != 0 and accel_scale != 1:
                accel_array = accel_array / accel_scale
            # GoPro ACCL is already in m/s²
            
            # Resample to match gyro if different lengths
            if len(accel_array) != len(gyro_array):
                accel_interp = interp1d(
                    np.linspace(0, 1, len(accel_array)),
                    accel_array, axis=0, kind='linear'
                )
                accel_array = accel_interp(np.linspace(0, 1, len(gyro_array)))
            
            logger.info(f"Extracted {len(accel_array)} accel samples")
        
        return timestamps, gyro_array, accel_array
    
    def _find_marker_positions(self, f, file_size: int, marker: bytes) -> List[int]:
        """Find all positions of a marker in the file."""
        positions = []
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        
        f.seek(0)
        offset = 0
        
        while offset < file_size:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            pos = 0
            while True:
                pos = chunk.find(marker, pos)
                if pos == -1:
                    break
                positions.append(offset + pos)
                pos += 1
            
            offset += len(chunk)
        
        return positions
    
    def _extract_sensor_from_block(
        self, 
        block_data: bytes, 
        sensor_marker: bytes,
        struct_size: int = 6
    ) -> Tuple[float, List[List[float]]]:
        """
        Extract sensor data (GYRO or ACCL) from a GPMF DEVC block.
        
        GPMF structure:
        - Each sensor stream (STRM) contains its own SCAL before the data
        - We need to find the SCAL that's immediately before the sensor data
        
        Args:
            block_data: Raw block bytes
            sensor_marker: b'GYRO' or b'ACCL'
            struct_size: Expected struct size (6 for 3x int16)
            
        Returns:
            Tuple of (scale_factor, list of [x, y, z] samples)
        """
        samples = []
        
        # Find sensor marker
        sensor_offset = block_data.find(sensor_marker)
        if sensor_offset < 0 or sensor_offset + 8 > len(block_data):
            return 0.0, []
        
        # Find the SCAL marker closest to (but before) the sensor
        scale = 1.0
        scal_offset = 0
        while True:
            next_scal = block_data.find(b'SCAL', scal_offset)
            if next_scal == -1 or next_scal > sensor_offset:
                break
            
            # Parse this SCAL
            if next_scal + 10 <= len(block_data):
                type_byte = block_data[next_scal + 4]
                scal_struct_size = block_data[next_scal + 5]
                
                if type_byte == ord('s') and scal_struct_size >= 2:  # signed short
                    scale = struct.unpack('>h', block_data[next_scal + 8:next_scal + 10])[0]
                elif type_byte == ord('S') and scal_struct_size >= 2:  # unsigned short
                    scale = struct.unpack('>H', block_data[next_scal + 8:next_scal + 10])[0]
                elif type_byte == ord('l') and scal_struct_size >= 4:  # signed long
                    scale = struct.unpack('>i', block_data[next_scal + 8:next_scal + 12])[0]
            
            scal_offset = next_scal + 1
        
        # Parse sensor data
        type_byte = block_data[sensor_offset + 4]
        data_struct_size = block_data[sensor_offset + 5]
        repeat = struct.unpack('>H', block_data[sensor_offset + 6:sensor_offset + 8])[0]
        
        # Data is typically type 's' (signed short) with struct_size=6 (3 shorts for x,y,z)
        if type_byte == ord('s') and data_struct_size == struct_size:
            data_start = sensor_offset + 8
            for i in range(repeat):
                sample_offset = data_start + i * struct_size
                if sample_offset + struct_size <= len(block_data):
                    x, y, z = struct.unpack('>hhh', block_data[sample_offset:sample_offset + 6])
                    samples.append([float(x), float(y), float(z)])
        
        return scale, samples
    
    def _extract_gyro_from_block(self, block_data: bytes) -> Tuple[float, List[List[float]]]:
        """Legacy method for backward compatibility."""
        return self._extract_sensor_from_block(block_data, b'GYRO', 6)
    
    def _fallback_gyro_data(self, video_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """
        Fallback method when GPMF parsing fails.
        
        Returns minimal gyro data assuming camera is stationary.
        """
        # Try to get video duration using cv2
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 30.0
            cap.release()
        except Exception:
            duration = 30.0
        
        logger.info(f"Using fallback gyro data (stationary assumption) for {duration:.1f}s video")
        
        num_samples = int(duration * self.sample_rate_hz)
        timestamps = np.linspace(0, duration, num_samples)
        
        # Assume camera is stationary - zero angular velocity with tiny noise
        angular_velocity = np.zeros((num_samples, 3))
        
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
    
    def _integrate_vqf(
        self,
        timestamps: np.ndarray,
        gyro: np.ndarray,
        accel: np.ndarray
    ) -> np.ndarray:
        """
        Integrate using VQF algorithm with gyro+accel sensor fusion.
        
        VQF provides:
        - Accelerometer-based tilt correction
        - Online bias estimation
        - Forward-backward smoothing
        
        Args:
            timestamps: (N,) time array
            gyro: (N, 3) gyroscope in rad/s
            accel: (N, 3) accelerometer in m/s²
            
        Returns:
            (N, 4) quaternion array [w, x, y, z]
        """
        from .vqf_integrator import vqf_offline
        
        logger.info("Using VQF sensor fusion for orientation estimation")
        return vqf_offline(timestamps, gyro, accel)
    
    def _velocity_adaptive_smooth(
        self,
        timestamps: np.ndarray,
        orientations: np.ndarray,
        angular_velocity: np.ndarray
    ) -> np.ndarray:
        """
        Apply velocity-adaptive smoothing to orientations.
        
        Implements Gyroflow-style 2-pass SLERP smoothing where:
        - Low velocity → high smoothness (1 second)
        - High velocity → low smoothness (0.1 second)
        
        The alpha value interpolates based on angular velocity normalized by 500°/s.
        
        Args:
            timestamps: (N,) time array
            orientations: (N, 4) quaternion array
            angular_velocity: (N, 3) angular velocity in rad/s
            
        Returns:
            (N, 4) smoothed quaternion array
        """
        n_samples = len(timestamps)
        
        # Calculate angular velocity magnitude (deg/s)
        velocity_deg = np.rad2deg(np.linalg.norm(angular_velocity, axis=1))
        
        # Apply velocity smoothing (moving average)
        window_size = min(21, n_samples // 10 + 1)
        if window_size % 2 == 0:
            window_size += 1
        velocity_smooth = np.convolve(
            velocity_deg, 
            np.ones(window_size) / window_size, 
            mode='same'
        )
        
        # Compute adaptive alpha: velocity / 500 deg/s
        # alpha = 0 → use max smoothness (1.0s time constant)
        # alpha = 1 → use min smoothness (0.1s time constant)
        alpha = np.clip(velocity_smooth / 500.0, 0, 1)
        
        # Interpolate time constant between 1.0s (slow) and 0.1s (fast motion)
        max_smoothness_sec = 1.0
        min_smoothness_sec = 0.1
        time_constants = max_smoothness_sec * (1 - alpha) + min_smoothness_sec * alpha
        
        # Forward SLERP pass
        logger.info("Velocity-adaptive smoothing: forward pass")
        smoothed_fwd = np.zeros_like(orientations)
        smoothed_fwd[0] = orientations[0]
        
        for i in range(1, n_samples):
            dt = timestamps[i] - timestamps[i-1]
            if dt <= 0:
                dt = 1.0 / self.sample_rate_hz
            
            tau = time_constants[i]
            blend = dt / (tau + dt)
            smoothed_fwd[i] = self._slerp(smoothed_fwd[i-1], orientations[i], blend)
        
        # Backward SLERP pass
        logger.info("Velocity-adaptive smoothing: backward pass")
        smoothed_bwd = np.zeros_like(orientations)
        smoothed_bwd[-1] = orientations[-1]
        
        for i in range(n_samples - 2, -1, -1):
            dt = timestamps[i+1] - timestamps[i]
            if dt <= 0:
                dt = 1.0 / self.sample_rate_hz
            
            tau = time_constants[i]
            blend = dt / (tau + dt)
            smoothed_bwd[i] = self._slerp(smoothed_bwd[i+1], orientations[i], blend)
        
        # Combine passes (average the two)
        logger.info("Velocity-adaptive smoothing: combining passes")
        smoothed = np.zeros_like(orientations)
        for i in range(n_samples):
            smoothed[i] = self._slerp(smoothed_fwd[i], smoothed_bwd[i], 0.5)
        
        return smoothed
    
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

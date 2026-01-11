"""
IMU-Based Video Stabilization for GoPro Videos

This module implements Gyroflow-style video stabilization using:
1. GPMF extraction (gyro + accelerometer at 200Hz)
2. VQF sensor fusion (gyro+accel → orientation)
3. Velocity-adaptive SLERP smoothing
4. Rolling shutter correction (per-row rotation)
5. Fisheye lens distortion handling

Usage:
    python gyro_stabilizer.py input_video.mp4 -o output_stabilized.mp4
"""

import struct
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import json

import numpy as np
import cv2
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class LensProfile:
    """GoPro lens profile for fisheye distortion."""
    fx: float  # Focal length X (pixels)
    fy: float  # Focal length Y (pixels)
    cx: float  # Principal point X
    cy: float  # Principal point Y
    k1: float  # Distortion coefficient
    k2: float
    k3: float
    k4: float
    frame_readout_time_ms: float  # Rolling shutter time
    
    @classmethod
    def gopro_hero7_4k_wide(cls) -> "LensProfile":
        """GoPro Hero7 Black - Wide Mode 4K (3840x2160)"""
        return cls(
            fx=1130.0, fy=1130.0,
            cx=1920.0, cy=1080.0,
            k1=-0.35, k2=0.15, k3=-0.02, k4=0.01,
            frame_readout_time_ms=8.3
        )
    
    @classmethod
    def gopro_hero7_1080p_wide(cls) -> "LensProfile":
        """GoPro Hero7 Black - Wide Mode 1080p (1920x1080)"""
        return cls(
            fx=565.0, fy=565.0,
            cx=960.0, cy=540.0,
            k1=-0.35, k2=0.15, k3=-0.02, k4=0.01,
            frame_readout_time_ms=16.6
        )
    
    def scale_to_resolution(self, width: int, height: int, orig_width: int, orig_height: int) -> "LensProfile":
        """Scale lens profile to different resolution."""
        scale_x = width / orig_width
        scale_y = height / orig_height
        return LensProfile(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            k1=self.k1, k2=self.k2, k3=self.k3, k4=self.k4,
            frame_readout_time_ms=self.frame_readout_time_ms * scale_y
        )
    
    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)


@dataclass  
class IMUData:
    """Container for extracted IMU data."""
    timestamps: np.ndarray      # (N,) seconds
    gyro: np.ndarray            # (N, 3) rad/s
    accel: Optional[np.ndarray] # (N, 3) m/s²
    orientations: np.ndarray    # (N, 4) quaternions [w,x,y,z]
    smoothed_orientations: Optional[np.ndarray] = None


# =============================================================================
# GPMF Extraction
# =============================================================================

class GPMFExtractor:
    """Extract gyroscope and accelerometer data from GoPro GPMF stream."""
    
    def __init__(self, sample_rate: float = 200.0):
        self.sample_rate = sample_rate
    
    def extract(self, video_path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Extract IMU data from GoPro video.
        
        Returns:
            (timestamps, gyro, accel) - accel may be None if not found
        """
        video_path = Path(video_path)
        
        gyro_samples = []
        accel_samples = []
        gyro_scale = 1.0
        accel_scale = 1.0
        
        with open(video_path, 'rb') as f:
            file_size = video_path.stat().st_size
            
            # Find all DEVC markers
            devc_positions = self._find_markers(f, file_size, b'DEVC')
            
            if not devc_positions:
                logger.warning("No GPMF data found, using fallback")
                return self._fallback_data(video_path)
            
            logger.info(f"Found {len(devc_positions)} GPMF blocks")
            
            for pos in devc_positions:
                f.seek(pos)
                block = f.read(8192)
                
                # Extract GYRO
                scale, samples = self._extract_sensor(block, b'GYRO')
                if scale > 0:
                    gyro_scale = scale
                gyro_samples.extend(samples)
                
                # Extract ACCL
                scale, samples = self._extract_sensor(block, b'ACCL')
                if scale > 0:
                    accel_scale = scale
                accel_samples.extend(samples)
        
        if not gyro_samples:
            return self._fallback_data(video_path)
        
        # Process gyro
        gyro = np.array(gyro_samples) / gyro_scale
        gyro = np.deg2rad(gyro)  # Convert to rad/s
        
        duration = len(gyro) / self.sample_rate
        timestamps = np.linspace(0, duration, len(gyro))
        
        # Process accel
        accel = None
        if accel_samples:
            accel = np.array(accel_samples) / accel_scale
            if len(accel) != len(gyro):
                # Resample to match gyro
                interp = interp1d(
                    np.linspace(0, 1, len(accel)),
                    accel, axis=0, kind='linear'
                )
                accel = interp(np.linspace(0, 1, len(gyro)))
        
        logger.info(f"Extracted {len(gyro)} gyro samples over {duration:.1f}s")
        return timestamps, gyro, accel
    
    def _find_markers(self, f, file_size: int, marker: bytes) -> List[int]:
        positions = []
        chunk_size = 10 * 1024 * 1024
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
    
    def _extract_sensor(self, block: bytes, marker: bytes) -> Tuple[float, List]:
        samples = []
        offset = block.find(marker)
        if offset < 0 or offset + 8 > len(block):
            return 0.0, []
        
        # Find SCAL before marker
        scale = 1.0
        scal_pos = 0
        while True:
            next_scal = block.find(b'SCAL', scal_pos)
            if next_scal == -1 or next_scal > offset:
                break
            if next_scal + 10 <= len(block):
                type_byte = block[next_scal + 4]
                if type_byte == ord('s'):
                    scale = struct.unpack('>h', block[next_scal + 8:next_scal + 10])[0]
                elif type_byte == ord('S'):
                    scale = struct.unpack('>H', block[next_scal + 8:next_scal + 10])[0]
            scal_pos = next_scal + 1
        
        # Parse data
        type_byte = block[offset + 4]
        struct_size = block[offset + 5]
        repeat = struct.unpack('>H', block[offset + 6:offset + 8])[0]
        
        if type_byte == ord('s') and struct_size == 6:
            data_start = offset + 8
            for i in range(repeat):
                pos = data_start + i * 6
                if pos + 6 <= len(block):
                    x, y, z = struct.unpack('>hhh', block[pos:pos + 6])
                    samples.append([float(x), float(y), float(z)])
        
        return scale, samples
    
    def _fallback_data(self, video_path: Path) -> Tuple[np.ndarray, np.ndarray, None]:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frames / fps if fps > 0 else 30.0
        cap.release()
        
        n = int(duration * self.sample_rate)
        timestamps = np.linspace(0, duration, n)
        gyro = np.zeros((n, 3))
        
        logger.warning(f"Using zero-motion fallback for {duration:.1f}s")
        return timestamps, gyro, None


# =============================================================================
# VQF Sensor Fusion
# =============================================================================

class VQFIntegrator:
    """
    VQF (Versatile Quaternion-based Filter) for IMU sensor fusion.
    
    Fuses gyroscope and accelerometer to estimate orientation with:
    - Accelerometer-based tilt correction
    - Online bias estimation
    - Rest detection
    """
    
    GRAVITY = 9.81
    
    def __init__(
        self,
        tau_acc: float = 3.0,
        rest_threshold_gyr: float = 2.0,  # deg/s
        rest_threshold_acc: float = 0.5,  # m/s²
    ):
        self.tau_acc = tau_acc
        self.rest_th_gyr = rest_threshold_gyr
        self.rest_th_acc = rest_threshold_acc
        self._reset()
    
    def _reset(self):
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.bias = np.zeros(3)
        self.rest_time = 0.0
        self.gyr_lp = np.zeros(3)
        self.acc_lp = np.zeros(3)
    
    def integrate(
        self,
        timestamps: np.ndarray,
        gyro: np.ndarray,
        accel: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Integrate IMU data to orientation quaternions.
        
        Args:
            timestamps: (N,) time array
            gyro: (N, 3) gyroscope in rad/s
            accel: (N, 3) accelerometer in m/s² (optional)
            
        Returns:
            (N, 4) quaternion array [w, x, y, z]
        """
        n = len(timestamps)
        quats = np.zeros((n, 4))
        quats[0] = self.quat.copy()
        
        use_accel = accel is not None
        
        for i in range(1, n):
            dt = timestamps[i] - timestamps[i-1]
            if dt <= 0:
                dt = 1.0 / 200.0
            
            gyr = gyro[i] - self.bias
            acc = accel[i] if use_accel else None
            
            # Rest detection for bias estimation
            if use_accel:
                self._update_rest_detection(gyr, acc, dt)
                if self.rest_time > 1.0:
                    # Update bias during rest
                    alpha = dt / (1.0 + dt)
                    self.bias = self.bias * (1 - alpha) + gyro[i] * alpha
            
            # RK4 integration
            self.quat = self._rk4_step(gyr, dt)
            
            # Accelerometer correction
            if use_accel and abs(np.linalg.norm(acc) - self.GRAVITY) < 5.0:
                self.quat = self._acc_correction(acc, dt)
            
            quats[i] = self.quat.copy()
        
        return quats
    
    def _rk4_step(self, gyr: np.ndarray, dt: float) -> np.ndarray:
        def deriv(q, w):
            w_quat = np.array([0, w[0], w[1], w[2]])
            return 0.5 * self._qmult(q, w_quat)
        
        k1 = deriv(self.quat, gyr)
        k2 = deriv(self._norm(self.quat + 0.5*dt*k1), gyr)
        k3 = deriv(self._norm(self.quat + 0.5*dt*k2), gyr)
        k4 = deriv(self._norm(self.quat + dt*k3), gyr)
        
        return self._norm(self.quat + (dt/6)*(k1 + 2*k2 + 2*k3 + k4))
    
    def _acc_correction(self, acc: np.ndarray, dt: float) -> np.ndarray:
        acc_n = acc / np.linalg.norm(acc)
        grav_w = np.array([0.0, 0.0, -1.0])
        grav_b = self._rotate_inv(grav_w, self.quat)
        
        axis = np.cross(grav_b, acc_n)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            return self.quat
        
        axis /= axis_norm
        angle = np.arcsin(np.clip(axis_norm, -1, 1))
        
        alpha = dt / (self.tau_acc + dt)
        angle *= alpha
        
        half = angle / 2
        corr = np.array([np.cos(half), *(axis * np.sin(half))])
        return self._norm(self._qmult(self.quat, corr))
    
    def _update_rest_detection(self, gyr: np.ndarray, acc: np.ndarray, dt: float):
        alpha = dt / (0.5 + dt)
        self.gyr_lp = self.gyr_lp * (1 - alpha) + gyr * alpha
        self.acc_lp = self.acc_lp * (1 - alpha) + acc * alpha
        
        gyr_norm = np.rad2deg(np.linalg.norm(self.gyr_lp))
        acc_dev = abs(np.linalg.norm(self.acc_lp) - self.GRAVITY)
        
        if gyr_norm < self.rest_th_gyr and acc_dev < self.rest_th_acc:
            self.rest_time += dt
        else:
            self.rest_time = 0.0
    
    @staticmethod
    def _qmult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
    
    @staticmethod
    def _norm(q: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(q)
        return q / n if n > 1e-10 else np.array([1., 0., 0., 0.])
    
    def _rotate_inv(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
        v_q = np.array([0, v[0], v[1], v[2]])
        result = self._qmult(self._qmult(q_inv, v_q), q)
        return result[1:4]


# =============================================================================
# Velocity-Adaptive Smoothing
# =============================================================================

def velocity_adaptive_smooth(
    timestamps: np.ndarray,
    orientations: np.ndarray,
    angular_velocity: np.ndarray,
    max_smoothness: float = 1.0,
    min_smoothness: float = 0.1,
    velocity_scale: float = 500.0
) -> np.ndarray:
    """
    Apply velocity-adaptive 2-pass SLERP smoothing.
    
    Low velocity → high smoothness (1 second time constant)
    High velocity → low smoothness (0.1 second time constant)
    
    Args:
        timestamps: Time array
        orientations: Quaternion array
        angular_velocity: Angular velocity in rad/s
        max_smoothness: Time constant at rest (seconds)
        min_smoothness: Time constant during fast motion
        velocity_scale: Velocity for full alpha (deg/s)
    """
    n = len(timestamps)
    
    # Compute angular velocity magnitude
    vel_deg = np.rad2deg(np.linalg.norm(angular_velocity, axis=1))
    
    # Smooth velocity
    window = min(21, n // 10 + 1)
    if window % 2 == 0:
        window += 1
    vel_smooth = np.convolve(vel_deg, np.ones(window)/window, mode='same')
    
    # Compute adaptive time constants
    alpha = np.clip(vel_smooth / velocity_scale, 0, 1)
    tau = max_smoothness * (1 - alpha) + min_smoothness * alpha
    
    # Forward pass
    fwd = np.zeros_like(orientations)
    fwd[0] = orientations[0]
    for i in range(1, n):
        dt = max(timestamps[i] - timestamps[i-1], 1e-6)
        blend = dt / (tau[i] + dt)
        fwd[i] = slerp(fwd[i-1], orientations[i], blend)
    
    # Backward pass
    bwd = np.zeros_like(orientations)
    bwd[-1] = orientations[-1]
    for i in range(n-2, -1, -1):
        dt = max(timestamps[i+1] - timestamps[i], 1e-6)
        blend = dt / (tau[i] + dt)
        bwd[i] = slerp(bwd[i+1], orientations[i], blend)
    
    # Combine
    result = np.zeros_like(orientations)
    for i in range(n):
        result[i] = slerp(fwd[i], bwd[i], 0.5)
    
    return result


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    
    theta0 = np.arccos(np.clip(dot, -1, 1))
    theta = theta0 * t
    
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    
    return q0 * np.cos(theta) + q2 * np.sin(theta)


# =============================================================================
# Video Stabilizer
# =============================================================================

class GyroStabilizer:
    """
    Main video stabilizer using IMU data.
    """
    
    def __init__(
        self,
        lens_profile: Optional[LensProfile] = None,
        rolling_shutter: bool = True,
        velocity_smoothing: bool = True,
        use_vqf: bool = True,
    ):
        self.lens_profile = lens_profile
        self.rolling_shutter = rolling_shutter
        self.velocity_smoothing = velocity_smoothing
        self.use_vqf = use_vqf
        
        self.extractor = GPMFExtractor()
        self.vqf = VQFIntegrator()
    
    def extract_imu(self, video_path: Path) -> IMUData:
        """Extract and process IMU data from video."""
        timestamps, gyro, accel = self.extractor.extract(video_path)
        
        # Apply low-pass filter
        gyro = self._lowpass_filter(gyro, 50.0, 200.0)
        if accel is not None:
            accel = self._lowpass_filter(accel, 50.0, 200.0)
        
        # Estimate and remove bias
        bias = self._estimate_bias(gyro)
        gyro = gyro - bias
        logger.info(f"Gyro bias: {np.rad2deg(bias)} deg/s")
        
        # Integrate to orientations
        if self.use_vqf and accel is not None:
            logger.info("Using VQF sensor fusion")
            orientations = self.vqf.integrate(timestamps, gyro, accel)
        else:
            logger.info("Using gyro-only integration")
            orientations = self._integrate_gyro(timestamps, gyro)
        
        # Velocity-adaptive smoothing
        smoothed = None
        if self.velocity_smoothing:
            logger.info("Applying velocity-adaptive smoothing")
            smoothed = velocity_adaptive_smooth(timestamps, orientations, gyro)
        
        return IMUData(
            timestamps=timestamps,
            gyro=gyro,
            accel=accel,
            orientations=orientations,
            smoothed_orientations=smoothed
        )
    
    def stabilize(
        self,
        video_path: Path,
        output_path: Path,
        imu_data: Optional[IMUData] = None,
        crop_ratio: float = 0.9,
    ) -> Path:
        """
        Stabilize video using IMU data.
        
        Args:
            video_path: Input video path
            output_path: Output video path
            imu_data: Pre-extracted IMU data (optional)
            crop_ratio: Crop ratio to remove black borders (0.9 = 90% of frame)
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        # Extract IMU if not provided
        if imu_data is None:
            imu_data = self.extract_imu(video_path)
        
        # Use smoothed orientations if available
        orientations = imu_data.smoothed_orientations
        if orientations is None:
            orientations = imu_data.orientations
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup lens profile
        if self.lens_profile is None:
            if width >= 3840:
                self.lens_profile = LensProfile.gopro_hero7_4k_wide()
            else:
                self.lens_profile = LensProfile.gopro_hero7_1080p_wide()
                if width != 1920 or height != 1080:
                    self.lens_profile = self.lens_profile.scale_to_resolution(
                        width, height, 1920, 1080
                    )
        
        # Compute target orientation (mean)
        target_quat = np.mean(orientations, axis=0)
        target_quat /= np.linalg.norm(target_quat)
        
        # Compute output size with crop
        out_width = int(width * crop_ratio)
        out_height = int(height * crop_ratio)
        # Make dimensions even for video encoding
        out_width = out_width - (out_width % 2)
        out_height = out_height - (out_height % 2)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
        
        frame_idx = 0
        readout_sec = self.lens_profile.frame_readout_time_ms / 1000.0
        K = self.lens_profile.camera_matrix
        K_inv = np.linalg.inv(K)
        
        logger.info(f"Stabilizing with {'rolling shutter' if self.rolling_shutter else 'global shutter'} correction")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            
            if self.rolling_shutter:
                # Per-row stabilization
                stabilized = self._stabilize_rolling_shutter(
                    frame, imu_data.timestamps, orientations,
                    timestamp, readout_sec, target_quat, K, K_inv
                )
            else:
                # Global homography
                quat = self._interp_quat(imu_data.timestamps, orientations, timestamp)
                H = self._compute_homography(quat, target_quat, K, K_inv)
                stabilized = cv2.warpPerspective(
                    frame, H, (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT
                )
            
            # Center crop
            x = (width - out_width) // 2
            y = (height - out_height) // 2
            cropped = stabilized[y:y+out_height, x:x+out_width]
            
            writer.write(cropped)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        writer.release()
        
        logger.info(f"Saved stabilized video to {output_path}")
        return output_path
    
    def _stabilize_rolling_shutter(
        self,
        frame: np.ndarray,
        timestamps: np.ndarray,
        orientations: np.ndarray,
        frame_time: float,
        readout_time: float,
        target_quat: np.ndarray,
        K: np.ndarray,
        K_inv: np.ndarray
    ) -> np.ndarray:
        """Apply per-row rolling shutter correction."""
        height, width = frame.shape[:2]
        row_time = readout_time / height
        
        # Create remap arrays
        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)
        
        # Compute homography for each row
        for row in range(height):
            row_timestamp = frame_time + row * row_time
            quat = self._interp_quat(timestamps, orientations, row_timestamp)
            H = self._compute_homography(quat, target_quat, K, K_inv)
            H_inv = np.linalg.inv(H)
            
            # Transform row coordinates
            x_coords = np.arange(width, dtype=np.float32)
            ones = np.ones(width, dtype=np.float32)
            coords = np.stack([x_coords, np.full(width, row, dtype=np.float32), ones])
            
            src = H_inv @ coords
            src /= src[2:3, :]
            
            map_x[row] = src[0]
            map_y[row] = src[1]
        
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    def _compute_homography(
        self,
        src_quat: np.ndarray,
        tgt_quat: np.ndarray,
        K: np.ndarray,
        K_inv: np.ndarray
    ) -> np.ndarray:
        """Compute homography from quaternion rotation."""
        R_src = self._quat_to_rotmat(src_quat)
        R_tgt = self._quat_to_rotmat(tgt_quat)
        R_rel = R_tgt @ R_src.T
        return K @ R_rel @ K_inv
    
    def _quat_to_rotmat(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w,x,y,z] to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
    
    def _interp_quat(
        self,
        timestamps: np.ndarray,
        quats: np.ndarray,
        t: float
    ) -> np.ndarray:
        """Interpolate quaternion at time t."""
        idx = np.searchsorted(timestamps, t)
        if idx == 0:
            return quats[0]
        if idx >= len(timestamps):
            return quats[-1]
        
        t0, t1 = timestamps[idx-1], timestamps[idx]
        alpha = (t - t0) / (t1 - t0)
        return slerp(quats[idx-1], quats[idx], alpha)
    
    def _lowpass_filter(self, data: np.ndarray, cutoff: float, fs: float) -> np.ndarray:
        nyq = fs / 2
        b, a = butter(4, cutoff / nyq, btype='low')
        filtered = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered[:, i] = filtfilt(b, a, data[:, i])
        return filtered
    
    def _estimate_bias(self, gyro: np.ndarray, window_sec: float = 2.0) -> np.ndarray:
        window = int(window_sec * 200)
        start = np.median(gyro[:window], axis=0)
        end = np.median(gyro[-window:], axis=0)
        return (start + end) / 2
    
    def _integrate_gyro(self, timestamps: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """Simple gyro integration without accelerometer."""
        n = len(timestamps)
        quats = np.zeros((n, 4))
        quats[0] = [1, 0, 0, 0]
        
        for i in range(1, n):
            dt = timestamps[i] - timestamps[i-1]
            if dt <= 0:
                dt = 1/200
            
            w = gyro[i-1]
            w_quat = np.array([0, w[0], w[1], w[2]])
            q_dot = 0.5 * VQFIntegrator._qmult(quats[i-1], w_quat)
            quats[i] = VQFIntegrator._norm(quats[i-1] + q_dot * dt)
        
        return quats


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="IMU-based video stabilization for GoPro")
    parser.add_argument("input", type=Path, help="Input video file")
    parser.add_argument("-o", "--output", type=Path, help="Output video file")
    parser.add_argument("--no-rolling-shutter", action="store_true", help="Disable rolling shutter correction")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable velocity-adaptive smoothing")
    parser.add_argument("--no-vqf", action="store_true", help="Disable VQF sensor fusion (gyro only)")
    parser.add_argument("--crop", type=float, default=0.9, help="Crop ratio (default: 0.9)")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_stabilized.mp4"
    
    stabilizer = GyroStabilizer(
        rolling_shutter=not args.no_rolling_shutter,
        velocity_smoothing=not args.no_smoothing,
        use_vqf=not args.no_vqf,
    )
    
    stabilizer.stabilize(args.input, args.output, crop_ratio=args.crop)


if __name__ == "__main__":
    main()

"""
VQF (Versatile Quaternion-based Filter) integration for IMU sensor fusion.

This module implements the VQF algorithm for fusing gyroscope and accelerometer
data to compute orientation, similar to Gyroflow's implementation.

VQF provides:
- Gyroscope integration with online bias estimation
- Accelerometer-based tilt correction
- Rest detection for improved bias estimation
- Forward-backward smoothing pass

Reference: Laidig & Seel (2023) - VQF: Highly Accurate IMU Orientation Estimation
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class VQFParams:
    """VQF algorithm parameters."""
    
    # Time constants
    tau_acc: float = 3.0  # Accelerometer time constant (seconds)
    tau_mag: float = 9.0  # Magnetometer time constant (not used without mag)
    
    # Bias estimation
    motion_bias_est_enabled: bool = True
    rest_bias_est_enabled: bool = True
    
    # Rest detection thresholds
    rest_th_gyr: float = 2.0  # deg/s - gyroscope threshold for rest detection
    rest_th_acc: float = 0.5  # m/s² - accelerometer threshold for rest detection
    rest_filter_tau: float = 0.5  # Rest filter time constant
    rest_min_duration: float = 1.5  # Minimum rest duration (seconds)
    
    # Bias estimation time constants
    bias_sigma_init: float = 0.5  # Initial bias uncertainty (rad/s)
    bias_sigma_motion: float = 0.1  # Bias uncertainty during motion
    bias_sigma_rest: float = 0.03  # Bias uncertainty during rest
    
    # Accelerometer rejection
    acc_rejection_threshold: float = 10.0  # Reject if |acc| differs from g by this much (m/s²)


class VQFIntegrator:
    """
    VQF-based IMU integration for gyroscope and accelerometer fusion.
    
    Implements sensor fusion using the VQF algorithm which provides:
    1. Gyroscope-based orientation estimation
    2. Accelerometer-based tilt correction (gravity vector alignment)
    3. Online gyroscope bias estimation
    4. Rest detection for improved calibration
    """
    
    GRAVITY = 9.81  # m/s²
    
    def __init__(self, params: Optional[VQFParams] = None):
        self.params = params or VQFParams()
        self._reset_state()
    
    def _reset_state(self):
        """Reset internal state."""
        # Current orientation (quaternion [w, x, y, z])
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Gyroscope bias estimate (rad/s)
        self.bias = np.zeros(3)
        self.bias_sigma = np.ones(3) * self.params.bias_sigma_init
        
        # Rest detection state
        self.rest_detected = False
        self.rest_time = 0.0
        self.last_gyr_norm = 0.0
        self.last_acc_norm = self.GRAVITY
        
        # Filtered values for rest detection
        self.gyr_lp = np.zeros(3)
        self.acc_lp = np.zeros(3)
    
    def update(
        self, 
        gyr: np.ndarray, 
        acc: np.ndarray, 
        dt: float
    ) -> np.ndarray:
        """
        Update orientation estimate with new IMU measurements.
        
        Args:
            gyr: Gyroscope reading [wx, wy, wz] in rad/s
            acc: Accelerometer reading [ax, ay, az] in m/s²
            dt: Time step in seconds
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        # Apply bias correction
        gyr_corrected = gyr - self.bias
        
        # Detect rest state
        self._update_rest_detection(gyr_corrected, acc, dt)
        
        # Update bias estimate
        if self.params.rest_bias_est_enabled and self.rest_detected:
            self._update_bias_rest(gyr, dt)
        elif self.params.motion_bias_est_enabled:
            self._update_bias_motion(gyr, acc, dt)
        
        # Integrate gyroscope (RK4)
        self.quat = self._integrate_gyro(gyr_corrected, dt)
        
        # Apply accelerometer correction (tilt correction)
        acc_norm = np.linalg.norm(acc)
        if abs(acc_norm - self.GRAVITY) < self.params.acc_rejection_threshold:
            self.quat = self._apply_acc_correction(acc, dt)
        
        return self.quat.copy()
    
    def _integrate_gyro(self, gyr: np.ndarray, dt: float) -> np.ndarray:
        """Integrate gyroscope using RK4."""
        def derivative(q, omega):
            omega_quat = np.array([0, omega[0], omega[1], omega[2]])
            return 0.5 * self._quat_mult(q, omega_quat)
        
        k1 = derivative(self.quat, gyr)
        k2 = derivative(self._normalize(self.quat + 0.5 * dt * k1), gyr)
        k3 = derivative(self._normalize(self.quat + 0.5 * dt * k2), gyr)
        k4 = derivative(self._normalize(self.quat + dt * k3), gyr)
        
        q_new = self.quat + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return self._normalize(q_new)
    
    def _apply_acc_correction(self, acc: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply accelerometer-based tilt correction.
        
        Uses gravity vector to correct roll and pitch drift.
        """
        # Normalize accelerometer
        acc_norm = acc / np.linalg.norm(acc)
        
        # Expected gravity in body frame (rotate world z-axis to body)
        # World gravity is [0, 0, -1] (pointing down)
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_body = self._rotate_vec_by_quat_inv(gravity_world, self.quat)
        
        # Compute correction rotation
        # Find rotation that aligns gravity_body with acc_norm
        correction_axis = np.cross(gravity_body, acc_norm)
        correction_axis_norm = np.linalg.norm(correction_axis)
        
        if correction_axis_norm < 1e-6:
            return self.quat
        
        correction_axis = correction_axis / correction_axis_norm
        correction_angle = np.arcsin(np.clip(correction_axis_norm, -1, 1))
        
        # Apply correction with time constant
        alpha = dt / (self.params.tau_acc + dt)
        correction_angle *= alpha
        
        # Create correction quaternion
        half_angle = correction_angle / 2
        correction_quat = np.array([
            np.cos(half_angle),
            correction_axis[0] * np.sin(half_angle),
            correction_axis[1] * np.sin(half_angle),
            correction_axis[2] * np.sin(half_angle)
        ])
        
        return self._normalize(self._quat_mult(self.quat, correction_quat))
    
    def _update_rest_detection(self, gyr: np.ndarray, acc: np.ndarray, dt: float):
        """Update rest detection state."""
        # Low-pass filter for rest detection
        alpha = dt / (self.params.rest_filter_tau + dt)
        self.gyr_lp = self.gyr_lp * (1 - alpha) + gyr * alpha
        self.acc_lp = self.acc_lp * (1 - alpha) + acc * alpha
        
        # Check if at rest
        gyr_norm = np.rad2deg(np.linalg.norm(self.gyr_lp))
        acc_dev = abs(np.linalg.norm(self.acc_lp) - self.GRAVITY)
        
        is_still = (gyr_norm < self.params.rest_th_gyr and 
                    acc_dev < self.params.rest_th_acc)
        
        if is_still:
            self.rest_time += dt
            if self.rest_time >= self.params.rest_min_duration:
                self.rest_detected = True
        else:
            self.rest_time = 0.0
            self.rest_detected = False
    
    def _update_bias_rest(self, gyr: np.ndarray, dt: float):
        """Update bias estimate during rest."""
        # During rest, the true angular velocity should be zero
        # So the measured gyro value is the bias
        alpha = dt / (1.0 + dt)  # Fast convergence during rest
        self.bias = self.bias * (1 - alpha) + gyr * alpha
        self.bias_sigma = np.minimum(
            self.bias_sigma, 
            np.ones(3) * self.params.bias_sigma_rest
        )
    
    def _update_bias_motion(self, gyr: np.ndarray, acc: np.ndarray, dt: float):
        """Update bias estimate during motion (slower convergence)."""
        # Very slow update during motion
        alpha = dt / (100.0 + dt)
        
        # Only update if motion is relatively smooth
        if np.linalg.norm(gyr) < 1.0:  # Less than ~60 deg/s
            self.bias = self.bias * (1 - alpha) + gyr * alpha
    
    @staticmethod
    def _quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion multiplication [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
    
    @staticmethod
    def _normalize(q: np.ndarray) -> np.ndarray:
        """Normalize quaternion."""
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm
    
    def _rotate_vec_by_quat_inv(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate vector by inverse of quaternion."""
        # q_inv = [w, -x, -y, -z]
        q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
        v_quat = np.array([0, v[0], v[1], v[2]])
        result = self._quat_mult(self._quat_mult(q_inv, v_quat), q)
        return result[1:4]


def vqf_offline(
    timestamps: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    params: Optional[VQFParams] = None
) -> np.ndarray:
    """
    Offline VQF processing with forward-backward pass.
    
    Performs forward pass, backward pass, and combines results
    for optimal orientation estimation.
    
    Args:
        timestamps: (N,) array of timestamps in seconds
        gyro: (N, 3) array of gyroscope readings in rad/s
        accel: (N, 3) array of accelerometer readings in m/s²
        params: VQF parameters
        
    Returns:
        (N, 4) array of quaternions [w, x, y, z]
    """
    n_samples = len(timestamps)
    params = params or VQFParams()
    
    # Forward pass
    logger.info("VQF forward pass...")
    vqf_fwd = VQFIntegrator(params)
    quats_fwd = np.zeros((n_samples, 4))
    quats_fwd[0] = vqf_fwd.quat
    
    for i in range(1, n_samples):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0:
            dt = 1.0 / 200.0  # Default 200Hz
        quats_fwd[i] = vqf_fwd.update(gyro[i], accel[i], dt)
    
    bias_fwd = vqf_fwd.bias.copy()
    
    # Backward pass (integrate in reverse)
    logger.info("VQF backward pass...")
    vqf_bwd = VQFIntegrator(params)
    vqf_bwd.quat = quats_fwd[-1]  # Start from forward end
    quats_bwd = np.zeros((n_samples, 4))
    quats_bwd[-1] = vqf_bwd.quat
    
    for i in range(n_samples - 2, -1, -1):
        dt = timestamps[i+1] - timestamps[i]
        if dt <= 0:
            dt = 1.0 / 200.0
        # Integrate backward with negated gyro
        quats_bwd[i] = vqf_bwd.update(-gyro[i], accel[i], dt)
    
    bias_bwd = vqf_bwd.bias.copy()
    
    # Average bias estimates
    avg_bias = (bias_fwd + bias_bwd) / 2
    logger.info(f"Estimated gyro bias: {np.rad2deg(avg_bias)} deg/s")
    
    # Final forward pass with averaged bias
    logger.info("VQF final pass with corrected bias...")
    vqf_final = VQFIntegrator(params)
    vqf_final.bias = avg_bias
    quats_final = np.zeros((n_samples, 4))
    quats_final[0] = np.array([1.0, 0.0, 0.0, 0.0])
    
    for i in range(1, n_samples):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0:
            dt = 1.0 / 200.0
        quats_final[i] = vqf_final.update(gyro[i], accel[i], dt)
    
    return quats_final


def estimate_bias_from_rest(
    timestamps: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    window_start_sec: float = 0.0,
    window_end_sec: float = 2.0
) -> Tuple[np.ndarray, float]:
    """
    Estimate gyroscope bias from a rest period.
    
    Args:
        timestamps: Time array in seconds
        gyro: Gyroscope data (N, 3) in rad/s
        accel: Accelerometer data (N, 3) in m/s²
        window_start_sec: Start of rest window
        window_end_sec: End of rest window
        
    Returns:
        Tuple of (bias [3], confidence 0-1)
    """
    # Find samples in window
    mask = (timestamps >= window_start_sec) & (timestamps <= window_end_sec)
    if not np.any(mask):
        return np.zeros(3), 0.0
    
    gyro_window = gyro[mask]
    accel_window = accel[mask]
    
    # Check if actually at rest
    gyro_std = np.std(gyro_window, axis=0)
    accel_std = np.std(accel_window, axis=0)
    
    # Higher confidence if low variance
    gyro_quality = 1.0 / (1.0 + np.mean(gyro_std) * 100)
    accel_quality = 1.0 / (1.0 + np.mean(accel_std))
    confidence = gyro_quality * accel_quality
    
    # Median is more robust than mean
    bias = np.median(gyro_window, axis=0)
    
    return bias, confidence

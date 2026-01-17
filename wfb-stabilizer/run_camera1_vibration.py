#!/usr/bin/python3
# Updated: January 17, 2026
"""
Vibration-compensated Video Stabilization using Orange Cube IMU data.

Uses high-frequency gyroscope data from MAVLink to detect and compensate
for camera vibrations in real-time.

Features:
- High-frequency gyro rate integration for vibration detection
- Bandpass filtering to separate vibration from intentional movement
- Real-time frame compensation

Based on wfb-stabilizer by ejowerks.
"""

import cv2
import numpy as np
import sys
import time
import threading
from collections import deque
import os

# Add parent directory for mavlink module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from mavlink.orange_cube_reader import OrangeCubeReader, AttitudeData, IMUData

#################### USER VARS ######################################
CAMERA_INDEX = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60

# Stabilization settings
zoomFactor = 0.85           # Zoom to hide edge bouncing
maxCorrectionDeg = 3.0      # Maximum correction in degrees

# Vibration filter settings
VIBRATION_ALPHA = 0.3       # High-pass filter for vibration (0-1, higher = more responsive)
DRIFT_ALPHA = 0.02          # Low-pass filter for drift compensation
GYRO_SCALE = 1.0            # Scale factor for gyro corrections

# Display settings
showFullScreen = 0
showOverlay = 1             # Show gyro data overlay
showVibrationGraph = 1      # Show vibration amplitude graph
delay_time = 1

# Orange Cube settings
MAVLINK_PORT = "COM6"       # Set to None for auto-detect
MAVLINK_BAUDRATE = 115200
IMU_RATE_HZ = 200           # High rate for vibration detection

# Camera FOV (degrees)
FOV_HORIZONTAL = 60
FOV_VERTICAL = 34

######################################################################


class VibrationFilter:
    """
    High-pass filter to extract vibration from gyro signal.
    Uses exponential moving average as a simple high-pass filter.
    """
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # Higher = more responsive to vibrations
        self.slow_avg = np.zeros(3)  # Low-pass filtered signal
        self.vibration = np.zeros(3)  # High-frequency component
        
    def update(self, gyro_rates):
        """
        Update filter with new gyro rates.
        
        Args:
            gyro_rates: [roll_rate, pitch_rate, yaw_rate] in rad/s
            
        Returns:
            vibration component [roll, pitch, yaw]
        """
        gyro = np.array(gyro_rates)
        
        # Low-pass filter to get slow movement (intentional motion)
        self.slow_avg = (1 - self.alpha) * self.slow_avg + self.alpha * gyro
        
        # High-pass = original - low-pass (this is the vibration)
        self.vibration = gyro - self.slow_avg
        
        return self.vibration
    
    def reset(self):
        self.slow_avg = np.zeros(3)
        self.vibration = np.zeros(3)


class GyroIntegrator:
    """
    Integrates gyro rates to get angular displacement.
    With drift compensation.
    """
    def __init__(self, drift_alpha=0.02):
        self.drift_alpha = drift_alpha
        self.angle = np.zeros(3)  # Integrated angle [roll, pitch, yaw]
        self.last_time = None
        
    def update(self, gyro_rates, timestamp=None):
        """
        Integrate gyro rates.
        
        Args:
            gyro_rates: [roll_rate, pitch_rate, yaw_rate] in rad/s
            timestamp: Current time in seconds
        """
        if timestamp is None:
            timestamp = time.time()
            
        if self.last_time is None:
            self.last_time = timestamp
            return self.angle
        
        dt = timestamp - self.last_time
        self.last_time = timestamp
        
        if dt <= 0 or dt > 0.1:  # Sanity check
            return self.angle
        
        # Integrate
        self.angle += np.array(gyro_rates) * dt
        
        # Drift compensation - slowly decay toward zero
        self.angle *= (1 - self.drift_alpha)
        
        return self.angle
    
    def reset(self):
        self.angle = np.zeros(3)
        self.last_time = None


class VibrationStabilizer:
    """Real-time vibration-compensated video stabilizer."""
    
    def __init__(self):
        self.cube_reader = None
        self.is_connected = False
        
        # Filters
        self.vib_filter = VibrationFilter(alpha=VIBRATION_ALPHA)
        self.integrator = GyroIntegrator(drift_alpha=DRIFT_ALPHA)
        
        # Current state
        self.current_imu = IMUData()
        self.current_attitude = AttitudeData()
        self.data_lock = threading.Lock()
        
        # Vibration history for visualization
        self.vib_history = deque(maxlen=200)
        
        # Correction state
        self.correction_roll = 0.0
        self.correction_pitch = 0.0
        self.correction_yaw = 0.0
        
        # Reader thread
        self.reader_thread = None
        self.running = False
        
        # Stats
        self.imu_count = 0
        self.last_stats_time = time.time()
        self.imu_rate = 0.0
        self.vibration_rms = 0.0
        
    def connect(self) -> bool:
        """Connect to Orange Cube."""
        print("Connecting to Orange Cube...")
        self.cube_reader = OrangeCubeReader(
            port=MAVLINK_PORT,
            baudrate=MAVLINK_BAUDRATE
        )
        
        if self.cube_reader.connect():
            self.cube_reader.request_data_streams(rate_hz=IMU_RATE_HZ)
            self.is_connected = True
            return True
        return False
    
    def start_reader(self):
        """Start background IMU reader thread."""
        self.running = True
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()
        
    def stop_reader(self):
        """Stop the reader thread."""
        self.running = False
        if self.reader_thread:
            self.reader_thread.join(timeout=1)
            
    def _reader_loop(self):
        """Background loop to read IMU data."""
        while self.running and self.cube_reader:
            msg = self.cube_reader.read_message(timeout=0.05)
            if msg:
                msg_type = self.cube_reader.process_message(msg)
                
                # Process high-rate IMU data for vibration
                if msg_type in ['RAW_IMU', 'SCALED_IMU', 'HIGHRES_IMU']:
                    with self.data_lock:
                        self.current_imu = self.cube_reader.imu_data
                        
                        # Get gyro rates
                        gyro_rates = [
                            self.current_imu.gyro_x,
                            self.current_imu.gyro_y,
                            self.current_imu.gyro_z
                        ]
                        
                        # Extract vibration component
                        vibration = self.vib_filter.update(gyro_rates)
                        
                        # Integrate vibration for angular correction
                        angles = self.integrator.update(vibration)
                        
                        # Store corrections
                        self.correction_roll = angles[0] * GYRO_SCALE
                        self.correction_pitch = angles[1] * GYRO_SCALE
                        self.correction_yaw = angles[2] * GYRO_SCALE
                        
                        # Store vibration magnitude for visualization
                        vib_magnitude = np.sqrt(np.sum(vibration**2))
                        self.vib_history.append(vib_magnitude)
                    
                    # Update stats
                    self.imu_count += 1
                    
                elif msg_type == 'ATTITUDE':
                    with self.data_lock:
                        self.current_attitude = self.cube_reader.attitude_data
                
                # Update rate calculation
                now = time.time()
                dt = now - self.last_stats_time
                if dt >= 1.0:
                    self.imu_rate = self.imu_count / dt
                    self.imu_count = 0
                    self.last_stats_time = now
                    
                    # Calculate RMS vibration
                    if len(self.vib_history) > 0:
                        self.vibration_rms = np.sqrt(np.mean(np.array(list(self.vib_history))**2))
    
    def reset(self):
        """Reset filters and integrators."""
        with self.data_lock:
            self.vib_filter.reset()
            self.integrator.reset()
            self.correction_roll = 0.0
            self.correction_pitch = 0.0
            self.correction_yaw = 0.0
        print("Vibration compensation reset")
    
    def get_correction(self) -> tuple:
        """
        Get current stabilization correction.
        
        Returns:
            (dx, dy, da) correction in pixels and radians
        """
        with self.data_lock:
            roll = self.correction_roll
            pitch = self.correction_pitch
            yaw = self.correction_yaw
        
        # Clamp corrections
        max_rad = np.radians(maxCorrectionDeg)
        roll = np.clip(roll, -max_rad, max_rad)
        pitch = np.clip(pitch, -max_rad, max_rad)
        yaw = np.clip(yaw, -max_rad, max_rad)
        
        # Convert to pixel offsets
        pixels_per_rad_h = CAMERA_WIDTH / np.radians(FOV_HORIZONTAL)
        pixels_per_rad_v = CAMERA_HEIGHT / np.radians(FOV_VERTICAL)
        
        # Roll -> rotation, Pitch -> vertical shift, Yaw -> horizontal shift
        dx = -yaw * pixels_per_rad_h
        dy = pitch * pixels_per_rad_v
        da = -roll  # rotation in radians
        
        return dx, dy, da
    
    def get_status(self) -> dict:
        """Get current status for overlay."""
        with self.data_lock:
            return {
                'connected': self.is_connected,
                'imu_rate': self.imu_rate,
                'vibration_rms': np.degrees(self.vibration_rms),
                'gyro_x': np.degrees(self.current_imu.gyro_x),
                'gyro_y': np.degrees(self.current_imu.gyro_y),
                'gyro_z': np.degrees(self.current_imu.gyro_z),
                'roll_deg': np.degrees(self.current_attitude.roll),
                'pitch_deg': np.degrees(self.current_attitude.pitch),
                'yaw_deg': np.degrees(self.current_attitude.yaw),
                'corr_roll': np.degrees(self.correction_roll),
                'corr_pitch': np.degrees(self.correction_pitch),
                'corr_yaw': np.degrees(self.correction_yaw),
                'vib_history': list(self.vib_history),
            }


def draw_vibration_graph(frame, vib_history, x, y, width, height):
    """Draw a mini graph of vibration amplitude."""
    if len(vib_history) < 2:
        return
    
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (20, 20, 20), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 1)
    
    # Convert to numpy and normalize
    vib = np.array(vib_history)
    if len(vib) > width:
        vib = vib[-width:]
    
    max_val = max(np.max(vib), 0.1)  # Prevent division by zero
    vib_normalized = vib / max_val
    
    # Draw line
    points = []
    for i, v in enumerate(vib_normalized):
        px = x + int(i * width / len(vib))
        py = y + height - int(v * height * 0.9)
        points.append((px, py))
    
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i+1], (0, 255, 255), 1)
    
    # Label
    cv2.putText(frame, f"Vib: {np.degrees(max_val):.1f} deg/s", 
               (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def main():
    print("=" * 60)
    print("Vibration-Compensated Video Stabilization")
    print("=" * 60)
    
    # Initialize stabilizer
    stabilizer = VibrationStabilizer()
    
    if not stabilizer.connect():
        print("Failed to connect to Orange Cube!")
        print("Running without vibration compensation...")
        use_gyro = False
    else:
        stabilizer.start_reader()
        use_gyro = True
        print("Vibration compensation active!")
        print(f"IMU rate: {IMU_RATE_HZ} Hz")
        print(f"Vibration filter alpha: {VIBRATION_ALPHA}")
        time.sleep(0.5)  # Let some data come in
    
    # Open camera
    print(f"\nOpening camera {CAMERA_INDEX}...")
    video = cv2.VideoCapture(CAMERA_INDEX)
    
    if not video.isOpened():
        print(f"ERROR: Could not open camera {CAMERA_INDEX}")
        return
    
    video.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    video.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    actual_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera: {actual_w}x{actual_h} @ {actual_fps} FPS")
    print("Press 'Q' to quit, 'R' to reset filters")
    print("=" * 60)
    
    window_name = "Vibration Stabilized - Camera 1"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # Get stabilization correction
            if use_gyro:
                dx, dy, da = stabilizer.get_correction()
            else:
                dx, dy, da = 0, 0, 0
            
            # Build transformation matrix
            center = (actual_w / 2, actual_h / 2)
            
            # Combine rotation and translation
            M = np.zeros((2, 3), dtype=np.float32)
            M[0, 0] = np.cos(da)
            M[0, 1] = -np.sin(da)
            M[1, 0] = np.sin(da)
            M[1, 1] = np.cos(da)
            
            # Rotate around center then translate
            M[0, 2] = dx + center[0] - center[0] * M[0, 0] - center[1] * M[0, 1]
            M[1, 2] = dy + center[1] - center[0] * M[1, 0] - center[1] * M[1, 1]
            
            # Apply stabilization transform
            stabilized = cv2.warpAffine(frame, M, (actual_w, actual_h))
            
            # Apply zoom to hide edges
            T = cv2.getRotationMatrix2D(center, 0, zoomFactor)
            stabilized = cv2.warpAffine(stabilized, T, (actual_w, actual_h))
            
            # Draw overlay
            if showOverlay and use_gyro:
                status = stabilizer.get_status()
                
                # Background for text
                cv2.rectangle(stabilized, (5, 5), (320, 160), (0, 0, 0), -1)
                cv2.rectangle(stabilized, (5, 5), (320, 160), (255, 255, 255), 1)
                
                # Status text
                color = (0, 255, 0) if status['connected'] else (0, 0, 255)
                y_pos = 25
                
                cv2.putText(stabilized, f"Gyro: {'Connected' if status['connected'] else 'Disconnected'}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += 20
                
                cv2.putText(stabilized, f"IMU Rate: {status['imu_rate']:.0f} Hz | Vib RMS: {status['vibration_rms']:.2f} deg/s", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y_pos += 18
                
                cv2.putText(stabilized, f"Gyro: X={status['gyro_x']:.1f} Y={status['gyro_y']:.1f} Z={status['gyro_z']:.1f} deg/s", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                y_pos += 18
                
                cv2.putText(stabilized, f"Attitude: R={status['roll_deg']:.1f} P={status['pitch_deg']:.1f} Y={status['yaw_deg']:.1f}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y_pos += 18
                
                cv2.putText(stabilized, f"Correction: R={status['corr_roll']:.2f} P={status['corr_pitch']:.2f} Y={status['corr_yaw']:.2f}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                y_pos += 18
                
                # FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(stabilized, f"FPS: {fps:.1f} | Press R=reset, Q=quit", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                
                # Draw vibration graph
                if showVibrationGraph and len(status['vib_history']) > 0:
                    draw_vibration_graph(stabilized, status['vib_history'], 
                                        actual_w - 210, 10, 200, 60)
            
            # Show frame
            if showFullScreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            cv2.imshow(window_name, stabilized)
            
            # Handle keys
            key = cv2.waitKey(delay_time) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                stabilizer.reset()
                
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("\nCleaning up...")
        if use_gyro:
            stabilizer.stop_reader()
        video.release()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    main()

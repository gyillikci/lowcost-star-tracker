#!/usr/bin/python3
# Updated: January 17, 2026
"""
Gyro-based Video Stabilization using Orange Cube IMU data.

Reads gyroscope data from Orange Cube flight controller via MAVLink
and uses it to stabilize the camera feed in real-time.

Based on wfb-stabilizer by ejowerks, modified for gyro-based stabilization.
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
from mavlink.orange_cube_reader import OrangeCubeReader, AttitudeData

#################### USER VARS ######################################
CAMERA_INDEX = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60

# Stabilization settings
zoomFactor = 0.85           # Zoom to hide edge bouncing
smoothingFactor = 0.95      # 0-1, higher = smoother but more lag
maxCorrectionDeg = 5.0      # Maximum correction in degrees

# Display settings
showFullScreen = 0
showOverlay = 1             # Show gyro data overlay
delay_time = 1

# Orange Cube settings
MAVLINK_PORT = "COM6"         # Auto-detect, or set to "COM3" etc.
MAVLINK_BAUDRATE = 115200
IMU_RATE_HZ = 100           # Request rate from IMU

######################################################################


class GyroStabilizer:
    """Real-time gyro-based video stabilizer."""
    
    def __init__(self):
        self.cube_reader = None
        self.is_connected = False
        
        # Attitude tracking
        self.base_roll = 0.0
        self.base_pitch = 0.0
        self.base_yaw = 0.0
        self.base_set = False
        
        # Smoothed corrections
        self.smooth_roll = 0.0
        self.smooth_pitch = 0.0
        self.smooth_yaw = 0.0
        
        # Latest attitude
        self.current_attitude = AttitudeData()
        self.attitude_lock = threading.Lock()
        
        # Reader thread
        self.reader_thread = None
        self.running = False
        
        # Stats
        self.imu_count = 0
        self.last_imu_time = time.time()
        self.imu_rate = 0.0
        
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
            msg = self.cube_reader.read_message(timeout=0.1)
            if msg:
                msg_type = self.cube_reader.process_message(msg)
                
                if msg_type == 'ATTITUDE':
                    with self.attitude_lock:
                        self.current_attitude = self.cube_reader.attitude_data
                        
                    # Set base attitude on first reading
                    if not self.base_set:
                        self.base_roll = self.current_attitude.roll
                        self.base_pitch = self.current_attitude.pitch
                        self.base_yaw = self.current_attitude.yaw
                        self.base_set = True
                        print(f"Base attitude set: Roll={np.degrees(self.base_roll):.1f}° "
                              f"Pitch={np.degrees(self.base_pitch):.1f}° "
                              f"Yaw={np.degrees(self.base_yaw):.1f}°")
                    
                    # Update rate calculation
                    self.imu_count += 1
                    now = time.time()
                    dt = now - self.last_imu_time
                    if dt >= 1.0:
                        self.imu_rate = self.imu_count / dt
                        self.imu_count = 0
                        self.last_imu_time = now
    
    def reset_base(self):
        """Reset the base attitude to current position."""
        self.base_set = False
        print("Base attitude reset - will recalibrate on next reading")
    
    def get_correction(self) -> tuple:
        """
        Get current stabilization correction.
        
        Returns:
            (dx, dy, da) correction in pixels and radians
        """
        if not self.base_set:
            return 0, 0, 0
            
        with self.attitude_lock:
            # Calculate delta from base attitude
            delta_roll = self.current_attitude.roll - self.base_roll
            delta_pitch = self.current_attitude.pitch - self.base_pitch
            delta_yaw = self.current_attitude.yaw - self.base_yaw
            
            # Handle yaw wraparound
            if delta_yaw > np.pi:
                delta_yaw -= 2 * np.pi
            elif delta_yaw < -np.pi:
                delta_yaw += 2 * np.pi
        
        # Clamp corrections
        max_rad = np.radians(maxCorrectionDeg)
        delta_roll = np.clip(delta_roll, -max_rad, max_rad)
        delta_pitch = np.clip(delta_pitch, -max_rad, max_rad)
        delta_yaw = np.clip(delta_yaw, -max_rad, max_rad)
        
        # Apply smoothing (exponential moving average)
        self.smooth_roll = smoothingFactor * self.smooth_roll + (1 - smoothingFactor) * delta_roll
        self.smooth_pitch = smoothingFactor * self.smooth_pitch + (1 - smoothingFactor) * delta_pitch
        self.smooth_yaw = smoothingFactor * self.smooth_yaw + (1 - smoothingFactor) * delta_yaw
        
        # Convert to pixel offsets
        # Assuming FOV of ~60 degrees horizontal
        fov_h = 60  # degrees
        pixels_per_degree = CAMERA_WIDTH / fov_h
        
        # Roll -> rotation, Pitch -> vertical shift, Yaw -> horizontal shift
        # Note: Signs may need adjustment based on camera/IMU orientation
        dx = -self.smooth_yaw * pixels_per_degree * (180 / np.pi)
        dy = self.smooth_pitch * pixels_per_degree * (180 / np.pi)
        da = -self.smooth_roll  # rotation in radians
        
        return dx, dy, da
    
    def get_status(self) -> dict:
        """Get current status for overlay."""
        with self.attitude_lock:
            return {
                'connected': self.is_connected,
                'imu_rate': self.imu_rate,
                'roll_deg': np.degrees(self.current_attitude.roll),
                'pitch_deg': np.degrees(self.current_attitude.pitch),
                'yaw_deg': np.degrees(self.current_attitude.yaw),
                'roll_delta': np.degrees(self.smooth_roll),
                'pitch_delta': np.degrees(self.smooth_pitch),
                'yaw_delta': np.degrees(self.smooth_yaw),
            }


def main():
    print("=" * 60)
    print("Gyro-based Video Stabilization")
    print("=" * 60)
    
    # Initialize gyro stabilizer
    stabilizer = GyroStabilizer()
    
    if not stabilizer.connect():
        print("Failed to connect to Orange Cube!")
        print("Running without gyro stabilization...")
        use_gyro = False
    else:
        stabilizer.start_reader()
        use_gyro = True
        print("Gyro stabilization active!")
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
    print("Press 'Q' to quit, 'R' to reset base attitude")
    print("=" * 60)
    
    window_name = "Gyro Stabilized - Camera 1"
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
            # Rotation around center
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
                cv2.rectangle(stabilized, (5, 5), (300, 130), (0, 0, 0), -1)
                cv2.rectangle(stabilized, (5, 5), (300, 130), (255, 255, 255), 1)
                
                # Status text
                color = (0, 255, 0) if status['connected'] else (0, 0, 255)
                cv2.putText(stabilized, f"Gyro: {'Connected' if status['connected'] else 'Disconnected'}", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(stabilized, f"IMU Rate: {status['imu_rate']:.1f} Hz", 
                           (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(stabilized, f"Attitude: R={status['roll_deg']:.1f} P={status['pitch_deg']:.1f} Y={status['yaw_deg']:.1f}", 
                           (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(stabilized, f"Correction: dR={status['roll_delta']:.2f} dP={status['pitch_delta']:.2f} dY={status['yaw_delta']:.2f}", 
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(stabilized, f"FPS: {fps:.1f}", 
                           (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(stabilized, "Press R to reset, Q to quit", 
                           (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            # Show frame
            if showFullScreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            cv2.imshow(window_name, stabilized)
            
            # Handle keys
            key = cv2.waitKey(delay_time) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                stabilizer.reset_base()
                
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

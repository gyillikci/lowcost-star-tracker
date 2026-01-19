#!/usr/bin/python3
# Updated: January 17, 2026
"""
Roll & Pitch Video Stabilization using Orange Cube Gyro

Reads roll and pitch angles from Orange Cube via MAVLink and compensates
for camera rotation in real-time. No template matching - just gyro data.

Controls:
  B - Reset roll/pitch baseline to current angles
  Q/ESC - Quit
"""

import cv2
import numpy as np
import time
import sys
import os
import threading
from collections import deque

# Add parent directory for mavlink module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from mavlink.orange_cube_reader import OrangeCubeReader, AttitudeData

#################### USER VARS ######################################
CAMERA_INDEX = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Orange Cube settings
MAVLINK_PORT = "COM6"
MAVLINK_BAUDRATE = 115200

# Stabilization settings
ZOOM_FACTOR = 1.0  # Slight zoom to hide edges during rotation

# Camera FOV settings (Harrier 10x varies with zoom)
# At 1x zoom: HFOV ~56°, VFOV ~34°
# At 10x zoom: HFOV ~6°, VFOV ~3.4°
# Adjust this based on your current zoom level
VFOV_DEGREES = 34.0  # Vertical field of view in degrees

# Calculated pitch scale: pixels per degree
PITCH_SCALE = CAMERA_HEIGHT / VFOV_DEGREES  # ~21.2 px/deg at 34° VFOV

# Delay compensation
CAMERA_DELAY_MS = 0  # Screen-to-screen delay in milliseconds
#####################################################################


class GyroReader:
    """Background thread to read gyro data from Orange Cube with delay buffer."""
    
    def __init__(self, port, baudrate, delay_ms=95):
        self.port = port
        self.baudrate = baudrate
        self.delay_ms = delay_ms
        self.cube = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Current values (latest)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.msg_count = 0
        self.connected = False
        
        # Delay buffer: list of (timestamp_ms, roll, pitch, yaw)
        self.buffer = deque(maxlen=500)  # ~5 seconds at 100Hz
    
    def connect(self):
        """Connect to Orange Cube."""
        self.cube = OrangeCubeReader(port=self.port, baudrate=self.baudrate)
        if self.cube.connect():
            self.cube.request_data_streams(rate_hz=100)
            self.connected = True
            return True
        return False
    
    def start(self):
        """Start background reading thread."""
        if not self.connected:
            return False
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True
    
    def _read_loop(self):
        """Background loop to read messages."""
        while self.running:
            try:
                msg = self.cube.read_message(timeout=0.05)
                if msg:
                    msg_type = self.cube.process_message(msg)
                    if msg_type == 'ATTITUDE':
                        now_ms = time.time() * 1000
                        with self.lock:
                            self.roll = self.cube.attitude_data.roll
                            self.pitch = self.cube.attitude_data.pitch
                            self.yaw = self.cube.attitude_data.yaw
                            self.msg_count += 1
                            # Store in buffer with timestamp
                            self.buffer.append((now_ms, self.roll, self.pitch, self.yaw))
            except Exception as e:
                pass
    
    def get_roll(self):
        """Get current roll angle."""
        with self.lock:
            return self.roll
    
    def get_attitude(self):
        """Get roll, pitch, yaw (latest values)."""
        with self.lock:
            return self.roll, self.pitch, self.yaw
    
    def get_delayed_attitude(self):
        """
        Get attitude from delay_ms ago to compensate for camera latency.
        Returns (roll, pitch, yaw) or current values if buffer insufficient.
        """
        target_time = time.time() * 1000 - self.delay_ms
        
        with self.lock:
            if len(self.buffer) < 2:
                return self.roll, self.pitch, self.yaw
            
            # Find the reading closest to target_time
            best = None
            for entry in self.buffer:
                ts, r, p, y = entry
                if ts <= target_time:
                    best = (r, p, y)
                else:
                    break
            
            if best is None:
                # All entries are newer than target, use oldest
                return self.buffer[0][1], self.buffer[0][2], self.buffer[0][3]
            
            return best
    
    def get_msg_count(self):
        """Get message count."""
        with self.lock:
            return self.msg_count
    
    def stop(self):
        """Stop reading and close connection."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cube:
            self.cube.close()


def main():
    print("=" * 60)
    print("Roll & Pitch Video Stabilization (Orange Cube Gyro)")
    print(f"Latency compensation: {CAMERA_DELAY_MS}ms")
    print("=" * 60)
    
    # Connect to Orange Cube
    print("\nConnecting to Orange Cube...")
    gyro = GyroReader(MAVLINK_PORT, MAVLINK_BAUDRATE, delay_ms=CAMERA_DELAY_MS)
    
    if not gyro.connect():
        print("ERROR: Could not connect to Orange Cube!")
        return
    
    print("✓ Connected to Orange Cube")
    
    # Start background reading
    gyro.start()
    print("Streaming attitude data...")
    
    # Wait for first attitude reading
    print("Waiting for attitude data...")
    timeout = time.time() + 5
    base_roll = None
    base_pitch = None
    while time.time() < timeout:
        if gyro.get_msg_count() > 0:
            base_roll, base_pitch, _ = gyro.get_attitude()
            print(f"✓ Got attitude! Base roll: {np.degrees(base_roll):.1f}°, pitch: {np.degrees(base_pitch):.1f}°")
            break
        time.sleep(0.01)
    
    if base_roll is None:
        print("ERROR: No attitude data received!")
        gyro.stop()
        return
    
    # Open camera
    print(f"\nOpening camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        gyro.stop()
        return
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: {actual_w}x{actual_h} @ {actual_fps} FPS")
    
    center = (actual_w // 2, actual_h // 2)
    
    print("\n" + "=" * 50)
    print("ROLL & PITCH STABILIZATION ACTIVE")
    print("=" * 50)
    print("Press 'B' to reset baseline")
    print("Press 'Q' or ESC to quit")
    print("=" * 50)
    
    cv2.namedWindow("Roll+Pitch Stabilized", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            frame_count += 1
            
            # Get current attitude from Orange Cube
            current_roll, current_pitch, _ = gyro.get_attitude()
            
            # Calculate corrections (negate to compensate)
            roll_correction = -(current_roll - base_roll)
            pitch_correction = -(current_pitch - base_pitch)
            
            # Convert to degrees
            roll_deg = np.degrees(roll_correction)
            pitch_deg = np.degrees(pitch_correction)
            current_roll_deg = np.degrees(current_roll)
            current_pitch_deg = np.degrees(current_pitch)
            base_roll_deg = np.degrees(base_roll)
            base_pitch_deg = np.degrees(base_pitch)
            
            # Calculate pitch as vertical translation (pixels)
            pitch_pixels = pitch_deg * PITCH_SCALE
            
            # Build combined transform: rotation + translation
            # Start with rotation matrix
            M = cv2.getRotationMatrix2D(center, roll_deg, ZOOM_FACTOR)
            # Add pitch translation (vertical shift)
            M[1, 2] += pitch_pixels
            
            # Apply combined transform
            stabilized = cv2.warpAffine(frame, M, (actual_w, actual_h),
                                        flags=cv2.INTER_LANCZOS4,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
            
            # Status overlay
            cv2.rectangle(stabilized, (5, 5), (380, 115), (0, 0, 0), -1)
            cv2.rectangle(stabilized, (5, 5), (380, 115), (255, 255, 255), 1)
            
            cv2.putText(stabilized, f"Roll:  {current_roll_deg:+.1f} -> correction: {roll_deg:+.1f} deg",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(stabilized, f"Pitch: {current_pitch_deg:+.1f} -> correction: {pitch_deg:+.1f} deg ({pitch_pixels:+.0f}px)",
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(stabilized, f"Base: roll={base_roll_deg:.1f}, pitch={base_pitch_deg:.1f}",
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            
            # FPS and IMU rate
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            imu_count = gyro.get_msg_count()
            imu_rate = imu_count / elapsed if elapsed > 0 else 0
            
            cv2.putText(stabilized, f"FPS: {fps:.1f}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(stabilized, f"IMU: {imu_rate:.0f}Hz",
                       (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(stabilized, f"Delay: {CAMERA_DELAY_MS}ms",
                       (200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(stabilized, f"[B] Reset baseline",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            
            cv2.imshow("Roll+Pitch Stabilized", stabilized)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('b'):
                # Reset baseline (use current/latest values for baseline)
                base_roll, base_pitch, _ = gyro.get_attitude()
                print(f"Baseline reset: roll={np.degrees(base_roll):.1f}°, pitch={np.degrees(base_pitch):.1f}°")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        gyro.stop()
        print("Done!")


if __name__ == "__main__":
    main()

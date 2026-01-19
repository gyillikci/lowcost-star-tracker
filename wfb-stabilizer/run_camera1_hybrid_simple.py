#!/usr/bin/python3
# Updated: January 17, 2026
"""
Hybrid Stabilization: Gyro Roll/Pitch + Simple Template XY
Simplified version to avoid freezing.
"""

import cv2
import numpy as np
import time
import sys
import os
import threading
from collections import deque

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from mavlink.orange_cube_reader import OrangeCubeReader

#################### USER VARS ######################################
CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 60

MAVLINK_PORT = "COM6"
MAVLINK_BAUDRATE = 115200

ZOOM_FACTOR = 0.9
VFOV_DEGREES = 34.0
PITCH_SCALE = CAMERA_HEIGHT / VFOV_DEGREES
SEARCH_MARGIN = 200  # Small margin for speed
#####################################################################


class GyroReader:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.cube = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.msg_count = 0
        self.connected = False
    
    def connect(self):
        self.cube = OrangeCubeReader(port=self.port, baudrate=self.baudrate)
        if self.cube.connect():
            self.cube.request_data_streams(rate_hz=100)
            self.connected = True
            return True
        return False
    
    def start(self):
        if not self.connected:
            return False
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True
    
    def _read_loop(self):
        while self.running:
            try:
                msg = self.cube.read_message(timeout=0.02)
                if msg:
                    msg_type = self.cube.process_message(msg)
                    if msg_type == 'ATTITUDE':
                        with self.lock:
                            self.roll = self.cube.attitude_data.roll
                            self.pitch = self.cube.attitude_data.pitch
                            self.yaw = self.cube.attitude_data.yaw
                            self.msg_count += 1
            except:
                pass
    
    def get_attitude(self):
        with self.lock:
            return self.roll, self.pitch, self.yaw
    
    def get_msg_count(self):
        with self.lock:
            return self.msg_count
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        if self.cube:
            self.cube.close()


def main():
    print("=" * 60)
    print("Hybrid Stabilization: Gyro + Template (Simple)")
    print("=" * 60)
    
    # Connect to Orange Cube
    print("\nConnecting to Orange Cube...")
    gyro = GyroReader(MAVLINK_PORT, MAVLINK_BAUDRATE)
    
    if not gyro.connect():
        print("ERROR: Could not connect to Orange Cube!")
        return
    
    print("✓ Connected")
    gyro.start()
    
    # Wait for attitude
    timeout = time.time() + 3
    base_roll, base_pitch = None, None
    while time.time() < timeout:
        if gyro.get_msg_count() > 0:
            base_roll, base_pitch, _ = gyro.get_attitude()
            print(f"✓ Base: roll={np.degrees(base_roll):.1f}°, pitch={np.degrees(base_pitch):.1f}°")
            break
        time.sleep(0.01)
    
    if base_roll is None:
        print("ERROR: No attitude data!")
        gyro.stop()
        return
    
    # Open camera
    print(f"\nOpening camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        gyro.stop()
        return
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {w}x{h}")
    
    center = (w // 2, h // 2)
    
    cv2.namedWindow("Stabilized", cv2.WINDOW_NORMAL)
    
    # Template variables
    template = None
    template_gray = None
    template_pos = None
    template_size = (0, 0)
    selecting = False
    sel_start = None
    sel_end = None
    
    # Mouse callback for template selection
    def mouse_cb(event, x, y, flags, param):
        nonlocal selecting, sel_start, sel_end
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            sel_start = (x, y)
            sel_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            sel_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            sel_end = (x, y)
    
    cv2.setMouseCallback("Stabilized", mouse_cb)
    
    print("\n" + "=" * 50)
    print("RUNNING - Draw rectangle to set template")
    print("B = reset gyro, R = clear template, Q = quit")
    print("=" * 50)
    
    frame_count = 0
    start_time = time.time()
    dx, dy, quality = 0.0, 0.0, 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # Get gyro data
            current_roll, current_pitch, _ = gyro.get_attitude()
            roll_deg = np.degrees(-(current_roll - base_roll))
            pitch_deg = np.degrees(-(current_pitch - base_pitch))
            pitch_px = pitch_deg * PITCH_SCALE
            
            # Apply gyro stabilization
            M = cv2.getRotationMatrix2D(center, roll_deg, ZOOM_FACTOR)
            M[1, 2] += pitch_px
            
            stabilized = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)
            
            # Template matching (if template exists)
            if template_gray is not None:
                th, tw = template_gray.shape
                tx, ty = template_pos
                
                # Search region
                margin = SEARCH_MARGIN
                x1 = max(0, tx - tw//2 - margin)
                y1 = max(0, ty - th//2 - margin)
                x2 = min(w, tx + tw//2 + margin)
                y2 = min(h, ty + th//2 + margin)
                
                if x2 - x1 >= tw and y2 - y1 >= th:
                    gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
                    search = gray[y1:y2, x1:x2]
                    
                    result = cv2.matchTemplate(search, template_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    quality = max_val
                    
                    if max_val > 0.5:
                        found_x = x1 + max_loc[0] + tw//2
                        found_y = y1 + max_loc[1] + th//2
                        dx = tx - found_x
                        dy = ty - found_y
                        
                        # Apply XY correction
                        M2 = np.float32([[1, 0, dx], [0, 1, dy]])
                        stabilized = cv2.warpAffine(stabilized, M2, (w, h))
                    
                    # Draw template box
                    color = (0, 255, 0) if quality > 0.7 else (0, 255, 255)
                    cv2.rectangle(stabilized, (tx - tw//2, ty - th//2), 
                                 (tx + tw//2, ty + th//2), color, 2)
            
            # Handle template selection (no visible rectangle)
            if sel_start and sel_end:
                x1, y1 = min(sel_start[0], sel_end[0]), min(sel_start[1], sel_end[1])
                x2, y2 = max(sel_start[0], sel_end[0]), max(sel_start[1], sel_end[1])
                
                # If mouse released, set template
                if not selecting and x2 > x1 + 10 and y2 > y1 + 10:
                    gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
                    template_gray = gray[y1:y2, x1:x2].copy()
                    template_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
                    template_size = (x2 - x1, y2 - y1)
                    print(f"Template set: {template_size[0]}x{template_size[1]} at {template_pos}")
                    sel_start, sel_end = None, None
            
            # Overlay
            cv2.rectangle(stabilized, (5, 5), (350, 80), (0, 0, 0), -1)
            cv2.putText(stabilized, f"Roll: {roll_deg:+.1f} Pitch: {pitch_deg:+.1f}",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(stabilized, f"Template: dx={dx:.1f} dy={dy:.1f} q={quality:.2f}",
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(stabilized, f"FPS: {fps:.1f}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow("Stabilized", stabilized)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('b'):
                base_roll, base_pitch, _ = gyro.get_attitude()
                template_gray = None
                template_pos = None
                dx, dy, quality = 0, 0, 0
                print(f"Baseline and template reset")
            elif key == ord('r'):
                template_gray = None
                template_pos = None
                dx, dy, quality = 0, 0, 0
                print("Template cleared")
    
    except KeyboardInterrupt:
        pass
    
    finally:
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        gyro.stop()
        print("Done!")


if __name__ == "__main__":
    main()

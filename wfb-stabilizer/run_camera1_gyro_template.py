#!/usr/bin/python3
# Updated: January 17, 2026
"""
Hybrid Video Stabilization: Gyro Roll/Pitch + Template Matching XY

Combines:
1. Orange Cube gyro for roll & pitch compensation (coarse/fast)
2. Template matching on the gyro-stabilized frame for fine XY correction

Controls:
  B - Reset gyro baseline
  R - Reselect template
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

# Gyro stabilization settings
ZOOM_FACTOR = 0.9  # Zoom to allow room for XY correction
VFOV_DEGREES = 34.0  # Vertical field of view in degrees
PITCH_SCALE = CAMERA_HEIGHT / VFOV_DEGREES

# Template matching settings
SEARCH_MARGIN = 40  # Pixels around template to search (smaller = faster)
MIN_MATCH_QUALITY = 0.5  # Minimum correlation to accept match

# Delay compensation
CAMERA_DELAY_MS = 0
#####################################################################


class GyroReader:
    """Background thread to read gyro data from Orange Cube."""
    
    def __init__(self, port, baudrate, delay_ms=0):
        self.port = port
        self.baudrate = baudrate
        self.delay_ms = delay_ms
        self.cube = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.msg_count = 0
        self.connected = False
        self.buffer = deque(maxlen=500)
    
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
                            self.buffer.append((now_ms, self.roll, self.pitch, self.yaw))
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
            self.thread.join(timeout=1.0)
        if self.cube:
            self.cube.close()


class TemplateTracker:
    """Template matching for fine XY stabilization - optimized with grayscale."""
    
    def __init__(self, search_margin=50, min_quality=0.5):
        self.template = None
        self.template_gray = None
        self.template_pos = None  # (x, y) of template center
        self.search_margin = search_margin
        self.min_quality = min_quality
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.quality = 0.0
        self.template_size = (0, 0)
    
    def set_template(self, frame, rect):
        """Set template from rectangle (x, y, w, h)."""
        x, y, w, h = rect
        self.template = frame[y:y+h, x:x+w].copy()
        # Convert to grayscale for faster matching
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.template_pos = (x + w//2, y + h//2)
        self.template_size = (w, h)
        self.last_dx = 0.0
        self.last_dy = 0.0
        print(f"Template set: {w}x{h} at ({x}, {y})")
    
    def track(self, frame):
        """
        Track template in frame using grayscale matching.
        Returns (dx, dy, quality) where dx/dy are pixels to shift.
        """
        if self.template_gray is None:
            return 0.0, 0.0, 0.0
        
        w, h = self.template_size
        fh, fw = frame.shape[:2]
        
        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Expected position (where template should be)
        exp_x, exp_y = self.template_pos
        
        # Search region around expected position
        margin = self.search_margin
        x1 = max(0, exp_x - w//2 - margin)
        y1 = max(0, exp_y - h//2 - margin)
        x2 = min(fw, exp_x + w//2 + margin)
        y2 = min(fh, exp_y + h//2 + margin)
        
        # Ensure search region is large enough
        if x2 - x1 < w or y2 - y1 < h:
            return self.last_dx, self.last_dy, 0.0
        
        search_region = frame_gray[y1:y2, x1:x2]
        
        # Template matching (grayscale is much faster)
        result = cv2.matchTemplate(search_region, self.template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        self.quality = max_val
        
        if max_val < self.min_quality:
            return self.last_dx, self.last_dy, max_val
        
        # Found position in search region
        found_x = x1 + max_loc[0] + w//2
        found_y = y1 + max_loc[1] + h//2
        
        # Calculate offset from expected
        dx = exp_x - found_x
        dy = exp_y - found_y
        
        self.last_dx = dx
        self.last_dy = dy
        
        return dx, dy, max_val
    
    def get_template_rect(self):
        """Get current template rectangle for visualization."""
        if self.template_gray is None:
            return None
        w, h = self.template_size
        x = self.template_pos[0] - w//2
        y = self.template_pos[1] - h//2
        return (x, y, w, h)


def select_template(frame, window_name):
    """Let user select template with mouse."""
    rect = [0, 0, 0, 0]
    drawing = False
    done = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal rect, drawing, done
        
        if event == cv2.EVENT_LBUTTONDOWN:
            rect[0], rect[1] = x, y
            rect[2], rect[3] = x, y
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect[2], rect[3] = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            rect[2], rect[3] = x, y
            drawing = False
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n" + "=" * 50)
    print("TEMPLATE SELECTION")
    print("=" * 50)
    print("Draw a rectangle around the feature to track")
    print("Press ENTER to confirm, ESC to cancel")
    print("=" * 50)
    
    while True:
        display = frame.copy()
        
        if rect[2] != rect[0] and rect[3] != rect[1]:
            x1, y1 = min(rect[0], rect[2]), min(rect[1], rect[3])
            x2, y2 = max(rect[0], rect[2]), max(rect[1], rect[3])
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter
            if rect[2] != rect[0] and rect[3] != rect[1]:
                x1, y1 = min(rect[0], rect[2]), min(rect[1], rect[3])
                x2, y2 = max(rect[0], rect[2]), max(rect[1], rect[3])
                cv2.setMouseCallback(window_name, lambda *args: None)
                return (x1, y1, x2-x1, y2-y1)
        elif key == 27:  # ESC
            cv2.setMouseCallback(window_name, lambda *args: None)
            return None
    
    return None


def main():
    print("=" * 60)
    print("Hybrid Stabilization: Gyro Roll/Pitch + Template XY")
    print("=" * 60)
    
    # Connect to Orange Cube
    print("\nConnecting to Orange Cube...")
    gyro = GyroReader(MAVLINK_PORT, MAVLINK_BAUDRATE, delay_ms=CAMERA_DELAY_MS)
    
    if not gyro.connect():
        print("ERROR: Could not connect to Orange Cube!")
        return
    
    print("✓ Connected to Orange Cube")
    gyro.start()
    
    # Wait for first attitude
    print("Waiting for attitude data...")
    timeout = time.time() + 5
    base_roll = None
    base_pitch = None
    while time.time() < timeout:
        if gyro.get_msg_count() > 0:
            base_roll, base_pitch, _ = gyro.get_attitude()
            print(f"✓ Base roll: {np.degrees(base_roll):.1f}°, pitch: {np.degrees(base_pitch):.1f}°")
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
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        gyro.stop()
        return
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: {actual_w}x{actual_h} @ {actual_fps} FPS")
    
    center = (actual_w // 2, actual_h // 2)
    
    # Create window
    cv2.namedWindow("Hybrid Stabilized", cv2.WINDOW_NORMAL)
    
    # Template tracker
    tracker = TemplateTracker(search_margin=SEARCH_MARGIN, min_quality=MIN_MATCH_QUALITY)
    
    # Get first frame and select template
    ret, frame = cap.read()
    if ret:
        # Apply initial gyro stabilization
        current_roll, current_pitch, _ = gyro.get_attitude()
        roll_deg = np.degrees(-(current_roll - base_roll))
        pitch_deg = np.degrees(-(current_pitch - base_pitch))
        pitch_pixels = pitch_deg * PITCH_SCALE
        
        M = cv2.getRotationMatrix2D(center, roll_deg, ZOOM_FACTOR)
        M[1, 2] += pitch_pixels
        gyro_stabilized = cv2.warpAffine(frame, M, (actual_w, actual_h))
        
        rect = select_template(gyro_stabilized, "Hybrid Stabilized")
        if rect:
            tracker.set_template(gyro_stabilized, rect)
        else:
            print("No template selected, running without template matching")
    
    print("\n" + "=" * 50)
    print("HYBRID STABILIZATION ACTIVE")
    print("=" * 50)
    print("Press 'B' to reset gyro baseline")
    print("Press 'R' to reselect template")
    print("Press 'Q' or ESC to quit")
    print("=" * 50)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            frame_count += 1
            
            # === STAGE 1: Gyro-based roll/pitch stabilization ===
            current_roll, current_pitch, _ = gyro.get_attitude()
            
            roll_correction = -(current_roll - base_roll)
            pitch_correction = -(current_pitch - base_pitch)
            
            roll_deg = np.degrees(roll_correction)
            pitch_deg = np.degrees(pitch_correction)
            pitch_pixels = pitch_deg * PITCH_SCALE
            
            M_gyro = cv2.getRotationMatrix2D(center, roll_deg, ZOOM_FACTOR)
            M_gyro[1, 2] += pitch_pixels
            
            # Apply gyro stabilization
            gyro_stabilized = cv2.warpAffine(frame, M_gyro, (actual_w, actual_h),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=(0, 0, 0))
            
            # === STAGE 2: Template matching for fine XY correction ===
            dx, dy, quality = 0, 0, 0
            if tracker.template_gray is not None:
                dx, dy, quality = tracker.track(gyro_stabilized)
                
                # Apply XY shift directly on the gyro-stabilized image
                if abs(dx) > 1 or abs(dy) > 1:
                    # Use numpy roll for fast integer shift (no interpolation)
                    stabilized = np.roll(gyro_stabilized, int(dx), axis=1)
                    stabilized = np.roll(stabilized, int(dy), axis=0)
                else:
                    stabilized = gyro_stabilized
            else:
                stabilized = gyro_stabilized
            
            # === Visualization ===
            # Draw template rectangle
            template_rect = tracker.get_template_rect()
            if template_rect:
                x, y, w, h = template_rect
                color = (0, 255, 0) if quality > 0.7 else (0, 255, 255) if quality > 0.5 else (0, 0, 255)
                cv2.rectangle(stabilized, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            
            # Status overlay
            cv2.rectangle(stabilized, (5, 5), (400, 135), (0, 0, 0), -1)
            cv2.rectangle(stabilized, (5, 5), (400, 135), (255, 255, 255), 1)
            
            cv2.putText(stabilized, f"Gyro Roll: {roll_deg:+.1f} deg",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(stabilized, f"Gyro Pitch: {pitch_deg:+.1f} deg ({pitch_pixels:+.0f}px)",
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(stabilized, f"Template XY: dx={dx:+.1f} dy={dy:+.1f} (match: {quality:.2f})",
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Quality bar
            bar_w = int(quality * 200)
            bar_color = (0, 255, 0) if quality > 0.7 else (0, 255, 255) if quality > 0.5 else (0, 0, 255)
            cv2.rectangle(stabilized, (10, 75), (10 + bar_w, 85), bar_color, -1)
            cv2.rectangle(stabilized, (10, 75), (210, 85), (255, 255, 255), 1)
            
            # FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            imu_rate = gyro.get_msg_count() / elapsed if elapsed > 0 else 0
            
            cv2.putText(stabilized, f"FPS: {fps:.1f}",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(stabilized, f"IMU: {imu_rate:.0f}Hz",
                       (100, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(stabilized, f"[B] Gyro reset  [R] Reselect template",
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            
            cv2.imshow("Hybrid Stabilized", stabilized)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('b'):
                base_roll, base_pitch, _ = gyro.get_attitude()
                print(f"Gyro baseline reset: roll={np.degrees(base_roll):.1f}°, pitch={np.degrees(base_pitch):.1f}°")
            elif key == ord('r'):
                # Reselect template
                ret2, frame2 = cap.read()
                if ret2:
                    current_roll, current_pitch, _ = gyro.get_attitude()
                    roll_deg = np.degrees(-(current_roll - base_roll))
                    pitch_deg = np.degrees(-(current_pitch - base_pitch))
                    pitch_pixels = pitch_deg * PITCH_SCALE
                    
                    M = cv2.getRotationMatrix2D(center, roll_deg, ZOOM_FACTOR)
                    M[1, 2] += pitch_pixels
                    gyro_stab = cv2.warpAffine(frame2, M, (actual_w, actual_h))
                    
                    rect = select_template(gyro_stab, "Hybrid Stabilized")
                    if rect:
                        tracker.set_template(gyro_stab, rect)
    
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

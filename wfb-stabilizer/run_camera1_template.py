#!/usr/bin/python3
# Updated: January 17, 2026
"""
Template Matching Video Stabilization

User selects a region of interest (template) with mouse.
The stabilizer tracks that template across frames and compensates for movement.

Usage:
1. Run the script
2. Draw a rectangle around a feature you want to track (e.g., a star, corner, etc.)
3. Press ENTER to confirm selection
4. Stabilization begins automatically

Press 'R' to reselect template, 'Q' to quit
"""

import cv2
import numpy as np
import time
import threading
import sys
import os

# Add parent directory for mavlink module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    from mavlink.orange_cube_reader import OrangeCubeReader, AttitudeData
    GYRO_AVAILABLE = True
except ImportError:
    print("Warning: Orange Cube support not available")
    GYRO_AVAILABLE = False

#################### USER VARS ######################################
CAMERA_INDEX = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60

# Stabilization settings
zoomFactor = 1.0            # No zoom - preserve full quality
smoothingFactor = 0.0       # No smoothing - direct response
searchMargin = 50           # How many pixels around template to search
minMatchQuality = 0.5       # Minimum correlation to accept match

# Gyro settings (for roll compensation)
USE_GYRO_ROLL = True        # Use Orange Cube for roll compensation
MAVLINK_PORT = "COM6"       # Orange Cube serial port
MAVLINK_BAUDRATE = 115200
IMU_RATE_HZ = 100

# Display settings
showFullScreen = 0
showOverlay = 1
showSearchArea = 1          # Show the search region
delay_time = 1

######################################################################


class GyroRollCompensator:
    """Reads roll from Orange Cube for rotation compensation."""
    
    def __init__(self):
        self.cube_reader = None
        self.is_connected = False
        self.base_roll = 0.0
        self.base_set = False
        self.current_roll = 0.0
        self.running = False
        self.reader_thread = None
        self.lock = threading.Lock()
        self.imu_rate = 0.0
        self.imu_count = 0
        self.last_imu_time = time.time()
        
    def connect(self) -> bool:
        if not GYRO_AVAILABLE:
            return False
        print("Connecting to Orange Cube for roll compensation...")
        self.cube_reader = OrangeCubeReader(port=MAVLINK_PORT, baudrate=MAVLINK_BAUDRATE)
        if self.cube_reader.connect():
            self.cube_reader.request_data_streams(rate_hz=IMU_RATE_HZ)
            self.is_connected = True
            return True
        return False
    
    def start(self):
        self.running = True
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()
        
    def stop(self):
        self.running = False
        if self.reader_thread:
            self.reader_thread.join(timeout=1)
            
    def _reader_loop(self):
        while self.running and self.cube_reader:
            msg = self.cube_reader.read_message(timeout=0.1)
            if msg:
                msg_type = self.cube_reader.process_message(msg)
                if msg_type == 'ATTITUDE':
                    with self.lock:
                        self.current_roll = self.cube_reader.attitude_data.roll
                        if not self.base_set:
                            self.base_roll = self.current_roll
                            self.base_set = True
                            print(f"Base roll set: {np.degrees(self.base_roll):.1f}°")
                    self.imu_count += 1
                    now = time.time()
                    if now - self.last_imu_time >= 1.0:
                        self.imu_rate = self.imu_count / (now - self.last_imu_time)
                        self.imu_count = 0
                        self.last_imu_time = now
    
    def reset_base(self):
        with self.lock:
            self.base_set = False
        print("Roll base reset")
    
    def get_roll_correction(self) -> float:
        """Get roll correction in radians."""
        if not self.base_set:
            return 0.0
        with self.lock:
            return -(self.current_roll - self.base_roll)
    
    def get_status(self) -> dict:
        with self.lock:
            roll_corr = -(self.current_roll - self.base_roll) if self.base_set else 0.0
            return {
                'connected': self.is_connected,
                'imu_rate': self.imu_rate,
                'roll_deg': np.degrees(self.current_roll),
                'roll_correction_deg': np.degrees(roll_corr),
            }


class TemplateStabilizer:
    """Template matching based video stabilizer."""
    
    def __init__(self):
        self.template = None
        self.template_pos = None      # (x, y) center of template in original frame
        self.template_size = None     # (w, h) size of template
        self.last_match_pos = None    # Last matched position
        
        # Smoothed correction
        self.smooth_dx = 0.0
        self.smooth_dy = 0.0
        
        # Stats
        self.match_quality = 0.0
        self.is_tracking = False
        
    def set_template(self, frame, rect):
        """
        Set the template from a frame region.
        
        Args:
            frame: Source frame (grayscale or color)
            rect: (x, y, w, h) rectangle defining template region
        """
        x, y, w, h = rect
        
        # Ensure valid region
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w < 10 or h < 10:
            print("Template too small!")
            return False
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Extract template
        self.template = gray[y:y+h, x:x+w].copy()
        self.template_size = (w, h)
        self.template_pos = (x + w//2, y + h//2)  # Center position
        self.last_match_pos = self.template_pos
        self.is_tracking = True
        
        # Reset smoothing
        self.smooth_dx = 0.0
        self.smooth_dy = 0.0
        
        print(f"Template set: {w}x{h} at ({x}, {y})")
        return True
    
    def track(self, frame) -> tuple:
        """
        Track template in frame and return correction.
        
        Args:
            frame: Current frame
            
        Returns:
            (dx, dy, quality) - correction and match quality
        """
        if self.template is None or not self.is_tracking:
            return 0, 0, 0
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Define search region around last known position
        tw, th = self.template_size
        cx, cy = self.last_match_pos
        
        # Search region bounds
        margin = searchMargin
        x1 = max(0, cx - tw//2 - margin)
        y1 = max(0, cy - th//2 - margin)
        x2 = min(gray.shape[1], cx + tw//2 + margin)
        y2 = min(gray.shape[0], cy + th//2 + margin)
        
        # Extract search region
        search_region = gray[y1:y2, x1:x2]
        
        # Check if search region is large enough
        if search_region.shape[0] < th or search_region.shape[1] < tw:
            self.match_quality = 0
            return 0, 0, 0
        
        # Template matching
        result = cv2.matchTemplate(search_region, self.template, cv2.TM_CCOEFF_NORMED)
        
        # Find best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        self.match_quality = max_val
        
        if max_val < minMatchQuality:
            # Poor match - don't update
            return self.smooth_dx, self.smooth_dy, max_val
        
        # Calculate new center position in frame coordinates
        match_x = x1 + max_loc[0] + tw//2
        match_y = y1 + max_loc[1] + th//2
        
        # Update last match position
        self.last_match_pos = (match_x, match_y)
        
        # Calculate offset from original position
        orig_x, orig_y = self.template_pos
        raw_dx = orig_x - match_x
        raw_dy = orig_y - match_y
        
        # Apply smoothing only if factor > 0
        if smoothingFactor > 0:
            self.smooth_dx = smoothingFactor * self.smooth_dx + (1 - smoothingFactor) * raw_dx
            self.smooth_dy = smoothingFactor * self.smooth_dy + (1 - smoothingFactor) * raw_dy
        else:
            # No smoothing - direct pixel-accurate tracking
            self.smooth_dx = raw_dx
            self.smooth_dy = raw_dy
        
        return self.smooth_dx, self.smooth_dy, max_val
    
    def get_search_rect(self) -> tuple:
        """Get the current search region rectangle."""
        if self.last_match_pos is None or self.template_size is None:
            return None
        
        tw, th = self.template_size
        cx, cy = self.last_match_pos
        margin = searchMargin
        
        x1 = int(cx - tw//2 - margin)
        y1 = int(cy - th//2 - margin)
        x2 = int(cx + tw//2 + margin)
        y2 = int(cy + th//2 + margin)
        
        return (x1, y1, x2, y2)
    
    def get_template_rect(self) -> tuple:
        """Get the current matched template rectangle."""
        if self.last_match_pos is None or self.template_size is None:
            return None
        
        tw, th = self.template_size
        cx, cy = self.last_match_pos
        
        x1 = int(cx - tw//2)
        y1 = int(cy - th//2)
        x2 = int(cx + tw//2)
        y2 = int(cy + th//2)
        
        return (x1, y1, x2, y2)
    
    def reset(self):
        """Reset tracking."""
        self.template = None
        self.template_pos = None
        self.template_size = None
        self.last_match_pos = None
        self.smooth_dx = 0.0
        self.smooth_dy = 0.0
        self.match_quality = 0.0
        self.is_tracking = False


# Global variables for mouse callback
drawing = False
ix, iy = -1, -1
rect_start = None
rect_end = None
selection_done = False


def mouse_callback(event, x, y, flags, param):
    """Mouse callback for rectangle selection."""
    global drawing, ix, iy, rect_start, rect_end, selection_done
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rect_start = (x, y)
        rect_end = None
        selection_done = False
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rect_end = (x, y)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect_end = (x, y)
        if rect_start and rect_end:
            # Ensure valid rectangle
            x1, y1 = rect_start
            x2, y2 = rect_end
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                selection_done = True


def select_template(video, window_name) -> tuple:
    """
    Let user select a template region.
    
    Returns:
        (frame, rect) or (None, None) if cancelled
    """
    global drawing, rect_start, rect_end, selection_done
    
    print("\n" + "=" * 50)
    print("TEMPLATE SELECTION")
    print("=" * 50)
    print("Draw a rectangle around the feature to track")
    print("Press ENTER to confirm, ESC to cancel")
    print("=" * 50)
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Reset state
    drawing = False
    rect_start = None
    rect_end = None
    selection_done = False
    
    selected_frame = None
    
    while True:
        ret, frame = video.read()
        if not ret:
            continue
        
        display = frame.copy()
        
        # Draw selection rectangle
        if rect_start and rect_end:
            cv2.rectangle(display, rect_start, rect_end, (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(display, "Draw rectangle around feature to track", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, "ENTER=Confirm, ESC=Cancel, R=Redraw", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if selection_done:
            cv2.putText(display, "Selection ready - Press ENTER to confirm", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == 27:  # ESC
            print("Selection cancelled")
            return None, None
            
        elif key == 13 and selection_done:  # ENTER
            # Calculate rectangle
            x1 = min(rect_start[0], rect_end[0])
            y1 = min(rect_start[1], rect_end[1])
            x2 = max(rect_start[0], rect_end[0])
            y2 = max(rect_start[1], rect_end[1])
            
            rect = (x1, y1, x2 - x1, y2 - y1)
            print(f"Template selected: {rect}")
            
            # Remove mouse callback
            cv2.setMouseCallback(window_name, lambda *args: None)
            
            return frame, rect
            
        elif key == ord('r'):
            # Reset selection
            rect_start = None
            rect_end = None
            selection_done = False


def main():
    global rect_start, rect_end, selection_done
    
    print("=" * 60)
    print("Template Matching Video Stabilization + Gyro Roll")
    print("=" * 60)
    
    # Initialize gyro for roll compensation
    gyro = None
    use_gyro = False
    if USE_GYRO_ROLL and GYRO_AVAILABLE:
        gyro = GyroRollCompensator()
        if gyro.connect():
            gyro.start()
            use_gyro = True
            print("Gyro roll compensation enabled!")
            time.sleep(0.5)
        else:
            print("Gyro not available, running without roll compensation")
    else:
        print("Gyro disabled or not available")
    
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
    
    window_name = "Template Stabilizer - Camera 1"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Initialize stabilizer
    stabilizer = TemplateStabilizer()
    
    # Select template
    frame, rect = select_template(video, window_name)
    if frame is None:
        print("No template selected, exiting")
        video.release()
        cv2.destroyAllWindows()
        return
    
    # Set template
    if not stabilizer.set_template(frame, rect):
        print("Failed to set template")
        video.release()
        cv2.destroyAllWindows()
        return
    
    print("\n" + "=" * 50)
    print("STABILIZATION ACTIVE")
    print("=" * 50)
    print("Press 'R' to reselect template")
    print("Press 'Q' to quit")
    print("=" * 50)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # Track template and get correction
            dx, dy, quality = stabilizer.track(frame)
            
            # Get roll correction from gyro
            roll_correction = 0.0
            if use_gyro:
                roll_correction = gyro.get_roll_correction()
            
            # Build combined transformation matrix (translation + rotation)
            center = (actual_w / 2, actual_h / 2)
            
            # Create rotation matrix around center with roll correction
            roll_deg = np.degrees(roll_correction)
            R = cv2.getRotationMatrix2D(center, roll_deg, zoomFactor)
            
            # Add translation to rotation matrix
            R[0, 2] += dx
            R[1, 2] += dy
            
            # Apply single combined transform (no double warp = no blur)
            stabilized = cv2.warpAffine(frame, R, (actual_w, actual_h), 
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
            
            # Draw overlay
            if showOverlay:
                # Show search area (on original position, adjusted by current correction)
                if showSearchArea:
                    search_rect = stabilizer.get_search_rect()
                    if search_rect:
                        x1, y1, x2, y2 = search_rect
                        # Adjust for stabilization
                        x1 = int(x1 + dx)
                        y1 = int(y1 + dy)
                        x2 = int(x2 + dx)
                        y2 = int(y2 + dy)
                        cv2.rectangle(stabilized, (x1, y1), (x2, y2), (255, 255, 0), 1)
                    
                    # Show tracked template position
                    template_rect = stabilizer.get_template_rect()
                    if template_rect:
                        x1, y1, x2, y2 = template_rect
                        x1 = int(x1 + dx)
                        y1 = int(y1 + dy)
                        x2 = int(x2 + dx)
                        y2 = int(y2 + dy)
                        color = (0, 255, 0) if quality > 0.7 else (0, 255, 255) if quality > 0.5 else (0, 0, 255)
                        cv2.rectangle(stabilized, (x1, y1), (x2, y2), color, 2)
                
                # Status overlay
                overlay_height = 115 if use_gyro else 95
                cv2.rectangle(stabilized, (5, 5), (320, overlay_height), (0, 0, 0), -1)
                cv2.rectangle(stabilized, (5, 5), (320, overlay_height), (255, 255, 255), 1)
                
                # Quality bar
                bar_w = int(quality * 200)
                bar_color = (0, 255, 0) if quality > 0.7 else (0, 255, 255) if quality > 0.5 else (0, 0, 255)
                cv2.rectangle(stabilized, (10, 25), (10 + bar_w, 35), bar_color, -1)
                cv2.rectangle(stabilized, (10, 25), (210, 35), (255, 255, 255), 1)
                cv2.putText(stabilized, f"Match: {quality:.2f}", 
                           (220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                cv2.putText(stabilized, f"XY: dx={dx:.1f} dy={dy:.1f}", 
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Gyro roll info
                if use_gyro:
                    gyro_status = gyro.get_status()
                    roll_deg = np.degrees(roll_correction)
                    cv2.putText(stabilized, f"Roll: {roll_deg:.2f} deg (Gyro: {gyro_status['imu_rate']:.0f}Hz)", 
                               (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset = 95
                else:
                    y_offset = 75
                
                # FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(stabilized, f"FPS: {fps:.1f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(stabilized, "R=Reselect, B=Reset Roll, Q=Quit", 
                           (10, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            # Show frame
            if showFullScreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            cv2.imshow(window_name, stabilized)
            
            # Handle keys
            key = cv2.waitKey(delay_time) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reselect template
                stabilizer.reset()
                frame, rect = select_template(video, window_name)
                if frame is not None:
                    stabilizer.set_template(frame, rect)
                    frame_count = 0
                    start_time = time.time()
                else:
                    print("Selection cancelled, continuing with current template")
            elif key == ord('b') and use_gyro:
                # Reset roll base
                gyro.reset_base()
                    
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("\nCleaning up...")
        if use_gyro:
            gyro.stop()
        video.release()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    main()

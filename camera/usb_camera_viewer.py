#!/usr/bin/env python3
"""
USB Camera Viewer

Reads video from a USB camera and displays it in a window.

Requirements:
    pip install opencv-python

Usage:
    python usb_camera_viewer.py           # Use default camera (0)
    python usb_camera_viewer.py 1         # Use camera index 1
    python usb_camera_viewer.py --list    # List available cameras

Controls:
    q / ESC  - Quit
    s        - Save screenshot
    f        - Toggle fullscreen
    r        - Toggle recording
"""

import cv2
import sys
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path


def list_cameras(max_cameras: int = 10):
    """List all available camera devices."""
    print("\nSearching for cameras...")
    available = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DirectShow for Windows
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                available.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps
                })
                print(f"  Camera {i}: {width}x{height} @ {fps:.1f} FPS")
            cap.release()
    
    if not available:
        print("  No cameras found!")
    else:
        print(f"\nFound {len(available)} camera(s)")
    
    return available


class USBCameraViewer:
    """USB Camera viewer with recording and screenshot capabilities."""
    
    def __init__(self, camera_index: int = 0, width: int = None, height: int = None, 
                 fps: int = None, window_width: int = None, window_height: int = None):
        self.camera_index = camera_index
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.window_width = window_width
        self.window_height = window_height
        
        self.cap = None
        self.is_recording = False
        self.video_writer = None
        self.fullscreen = False
        self.window_name = "USB Camera Viewer"
        
        # Stats
        self.frame_count = 0
        self.start_time = None
        self.fps_actual = 0
        
        # Output directory
        self.output_dir = Path("camera_output")
        self.output_dir.mkdir(exist_ok=True)
    
    def open(self) -> bool:
        """Open the camera."""
        print(f"\nOpening camera {self.camera_index}...")
        
        # Use DirectShow on Windows for better compatibility
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        # Set resolution if specified
        if self.requested_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        if self.requested_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        if self.requested_fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
        
        # Get actual settings
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera opened: {self.width}x{self.height} @ {self.fps:.1f} FPS")
        return True
    
    def save_screenshot(self, frame):
        """Save current frame as screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"screenshot_{timestamp}.png"
        cv2.imwrite(str(filename), frame)
        print(f"Screenshot saved: {filename}")
    
    def start_recording(self, frame):
        """Start video recording."""
        if self.is_recording:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"recording_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(filename), fourcc, self.fps, 
            (frame.shape[1], frame.shape[0])
        )
        
        if self.video_writer.isOpened():
            self.is_recording = True
            print(f"Recording started: {filename}")
        else:
            print("Error: Could not start recording")
    
    def stop_recording(self):
        """Stop video recording."""
        if not self.is_recording:
            return
        
        self.video_writer.release()
        self.video_writer = None
        self.is_recording = False
        print("Recording stopped")
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    def draw_overlay(self, frame):
        """Draw information overlay on frame."""
        h, w = frame.shape[:2]
        
        # FPS display
        fps_text = f"FPS: {self.fps_actual:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # Resolution
        res_text = f"{w}x{h}"
        cv2.putText(frame, res_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        
        # Recording indicator
        if self.is_recording:
            cv2.circle(frame, (w - 30, 30), 15, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 80, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
        
        # Controls help (bottom)
        help_text = "Q:Quit | S:Screenshot | F:Fullscreen | R:Record"
        cv2.putText(frame, help_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        
        return frame
    
    def center_frame(self, frame):
        """Center the frame in a larger canvas if window size is specified."""
        if self.window_width is None or self.window_height is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Create black canvas of window size
        canvas = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        
        # Calculate position to center the frame
        x_offset = (self.window_width - w) // 2
        y_offset = (self.window_height - h) // 2
        
        # Ensure offsets are non-negative
        x_offset = max(0, x_offset)
        y_offset = max(0, y_offset)
        
        # Place frame on canvas
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = frame
        
        return canvas
    
    def run(self):
        """Main loop - capture and display frames."""
        if not self.open():
            return
        
        # Determine display size
        display_width = self.window_width if self.window_width else self.width
        display_height = self.window_height if self.window_height else self.height
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, display_width, display_height)
        
        self.start_time = time.time()
        fps_update_time = self.start_time
        fps_frame_count = 0
        
        print("\nViewer running. Press 'q' or ESC to quit.")
        print("Controls: s=screenshot, f=fullscreen, r=record")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to read frame")
                    break
                
                self.frame_count += 1
                fps_frame_count += 1
                
                # Calculate actual FPS every second
                current_time = time.time()
                if current_time - fps_update_time >= 1.0:
                    self.fps_actual = fps_frame_count / (current_time - fps_update_time)
                    fps_frame_count = 0
                    fps_update_time = current_time
                
                # Record if active
                if self.is_recording and self.video_writer:
                    self.video_writer.write(frame)
                
                # Draw overlay and center in canvas
                display_frame = self.draw_overlay(frame.copy())
                display_frame = self.center_frame(display_frame)
                cv2.imshow(self.window_name, display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # q or ESC
                    break
                elif key == ord('s'):
                    self.save_screenshot(frame)
                elif key == ord('f'):
                    self.toggle_fullscreen()
                elif key == ord('r'):
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording(frame)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.close()
    
    def close(self):
        """Clean up resources."""
        if self.is_recording:
            self.stop_recording()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"\nViewer closed. {self.frame_count} frames in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="USB Camera Viewer - Display video from USB camera"
    )
    parser.add_argument('camera', nargs='?', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--width', '-W', type=int, default=None,
                        help='Requested frame width')
    parser.add_argument('--height', '-H', type=int, default=None,
                        help='Requested frame height')
    parser.add_argument('--fps', type=int, default=None,
                        help='Requested FPS')
    parser.add_argument('--window-width', type=int, default=None,
                        help='Window width (frame will be centered)')
    parser.add_argument('--window-height', type=int, default=None,
                        help='Window height (frame will be centered)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available cameras and exit')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("USB Camera Viewer")
    print("=" * 50)
    
    if args.list:
        list_cameras()
        return
    
    viewer = USBCameraViewer(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        window_width=args.window_width,
        window_height=args.window_height
    )
    viewer.run()


if __name__ == '__main__':
    main()

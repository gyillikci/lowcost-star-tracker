#!/usr/bin/python3
# Updated: January 17, 2026
"""
Stabilize + Stack: Gyro/Template stabilization feeding into light stacker
Combines hybrid stabilization with frame stacking for star intensification.
"""

import cv2
import numpy as np
import time
import sys
import os
import threading
import tkinter as tk
from tkinter import ttk
from collections import deque

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from mavlink.orange_cube_reader import OrangeCubeReader

#################### USER VARS ######################################
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60

MAVLINK_PORT = "COM6"
MAVLINK_BAUDRATE = 115200

ZOOM_FACTOR = 0.9
VFOV_DEGREES = 34.0
PITCH_SCALE = CAMERA_HEIGHT / VFOV_DEGREES
SEARCH_MARGIN = 200
#####################################################################


class GyroReader:
    """Threaded gyro reader from Orange Cube."""
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


class StabilizeStackApp:
    """Combined stabilizer + stacker application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Stabilize + Stack")
        self.root.geometry("400x550")
        
        # Components
        self.gyro = None
        self.cap = None
        self.running = False
        self.thread = None
        
        # Stabilization state
        self.base_roll = 0.0
        self.base_pitch = 0.0
        self.template_gray = None
        self.template_pos = None
        self.selecting = False
        self.sel_start = None
        self.sel_end = None
        
        # Stacking parameters
        self.stack_count = tk.IntVar(value=10)
        self.intensity = tk.DoubleVar(value=2.0)
        self.gamma = tk.DoubleVar(value=1.5)
        self.stack_mode = tk.StringVar(value="max")
        self.stacking_enabled = tk.BooleanVar(value=True)
        
        # Frame buffer for stacking
        self.frame_buffer = deque(maxlen=100)
        self.stacked_frame = None
        self.lock = threading.Lock()
        
        # Stats
        self.frame_count = 0
        self.start_time = 0
        self.dx, self.dy, self.quality = 0.0, 0.0, 0.0
        self.roll_deg, self.pitch_deg = 0.0, 0.0
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(main_frame, text="Stabilize + Stack", font=('Helvetica', 16, 'bold'))
        title.pack(pady=(0, 10))
        
        # Status
        self.status_var = tk.StringVar(value="Not connected")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(pady=(0, 10))
        
        # Separator - Stacking controls
        ttk.Label(main_frame, text="── Stacking Controls ──", font=('Helvetica', 10, 'bold')).pack(pady=5)
        
        # Enable stacking checkbox
        self.stack_check = ttk.Checkbutton(main_frame, text="Enable Stacking", 
                                           variable=self.stacking_enabled)
        self.stack_check.pack(anchor=tk.W, pady=2)
        
        # Stack count
        frame1 = ttk.Frame(main_frame)
        frame1.pack(fill=tk.X, pady=3)
        ttk.Label(frame1, text="Stack Frames:", width=12).pack(side=tk.LEFT)
        self.stack_slider = ttk.Scale(frame1, from_=2, to=100, variable=self.stack_count,
                                      orient=tk.HORIZONTAL, command=self.on_stack_change)
        self.stack_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.stack_label = ttk.Label(frame1, text="10", width=4)
        self.stack_label.pack(side=tk.LEFT)
        
        # Intensity
        frame2 = ttk.Frame(main_frame)
        frame2.pack(fill=tk.X, pady=3)
        ttk.Label(frame2, text="Intensity:", width=12).pack(side=tk.LEFT)
        self.intensity_slider = ttk.Scale(frame2, from_=1.0, to=10.0, variable=self.intensity,
                                          orient=tk.HORIZONTAL, command=self.on_intensity_change)
        self.intensity_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.intensity_label = ttk.Label(frame2, text="2.0", width=4)
        self.intensity_label.pack(side=tk.LEFT)
        
        # Gamma
        frame3 = ttk.Frame(main_frame)
        frame3.pack(fill=tk.X, pady=3)
        ttk.Label(frame3, text="Gamma:", width=12).pack(side=tk.LEFT)
        self.gamma_slider = ttk.Scale(frame3, from_=0.5, to=3.0, variable=self.gamma,
                                      orient=tk.HORIZONTAL, command=self.on_gamma_change)
        self.gamma_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.gamma_label = ttk.Label(frame3, text="1.5", width=4)
        self.gamma_label.pack(side=tk.LEFT)
        
        # Stack mode
        frame4 = ttk.Frame(main_frame)
        frame4.pack(fill=tk.X, pady=3)
        ttk.Label(frame4, text="Stack Mode:", width=12).pack(side=tk.LEFT)
        modes = ["max", "average", "sum", "median"]
        self.mode_combo = ttk.Combobox(frame4, textvariable=self.stack_mode,
                                       values=modes, state="readonly", width=10)
        self.mode_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Buffer info
        self.buffer_var = tk.StringVar(value="Buffer: 0 frames")
        ttk.Label(main_frame, textvariable=self.buffer_var).pack(pady=5)
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Start/Stop buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="Start", command=self.start, width=12)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop, 
                                   width=12, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        btn_frame2 = ttk.Frame(main_frame)
        btn_frame2.pack(pady=5)
        
        self.reset_btn = ttk.Button(btn_frame2, text="Reset (B)", 
                                    command=self.reset_baseline, width=12)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(btn_frame2, text="Clear Buffer",
                                    command=self.clear_buffer, width=12)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Save button
        self.save_btn = ttk.Button(main_frame, text="Save Stacked Image", 
                                   command=self.save_image, width=20)
        self.save_btn.pack(pady=10)
        
        # Info
        info_text = """Controls (in video window):
• Draw rectangle: Set template
• B: Reset gyro + template
• R: Clear template only
• Q: Quit

Stack modes: max (stars), average, sum, median"""
        info = ttk.Label(main_frame, text=info_text, font=('Helvetica', 9), 
                        foreground='gray', justify=tk.LEFT)
        info.pack(pady=(10, 0))
        
    def on_stack_change(self, value):
        val = int(float(value))
        self.stack_label.config(text=str(val))
        
    def on_intensity_change(self, value):
        self.intensity_label.config(text=f"{float(value):.1f}")
        
    def on_gamma_change(self, value):
        self.gamma_label.config(text=f"{float(value):.1f}")
        
    def start(self):
        """Start the stabilizer + stacker."""
        print("=" * 60)
        print("Stabilize + Stack")
        print("=" * 60)
        
        # Connect to Orange Cube
        print("\nConnecting to Orange Cube...")
        self.gyro = GyroReader(MAVLINK_PORT, MAVLINK_BAUDRATE)
        
        if not self.gyro.connect():
            self.status_var.set("ERROR: Could not connect to Orange Cube!")
            return
        
        print("✓ Connected to Orange Cube")
        self.gyro.start()
        
        # Wait for attitude
        timeout = time.time() + 3
        while time.time() < timeout:
            if self.gyro.get_msg_count() > 0:
                self.base_roll, self.base_pitch, _ = self.gyro.get_attitude()
                print(f"✓ Base: roll={np.degrees(self.base_roll):.1f}°, pitch={np.degrees(self.base_pitch):.1f}°")
                break
            time.sleep(0.01)
        else:
            self.status_var.set("ERROR: No attitude data!")
            self.gyro.stop()
            return
        
        # Open camera
        print(f"\nOpening camera {CAMERA_INDEX}...")
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            self.status_var.set("ERROR: Could not open camera!")
            self.gyro.stop()
            return
        
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Camera: {w}x{h}")
        
        self.status_var.set(f"Running: {w}x{h}")
        
        # Setup window
        cv2.namedWindow("Stabilized + Stacked", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Stabilized + Stacked", self.mouse_callback)
        
        self.running = True
        self.frame_count = 0
        self.start_time = time.time()
        
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Start processing loop
        self.process_loop()
        
    def stop(self):
        """Stop the application."""
        self.running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.gyro:
            self.gyro.stop()
            self.gyro = None
        
        cv2.destroyAllWindows()
        
        self.status_var.set("Stopped")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for template selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.sel_start = (x, y)
            self.sel_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.sel_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.sel_end = (x, y)
            
    def reset_baseline(self):
        """Reset gyro baseline and clear template."""
        if self.gyro:
            self.base_roll, self.base_pitch, _ = self.gyro.get_attitude()
        self.template_gray = None
        self.template_pos = None
        self.dx, self.dy, self.quality = 0, 0, 0
        self.clear_buffer()
        print("Baseline and template reset")
        
    def clear_buffer(self):
        """Clear the frame buffer."""
        with self.lock:
            self.frame_buffer.clear()
        self.buffer_var.set("Buffer: 0 frames")
        
    def stack_frames(self, frames):
        """Stack frames based on selected mode."""
        if len(frames) == 0:
            return None
        
        mode = self.stack_mode.get()
        
        if mode == "max":
            stacked = np.max(frames, axis=0)
        elif mode == "average":
            stacked = np.mean(frames, axis=0)
        elif mode == "sum":
            stacked = np.sum(frames, axis=0)
            stacked = np.clip(stacked, 0, 255)
        elif mode == "median":
            stacked = np.median(frames, axis=0)
        else:
            stacked = np.max(frames, axis=0)
        
        return stacked.astype(np.float32)
    
    def apply_intensity(self, image):
        """Apply intensity boost and gamma correction."""
        if image is None:
            return None
        
        intensity = self.intensity.get()
        gamma = self.gamma.get()
        
        enhanced = image * intensity
        enhanced = enhanced / 255.0
        enhanced = np.clip(enhanced, 0, 1)
        enhanced = np.power(enhanced, 1.0 / gamma)
        enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced
    
    def process_loop(self):
        """Main processing loop."""
        if not self.running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(1, self.process_loop)
            return
        
        self.frame_count += 1
        w, h = frame.shape[1], frame.shape[0]
        center = (w // 2, h // 2)
        
        # Get gyro data
        current_roll, current_pitch, _ = self.gyro.get_attitude()
        self.roll_deg = np.degrees(-(current_roll - self.base_roll))
        self.pitch_deg = np.degrees(-(current_pitch - self.base_pitch))
        pitch_px = self.pitch_deg * PITCH_SCALE
        
        # Apply gyro stabilization
        M = cv2.getRotationMatrix2D(center, self.roll_deg, ZOOM_FACTOR)
        M[1, 2] += pitch_px
        stabilized = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)
        
        # Template matching
        if self.template_gray is not None:
            th, tw = self.template_gray.shape
            tx, ty = self.template_pos
            
            margin = SEARCH_MARGIN
            x1 = max(0, tx - tw//2 - margin)
            y1 = max(0, ty - th//2 - margin)
            x2 = min(w, tx + tw//2 + margin)
            y2 = min(h, ty + th//2 + margin)
            
            if x2 - x1 >= tw and y2 - y1 >= th:
                gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
                search = gray[y1:y2, x1:x2]
                
                result = cv2.matchTemplate(search, self.template_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                self.quality = max_val
                
                if max_val > 0.3:
                    found_x = x1 + max_loc[0] + tw//2
                    found_y = y1 + max_loc[1] + th//2
                    self.dx = tx - found_x
                    self.dy = ty - found_y
                    
                    M2 = np.float32([[1, 0, self.dx], [0, 1, self.dy]])
                    stabilized = cv2.warpAffine(stabilized, M2, (w, h))
        
        # Handle template selection
        if self.sel_start and self.sel_end and not self.selecting:
            x1, y1 = min(self.sel_start[0], self.sel_end[0]), min(self.sel_start[1], self.sel_end[1])
            x2, y2 = max(self.sel_start[0], self.sel_end[0]), max(self.sel_start[1], self.sel_end[1])
            
            if x2 > x1 + 10 and y2 > y1 + 10:
                gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
                self.template_gray = gray[y1:y2, x1:x2].copy()
                self.template_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
                print(f"Template set: {x2-x1}x{y2-y1} at {self.template_pos}")
            
            self.sel_start, self.sel_end = None, None
        
        # Add to stack buffer
        with self.lock:
            self.frame_buffer.append(stabilized.astype(np.float32))
            buffer_len = len(self.frame_buffer)
        
        self.buffer_var.set(f"Buffer: {buffer_len} frames")
        
        # Apply stacking if enabled
        if self.stacking_enabled.get() and buffer_len >= 2:
            stack_n = min(self.stack_count.get(), buffer_len)
            with self.lock:
                frames = list(self.frame_buffer)[-stack_n:]
            stacked = self.stack_frames(frames)
            display = self.apply_intensity(stacked)
        else:
            display = stabilized.copy()
        
        self.stacked_frame = display.copy()
        
        # Draw template box
        if self.template_gray is not None:
            th, tw = self.template_gray.shape
            tx, ty = self.template_pos
            color = (0, 255, 0) if self.quality > 0.7 else (0, 255, 255)
            cv2.rectangle(display, (tx - tw//2, ty - th//2), (tx + tw//2, ty + th//2), color, 2)
        
        # Overlay
        cv2.rectangle(display, (5, 5), (380, 90), (0, 0, 0), -1)
        cv2.putText(display, f"Roll: {self.roll_deg:+.1f} Pitch: {self.pitch_deg:+.1f}",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"Template: dx={self.dx:.1f} dy={self.dy:.1f} q={self.quality:.2f}",
                   (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        stack_n = min(self.stack_count.get(), buffer_len) if self.stacking_enabled.get() else 0
        cv2.putText(display, f"Stack: {stack_n} frames ({self.stack_mode.get()}) I={self.intensity.get():.1f} G={self.gamma.get():.1f}",
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(display, f"FPS: {fps:.1f}",
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow("Stabilized + Stacked", display)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            self.stop()
            return
        elif key == ord('b'):
            self.reset_baseline()
        elif key == ord('r'):
            self.template_gray = None
            self.template_pos = None
            self.dx, self.dy, self.quality = 0, 0, 0
            print("Template cleared")
        
        # Continue loop
        self.root.after(1, self.process_loop)
        
    def save_image(self):
        """Save the current stacked image."""
        if self.stacked_frame is None:
            self.status_var.set("No image to save!")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"stabilized_stacked_{timestamp}.png"
        
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        cv2.imwrite(filepath, self.stacked_frame)
        self.status_var.set(f"Saved: {filename}")
        print(f"Saved: {filepath}")
        
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
        
    def on_close(self):
        self.stop()
        self.root.destroy()


def main():
    print("=" * 50)
    print("Stabilize + Stack Application")
    print("=" * 50)
    
    app = StabilizeStackApp()
    app.run()


if __name__ == "__main__":
    main()

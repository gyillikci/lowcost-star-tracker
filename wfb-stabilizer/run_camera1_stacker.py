#!/usr/bin/python3
# Updated: January 17, 2026
"""
Light Stacker GUI
Stacks frames to intensify faint lights (stars, etc.)
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
from PIL import Image, ImageTk

#################### USER VARS ######################################
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
#####################################################################


class LightStackerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Light Stacker")
        self.root.geometry("400x500")
        
        # Camera
        self.cap = None
        self.running = False
        self.thread = None
        
        # Stacking parameters
        self.stack_count = tk.IntVar(value=10)
        self.intensity = tk.DoubleVar(value=2.0)
        self.gamma = tk.DoubleVar(value=1.5)
        self.stack_mode = tk.StringVar(value="max")  # max, average, sum
        self.auto_stack = tk.BooleanVar(value=True)
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=100)
        self.current_frame = None
        self.stacked_frame = None
        self.lock = threading.Lock()
        
        # Display window
        self.display_window = None
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(main_frame, text="Light Stacker", font=('Helvetica', 16, 'bold'))
        title.pack(pady=(0, 15))
        
        # Status
        self.status_var = tk.StringVar(value="Camera: Not connected")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(pady=(0, 10))
        
        # Stack count slider
        frame1 = ttk.Frame(main_frame)
        frame1.pack(fill=tk.X, pady=5)
        ttk.Label(frame1, text="Stack Frames:", width=15).pack(side=tk.LEFT)
        self.stack_slider = ttk.Scale(frame1, from_=2, to=100, variable=self.stack_count, 
                                       orient=tk.HORIZONTAL, command=self.on_stack_change)
        self.stack_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        self.stack_label = ttk.Label(frame1, text="10", width=5)
        self.stack_label.pack(side=tk.LEFT)
        
        # Intensity slider
        frame2 = ttk.Frame(main_frame)
        frame2.pack(fill=tk.X, pady=5)
        ttk.Label(frame2, text="Intensity:", width=15).pack(side=tk.LEFT)
        self.intensity_slider = ttk.Scale(frame2, from_=1.0, to=10.0, variable=self.intensity,
                                          orient=tk.HORIZONTAL, command=self.on_intensity_change)
        self.intensity_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        self.intensity_label = ttk.Label(frame2, text="2.0", width=5)
        self.intensity_label.pack(side=tk.LEFT)
        
        # Gamma slider
        frame3 = ttk.Frame(main_frame)
        frame3.pack(fill=tk.X, pady=5)
        ttk.Label(frame3, text="Gamma:", width=15).pack(side=tk.LEFT)
        self.gamma_slider = ttk.Scale(frame3, from_=0.5, to=3.0, variable=self.gamma,
                                      orient=tk.HORIZONTAL, command=self.on_gamma_change)
        self.gamma_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        self.gamma_label = ttk.Label(frame3, text="1.5", width=5)
        self.gamma_label.pack(side=tk.LEFT)
        
        # Stack mode dropdown
        frame4 = ttk.Frame(main_frame)
        frame4.pack(fill=tk.X, pady=5)
        ttk.Label(frame4, text="Stack Mode:", width=15).pack(side=tk.LEFT)
        modes = ["max", "average", "sum", "median"]
        self.mode_combo = ttk.Combobox(frame4, textvariable=self.stack_mode, 
                                        values=modes, state="readonly", width=12)
        self.mode_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Auto stack checkbox
        frame5 = ttk.Frame(main_frame)
        frame5.pack(fill=tk.X, pady=5)
        self.auto_check = ttk.Checkbutton(frame5, text="Auto-stack (continuous)", 
                                          variable=self.auto_stack)
        self.auto_check.pack(side=tk.LEFT)
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # Buffer info
        self.buffer_var = tk.StringVar(value="Buffer: 0 frames")
        buffer_label = ttk.Label(main_frame, textvariable=self.buffer_var)
        buffer_label.pack(pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=15)
        
        self.start_btn = ttk.Button(btn_frame, text="Start Camera", 
                                    command=self.start_camera, width=15)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="Stop Camera",
                                   command=self.stop_camera, width=15, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        btn_frame2 = ttk.Frame(main_frame)
        btn_frame2.pack(pady=10)
        
        self.clear_btn = ttk.Button(btn_frame2, text="Clear Buffer",
                                    command=self.clear_buffer, width=15)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(btn_frame2, text="Save Image",
                                   command=self.save_image, width=15)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Manual stack button
        self.stack_btn = ttk.Button(main_frame, text="Stack Now", 
                                    command=self.manual_stack, width=20)
        self.stack_btn.pack(pady=10)
        
        # Info
        info = ttk.Label(main_frame, text="Stack Mode:\n• max: brightest pixel wins\n• average: mean of all frames\n• sum: add all (clipped)\n• median: middle value",
                        font=('Helvetica', 9), foreground='gray', justify=tk.LEFT)
        info.pack(pady=(15, 0))
        
    def on_stack_change(self, value):
        val = int(float(value))
        self.stack_label.config(text=str(val))
        self.frame_buffer = deque(maxlen=val + 50)  # Extra buffer room
        
    def on_intensity_change(self, value):
        self.intensity_label.config(text=f"{float(value):.1f}")
        
    def on_gamma_change(self, value):
        self.gamma_label.config(text=f"{float(value):.1f}")
        
    def start_camera(self):
        print(f"Opening camera {CAMERA_INDEX}...")
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            self.status_var.set("ERROR: Could not open camera!")
            return
            
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.status_var.set(f"Camera: {w}x{h} @ {CAMERA_FPS}fps")
        
        self.running = True
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()
        
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Start display update
        self.update_display()
        
    def stop_camera(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.status_var.set("Camera: Stopped")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        cv2.destroyAllWindows()
        
    def capture_loop(self):
        """Background thread to capture frames."""
        while self.running:
            if self.cap is None:
                break
                
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            with self.lock:
                self.current_frame = frame.copy()
                self.frame_buffer.append(frame.astype(np.float32))
                
            time.sleep(0.001)  # Small delay to prevent CPU spinning
            
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
        
        # Apply intensity multiplier
        enhanced = image * intensity
        
        # Normalize to 0-1 range
        enhanced = enhanced / 255.0
        enhanced = np.clip(enhanced, 0, 1)
        
        # Apply gamma correction
        enhanced = np.power(enhanced, 1.0 / gamma)
        
        # Convert back to 0-255
        enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced
        
    def update_display(self):
        """Update the OpenCV display window."""
        if not self.running:
            return
            
        with self.lock:
            buffer_len = len(self.frame_buffer)
            self.buffer_var.set(f"Buffer: {buffer_len} frames")
            
            if self.current_frame is None:
                self.root.after(33, self.update_display)
                return
                
            # Get frames for stacking
            stack_n = min(self.stack_count.get(), buffer_len)
            
            if self.auto_stack.get() and stack_n >= 2:
                # Get last N frames
                frames = list(self.frame_buffer)[-stack_n:]
                stacked = self.stack_frames(frames)
                display = self.apply_intensity(stacked)
            else:
                display = self.current_frame.copy()
                
        if display is not None:
            # Add overlay text
            cv2.rectangle(display, (5, 5), (300, 70), (0, 0, 0), -1)
            cv2.putText(display, f"Stacking: {stack_n} frames ({self.stack_mode.get()})",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, f"Intensity: {self.intensity.get():.1f}  Gamma: {self.gamma.get():.1f}",
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(display, f"Buffer: {buffer_len} frames",
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Store for saving
            self.stacked_frame = display.copy()
            
            cv2.imshow("Light Stacker - Live", display)
            cv2.waitKey(1)
            
        self.root.after(33, self.update_display)  # ~30fps
        
    def manual_stack(self):
        """Manually trigger a stack operation."""
        with self.lock:
            buffer_len = len(self.frame_buffer)
            stack_n = min(self.stack_count.get(), buffer_len)
            
            if stack_n < 2:
                self.status_var.set("Not enough frames to stack!")
                return
                
            frames = list(self.frame_buffer)[-stack_n:]
            
        stacked = self.stack_frames(frames)
        display = self.apply_intensity(stacked)
        
        if display is not None:
            self.stacked_frame = display.copy()
            cv2.imshow("Stacked Result", display)
            self.status_var.set(f"Stacked {stack_n} frames!")
            
    def clear_buffer(self):
        """Clear the frame buffer."""
        with self.lock:
            self.frame_buffer.clear()
        self.buffer_var.set("Buffer: 0 frames")
        self.status_var.set("Buffer cleared!")
        
    def save_image(self):
        """Save the current stacked image."""
        if self.stacked_frame is None:
            self.status_var.set("No stacked image to save!")
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"stacked_{timestamp}.png"
        
        # Save to output folder
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
        self.stop_camera()
        self.root.destroy()


def main():
    print("=" * 50)
    print("Light Stacker GUI")
    print("=" * 50)
    
    app = LightStackerGUI()
    app.run()


if __name__ == "__main__":
    main()

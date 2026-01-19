#!/usr/bin/python3
"""
Dual GUI: Stabilization + Stacking
- Stabilization runs at full speed with OpenCV window
- Stacking GUI receives stabilized frames via shared queue

Created: January 17, 2026
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
from queue import Queue, Empty
import multiprocessing as mp

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

# Shared frame queue between stabilizer and stacker
# Sends tuples: (frame, template_info) where template_info is (template_gray, template_pos) or None
frame_queue = Queue(maxsize=5)


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


def run_stabilizer(frame_queue, stop_event):
    """Stabilization loop - runs at full camera speed."""
    print("=" * 60)
    print("STABILIZER - Real-time gyro + template stabilization")
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
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        gyro.stop()
        return
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera: {w}x{h}")
    
    center = (w // 2, h // 2)
    cv2.namedWindow("Stabilized (Real-time)", cv2.WINDOW_NORMAL)
    
    # Template variables
    template_gray = None
    template_pos = None
    selecting = False
    sel_start = None
    sel_end = None
    
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
    
    cv2.setMouseCallback("Stabilized (Real-time)", mouse_cb)
    
    print("\n" + "=" * 50)
    print("STABILIZER RUNNING")
    print("Draw rectangle = set template")
    print("B = reset gyro+template, R = clear template, Q = quit")
    print("=" * 50)
    
    frame_count = 0
    start_time = time.time()
    dx, dy, quality = 0.0, 0.0, 0.0
    
    try:
        while not stop_event.is_set():
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
            
            # Template matching
            if template_gray is not None:
                th, tw = template_gray.shape
                tx, ty = template_pos
                
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
                    
                    if max_val > 0.4:
                        found_x = x1 + max_loc[0] + tw//2
                        found_y = y1 + max_loc[1] + th//2
                        dx = tx - found_x
                        dy = ty - found_y
                        
                        M2 = np.float32([[1, 0, dx], [0, 1, dy]])
                        stabilized = cv2.warpAffine(stabilized, M2, (w, h))
                    
                    # Draw template box
                    color = (0, 255, 0) if quality > 0.7 else (0, 255, 255)
                    cv2.rectangle(stabilized, (tx - tw//2, ty - th//2),
                                 (tx + tw//2, ty + th//2), color, 2)
            
            # Handle template selection
            if sel_start and sel_end and not selecting:
                x1, y1 = min(sel_start[0], sel_end[0]), min(sel_start[1], sel_end[1])
                x2, y2 = max(sel_start[0], sel_end[0]), max(sel_start[1], sel_end[1])
                
                if x2 > x1 + 10 and y2 > y1 + 10:
                    gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
                    template_gray = gray[y1:y2, x1:x2].copy()
                    template_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
                    print(f"Template set: {x2-x1}x{y2-y1} at {template_pos}")
                
                sel_start, sel_end = None, None
            
            # Send stabilized frame to stacker with template info (non-blocking)
            try:
                template_info = None
                if template_gray is not None:
                    template_info = (template_gray.copy(), template_pos)
                frame_queue.put_nowait((stabilized.copy(), template_info))
            except:
                pass  # Queue full, skip frame
            
            # Display
            display = stabilized.copy()
            cv2.rectangle(display, (5, 5), (350, 80), (0, 0, 0), -1)
            cv2.putText(display, f"Roll: {roll_deg:+.1f} Pitch: {pitch_deg:+.1f}",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, f"Template: dx={dx:.1f} dy={dy:.1f} q={quality:.2f}",
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(display, f"FPS: {fps:.1f} | Sending to stacker...",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow("Stabilized (Real-time)", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                stop_event.set()
                break
            elif key == ord('b'):
                base_roll, base_pitch, _ = gyro.get_attitude()
                template_gray = None
                template_pos = None
                dx, dy, quality = 0, 0, 0
                print("Baseline and template reset")
            elif key == ord('r'):
                template_gray = None
                template_pos = None
                dx, dy, quality = 0, 0, 0
                print("Template cleared")
    
    except KeyboardInterrupt:
        pass
    
    finally:
        print("\nStabilizer shutting down...")
        cap.release()
        cv2.destroyAllWindows()
        gyro.stop()
        stop_event.set()


class StackerGUI:
    """Stacking GUI that receives frames from stabilizer and uses stabilizer's template for alignment."""
    
    def __init__(self, frame_queue, stop_event):
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        
        self.root = tk.Tk()
        self.root.title("Stacker GUI")
        self.root.geometry("400x580")
        
        # Stacking parameters
        self.stack_count = tk.IntVar(value=10)
        self.intensity = tk.DoubleVar(value=2.0)
        self.gamma = tk.DoubleVar(value=1.5)
        self.stack_mode = tk.StringVar(value="max")
        self.stacking_enabled = tk.BooleanVar(value=True)
        
        # Adaptive enhancement parameters
        self.enhance_mode = tk.StringVar(value="none")
        self.clahe_clip = tk.DoubleVar(value=2.0)
        self.clahe_grid = tk.IntVar(value=8)
        self.asinh_stretch = tk.DoubleVar(value=1.0)
        
        # Template from stabilizer (received via queue)
        self.template_gray = None
        self.template_pos = None
        
        # Frame buffer (stores cropped template regions)
        self.frame_buffer = deque(maxlen=200)
        self.stacked_frame = None
        self.lock = threading.Lock()
        self.frames_received = 0
        self.frames_aligned = 0
        self.crop_size = (0, 0)
        self.latest_full_frame = None
        
        self.setup_ui()
        
        # Start receiver thread
        self.receiver_thread = threading.Thread(target=self.receive_frames, daemon=True)
        self.receiver_thread.start()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(main_frame, text="Frame Stacker", font=('Helvetica', 16, 'bold'))
        title.pack(pady=(0, 10))
        
        # Status
        self.status_var = tk.StringVar(value="Waiting for frames...")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(pady=(0, 10))
        
        # Enable stacking
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
        
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=8)
        ttk.Label(main_frame, text="── Adaptive Enhancement ──", font=('Helvetica', 9, 'bold')).pack()
        
        # Enhancement mode
        frame5 = ttk.Frame(main_frame)
        frame5.pack(fill=tk.X, pady=3)
        ttk.Label(frame5, text="Enhance:", width=12).pack(side=tk.LEFT)
        enhance_modes = ["none", "CLAHE", "asinh", "log", "sqrt"]
        self.enhance_combo = ttk.Combobox(frame5, textvariable=self.enhance_mode,
                                          values=enhance_modes, state="readonly", width=10)
        self.enhance_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # CLAHE clip limit
        frame6 = ttk.Frame(main_frame)
        frame6.pack(fill=tk.X, pady=3)
        ttk.Label(frame6, text="CLAHE Clip:", width=12).pack(side=tk.LEFT)
        self.clahe_slider = ttk.Scale(frame6, from_=0.5, to=10.0, variable=self.clahe_clip,
                                      orient=tk.HORIZONTAL, command=self.on_clahe_change)
        self.clahe_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.clahe_label = ttk.Label(frame6, text="2.0", width=4)
        self.clahe_label.pack(side=tk.LEFT)
        
        # CLAHE grid size
        frame7 = ttk.Frame(main_frame)
        frame7.pack(fill=tk.X, pady=3)
        ttk.Label(frame7, text="CLAHE Grid:", width=12).pack(side=tk.LEFT)
        self.grid_slider = ttk.Scale(frame7, from_=2, to=16, variable=self.clahe_grid,
                                     orient=tk.HORIZONTAL, command=self.on_grid_change)
        self.grid_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.grid_label = ttk.Label(frame7, text="8", width=4)
        self.grid_label.pack(side=tk.LEFT)
        
        # Asinh/log stretch factor
        frame8 = ttk.Frame(main_frame)
        frame8.pack(fill=tk.X, pady=3)
        ttk.Label(frame8, text="Stretch:", width=12).pack(side=tk.LEFT)
        self.stretch_slider = ttk.Scale(frame8, from_=0.1, to=5.0, variable=self.asinh_stretch,
                                        orient=tk.HORIZONTAL, command=self.on_stretch_change)
        self.stretch_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.stretch_label = ttk.Label(frame8, text="1.0", width=4)
        self.stretch_label.pack(side=tk.LEFT)
        
        # Buffer info
        self.buffer_var = tk.StringVar(value="Buffer: 0 frames")
        ttk.Label(main_frame, textvariable=self.buffer_var).pack(pady=5)
        
        # Frames received
        self.received_var = tk.StringVar(value="Received: 0")
        ttk.Label(main_frame, textvariable=self.received_var).pack(pady=2)
        
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)
        
        self.clear_btn = ttk.Button(btn_frame, text="Clear Buffer",
                                    command=self.clear_buffer, width=12)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(btn_frame, text="Save Image",
                                   command=self.save_image, width=12)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Info
        info_text = """Template comes from Stabilizer window.
Set template there, stacker will use same region.
Cropped regions are aligned and stacked."""
        info = ttk.Label(main_frame, text=info_text, font=('Helvetica', 9),
                        foreground='gray', justify=tk.LEFT)
        info.pack(pady=(15, 0))
        
        # Alignment info
        self.align_var = tk.StringVar(value="Template: Waiting for stabilizer...")
        ttk.Label(main_frame, textvariable=self.align_var).pack(pady=5)
        
    def on_stack_change(self, value):
        self.stack_label.config(text=str(int(float(value))))
        
    def on_intensity_change(self, value):
        self.intensity_label.config(text=f"{float(value):.1f}")
        
    def on_gamma_change(self, value):
        self.gamma_label.config(text=f"{float(value):.1f}")
        
    def on_clahe_change(self, value):
        self.clahe_label.config(text=f"{float(value):.1f}")
        
    def on_grid_change(self, value):
        self.grid_label.config(text=str(int(float(value))))
        
    def on_stretch_change(self, value):
        self.stretch_label.config(text=f"{float(value):.1f}")
        
    def receive_frames(self):
        """Background thread to receive frames and crop template region from stabilizer."""
        CROP_MARGIN = 50  # Extra margin around template for stacking
        
        while not self.stop_event.is_set():
            try:
                data = self.frame_queue.get(timeout=0.1)
                frame, template_info = data
                self.frames_received += 1
                
                # Store latest full frame for display when no template
                self.latest_full_frame = frame.copy()
                
                # If no template from stabilizer, skip
                if template_info is None:
                    self.template_gray = None
                    self.template_pos = None
                    continue
                
                # Use template info from stabilizer
                self.template_gray, self.template_pos = template_info
                
                # Get template info
                h, w = frame.shape[:2]
                th, tw = self.template_gray.shape
                tx, ty = self.template_pos
                
                # Crop region around template position (already stabilized, so template is at template_pos)
                crop_x1 = max(0, tx - tw//2 - CROP_MARGIN)
                crop_y1 = max(0, ty - th//2 - CROP_MARGIN)
                crop_x2 = min(w, tx + tw//2 + CROP_MARGIN)
                crop_y2 = min(h, ty + th//2 + CROP_MARGIN)
                
                # Extract the template region
                cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                
                # Store crop size
                self.crop_size = (crop_x2 - crop_x1, crop_y2 - crop_y1)
                
                with self.lock:
                    self.frame_buffer.append(cropped.astype(np.float32))
                    self.frames_aligned += 1
                        
            except Empty:
                continue
            except Exception as e:
                print(f"Receive error: {e}")
                continue
                
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
        """Apply intensity boost, gamma correction, and adaptive enhancement."""
        if image is None:
            return None
        
        intensity = self.intensity.get()
        gamma = self.gamma.get()
        
        # Apply intensity and gamma first
        enhanced = image * intensity
        enhanced = enhanced / 255.0
        enhanced = np.clip(enhanced, 0, 1)
        enhanced = np.power(enhanced, 1.0 / gamma)
        enhanced = (enhanced * 255).astype(np.uint8)
        
        # Apply adaptive enhancement
        enhance_mode = self.enhance_mode.get()
        
        if enhance_mode == "CLAHE":
            enhanced = self.apply_clahe(enhanced)
        elif enhance_mode == "asinh":
            enhanced = self.apply_asinh_stretch(enhanced)
        elif enhance_mode == "log":
            enhanced = self.apply_log_stretch(enhanced)
        elif enhance_mode == "sqrt":
            enhanced = self.apply_sqrt_stretch(enhanced)
        
        return enhanced
    
    def apply_clahe(self, image):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Great for enhancing faint stars without over-saturating bright ones."""
        clip_limit = self.clahe_clip.get()
        grid_size = int(self.clahe_grid.get())
        
        # Convert to LAB color space for better results
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            l = clahe.apply(l)
            
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            result = clahe.apply(image)
        
        return result
    
    def apply_asinh_stretch(self, image):
        """Apply asinh (inverse hyperbolic sine) stretch.
        Common in astronomy - compresses bright regions, expands faint regions."""
        stretch = self.asinh_stretch.get()
        
        # Normalize to 0-1
        img_float = image.astype(np.float32) / 255.0
        
        # Apply asinh stretch: asinh(a * x) / asinh(a)
        a = stretch * 10  # Scale factor
        if a > 0:
            stretched = np.arcsinh(a * img_float) / np.arcsinh(a)
        else:
            stretched = img_float
        
        # Back to 0-255
        result = np.clip(stretched * 255, 0, 255).astype(np.uint8)
        return result
    
    def apply_log_stretch(self, image):
        """Apply logarithmic stretch to compress dynamic range."""
        stretch = self.asinh_stretch.get()
        
        img_float = image.astype(np.float32) / 255.0
        
        # log(1 + a*x) / log(1 + a)
        a = stretch * 100
        if a > 0:
            stretched = np.log1p(a * img_float) / np.log1p(a)
        else:
            stretched = img_float
        
        result = np.clip(stretched * 255, 0, 255).astype(np.uint8)
        return result
    
    def apply_sqrt_stretch(self, image):
        """Apply square root stretch - simple but effective for stars."""
        stretch = self.asinh_stretch.get()
        
        img_float = image.astype(np.float32) / 255.0
        
        # Apply power law with adjustable exponent
        exponent = 1.0 / (1.0 + stretch)  # stretch=0 -> 1.0, stretch=5 -> 0.17
        stretched = np.power(img_float, exponent)
        
        result = np.clip(stretched * 255, 0, 255).astype(np.uint8)
        return result
    
    def update_display(self):
        """Update the stacked display."""
        if self.stop_event.is_set():
            self.root.quit()
            return
        
        with self.lock:
            buffer_len = len(self.frame_buffer)
            self.buffer_var.set(f"Buffer: {buffer_len} frames")
            self.received_var.set(f"Received: {self.frames_received} | Stacked: {self.frames_aligned}")
        
        # If no template from stabilizer, show full frame with message
        if self.template_gray is None:
            if self.latest_full_frame is not None:
                display = self.latest_full_frame.copy()
                cv2.rectangle(display, (5, 5), (450, 50), (0, 0, 0), -1)
                cv2.putText(display, "Set template in STABILIZER window first",
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display, f"Received: {self.frames_received} frames",
                           (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.imshow("Stacked Output", display)
                cv2.waitKey(1)
            self.status_var.set("Waiting for template from stabilizer...")
            self.align_var.set("Template: Not set in stabilizer")
            self.root.after(50, self.update_display)
            return
        
        with self.lock:
            if buffer_len < 1:
                self.status_var.set("Waiting for frames...")
                self.root.after(50, self.update_display)
                return
            
            if self.stacking_enabled.get() and buffer_len >= 2:
                stack_n = min(self.stack_count.get(), buffer_len)
                frames = list(self.frame_buffer)[-stack_n:]
                self.status_var.set(f"Stacking {stack_n} frames ({self.stack_mode.get()})")
            else:
                frames = [self.frame_buffer[-1]]
                self.status_var.set("Stacking disabled - showing latest")
        
        # Ensure all frames are same size for stacking
        if len(frames) > 0:
            min_h = min(f.shape[0] for f in frames)
            min_w = min(f.shape[1] for f in frames)
            frames = [f[:min_h, :min_w] for f in frames]
        
        stacked = self.stack_frames(frames)
        display = self.apply_intensity(stacked)
        
        if display is not None:
            self.stacked_frame = display.copy()
            
            # Add overlay
            h, w = display.shape[:2]
            cv2.rectangle(display, (5, 5), (min(w-5, 320), min(h-5, 50)), (0, 0, 0), -1)
            stack_n = len(frames)
            cv2.putText(display, f"Stacked: {stack_n} frames ({self.stack_mode.get()})",
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(display, f"I={self.intensity.get():.1f} G={self.gamma.get():.1f} | Size: {w}x{h}",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            
            cv2.imshow("Stacked Output", display)
            cv2.waitKey(1)
        
        # Update alignment info in GUI
        if self.template_gray is not None:
            th, tw = self.template_gray.shape
            self.align_var.set(f"Template: {tw}x{th} | Crop: {self.crop_size}")
        
        self.root.after(50, self.update_display)
        
    def clear_buffer(self):
        """Clear the frame buffer."""
        with self.lock:
            self.frame_buffer.clear()
            self.frames_aligned = 0
        self.buffer_var.set("Buffer: 0 frames")
        
    def save_image(self):
        """Save the current stacked image."""
        if self.stacked_frame is None:
            self.status_var.set("No image to save!")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"stacked_{timestamp}.png"
        
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        cv2.imwrite(filepath, self.stacked_frame)
        self.status_var.set(f"Saved: {filename}")
        print(f"Saved: {filepath}")
        
    def run(self):
        cv2.namedWindow("Stacked Output", cv2.WINDOW_NORMAL)
        self.root.after(100, self.update_display)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
        
    def on_close(self):
        self.stop_event.set()
        cv2.destroyWindow("Stacked Output")
        self.root.destroy()


def main():
    print("=" * 60)
    print("Dual GUI: Stabilizer + Stacker")
    print("=" * 60)
    
    stop_event = threading.Event()
    
    # Start stabilizer in a separate thread
    stabilizer_thread = threading.Thread(
        target=run_stabilizer,
        args=(frame_queue, stop_event),
        daemon=True
    )
    stabilizer_thread.start()
    
    # Give stabilizer time to start
    time.sleep(1)
    
    # Run stacker GUI in main thread (tkinter requires main thread)
    stacker = StackerGUI(frame_queue, stop_event)
    stacker.run()
    
    # Cleanup
    stop_event.set()
    stabilizer_thread.join(timeout=2)
    print("\nDone!")


if __name__ == "__main__":
    main()

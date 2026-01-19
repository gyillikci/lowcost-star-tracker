#!/usr/bin/env python3
"""
Stellarium Shake Effect Controller

Creates a shaking effect on Stellarium's view with adjustable frequency and amplitude.
Uses the Stellarium Remote Control API.
"""

import tkinter as tk
from tkinter import ttk
import requests
import threading
import time
import math
import json

STELLARIUM_URL = "http://localhost:8090"

class StellariumShaker:
    def __init__(self):
        self.running = False
        self.thread = None
        self.frequency = 2.0  # Hz
        self.amplitude = 1.0  # degrees
        self.base_ra = None
        self.base_dec = None
    
    def set_magnitude_limit(self, mag_value):
        """Set star magnitude limit in Stellarium."""
        try:
            if mag_value == "All":
                # Disable magnitude limit
                requests.post(f"{STELLARIUM_URL}/api/stelproperty/set",
                            data={'id': 'StelSkyDrawer.flagStarMagnitudeLimit', 'value': 'false'},
                            timeout=1)
                print("Magnitude limit disabled - showing all stars")
            else:
                # Enable and set magnitude limit
                mag = float(mag_value)
                requests.post(f"{STELLARIUM_URL}/api/stelproperty/set",
                            data={'id': 'StelSkyDrawer.flagStarMagnitudeLimit', 'value': 'true'},
                            timeout=1)
                requests.post(f"{STELLARIUM_URL}/api/stelproperty/set",
                            data={'id': 'StelSkyDrawer.customStarMagLimit', 'value': str(mag)},
                            timeout=1)
                print(f"Magnitude limit set to {mag}")
            return True
        except Exception as e:
            print(f"Error setting magnitude: {e}")
            return False
    
    def nudge_view(self, delta_ra, delta_dec):
        """Nudge the current view by delta RA/Dec degrees."""
        try:
            ra, dec = self.get_current_view()
            if ra is not None:
                new_ra = ra + delta_ra
                new_dec = max(-90, min(90, dec + delta_dec))
                self.move_view(new_ra, new_dec)
                return new_ra, new_dec
        except Exception as e:
            print(f"Error nudging view: {e}")
        return None, None
    
    def set_labels_visible(self, visible):
        """Show or hide star and planet labels."""
        try:
            value = 'true' if visible else 'false'
            # Star labels
            requests.post(f"{STELLARIUM_URL}/api/stelproperty/set",
                        data={'id': 'StarMgr.flagLabelsDisplayed', 'value': value},
                        timeout=1)
            # Planet labels
            requests.post(f"{STELLARIUM_URL}/api/stelproperty/set",
                        data={'id': 'SolarSystem.flagLabels', 'value': value},
                        timeout=1)
            # Constellation names
            requests.post(f"{STELLARIUM_URL}/api/stelproperty/set",
                        data={'id': 'ConstellationMgr.namesDisplayed', 'value': value},
                        timeout=1)
            print(f"Labels {'shown' if visible else 'hidden'}")
            return True
        except Exception as e:
            print(f"Error setting labels: {e}")
            return False
        
    def get_current_view(self):
        """Get current RA/Dec from Stellarium."""
        try:
            resp = requests.get(f"{STELLARIUM_URL}/api/main/view", timeout=1)
            if resp.status_code == 200:
                data = resp.json()
                # j2000 is a JSON string "[x, y, z]" that needs parsing
                j2000_str = data.get('j2000', '[0, 0, 0]')
                j2000 = json.loads(j2000_str)
                x = float(j2000[0])
                y = float(j2000[1])
                z = float(j2000[2])
                # Convert unit vector to RA/Dec
                # RA = atan2(y, x), Dec = asin(z)
                ra_rad = math.atan2(y, x)
                dec_rad = math.asin(max(-1, min(1, z)))  # Clamp z to [-1,1]
                return math.degrees(ra_rad), math.degrees(dec_rad)
        except Exception as e:
            print(f"Error getting view: {e}")
        return None, None
    
    def move_view(self, ra_deg, dec_deg):
        """Move Stellarium view to specified RA/Dec."""
        try:
            # Convert RA/Dec to unit vector
            ra_rad = math.radians(ra_deg)
            dec_rad = math.radians(dec_deg)
            
            # Unit vector from RA/Dec
            x = math.cos(dec_rad) * math.cos(ra_rad)
            y = math.cos(dec_rad) * math.sin(ra_rad)
            z = math.sin(dec_rad)
            
            # Send as j2000 unit vector (as JSON array string)
            data = {
                'j2000': f'[{x}, {y}, {z}]'
            }
            resp = requests.post(f"{STELLARIUM_URL}/api/main/view", data=data, timeout=0.5)
            return resp.status_code == 200
        except Exception as e:
            print(f"Error moving view: {e}")
            return False
    
    def move_view_alt(self, delta_x, delta_y):
        """Alternative: use mouse move simulation."""
        try:
            # Use stelaction for view movement
            # Send multiple small movements
            if delta_x > 0:
                action = "actionMove_View_Right_Slowly"
            elif delta_x < 0:
                action = "actionMove_View_Left_Slowly"
            else:
                action = None
                
            if action:
                requests.post(f"{STELLARIUM_URL}/api/stelaction/do", 
                            data={'id': action}, timeout=0.2)
            
            if delta_y > 0:
                action = "actionMove_View_Up_Slowly"
            elif delta_y < 0:
                action = "actionMove_View_Down_Slowly"
            else:
                action = None
                
            if action:
                requests.post(f"{STELLARIUM_URL}/api/stelaction/do",
                            data={'id': action}, timeout=0.2)
            return True
        except Exception as e:
            print(f"Error with action: {e}")
            return False
    
    def shake_loop(self):
        """Main shake loop running in background thread."""
        # Get initial position
        self.base_ra, self.base_dec = self.get_current_view()
        if self.base_ra is None:
            print("Could not get initial view position")
            self.running = False
            return
            
        print(f"Base position: RA={self.base_ra:.2f}°, Dec={self.base_dec:.2f}°")
        
        start_time = time.time()
        phase = 0
        
        while self.running:
            t = time.time() - start_time
            freq = self.frequency
            amp = self.amplitude
            
            # Create 2D shake pattern using sin/cos with different frequencies
            offset_ra = amp * math.sin(2 * math.pi * freq * t)
            offset_dec = amp * math.sin(2 * math.pi * freq * t * 1.3 + 0.5)  # Slightly different freq for natural feel
            
            # Calculate new position
            new_ra = self.base_ra + offset_ra
            new_dec = self.base_dec + offset_dec
            
            # Clamp dec to valid range
            new_dec = max(-90, min(90, new_dec))
            
            # Move the view
            self.move_view(new_ra, new_dec)
            
            # Sleep based on frequency (aim for smooth motion)
            time.sleep(0.03)  # ~30 fps update rate
    
    def start(self):
        """Start the shake effect."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.shake_loop, daemon=True)
            self.thread.start()
            print("Shake started!")
    
    def stop(self):
        """Stop the shake effect and return to base position."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        # Return to base position
        if self.base_ra is not None:
            self.move_view(self.base_ra, self.base_dec)
            print(f"Returned to base: RA={self.base_ra:.2f}°, Dec={self.base_dec:.2f}°")
        print("Shake stopped!")


class StellariumShakeGUI:
    def __init__(self):
        self.shaker = StellariumShaker()
        self.root = tk.Tk()
        self.root.title("Stellarium Shake Controller")
        self.root.geometry("450x450")
        self.root.resizable(False, False)
        self.nudge_step = 1.0  # degrees per key press
        
        self.setup_ui()
        self.setup_keybindings()
        
    def setup_keybindings(self):
        """Setup keyboard bindings for RA/Dec control."""
        self.root.bind('<Left>', lambda e: self.nudge_ra(-self.nudge_step))
        self.root.bind('<Right>', lambda e: self.nudge_ra(self.nudge_step))
        self.root.bind('<Up>', lambda e: self.nudge_dec(self.nudge_step))
        self.root.bind('<Down>', lambda e: self.nudge_dec(-self.nudge_step))
        self.root.bind('a', lambda e: self.nudge_ra(-self.nudge_step))
        self.root.bind('d', lambda e: self.nudge_ra(self.nudge_step))
        self.root.bind('w', lambda e: self.nudge_dec(self.nudge_step))
        self.root.bind('s', lambda e: self.nudge_dec(-self.nudge_step))
        self.root.bind('A', lambda e: self.nudge_ra(-self.nudge_step * 5))
        self.root.bind('D', lambda e: self.nudge_ra(self.nudge_step * 5))
        self.root.bind('W', lambda e: self.nudge_dec(self.nudge_step * 5))
        self.root.bind('S', lambda e: self.nudge_dec(-self.nudge_step * 5))
    
    def nudge_ra(self, delta):
        """Nudge RA by delta degrees."""
        if not self.shaker.running:
            ra, dec = self.shaker.nudge_view(delta, 0)
            if ra is not None:
                self.status_var.set(f"RA={ra:.1f}°, Dec={dec:.1f}°")
    
    def nudge_dec(self, delta):
        """Nudge Dec by delta degrees."""
        if not self.shaker.running:
            ra, dec = self.shaker.nudge_view(0, delta)
            if ra is not None:
                self.status_var.set(f"RA={ra:.1f}°, Dec={dec:.1f}°")
        
    def setup_ui(self):
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Stellarium Shake Effect", 
                               font=('Helvetica', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Status label
        self.status_var = tk.StringVar(value="Status: Stopped")
        status_label = ttk.Label(main_frame, textvariable=self.status_var,
                                font=('Helvetica', 10))
        status_label.pack(pady=(0, 15))
        
        # Frequency slider
        freq_frame = ttk.Frame(main_frame)
        freq_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(freq_frame, text="Frequency (Hz):", width=15).pack(side=tk.LEFT)
        self.freq_var = tk.DoubleVar(value=2.0)
        self.freq_slider = ttk.Scale(freq_frame, from_=0.5, to=10.0, 
                                     variable=self.freq_var, orient=tk.HORIZONTAL,
                                     command=self.on_freq_change)
        self.freq_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        self.freq_label = ttk.Label(freq_frame, text="2.0", width=5)
        self.freq_label.pack(side=tk.LEFT)
        
        # Amplitude slider  
        amp_frame = ttk.Frame(main_frame)
        amp_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(amp_frame, text="Amplitude (°):", width=15).pack(side=tk.LEFT)
        self.amp_var = tk.DoubleVar(value=1.0)
        self.amp_slider = ttk.Scale(amp_frame, from_=0.1, to=5.0,
                                    variable=self.amp_var, orient=tk.HORIZONTAL,
                                    command=self.on_amp_change)
        self.amp_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        self.amp_label = ttk.Label(amp_frame, text="1.0", width=5)
        self.amp_label.pack(side=tk.LEFT)
        
        # Magnitude dropdown
        mag_frame = ttk.Frame(main_frame)
        mag_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mag_frame, text="Star Magnitude:", width=15).pack(side=tk.LEFT)
        self.mag_var = tk.StringVar(value="All")
        mag_options = ["All", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0"]
        self.mag_dropdown = ttk.Combobox(mag_frame, textvariable=self.mag_var,
                                         values=mag_options, state="readonly", width=10)
        self.mag_dropdown.pack(side=tk.LEFT, padx=(10, 10))
        self.mag_dropdown.bind("<<ComboboxSelected>>", self.on_mag_change)
        
        # Nudge step slider
        nudge_frame = ttk.Frame(main_frame)
        nudge_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(nudge_frame, text="Nudge Step (°):", width=15).pack(side=tk.LEFT)
        self.nudge_var = tk.DoubleVar(value=1.0)
        self.nudge_slider = ttk.Scale(nudge_frame, from_=0.1, to=10.0,
                                      variable=self.nudge_var, orient=tk.HORIZONTAL,
                                      command=self.on_nudge_change)
        self.nudge_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        self.nudge_label = ttk.Label(nudge_frame, text="1.0", width=5)
        self.nudge_label.pack(side=tk.LEFT)
        
        # Labels toggle checkbox
        labels_frame = ttk.Frame(main_frame)
        labels_frame.pack(fill=tk.X, pady=5)
        
        self.labels_var = tk.BooleanVar(value=True)
        self.labels_check = ttk.Checkbutton(labels_frame, text="Show Star/Planet/Constellation Names",
                                            variable=self.labels_var, command=self.on_labels_toggle)
        self.labels_check.pack(side=tk.LEFT)
        
        # Buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=30)
        
        self.start_btn = ttk.Button(btn_frame, text="Start Shake", 
                                    command=self.start_shake, width=15)
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        self.stop_btn = ttk.Button(btn_frame, text="Stop Shake",
                                   command=self.stop_shake, width=15, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        
        # Test connection button
        test_btn = ttk.Button(main_frame, text="Test Stellarium Connection",
                             command=self.test_connection)
        test_btn.pack(pady=10)
        
        # Info label
        info_label = ttk.Label(main_frame, 
                              text="Keys: Arrow/WASD to nudge view (Shift=5x)\nMake sure Stellarium Remote Control is enabled on port 8090",
                              font=('Helvetica', 9), foreground='gray')
        info_label.pack(pady=(20, 0))
    
    def on_mag_change(self, event=None):
        mag = self.mag_var.get()
        self.shaker.set_magnitude_limit(mag)
        self.status_var.set(f"Magnitude set to: {mag}")
    
    def on_nudge_change(self, value):
        step = float(value)
        self.nudge_label.config(text=f"{step:.1f}")
        self.nudge_step = step
    
    def on_labels_toggle(self):
        visible = self.labels_var.get()
        self.shaker.set_labels_visible(visible)
        self.status_var.set(f"Labels {'shown' if visible else 'hidden'}")
        
    def on_freq_change(self, value):
        freq = float(value)
        self.freq_label.config(text=f"{freq:.1f}")
        self.shaker.frequency = freq
        
    def on_amp_change(self, value):
        amp = float(value)
        self.amp_label.config(text=f"{amp:.1f}")
        self.shaker.amplitude = amp
        
    def start_shake(self):
        self.shaker.start()
        self.status_var.set("Status: Shaking...")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
    def stop_shake(self):
        self.shaker.stop()
        self.status_var.set("Status: Stopped")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
    def test_connection(self):
        ra, dec = self.shaker.get_current_view()
        if ra is not None:
            self.status_var.set(f"Connected! View: RA={ra:.1f}°, Dec={dec:.1f}°")
        else:
            self.status_var.set("Connection failed! Check Stellarium Remote Control")
            
    def run(self):
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
        
    def on_close(self):
        self.shaker.stop()
        self.root.destroy()


def main():
    print("=" * 50)
    print("Stellarium Shake Controller")
    print("=" * 50)
    print("Make sure Stellarium is running with Remote Control")
    print("plugin enabled on port 8090")
    print("=" * 50)
    
    app = StellariumShakeGUI()
    app.run()


if __name__ == "__main__":
    main()

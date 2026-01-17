#!/usr/bin/env python3
"""
Live Tetra3 plate solving loop.

Captures images from the Harrier camera and runs tetra3 plate solving continuously.
Press Ctrl+C to stop.
"""

import sys
import time
import cv2
import numpy as np
import requests
import math
from PIL import Image

# Add external tetra3 to path
sys.path.insert(0, 'external/tetra3')

try:
    from tetra3 import Tetra3
except ImportError as e:
    print(f"Error importing tetra3: {e}")
    print("Trying alternative import...")
    try:
        from tetra3.tetra3 import Tetra3
    except ImportError as e2:
        print(f"Still failed: {e2}")
        sys.exit(1)

STELLARIUM_URL = "http://localhost:8090"

def get_stellarium_view():
    """Get current RA/Dec from Stellarium's view center."""
    try:
        # Get the current view info
        resp = requests.get(f"{STELLARIUM_URL}/api/main/view", timeout=0.5)
        if resp.status_code == 200:
            data = resp.json()
            # j2000 contains [RA, Dec] in radians
            j2000 = data.get('j2000', [0, 0])
            ra_rad = j2000[0]
            dec_rad = j2000[1]
            # Convert to degrees
            ra_deg = math.degrees(ra_rad)
            dec_deg = math.degrees(dec_rad)
            # Normalize RA to 0-360
            if ra_deg < 0:
                ra_deg += 360
            return ra_deg, dec_deg
    except Exception:
        pass
    return None, None

def format_ra_dec(ra_deg, dec_deg):
    """Format RA/Dec for display."""
    if ra_deg is None:
        return "N/A", "N/A"
    # RA to hours
    ra_h = ra_deg / 15.0
    ra_m = (ra_h - int(ra_h)) * 60
    ra_s = (ra_m - int(ra_m)) * 60
    ra_str = f"{int(ra_h):02d}h{int(ra_m):02d}m{ra_s:05.2f}s"
    # Dec to dms
    dec_sign = '+' if dec_deg >= 0 else '-'
    dec_abs = abs(dec_deg)
    dec_d = int(dec_abs)
    dec_m = (dec_abs - dec_d) * 60
    dec_s = (dec_m - int(dec_m)) * 60
    dec_str = f"{dec_sign}{dec_d:02d} {int(dec_m):02d}' {dec_s:04.1f}\""
    return ra_str, dec_str

def main():
    camera_index = 0
    width = 1280
    height = 720
    fov_estimate = 57.0  # degrees
    
    print("=" * 60)
    print("Live Tetra3 Plate Solving")
    print("=" * 60)
    print(f"Camera index: {camera_index}")
    print(f"Resolution: {width}x{height}")
    print(f"FOV estimate: {fov_estimate}°")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Initialize camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_index}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Actual resolution: {actual_w}x{actual_h}")
    
    # Initialize tetra3
    print("\nLoading tetra3 database...")
    try:
        # Try to load default database
        t3 = Tetra3()
        print("Tetra3 initialized with default database")
    except Exception as e:
        print(f"Error initializing tetra3: {e}")
        print("Trying to create database...")
        try:
            t3 = Tetra3(load_database=None)
            print("Tetra3 initialized without database - will need to generate one")
        except Exception as e2:
            print(f"Failed to initialize tetra3: {e2}")
            cap.release()
            return
    
    print("\nStarting live plate solving loop...")
    print("-" * 60)
    
    frame_count = 0
    solve_count = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.5)
                continue
            
            frame_count += 1
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Convert to PIL Image (tetra3 expects PIL Image)
            pil_image = Image.fromarray(gray)
            
            # Create display frame
            display = frame.copy()
            
            # Get Stellarium view
            stell_ra, stell_dec = get_stellarium_view()
            stell_ra_str, stell_dec_str = format_ra_dec(stell_ra, stell_dec)
            
            # Run plate solving
            start_time = time.time()
            try:
                result = t3.solve_from_image(
                    pil_image,
                    fov_estimate=fov_estimate,
                    fov_max_error=10.0,
                    match_radius=0.01,
                    return_matches=True
                )
                solve_time = time.time() - start_time
                
                if result and result.get('RA') is not None:
                    solve_count += 1
                    ra = result.get('RA', 0)
                    dec = result.get('Dec', 0)
                    roll = result.get('Roll', 0)
                    fov = result.get('FOV', fov_estimate)
                    matched = result.get('Matches', 0)
                    
                    # Convert RA to hours
                    ra_h = ra / 15.0
                    ra_m = (ra_h - int(ra_h)) * 60
                    ra_s = (ra_m - int(ra_m)) * 60
                    
                    # Convert Dec to degrees/arcmin/arcsec
                    dec_sign = '+' if dec >= 0 else '-'
                    dec_abs = abs(dec)
                    dec_d = int(dec_abs)
                    dec_m = (dec_abs - dec_d) * 60
                    dec_s = (dec_m - int(dec_m)) * 60
                    
                    # Calculate error vs Stellarium
                    err_str = ""
                    if stell_ra is not None:
                        ra_err = abs(ra - stell_ra)
                        if ra_err > 180:
                            ra_err = 360 - ra_err
                        dec_err = abs(dec - stell_dec)
                        err_str = f" | Err: {ra_err:.2f}°/{dec_err:.2f}°"
                    
                    print(f"[{frame_count:4d}] SOLVED | "
                          f"T3: {int(ra_h):02d}h{int(ra_m):02d}m/{dec_sign}{dec_d:02d}°{int(dec_m):02d}' | "
                          f"Stell: {stell_ra_str}/{stell_dec_str}{err_str}")
                    
                    # Draw solution on display
                    cv2.putText(display, "SOLVED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display, f"RA: {int(ra_h):02d}h{int(ra_m):02d}m{ra_s:05.2f}s  ({ra:.4f} deg)", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display, f"Dec: {dec_sign}{dec_d:02d} {int(dec_m):02d}' {dec_s:04.1f}\"  ({dec:+.4f} deg)", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display, f"Roll: {roll:.1f}  FOV: {fov:.1f}  Stars: {matched}", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # Draw Stellarium info
                    stell_ra_deg = f"({stell_ra:.4f} deg)" if stell_ra is not None else ""
                    stell_dec_deg = f"({stell_dec:+.4f} deg)" if stell_dec is not None else ""
                    cv2.putText(display, "--- Stellarium ---", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(display, f"RA: {stell_ra_str}  {stell_ra_deg}", (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                    cv2.putText(display, f"Dec: {stell_dec_str}  {stell_dec_deg}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                else:
                    print(f"[{frame_count:4d}] No solution | Stell: {stell_ra_str}/{stell_dec_str}")
                    cv2.putText(display, "No solution", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Still show Stellarium info
                    stell_ra_deg = f"({stell_ra:.4f} deg)" if stell_ra is not None else ""
                    stell_dec_deg = f"({stell_dec:+.4f} deg)" if stell_dec is not None else ""
                    cv2.putText(display, "--- Stellarium ---", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(display, f"RA: {stell_ra_str}  {stell_ra_deg}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                    cv2.putText(display, f"Dec: {stell_dec_str}  {stell_dec_deg}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                    
            except Exception as e:
                solve_time = time.time() - start_time
                print(f"[{frame_count:4d}] Solve error: {e} ({solve_time:.2f}s)")
                cv2.putText(display, f"Error: {str(e)[:50]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show the frame
            cv2.imshow("Tetra3 Live Plate Solving", display)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Stopping...")
        print(f"Total frames: {frame_count}")
        print(f"Solved: {solve_count} ({100*solve_count/max(1,frame_count):.1f}%)")
        print("=" * 60)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")

if __name__ == "__main__":
    main()

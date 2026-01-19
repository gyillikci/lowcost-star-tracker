#!/usr/bin/env python3
"""
Display-Aware Camera Calibration Utility

Uses known display specifications for accurate on-screen calibration patterns
and proper checkerboard sizing.

Supports HP ZBook Fury 16 G10 and other workstation displays.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import subprocess
import platform


# =============================================================================
# Display Profiles Database
# =============================================================================

@dataclass
class DisplayProfile:
    """Display specifications."""
    name: str
    diagonal_inches: float
    width_pixels: int
    height_pixels: int
    aspect_ratio: Tuple[int, int]  # e.g., (16, 10)

    @property
    def ppi(self) -> float:
        """Pixels per inch."""
        # Calculate physical dimensions from diagonal and aspect ratio
        w_ratio, h_ratio = self.aspect_ratio
        diag_ratio = np.sqrt(w_ratio**2 + h_ratio**2)
        width_inches = self.diagonal_inches * w_ratio / diag_ratio
        return self.width_pixels / width_inches

    @property
    def ppcm(self) -> float:
        """Pixels per centimeter."""
        return self.ppi / 2.54

    @property
    def cm_per_pixel(self) -> float:
        """Centimeters per pixel."""
        return 1.0 / self.ppcm

    @property
    def mm_per_pixel(self) -> float:
        """Millimeters per pixel."""
        return 10.0 / self.ppcm

    @property
    def width_cm(self) -> float:
        """Physical screen width in cm."""
        return self.width_pixels / self.ppcm

    @property
    def height_cm(self) -> float:
        """Physical screen height in cm."""
        return self.height_pixels / self.ppcm

    def pixels_for_mm(self, mm: float) -> int:
        """Convert millimeters to pixels."""
        return int(mm * self.ppcm / 10.0)

    def pixels_for_cm(self, cm: float) -> int:
        """Convert centimeters to pixels."""
        return int(cm * self.ppcm)

    def mm_for_pixels(self, pixels: int) -> float:
        """Convert pixels to millimeters."""
        return pixels * self.mm_per_pixel

    def print_info(self):
        """Print display specifications."""
        w_ratio, h_ratio = self.aspect_ratio
        diag_ratio = np.sqrt(w_ratio**2 + h_ratio**2)
        width_inches = self.diagonal_inches * w_ratio / diag_ratio
        height_inches = self.diagonal_inches * h_ratio / diag_ratio

        print(f"\n{'='*60}")
        print(f"DISPLAY: {self.name}")
        print(f"{'='*60}")
        print(f"Resolution:     {self.width_pixels} x {self.height_pixels}")
        print(f"Aspect Ratio:   {w_ratio}:{h_ratio}")
        print(f"Diagonal:       {self.diagonal_inches}\"")
        print(f"Physical Size:  {width_inches:.2f}\" x {height_inches:.2f}\"")
        print(f"                ({self.width_cm:.1f} cm x {self.height_cm:.1f} cm)")
        print(f"Pixel Density:  {self.ppi:.1f} PPI ({self.ppcm:.1f} px/cm)")
        print(f"Pixel Size:     {self.mm_per_pixel:.4f} mm ({self.cm_per_pixel:.5f} cm)")
        print(f"{'='*60}")


# Pre-defined display profiles
DISPLAY_PROFILES = {
    # HP ZBook Fury 16 G10 variants
    'zbook_fury_16_g10_fhd': DisplayProfile(
        name='HP ZBook Fury 16 G10 (WUXGA)',
        diagonal_inches=16.0,
        width_pixels=1920,
        height_pixels=1200,
        aspect_ratio=(16, 10)
    ),
    'zbook_fury_16_g10_4k': DisplayProfile(
        name='HP ZBook Fury 16 G10 (WQUXGA/4K+)',
        diagonal_inches=16.0,
        width_pixels=3840,
        height_pixels=2400,
        aspect_ratio=(16, 10)
    ),

    # Other common workstation displays
    'zbook_15_g6_fhd': DisplayProfile(
        name='HP ZBook 15 G6 (FHD)',
        diagonal_inches=15.6,
        width_pixels=1920,
        height_pixels=1080,
        aspect_ratio=(16, 9)
    ),
    'zbook_15_g6_4k': DisplayProfile(
        name='HP ZBook 15 G6 (4K UHD)',
        diagonal_inches=15.6,
        width_pixels=3840,
        height_pixels=2160,
        aspect_ratio=(16, 9)
    ),
    'thinkpad_p16_fhd': DisplayProfile(
        name='Lenovo ThinkPad P16 (WUXGA)',
        diagonal_inches=16.0,
        width_pixels=1920,
        height_pixels=1200,
        aspect_ratio=(16, 10)
    ),
    'thinkpad_p16_4k': DisplayProfile(
        name='Lenovo ThinkPad P16 (WQUXGA)',
        diagonal_inches=16.0,
        width_pixels=3840,
        height_pixels=2400,
        aspect_ratio=(16, 10)
    ),
    'macbook_pro_16': DisplayProfile(
        name='MacBook Pro 16"',
        diagonal_inches=16.2,
        width_pixels=3456,
        height_pixels=2234,
        aspect_ratio=(3456, 2234)  # Exact ratio
    ),
    'dell_precision_17': DisplayProfile(
        name='Dell Precision 7780 (4K)',
        diagonal_inches=17.0,
        width_pixels=3840,
        height_pixels=2400,
        aspect_ratio=(16, 10)
    ),

    # Generic displays
    'generic_1080p_15': DisplayProfile(
        name='Generic 15.6" FHD',
        diagonal_inches=15.6,
        width_pixels=1920,
        height_pixels=1080,
        aspect_ratio=(16, 9)
    ),
    'generic_1080p_24': DisplayProfile(
        name='Generic 24" FHD',
        diagonal_inches=24.0,
        width_pixels=1920,
        height_pixels=1080,
        aspect_ratio=(16, 9)
    ),
    'generic_4k_27': DisplayProfile(
        name='Generic 27" 4K',
        diagonal_inches=27.0,
        width_pixels=3840,
        height_pixels=2160,
        aspect_ratio=(16, 9)
    ),
}


def detect_display_resolution() -> Tuple[int, int]:
    """Detect current display resolution."""
    try:
        if platform.system() == 'Linux':
            result = subprocess.run(['xdpyinfo'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'dimensions:' in line:
                    parts = line.split()[1].split('x')
                    return int(parts[0]), int(parts[1])
        elif platform.system() == 'Darwin':  # macOS
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'],
                                   capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'Resolution:' in line:
                    parts = line.split(':')[1].strip().split(' x ')
                    return int(parts[0]), int(parts[1].split()[0])
    except:
        pass

    # Fallback: try OpenCV
    try:
        import tkinter as tk
        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except:
        pass

    return 1920, 1080  # Default fallback


def guess_display_profile() -> Optional[DisplayProfile]:
    """Try to guess the display profile from resolution."""
    width, height = detect_display_resolution()
    print(f"Detected resolution: {width}x{height}")

    # Find matching profile
    for name, profile in DISPLAY_PROFILES.items():
        if profile.width_pixels == width and profile.height_pixels == height:
            print(f"Matched profile: {profile.name}")
            return profile

    print("No exact profile match found")
    return None


def create_custom_profile(width: int, height: int, diagonal: float,
                          name: str = "Custom Display") -> DisplayProfile:
    """Create a custom display profile."""
    # Determine aspect ratio
    from math import gcd
    g = gcd(width, height)
    aspect = (width // g, height // g)

    return DisplayProfile(
        name=name,
        diagonal_inches=diagonal,
        width_pixels=width,
        height_pixels=height,
        aspect_ratio=aspect
    )


# =============================================================================
# On-Screen Calibration Pattern Generator
# =============================================================================

class OnScreenCalibrationPattern:
    """
    Generate calibration patterns sized correctly for a specific display.

    The pattern can be displayed on-screen at the correct physical size,
    allowing calibration without printing.
    """

    def __init__(self, display: DisplayProfile):
        self.display = display

    def create_checkerboard(self,
                            cols: int = 9,
                            rows: int = 6,
                            square_size_mm: float = 25.0) -> np.ndarray:
        """
        Create a checkerboard pattern at exact physical size.

        Args:
            cols: Number of inner corners (columns)
            rows: Number of inner corners (rows)
            square_size_mm: Physical size of each square in mm

        Returns:
            BGR image of the checkerboard at correct pixel size
        """
        # Convert mm to pixels
        square_size_px = self.display.pixels_for_mm(square_size_mm)

        # Pattern dimensions (cols+1 x rows+1 squares)
        pattern_width = (cols + 1) * square_size_px
        pattern_height = (rows + 1) * square_size_px

        # Create pattern
        pattern = np.ones((pattern_height, pattern_width), dtype=np.uint8) * 255

        # Draw squares
        for i in range(rows + 1):
            for j in range(cols + 1):
                if (i + j) % 2 == 0:
                    x1 = j * square_size_px
                    y1 = i * square_size_px
                    x2 = x1 + square_size_px
                    y2 = y1 + square_size_px
                    pattern[y1:y2, x1:x2] = 0

        # Convert to BGR
        pattern_bgr = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)

        return pattern_bgr

    def create_ruler_overlay(self, pattern: np.ndarray) -> np.ndarray:
        """Add ruler markings to pattern for verification."""
        result = pattern.copy()
        h, w = result.shape[:2]

        # Draw cm markings on top edge
        cm_px = self.display.pixels_for_cm(1.0)
        for i in range(int(w / cm_px) + 1):
            x = i * cm_px
            if x < w:
                # Major tick every cm
                cv2.line(result, (int(x), 0), (int(x), 20), (0, 0, 255), 2)
                cv2.putText(result, f"{i}cm", (int(x) + 2, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                # Minor ticks every 5mm
                for j in range(1, 10):
                    x_mm = x + j * cm_px / 10
                    if x_mm < w:
                        tick_len = 10 if j == 5 else 5
                        cv2.line(result, (int(x_mm), 0), (int(x_mm), tick_len), (0, 0, 255), 1)

        # Draw cm markings on left edge
        for i in range(int(h / cm_px) + 1):
            y = i * cm_px
            if y < h:
                cv2.line(result, (0, int(y)), (20, int(y)), (0, 0, 255), 2)
                cv2.putText(result, f"{i}", (25, int(y) + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return result

    def display_fullscreen_pattern(self,
                                   cols: int = 9,
                                   rows: int = 6,
                                   square_size_mm: float = 25.0,
                                   show_ruler: bool = True):
        """
        Display pattern fullscreen for on-screen calibration.

        Press 'q' to quit, 's' to save pattern, '+'/'-' to adjust size.
        """
        current_size = square_size_mm

        print(f"\n{'='*60}")
        print("ON-SCREEN CALIBRATION PATTERN")
        print(f"{'='*60}")
        print(f"Display: {self.display.name}")
        print(f"Pixel density: {self.display.ppi:.1f} PPI")
        print(f"Square size: {current_size:.1f} mm")
        print(f"\nControls:")
        print("  +/- : Adjust square size")
        print("  R   : Toggle ruler overlay")
        print("  S   : Save pattern to file")
        print("  F   : Toggle fullscreen")
        print("  Q   : Quit")
        print(f"{'='*60}")

        cv2.namedWindow('Calibration Pattern', cv2.WINDOW_NORMAL)

        fullscreen = False
        ruler = show_ruler

        while True:
            # Generate pattern
            pattern = self.create_checkerboard(cols, rows, current_size)

            if ruler:
                pattern = self.create_ruler_overlay(pattern)

            # Add info text
            info_h = 60
            info_bar = np.ones((info_h, pattern.shape[1], 3), dtype=np.uint8) * 40
            cv2.putText(info_bar, f"Square: {current_size:.1f}mm | {self.display.pixels_for_mm(current_size)}px",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(info_bar, f"[+/-] Size  [R] Ruler  [S] Save  [F] Fullscreen  [Q] Quit",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            display = np.vstack([pattern, info_bar])

            cv2.imshow('Calibration Pattern', display)

            if fullscreen:
                cv2.setWindowProperty('Calibration Pattern',
                                     cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty('Calibration Pattern',
                                     cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

            key = cv2.waitKey(100) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('+') or key == ord('='):
                current_size += 1.0
                print(f"Square size: {current_size:.1f} mm")
            elif key == ord('-') or key == ord('_'):
                current_size = max(5.0, current_size - 1.0)
                print(f"Square size: {current_size:.1f} mm")
            elif key == ord('r') or key == ord('R'):
                ruler = not ruler
            elif key == ord('f') or key == ord('F'):
                fullscreen = not fullscreen
            elif key == ord('s') or key == ord('S'):
                filename = f"calibration_pattern_{cols}x{rows}_{current_size:.0f}mm.png"
                cv2.imwrite(filename, pattern)
                print(f"Pattern saved to: {filename}")

        cv2.destroyAllWindows()


# =============================================================================
# Integrated Calibration with Display Awareness
# =============================================================================

class DisplayAwareCalibrator:
    """
    Camera calibrator that uses display specifications for accurate sizing.
    """

    def __init__(self, display: DisplayProfile, camera_id: int = 0):
        self.display = display
        self.camera_id = camera_id
        self.pattern_generator = OnScreenCalibrationPattern(display)

        # Calibration state
        from camera_calibration import CameraCalibrator
        self.calibrator = CameraCalibrator(camera_id=camera_id)

    def run_onscreen_calibration(self,
                                 cols: int = 9,
                                 rows: int = 6,
                                 square_size_mm: float = 25.0):
        """
        Run calibration using on-screen pattern.

        Displays pattern on screen while capturing from camera.
        """
        print(f"\n{'='*60}")
        print("ON-SCREEN CAMERA CALIBRATION")
        print(f"{'='*60}")
        print(f"Display: {self.display.name}")
        print(f"Pattern: {cols}x{rows} inner corners")
        print(f"Square size: {square_size_mm:.1f} mm")
        print(f"\nInstructions:")
        print("1. Position camera to view the screen")
        print("2. Pattern will be displayed on screen")
        print("3. Move camera to capture from different angles")
        print("4. Press SPACE to capture when pattern detected")
        print(f"{'='*60}")

        # Open camera
        if not self.calibrator.open_camera():
            return None

        self.calibrator.checkerboard_size = (cols, rows)
        self.calibrator.square_size = square_size_mm

        # Generate pattern
        pattern = self.pattern_generator.create_checkerboard(cols, rows, square_size_mm)
        pattern = self.pattern_generator.create_ruler_overlay(pattern)

        # Create windows
        cv2.namedWindow('Calibration Pattern', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Camera View', cv2.WINDOW_NORMAL)

        # Position windows side by side
        cv2.moveWindow('Calibration Pattern', 0, 0)
        cv2.moveWindow('Camera View', pattern.shape[1] + 10, 0)

        while True:
            ret, frame = self.calibrator.cap.read()
            if not ret:
                break

            # Detect checkerboard in camera view
            found, corners, display = self.calibrator.detect_checkerboard(frame)

            # Add status
            status = f"Images: {len(self.calibrator.image_points)}/10"
            cv2.putText(display, status, (10, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show windows
            cv2.imshow('Calibration Pattern', pattern)
            cv2.imshow('Camera View', display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord(' ') and found:
                if self.calibrator.add_calibration_image(corners):
                    # Flash effect
                    cv2.imshow('Camera View', np.ones_like(display) * 255)
                    cv2.waitKey(100)

                    if len(self.calibrator.image_points) >= 10:
                        print("\nSufficient images captured!")
                        break
            elif key == ord('c') or key == ord('C'):
                if len(self.calibrator.image_points) >= 5:
                    break

        cv2.destroyAllWindows()
        self.calibrator.close_camera()

        # Run calibration
        if len(self.calibrator.image_points) >= 5:
            return self.calibrator.calibrate()
        else:
            print("Not enough images captured")
            return None


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Display-Aware Camera Calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available display profiles
  python display_calibration.py --list

  # Use HP ZBook Fury 16 G10 (4K) profile
  python display_calibration.py --profile zbook_fury_16_g10_4k

  # Show pattern on screen
  python display_calibration.py --profile zbook_fury_16_g10_4k --pattern

  # Custom display (provide diagonal in inches)
  python display_calibration.py --custom 1920x1200 --diagonal 16.0

  # Run full calibration with on-screen pattern
  python display_calibration.py --profile zbook_fury_16_g10_4k --calibrate
        """
    )

    parser.add_argument('--list', '-l', action='store_true',
                       help='List available display profiles')
    parser.add_argument('--profile', '-p', type=str,
                       help='Display profile name')
    parser.add_argument('--custom', type=str,
                       help='Custom resolution as WIDTHxHEIGHT')
    parser.add_argument('--diagonal', '-d', type=float, default=16.0,
                       help='Screen diagonal in inches (for custom)')
    parser.add_argument('--pattern', action='store_true',
                       help='Display calibration pattern on screen')
    parser.add_argument('--calibrate', action='store_true',
                       help='Run full calibration with on-screen pattern')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera ID for calibration')
    parser.add_argument('--square-size', '-s', type=float, default=25.0,
                       help='Square size in mm (default: 25)')
    parser.add_argument('--detect', action='store_true',
                       help='Detect display and show info')

    args = parser.parse_args()

    # List profiles
    if args.list:
        print("\nAvailable Display Profiles:")
        print("-" * 60)
        for name, profile in DISPLAY_PROFILES.items():
            print(f"  {name:30s} {profile.width_pixels}x{profile.height_pixels} "
                  f"({profile.diagonal_inches}\" {profile.ppi:.0f}PPI)")
        print()
        return 0

    # Detect display
    if args.detect:
        profile = guess_display_profile()
        if profile:
            profile.print_info()
        else:
            width, height = detect_display_resolution()
            print(f"\nDetected: {width}x{height}")
            print("No matching profile. Use --custom to create one.")
        return 0

    # Get display profile
    display = None

    if args.profile:
        if args.profile in DISPLAY_PROFILES:
            display = DISPLAY_PROFILES[args.profile]
        else:
            print(f"Unknown profile: {args.profile}")
            print("Use --list to see available profiles")
            return 1

    elif args.custom:
        try:
            w, h = map(int, args.custom.lower().split('x'))
            display = create_custom_profile(w, h, args.diagonal)
        except:
            print(f"Invalid resolution format: {args.custom}")
            print("Use format: WIDTHxHEIGHT (e.g., 1920x1200)")
            return 1

    else:
        # Try to detect
        display = guess_display_profile()
        if display is None:
            # Default to ZBook Fury 16 G10 FHD
            print("Using default profile: HP ZBook Fury 16 G10 (WUXGA)")
            display = DISPLAY_PROFILES['zbook_fury_16_g10_fhd']

    # Show display info
    display.print_info()

    # Show pattern
    if args.pattern:
        generator = OnScreenCalibrationPattern(display)
        generator.display_fullscreen_pattern(square_size_mm=args.square_size)
        return 0

    # Run calibration
    if args.calibrate:
        calibrator = DisplayAwareCalibrator(display, args.camera)
        result = calibrator.run_onscreen_calibration(square_size_mm=args.square_size)
        if result:
            filename = f"calibration_result.json"
            result.save(filename)
        return 0

    # Default: just show info
    print("\nUse --pattern to display calibration pattern")
    print("Use --calibrate to run full calibration")

    return 0


if __name__ == '__main__':
    exit(main())

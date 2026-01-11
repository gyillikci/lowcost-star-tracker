#!/usr/bin/env python3
"""
Interactive Camera Calibration Tool

Calibrates camera intrinsic parameters (focal length, principal point, distortion)
using checkerboard pattern detection. Designed for creating custom action cameras
with external IMU (e.g., Orange Cube flight controller).

Features:
- Live camera preview with checkerboard detection
- Automatic/manual image capture
- Real-time calibration feedback
- Undistortion preview
- FOV calculation
- Save/load calibration profiles (JSON/YAML)

Usage:
    python camera_calibration.py [camera_id] [--checkerboard 9x6]

Requirements:
    - Printed checkerboard pattern (default 9x6 inner corners)
    - Download: https://docs.opencv.org/4.x/pattern.png
"""

import cv2
import numpy as np
import json
import yaml
import os
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime


# =============================================================================
# Calibration Data Structures
# =============================================================================

@dataclass
class CalibrationResult:
    """Camera calibration result."""
    # Camera matrix (intrinsics)
    fx: float  # Focal length X (pixels)
    fy: float  # Focal length Y (pixels)
    cx: float  # Principal point X
    cy: float  # Principal point Y

    # Distortion coefficients [k1, k2, p1, p2, k3]
    k1: float = 0.0  # Radial distortion 1
    k2: float = 0.0  # Radial distortion 2
    p1: float = 0.0  # Tangential distortion 1
    p2: float = 0.0  # Tangential distortion 2
    k3: float = 0.0  # Radial distortion 3

    # Image dimensions
    image_width: int = 0
    image_height: int = 0

    # Calibration quality
    rms_error: float = 0.0
    num_images: int = 0

    # Computed properties
    fov_horizontal: float = 0.0
    fov_vertical: float = 0.0
    fov_diagonal: float = 0.0

    # Metadata
    camera_name: str = ""
    calibration_date: str = ""
    checkerboard_size: str = ""

    @property
    def camera_matrix(self) -> np.ndarray:
        """Get 3x3 camera matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    @property
    def dist_coeffs(self) -> np.ndarray:
        """Get distortion coefficients array."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    def compute_fov(self):
        """
        Compute field of view from intrinsics, accounting for lens distortion.

        For cameras with barrel distortion (typical of wide-angle lenses),
        the actual FOV is larger than what the simple pinhole formula gives.
        We compute the true FOV by tracing rays through the distorted edge pixels.
        """
        if self.image_width <= 0 or self.fx <= 0:
            return

        w, h = self.image_width, self.image_height
        K = self.camera_matrix
        dist = self.dist_coeffs

        # Check if we have significant distortion
        has_distortion = abs(self.k1) > 1e-6 or abs(self.k2) > 1e-6

        if has_distortion:
            # Use ray tracing through distorted edge pixels for accurate FOV
            # Edge points in the distorted image (what the sensor actually captures)
            edge_points = np.array([
                [0, h / 2],           # Left edge center
                [w - 1, h / 2],       # Right edge center
                [w / 2, 0],           # Top edge center
                [w / 2, h - 1],       # Bottom edge center
                [0, 0],               # Top-left corner
                [w - 1, h - 1],       # Bottom-right corner
            ], dtype=np.float32).reshape(-1, 1, 2)

            # Undistort to normalized camera coordinates
            # This gives the direction vector (x, y, 1) for each pixel in world space
            normalized = cv2.undistortPoints(edge_points, K, dist)

            # Horizontal FOV: angle between left and right edge rays
            x_left = normalized[0][0][0]    # Normalized x for left edge (negative)
            x_right = normalized[1][0][0]   # Normalized x for right edge (positive)
            angle_left = np.arctan(x_left)
            angle_right = np.arctan(x_right)
            self.fov_horizontal = np.degrees(angle_right - angle_left)

            # Vertical FOV: angle between top and bottom edge rays
            y_top = normalized[2][0][1]     # Normalized y for top edge (negative)
            y_bottom = normalized[3][0][1]  # Normalized y for bottom edge (positive)
            angle_top = np.arctan(y_top)
            angle_bottom = np.arctan(y_bottom)
            self.fov_vertical = np.degrees(angle_bottom - angle_top)

            # Diagonal FOV: angle between opposite corners
            # Top-left corner direction
            tl_x, tl_y = normalized[4][0]
            # Bottom-right corner direction
            br_x, br_y = normalized[5][0]

            # Angle from optical axis for each corner
            angle_tl = np.arctan(np.sqrt(tl_x**2 + tl_y**2))
            angle_br = np.arctan(np.sqrt(br_x**2 + br_y**2))

            # Full diagonal is sum of both half-angles (they point opposite directions)
            self.fov_diagonal = np.degrees(angle_tl + angle_br)
        else:
            # No significant distortion - use simple pinhole formula
            self.fov_horizontal = 2 * np.degrees(np.arctan(w / (2 * self.fx)))
            self.fov_vertical = 2 * np.degrees(np.arctan(h / (2 * self.fy)))
            diag = np.sqrt(w**2 + h**2)
            f_avg = (self.fx + self.fy) / 2
            self.fov_diagonal = 2 * np.degrees(np.arctan(diag / (2 * f_avg)))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationResult':
        """Create from dictionary."""
        return cls(**data)

    def save(self, filepath: str):
        """Save calibration to file (JSON or YAML)."""
        data = self.to_dict()
        filepath = Path(filepath)

        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

        print(f"Calibration saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'CalibrationResult':
        """Load calibration from file."""
        filepath = Path(filepath)

        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)

        return cls.from_dict(data)


# =============================================================================
# Calibration Tool
# =============================================================================

class CameraCalibrator:
    """Interactive camera calibration tool."""

    def __init__(self,
                 camera_id: int = 0,
                 checkerboard_size: Tuple[int, int] = (9, 6),
                 square_size: float = 25.0,  # mm
                 min_images: int = 10,
                 max_images: int = 30):
        """
        Initialize calibrator.

        Args:
            camera_id: Camera device ID (0 for default)
            checkerboard_size: Inner corners (columns, rows)
            square_size: Physical size of squares in mm
            min_images: Minimum images for calibration
            max_images: Maximum images to capture
        """
        self.camera_id = camera_id
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.min_images = min_images
        self.max_images = max_images

        # Calibration data
        self.image_points: List[np.ndarray] = []  # 2D points in image
        self.object_points: List[np.ndarray] = []  # 3D points in world
        self.image_size: Optional[Tuple[int, int]] = None

        # Generate object points for checkerboard
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

        # Calibration result
        self.result: Optional[CalibrationResult] = None

        # Corner refinement criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # State
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_name = f"Camera_{camera_id}"

    def open_camera(self) -> bool:
        """Open camera device."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_id}")
            return False

        # Try to set higher resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Get actual resolution
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.image_size = (width, height)

        print(f"Camera opened: {width}x{height}")
        return True

    def close_camera(self):
        """Close camera device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def detect_checkerboard(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], np.ndarray]:
        """
        Detect checkerboard corners in frame.

        Returns:
            Tuple of (found, corners, display_frame)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        # Find checkerboard corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, flags)

        if found:
            # Refine corners to sub-pixel accuracy
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

            # Draw corners
            cv2.drawChessboardCorners(display, self.checkerboard_size, corners, found)

            # Draw quality indicator
            cv2.putText(display, "CHECKERBOARD DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Searching for checkerboard...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return found, corners, display

    def add_calibration_image(self, corners: np.ndarray) -> bool:
        """
        Add detected corners to calibration set.

        Returns:
            True if image was added
        """
        if len(self.image_points) >= self.max_images:
            print(f"Maximum images ({self.max_images}) reached")
            return False

        self.image_points.append(corners)
        self.object_points.append(self.objp)

        print(f"Image {len(self.image_points)}/{self.max_images} captured")
        return True

    def calibrate(self) -> Optional[CalibrationResult]:
        """
        Run camera calibration.

        Returns:
            CalibrationResult or None if failed
        """
        if len(self.image_points) < self.min_images:
            print(f"Need at least {self.min_images} images (have {len(self.image_points)})")
            return None

        print(f"\nCalibrating with {len(self.image_points)} images...")

        # Run calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            self.image_size,
            None, None,
            flags=cv2.CALIB_RATIONAL_MODEL
        )

        if not ret:
            print("Calibration failed!")
            return None

        # Extract parameters
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Distortion coefficients
        k1, k2, p1, p2, k3 = dist_coeffs.flatten()[:5]

        # Create result
        self.result = CalibrationResult(
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            k1=float(k1),
            k2=float(k2),
            p1=float(p1),
            p2=float(p2),
            k3=float(k3),
            image_width=self.image_size[0],
            image_height=self.image_size[1],
            rms_error=float(ret),
            num_images=len(self.image_points),
            camera_name=self.camera_name,
            calibration_date=datetime.now().isoformat(),
            checkerboard_size=f"{self.checkerboard_size[0]}x{self.checkerboard_size[1]}"
        )

        # Compute FOV (with distortion correction)
        self.result.compute_fov()

        # Also compute simple pinhole FOV for comparison
        w, h = self.image_size
        pinhole_fov_h = 2 * np.degrees(np.arctan(w / (2 * fx)))
        pinhole_fov_v = 2 * np.degrees(np.arctan(h / (2 * fy)))

        print("\n" + "=" * 50)
        print("CALIBRATION COMPLETE")
        print("=" * 50)
        print(f"RMS Error: {ret:.4f} pixels")
        print(f"\nIntrinsic Matrix:")
        print(f"  fx = {fx:.2f} px")
        print(f"  fy = {fy:.2f} px")
        print(f"  cx = {cx:.2f} px")
        print(f"  cy = {cy:.2f} px")
        print(f"\nDistortion Coefficients:")
        print(f"  k1 = {k1:.6f}")
        print(f"  k2 = {k2:.6f}")
        print(f"  p1 = {p1:.6f}")
        print(f"  p2 = {p2:.6f}")
        print(f"  k3 = {k3:.6f}")
        print(f"\nField of View (with distortion correction):")
        print(f"  Horizontal: {self.result.fov_horizontal:.1f}°")
        print(f"  Vertical:   {self.result.fov_vertical:.1f}°")
        print(f"  Diagonal:   {self.result.fov_diagonal:.1f}°")

        # Show pinhole comparison if there's significant distortion
        if abs(k1) > 1e-6:
            print(f"\n  (Pinhole model would give: {pinhole_fov_h:.1f}° x {pinhole_fov_v:.1f}°)")
            print(f"  (Distortion adds {self.result.fov_horizontal - pinhole_fov_h:.1f}° horizontal)")
        print("=" * 50)

        return self.result

    def compute_reprojection_error(self) -> float:
        """Compute mean reprojection error."""
        if self.result is None:
            return float('inf')

        total_error = 0
        total_points = 0

        camera_matrix = self.result.camera_matrix
        dist_coeffs = self.result.dist_coeffs

        for i, (obj_pts, img_pts) in enumerate(zip(self.object_points, self.image_points)):
            # Get rotation and translation for this image
            _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)

            # Project points
            projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)

            # Compute error
            error = cv2.norm(img_pts, projected, cv2.NORM_L2) / len(projected)
            total_error += error
            total_points += 1

        return total_error / total_points if total_points > 0 else float('inf')

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply undistortion to frame."""
        if self.result is None:
            return frame

        camera_matrix = self.result.camera_matrix
        dist_coeffs = self.result.dist_coeffs

        # Get optimal new camera matrix
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )

        # Undistort
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # Crop to ROI
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y:y+h, x:x+w]
            undistorted = cv2.resize(undistorted, (frame.shape[1], frame.shape[0]))

        return undistorted

    def run_interactive(self):
        """Run interactive calibration session."""
        if not self.open_camera():
            return

        print("\n" + "=" * 60)
        print("INTERACTIVE CAMERA CALIBRATION")
        print("=" * 60)
        print(f"Camera: {self.camera_id}")
        print(f"Resolution: {self.image_size[0]}x{self.image_size[1]}")
        print(f"Checkerboard: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} inner corners")
        print(f"Required images: {self.min_images}-{self.max_images}")
        print("\nControls:")
        print("  SPACE  - Capture image (when checkerboard detected)")
        print("  A      - Toggle auto-capture mode")
        print("  C      - Run calibration")
        print("  U      - Toggle undistortion preview")
        print("  S      - Save calibration")
        print("  R      - Reset (clear all images)")
        print("  Q/ESC  - Quit")
        print("=" * 60 + "\n")

        cv2.namedWindow('Camera Calibration', cv2.WINDOW_NORMAL)

        auto_capture = False
        auto_capture_delay = 1.5  # seconds between auto captures
        last_capture_time = 0
        show_undistorted = False

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # Detect checkerboard
            found, corners, display = self.detect_checkerboard(frame)

            # Auto capture
            if auto_capture and found:
                current_time = time.time()
                if current_time - last_capture_time > auto_capture_delay:
                    if self.add_calibration_image(corners):
                        last_capture_time = current_time
                        # Flash effect
                        display = cv2.addWeighted(display, 0.5,
                                                  np.ones_like(display) * 255, 0.5, 0)

            # Show undistorted if calibrated and enabled
            if show_undistorted and self.result is not None:
                undistorted = self.undistort_frame(frame)
                # Side by side
                h = display.shape[0]
                w = display.shape[1]
                display_small = cv2.resize(display, (w // 2, h // 2))
                undist_small = cv2.resize(undistorted, (w // 2, h // 2))

                # Labels
                cv2.putText(display_small, "Original", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(undist_small, "Undistorted", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                display = np.hstack([display_small, undist_small])

            # Status bar
            status_bar = np.zeros((60, display.shape[1], 3), dtype=np.uint8)

            # Image count
            cv2.putText(status_bar, f"Images: {len(self.image_points)}/{self.max_images}",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Auto capture indicator
            auto_text = "AUTO: ON" if auto_capture else "AUTO: OFF"
            auto_color = (0, 255, 0) if auto_capture else (128, 128, 128)
            cv2.putText(status_bar, auto_text, (200, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, auto_color, 1)

            # Calibration status
            if self.result is not None:
                cv2.putText(status_bar, f"FOV: {self.result.fov_horizontal:.1f}° x {self.result.fov_vertical:.1f}°",
                           (350, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(status_bar, f"RMS: {self.result.rms_error:.3f}px",
                           (600, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            # Help text
            cv2.putText(status_bar, "[SPACE] Capture  [A] Auto  [C] Calibrate  [U] Undistort  [S] Save  [Q] Quit",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            display = np.vstack([display, status_bar])

            cv2.imshow('Camera Calibration', display)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                break

            elif key == ord(' '):  # Space - capture
                if found:
                    if self.add_calibration_image(corners):
                        # Flash effect
                        flash = cv2.addWeighted(display, 0.5,
                                               np.ones_like(display) * 255, 0.5, 0)
                        cv2.imshow('Camera Calibration', flash)
                        cv2.waitKey(100)
                        print(f">>> Image captured! Total: {len(self.image_points)}")
                else:
                    print(">>> No checkerboard detected! Make sure the pattern is visible.")

            elif key == ord('a') or key == ord('A'):  # Toggle auto capture
                auto_capture = not auto_capture
                print(f"Auto capture: {'ON' if auto_capture else 'OFF'}")

            elif key == ord('c') or key == ord('C'):  # Calibrate
                self.calibrate()

            elif key == ord('u') or key == ord('U'):  # Toggle undistortion
                if self.result is not None:
                    show_undistorted = not show_undistorted
                    print(f"Undistortion preview: {'ON' if show_undistorted else 'OFF'}")
                else:
                    print("Run calibration first!")

            elif key == ord('s') or key == ord('S'):  # Save
                if self.result is not None:
                    filename = f"calibration_{self.camera_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    self.result.save(filename)
                else:
                    print("Run calibration first!")

            elif key == ord('r') or key == ord('R'):  # Reset
                self.image_points = []
                self.object_points = []
                self.result = None
                show_undistorted = False
                print("Calibration data cleared")

        self.close_camera()
        cv2.destroyAllWindows()

        # Offer to save if calibrated
        if self.result is not None:
            save = input("\nSave calibration before exit? (y/n): ").lower()
            if save == 'y':
                filename = f"calibration_{self.camera_name}.json"
                self.result.save(filename)


# =============================================================================
# Calibration Pattern Generator
# =============================================================================

def generate_checkerboard_pattern(cols: int = 9, rows: int = 6,
                                   square_size_px: int = 100,
                                   output_path: str = "checkerboard.png"):
    """
    Generate a printable checkerboard calibration pattern.

    Args:
        cols: Number of inner corners (columns)
        rows: Number of inner corners (rows)
        square_size_px: Size of each square in pixels
        output_path: Output file path
    """
    # Pattern has (cols+1) x (rows+1) squares
    width = (cols + 1) * square_size_px
    height = (rows + 1) * square_size_px

    # Add margin
    margin = square_size_px
    total_width = width + 2 * margin
    total_height = height + 2 * margin

    # Create white image
    pattern = np.ones((total_height, total_width), dtype=np.uint8) * 255

    # Draw checkerboard
    for i in range(rows + 1):
        for j in range(cols + 1):
            if (i + j) % 2 == 0:
                x1 = margin + j * square_size_px
                y1 = margin + i * square_size_px
                x2 = x1 + square_size_px
                y2 = y1 + square_size_px
                pattern[y1:y2, x1:x2] = 0

    # Add border
    cv2.rectangle(pattern, (margin - 2, margin - 2),
                  (margin + width + 2, margin + height + 2), 0, 2)

    # Add text
    text = f"Calibration Pattern: {cols}x{rows} inner corners"
    cv2.putText(pattern, text, (margin, total_height - margin // 2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)

    cv2.imwrite(output_path, pattern)
    print(f"Checkerboard pattern saved to: {output_path}")
    print(f"  Inner corners: {cols}x{rows}")
    print(f"  Print at actual size for best results")

    return pattern


# =============================================================================
# Utility Functions
# =============================================================================

def list_cameras(max_cameras: int = 10) -> List[int]:
    """List available camera devices."""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Camera {i}: {width}x{height}")
                available.append(i)
            cap.release()
    return available


def quick_fov_estimate(camera_id: int = 0) -> Optional[float]:
    """
    Quick FOV estimation using checkerboard at known distance.

    This is a simplified method - full calibration is more accurate.
    """
    print("\n" + "=" * 50)
    print("QUICK FOV ESTIMATION")
    print("=" * 50)
    print("\nInstructions:")
    print("1. Place checkerboard at a known distance from camera")
    print("2. Measure the physical width of the checkerboard")
    print("3. The tool will calculate approximate FOV")
    print("=" * 50)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nCamera resolution: {width}x{height}")

    try:
        distance_mm = float(input("Enter distance from camera to checkerboard (mm): "))
        board_width_mm = float(input("Enter physical width of checkerboard (mm): "))
    except ValueError:
        print("Invalid input")
        cap.release()
        return None

    print("\nPoint camera at checkerboard and press SPACE when ready...")

    cv2.namedWindow('FOV Estimation', cv2.WINDOW_NORMAL)

    checkerboard_size = (9, 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        display = frame.copy()
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, checkerboard_size, corners, found)

            # Measure board width in pixels
            left_corners = corners[::checkerboard_size[0]]  # First column
            right_corners = corners[checkerboard_size[0]-1::checkerboard_size[0]]  # Last column

            board_width_px = np.mean([
                np.linalg.norm(right_corners[i] - left_corners[i])
                for i in range(len(left_corners))
            ])

            cv2.putText(display, f"Board width: {board_width_px:.1f} px", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Press SPACE to calculate FOV", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Searching for checkerboard...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('FOV Estimation', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and found:
            # Calculate FOV
            # board_width_mm / board_width_px = sensor_width_mm / image_width_px
            # FOV = 2 * atan(sensor_width / (2 * focal_length))
            # focal_length = distance * board_width_px / board_width_mm

            focal_length_px = distance_mm * board_width_px / board_width_mm
            fov_h = 2 * np.degrees(np.arctan(width / (2 * focal_length_px)))
            fov_v = 2 * np.degrees(np.arctan(height / (2 * focal_length_px)))

            print(f"\n" + "=" * 50)
            print("ESTIMATED FOV")
            print("=" * 50)
            print(f"Focal length: {focal_length_px:.1f} pixels")
            print(f"Horizontal FOV: {fov_h:.1f}°")
            print(f"Vertical FOV: {fov_v:.1f}°")
            print("=" * 50)
            print("\nNote: Full calibration provides more accurate results")

            cap.release()
            cv2.destroyAllWindows()
            return fov_h

        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return None


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Interactive Camera Calibration Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python camera_calibration.py                    # Use default camera (0)
  python camera_calibration.py 1                  # Use camera 1
  python camera_calibration.py --checkerboard 7x5 # Custom checkerboard size
  python camera_calibration.py --generate         # Generate calibration pattern
  python camera_calibration.py --list             # List available cameras
  python camera_calibration.py --quick-fov        # Quick FOV estimation
        """
    )

    parser.add_argument('camera', nargs='?', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--checkerboard', '-c', type=str, default='9x6',
                       help='Checkerboard size as COLSxROWS (default: 9x6)')
    parser.add_argument('--square-size', '-s', type=float, default=25.0,
                       help='Square size in mm (default: 25.0)')
    parser.add_argument('--generate', '-g', action='store_true',
                       help='Generate checkerboard pattern image')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available cameras')
    parser.add_argument('--quick-fov', '-q', action='store_true',
                       help='Quick FOV estimation mode')
    parser.add_argument('--load', type=str,
                       help='Load existing calibration file')
    parser.add_argument('--name', type=str, default='',
                       help='Camera name for calibration file')

    args = parser.parse_args()

    # Parse checkerboard size
    try:
        cols, rows = map(int, args.checkerboard.lower().split('x'))
        checkerboard_size = (cols, rows)
    except:
        print(f"Invalid checkerboard size: {args.checkerboard}")
        print("Use format: COLSxROWS (e.g., 9x6)")
        return 1

    # List cameras
    if args.list:
        print("Scanning for cameras...")
        cameras = list_cameras()
        if not cameras:
            print("No cameras found")
        return 0

    # Generate pattern
    if args.generate:
        generate_checkerboard_pattern(cols, rows)
        return 0

    # Quick FOV estimation
    if args.quick_fov:
        quick_fov_estimate(args.camera)
        return 0

    # Load existing calibration
    if args.load:
        try:
            result = CalibrationResult.load(args.load)
            print(f"\nLoaded calibration from: {args.load}")
            print(f"  Camera: {result.camera_name}")
            print(f"  Resolution: {result.image_width}x{result.image_height}")
            print(f"  FOV: {result.fov_horizontal:.1f}° x {result.fov_vertical:.1f}°")
            print(f"  RMS Error: {result.rms_error:.4f}")
            return 0
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return 1

    # Run interactive calibration
    calibrator = CameraCalibrator(
        camera_id=args.camera,
        checkerboard_size=checkerboard_size,
        square_size=args.square_size
    )

    if args.name:
        calibrator.camera_name = args.name

    calibrator.run_interactive()

    return 0


if __name__ == '__main__':
    exit(main())

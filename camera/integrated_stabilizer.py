#!/usr/bin/env python3
"""
Integrated Camera + Orange Cube IMU Stabilizer

Combines USB camera feed with Orange Cube flight controller attitude data
to create a real-time software gimbal stabilizer.

The camera feed (e.g., 640x480) is placed in a larger canvas (e.g., 1920x1080)
and rotated/translated opposite to the measured attitude to maintain a stable view.

Requirements:
    pip install opencv-python numpy pymavlink pyserial scipy

Usage:
    python integrated_stabilizer.py                    # Auto-detect camera and FC
    python integrated_stabilizer.py --camera 0 --port COM3
    python integrated_stabilizer.py --camera 0 --port /dev/ttyUSB0 --canvas 1920x1080
"""

import cv2
import numpy as np
import sys
import time
import argparse
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation
from collections import deque

# Import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from mavlink.orange_cube_reader import OrangeCubeReader, AttitudeData, IMUData
from src.imu_stabilizer import CameraIntrinsics


@dataclass
class AxisAlignment:
    """
    Configuration for aligning camera axes with flight controller axes.

    The flight controller (Orange Cube) has its own coordinate frame.
    The camera has its own coordinate frame.
    This configuration maps between them based on physical mounting.

    FC Frame (NED - typical):
        X = Forward (North)
        Y = Right (East)
        Z = Down

    Camera Frame (typical):
        X = Right
        Y = Down
        Z = Forward (into scene)

    Attitude angles:
        Roll  = rotation around X (forward axis)
        Pitch = rotation around Y (right axis)
        Yaw   = rotation around Z (down axis)
    """
    # Which FC axis corresponds to camera roll rotation
    # Options: 'roll', 'pitch', 'yaw', '-roll', '-pitch', '-yaw'
    camera_roll_from: str = 'roll'
    camera_pitch_from: str = 'pitch'
    camera_yaw_from: str = 'yaw'

    # Offset angles (for fine-tuning alignment)
    roll_offset_deg: float = 0.0
    pitch_offset_deg: float = 0.0
    yaw_offset_deg: float = 0.0

    def get_camera_angles(self, fc_roll: float, fc_pitch: float, fc_yaw: float) -> Tuple[float, float, float]:
        """
        Convert flight controller attitude to camera frame angles.

        Args:
            fc_roll, fc_pitch, fc_yaw: FC attitude in radians

        Returns:
            camera_roll, camera_pitch, camera_yaw in radians
        """
        fc_angles = {
            'roll': fc_roll,
            'pitch': fc_pitch,
            'yaw': fc_yaw,
            '-roll': -fc_roll,
            '-pitch': -fc_pitch,
            '-yaw': -fc_yaw,
        }

        cam_roll = fc_angles.get(self.camera_roll_from, 0.0) + np.radians(self.roll_offset_deg)
        cam_pitch = fc_angles.get(self.camera_pitch_from, 0.0) + np.radians(self.pitch_offset_deg)
        cam_yaw = fc_angles.get(self.camera_yaw_from, 0.0) + np.radians(self.yaw_offset_deg)

        return cam_roll, cam_pitch, cam_yaw


class IntegratedStabilizer:
    """
    Real-time video stabilizer using Orange Cube IMU data.

    The camera feed is placed in a larger canvas and transformed
    opposite to the measured attitude to maintain stability.
    """

    def __init__(self,
                 camera_id: int = 0,
                 camera_width: int = 640,
                 camera_height: int = 480,
                 canvas_width: int = 1920,
                 canvas_height: int = 1080,
                 fc_port: Optional[str] = None,
                 fc_baudrate: int = 115200,
                 axis_alignment: Optional[AxisAlignment] = None,
                 fov_horizontal_deg: float = 60.0):
        """
        Initialize the integrated stabilizer.

        Args:
            camera_id: USB camera device ID
            camera_width: Camera capture width
            camera_height: Camera capture height
            canvas_width: Output canvas width (should be larger than camera)
            canvas_height: Output canvas height (should be larger than camera)
            fc_port: Flight controller serial port (auto-detect if None)
            fc_baudrate: FC baud rate
            axis_alignment: Camera-to-FC axis mapping
            fov_horizontal_deg: Camera horizontal FOV for intrinsics
        """
        self.camera_id = camera_id
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.fc_port = fc_port
        self.fc_baudrate = fc_baudrate
        self.axis_alignment = axis_alignment or AxisAlignment()

        # Camera intrinsics (for proper projection)
        self.intrinsics = CameraIntrinsics.from_fov(
            camera_width, camera_height, fov_horizontal_deg
        )
        self.K = self.intrinsics.matrix
        self.K_inv = np.linalg.inv(self.K)

        # Reference attitude (the "zero" position to stabilize to)
        self.reference_attitude: Optional[Tuple[float, float, float]] = None
        self.reference_set = False

        # Current attitude from FC (updated by reader thread)
        self.current_attitude = AttitudeData()
        self.attitude_lock = threading.Lock()

        # Hardware handles
        self.cap: Optional[cv2.VideoCapture] = None
        self.fc_reader: Optional[OrangeCubeReader] = None

        # State
        self.running = False
        self.stabilization_enabled = True
        self.show_debug = True

        # Stats
        self.frame_count = 0
        self.fc_update_count = 0
        self.start_time = 0

        # Smoothing filter for attitude (optional)
        self.attitude_filter_alpha = 0.3  # Lower = more smoothing
        self.filtered_roll = 0.0
        self.filtered_pitch = 0.0
        self.filtered_yaw = 0.0

    def open_camera(self) -> bool:
        """Open USB camera."""
        print(f"\nOpening camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"ERROR: Cannot open camera {self.camera_id}")
            return False

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        # Get actual resolution
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

        # Update intrinsics if resolution differs
        if actual_w != self.camera_width or actual_h != self.camera_height:
            self.camera_width = actual_w
            self.camera_height = actual_h
            self.intrinsics = CameraIntrinsics.from_fov(
                actual_w, actual_h, self.intrinsics.fov_horizontal
            )
            self.K = self.intrinsics.matrix
            self.K_inv = np.linalg.inv(self.K)

        return True

    def connect_fc(self) -> bool:
        """Connect to flight controller."""
        print(f"\nConnecting to flight controller...")
        self.fc_reader = OrangeCubeReader(port=self.fc_port, baudrate=self.fc_baudrate)

        if not self.fc_reader.connect():
            print("WARNING: Flight controller not connected. Running without stabilization.")
            self.fc_reader = None
            return False

        # Request attitude data at high rate
        self.fc_reader.request_data_streams(rate_hz=100)
        return True

    def fc_reader_thread(self):
        """Background thread to read FC attitude data."""
        while self.running and self.fc_reader:
            try:
                msg = self.fc_reader.read_message(timeout=0.01)
                if msg:
                    msg_type = self.fc_reader.process_message(msg)
                    if msg_type in ['ATTITUDE', 'ATTITUDE_QUATERNION']:
                        with self.attitude_lock:
                            self.current_attitude = self.fc_reader.attitude_data
                            self.fc_update_count += 1
            except Exception as e:
                pass

    def get_current_attitude(self) -> Tuple[float, float, float]:
        """Get current attitude (thread-safe) with optional filtering."""
        with self.attitude_lock:
            roll = self.current_attitude.roll
            pitch = self.current_attitude.pitch
            yaw = self.current_attitude.yaw

        # Apply exponential moving average filter
        alpha = self.attitude_filter_alpha
        self.filtered_roll = alpha * roll + (1 - alpha) * self.filtered_roll
        self.filtered_pitch = alpha * pitch + (1 - alpha) * self.filtered_pitch
        self.filtered_yaw = alpha * yaw + (1 - alpha) * self.filtered_yaw

        return self.filtered_roll, self.filtered_pitch, self.filtered_yaw

    def set_reference(self):
        """Set current attitude as the reference (stable) position."""
        roll, pitch, yaw = self.get_current_attitude()
        self.reference_attitude = (roll, pitch, yaw)
        self.reference_set = True
        print(f"Reference set: Roll={np.degrees(roll):.2f}° Pitch={np.degrees(pitch):.2f}° Yaw={np.degrees(yaw):.2f}°")

    def compute_stabilization_transform(self) -> np.ndarray:
        """
        Compute the homography to stabilize the current frame.

        Returns:
            3x3 homography matrix
        """
        if not self.reference_set or not self.stabilization_enabled:
            return np.eye(3)

        # Get current and reference attitudes
        current_roll, current_pitch, current_yaw = self.get_current_attitude()
        ref_roll, ref_pitch, ref_yaw = self.reference_attitude

        # Map to camera frame
        cam_roll, cam_pitch, cam_yaw = self.axis_alignment.get_camera_angles(
            current_roll, current_pitch, current_yaw
        )
        ref_cam_roll, ref_cam_pitch, ref_cam_yaw = self.axis_alignment.get_camera_angles(
            ref_roll, ref_pitch, ref_yaw
        )

        # Compute relative rotation (current -> reference)
        # We want to undo the rotation, so we compute ref * current^(-1)
        current_rot = Rotation.from_euler('xyz', [cam_roll, cam_pitch, cam_yaw])
        ref_rot = Rotation.from_euler('xyz', [ref_cam_roll, ref_cam_pitch, ref_cam_yaw])

        relative_rot = ref_rot * current_rot.inv()
        R = relative_rot.as_matrix()

        # Homography for pure rotation: H = K * R * K^(-1)
        H = self.K @ R @ self.K_inv

        return H

    def apply_stabilization(self, frame: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Apply stabilization transform and place in canvas.

        Args:
            frame: Camera frame (camera_width x camera_height)
            H: Homography matrix

        Returns:
            Stabilized frame in canvas (canvas_width x canvas_height)
        """
        h, w = frame.shape[:2]

        # Compute offset to center camera frame in canvas
        x_offset = (self.canvas_width - w) // 2
        y_offset = (self.canvas_height - h) // 2

        # Modify homography to account for canvas offset
        # Translation to center: T_center = [[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]]
        T_center = np.array([
            [1, 0, x_offset],
            [0, 1, y_offset],
            [0, 0, 1]
        ], dtype=np.float64)

        # We want to: 1) translate to canvas center, 2) apply rotation around canvas center
        # H_canvas = T_center @ H @ T_center^(-1) ... but simpler approach:

        # Create canvas
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)

        # First place frame in canvas
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = frame

        # Modify H to rotate around canvas center
        cx, cy = self.canvas_width / 2, self.canvas_height / 2
        T_to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
        T_from_origin = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)

        # Rotation around canvas center
        H_canvas = T_from_origin @ H @ T_to_origin

        # Apply transformation
        stabilized = cv2.warpPerspective(
            canvas, H_canvas,
            (self.canvas_width, self.canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return stabilized

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw debug information overlay."""
        if not self.show_debug:
            return frame

        h, w = frame.shape[:2]

        # Get current attitude
        roll, pitch, yaw = self.get_current_attitude()
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)

        # Draw crosshair at center
        cx, cy = w // 2, h // 2
        color = (0, 255, 0) if self.stabilization_enabled else (0, 255, 255)
        cv2.line(frame, (cx - 50, cy), (cx + 50, cy), color, 2)
        cv2.line(frame, (cx, cy - 50), (cx, cy + 50), color, 2)
        cv2.circle(frame, (cx, cy), 5, color, -1)

        # Draw artificial horizon indicator
        horizon_length = 200
        horizon_y_offset = int(pitch_deg * 5)  # 5 pixels per degree
        roll_rad = roll

        # Horizon line endpoints
        dx = int(horizon_length * np.cos(roll_rad))
        dy = int(horizon_length * np.sin(roll_rad))
        pt1 = (cx - dx, cy + horizon_y_offset - dy)
        pt2 = (cx + dx, cy + horizon_y_offset + dy)
        cv2.line(frame, pt1, pt2, (0, 200, 255), 2)

        # Info box
        info_y = 30
        cv2.rectangle(frame, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 180), color, 2)

        status = "STABILIZED" if self.stabilization_enabled else "PASSTHROUGH"
        cv2.putText(frame, f"Mode: {status}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        info_y += 25

        cv2.putText(frame, f"Roll:  {roll_deg:+8.3f} deg", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        info_y += 22
        cv2.putText(frame, f"Pitch: {pitch_deg:+8.3f} deg", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        info_y += 22
        cv2.putText(frame, f"Yaw:   {yaw_deg:+8.3f} deg", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        info_y += 25

        # Reference status
        if self.reference_set:
            ref_roll, ref_pitch, ref_yaw = self.reference_attitude
            delta_roll = np.degrees(roll - ref_roll)
            delta_pitch = np.degrees(pitch - ref_pitch)
            cv2.putText(frame, f"Delta R/P: {delta_roll:+.2f} / {delta_pitch:+.2f}", (20, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "Press 'R' to set reference", (20, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        info_y += 22

        # FC status
        fc_status = "CONNECTED" if self.fc_reader else "NOT CONNECTED"
        fc_color = (0, 255, 0) if self.fc_reader else (0, 0, 255)
        cv2.putText(frame, f"FC: {fc_status}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fc_color, 1)

        # Controls help at bottom
        help_text = "[R] Set Ref | [S] Toggle Stab | [D] Debug | [Q] Quit"
        cv2.putText(frame, help_text, (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # FPS
        if self.frame_count > 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def run(self):
        """Main loop - capture, stabilize, display."""
        # Open camera
        if not self.open_camera():
            return

        # Connect to FC (optional - will work without it)
        self.connect_fc()

        # Start FC reader thread
        self.running = True
        if self.fc_reader:
            fc_thread = threading.Thread(target=self.fc_reader_thread, daemon=True)
            fc_thread.start()

        # Create window
        window_name = "Integrated Stabilizer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.canvas_width, self.canvas_height)

        print("\n" + "=" * 60)
        print("INTEGRATED STABILIZER RUNNING")
        print("=" * 60)
        print(f"Camera: {self.camera_width}x{self.camera_height}")
        print(f"Canvas: {self.canvas_width}x{self.canvas_height}")
        print(f"FOV: {self.intrinsics.fov_horizontal:.1f}° horizontal")
        print("\nControls:")
        print("  R     - Set current attitude as reference")
        print("  S     - Toggle stabilization on/off")
        print("  D     - Toggle debug overlay")
        print("  +/-   - Adjust smoothing filter")
        print("  Q/ESC - Quit")
        print("=" * 60)

        self.start_time = time.time()

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to read frame")
                    break

                self.frame_count += 1

                # Compute stabilization transform
                H = self.compute_stabilization_transform()

                # Apply stabilization and place in canvas
                stabilized = self.apply_stabilization(frame, H)

                # Draw overlay
                display = self.draw_overlay(stabilized)

                cv2.imshow(window_name, display)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    break
                elif key == ord('r') or key == ord('R'):
                    self.set_reference()
                elif key == ord('s') or key == ord('S'):
                    self.stabilization_enabled = not self.stabilization_enabled
                    print(f"Stabilization: {'ON' if self.stabilization_enabled else 'OFF'}")
                elif key == ord('d') or key == ord('D'):
                    self.show_debug = not self.show_debug
                elif key == ord('+') or key == ord('='):
                    self.attitude_filter_alpha = min(1.0, self.attitude_filter_alpha + 0.1)
                    print(f"Smoothing: {1.0 - self.attitude_filter_alpha:.1f}")
                elif key == ord('-') or key == ord('_'):
                    self.attitude_filter_alpha = max(0.1, self.attitude_filter_alpha - 0.1)
                    print(f"Smoothing: {1.0 - self.attitude_filter_alpha:.1f}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.close()

    def close(self):
        """Clean up resources."""
        self.running = False

        if self.cap:
            self.cap.release()

        if self.fc_reader:
            self.fc_reader.close()

        cv2.destroyAllWindows()

        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"\nSession ended. {self.frame_count} frames in {elapsed:.1f}s")
        if self.fc_reader:
            print(f"FC updates: {self.fc_update_count} ({self.fc_update_count/elapsed:.1f} Hz)")


def main():
    parser = argparse.ArgumentParser(
        description="Integrated Camera + Orange Cube IMU Stabilizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python integrated_stabilizer.py
  python integrated_stabilizer.py --camera 0 --port COM3
  python integrated_stabilizer.py --camera 0 --canvas 1920x1080 --cam-res 640x480
  python integrated_stabilizer.py --fov 90  # Wide angle lens
        """
    )

    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--port', '-p', type=str, default=None,
                       help='Flight controller serial port (auto-detect if not specified)')
    parser.add_argument('--baud', '-b', type=int, default=115200,
                       help='FC baud rate (default: 115200)')
    parser.add_argument('--canvas', type=str, default='1920x1080',
                       help='Canvas size as WxH (default: 1920x1080)')
    parser.add_argument('--cam-res', type=str, default='640x480',
                       help='Camera resolution as WxH (default: 640x480)')
    parser.add_argument('--fov', type=float, default=60.0,
                       help='Camera horizontal FOV in degrees (default: 60)')
    parser.add_argument('--roll-from', type=str, default='roll',
                       choices=['roll', 'pitch', 'yaw', '-roll', '-pitch', '-yaw'],
                       help='FC axis for camera roll (default: roll)')
    parser.add_argument('--pitch-from', type=str, default='pitch',
                       choices=['roll', 'pitch', 'yaw', '-roll', '-pitch', '-yaw'],
                       help='FC axis for camera pitch (default: pitch)')
    parser.add_argument('--yaw-from', type=str, default='yaw',
                       choices=['roll', 'pitch', 'yaw', '-roll', '-pitch', '-yaw'],
                       help='FC axis for camera yaw (default: yaw)')

    args = parser.parse_args()

    # Parse canvas size
    try:
        canvas_w, canvas_h = map(int, args.canvas.lower().split('x'))
    except:
        print(f"Invalid canvas size: {args.canvas}")
        return 1

    # Parse camera resolution
    try:
        cam_w, cam_h = map(int, args.cam_res.lower().split('x'))
    except:
        print(f"Invalid camera resolution: {args.cam_res}")
        return 1

    # Create axis alignment
    axis_alignment = AxisAlignment(
        camera_roll_from=args.roll_from,
        camera_pitch_from=args.pitch_from,
        camera_yaw_from=args.yaw_from,
    )

    print("=" * 60)
    print("INTEGRATED CAMERA + IMU STABILIZER")
    print("=" * 60)

    # Create and run stabilizer
    stabilizer = IntegratedStabilizer(
        camera_id=args.camera,
        camera_width=cam_w,
        camera_height=cam_h,
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        fc_port=args.port,
        fc_baudrate=args.baud,
        axis_alignment=axis_alignment,
        fov_horizontal_deg=args.fov
    )

    stabilizer.run()

    return 0


if __name__ == '__main__':
    exit(main())

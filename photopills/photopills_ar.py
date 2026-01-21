#!/usr/bin/env python3
"""
PhotoPills-like Night AR Application.

Augmented Reality night sky visualization using:
- Harrier 10x AF Zoom Camera
- Orange Cube Flight Controller (MAVLink IMU)

Features:
- Real-time celestial overlay on camera feed
- Milky Way band and Galactic Center position
- Sun and Moon positions with phase
- Celestial equator and poles
- Star trails pattern preview
- Time manipulation (plan future shots)

Usage:
    python photopills_ar.py --lat 34.05 --lon -118.24

Press 'H' for help with keyboard shortcuts.
"""

import sys
import time
import argparse
import cv2
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional, Tuple
import threading
import math

# Try to import MAVLink for Orange Cube
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False
    print("Warning: pymavlink not available. Install with: pip install pymavlink")

from .night_ar import (
    NightARRenderer, create_renderer, OverlayLayer, IMUOrientation
)
from .celestial import create_calculator


@dataclass
class Config:
    """Application configuration."""
    # Observer location
    latitude: float = 34.0522      # Los Angeles default
    longitude: float = -118.2437

    # Camera settings
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_hfov: float = 50.0      # Harrier 10x at 1x zoom
    camera_vfov: float = 34.0

    # Orange Cube MAVLink settings
    mavlink_port: str = "COM6"     # Windows COM port
    mavlink_baud: int = 115200

    # Display settings
    fullscreen: bool = False
    window_name: str = "PhotoPills Night AR"


class OrangeCubeIMU:
    """
    Orange Cube IMU reader via MAVLink.

    Reads ATTITUDE messages from Orange Cube flight controller.
    """

    def __init__(self, port: str = "COM6", baudrate: int = 115200):
        """
        Initialize Orange Cube connection.

        Args:
            port: Serial port (e.g., "COM6" on Windows, "/dev/ttyACM0" on Linux)
            baudrate: Baud rate (typically 115200)
        """
        self.port = port
        self.baudrate = baudrate
        self._connection = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Current orientation
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._last_update = 0.0

    def connect(self) -> bool:
        """
        Connect to Orange Cube.

        Returns:
            True if connected successfully
        """
        if not MAVLINK_AVAILABLE:
            print("MAVLink not available")
            return False

        try:
            connection_string = f"serial:{self.port}:{self.baudrate}"
            print(f"Connecting to Orange Cube: {connection_string}")

            self._connection = mavutil.mavlink_connection(connection_string)

            # Wait for heartbeat
            print("Waiting for heartbeat...")
            self._connection.wait_heartbeat(timeout=10)
            print(f"Heartbeat received from system {self._connection.target_system}")

            # Request attitude stream
            self._connection.mav.request_data_stream_send(
                self._connection.target_system,
                self._connection.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,  # Attitude
                50,  # 50 Hz
                1    # Start
            )

            return True

        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def start(self):
        """Start reading attitude data in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop reading."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._connection:
            self._connection.close()

    def _read_loop(self):
        """Background thread for reading MAVLink messages."""
        while self._running and self._connection:
            try:
                msg = self._connection.recv_match(
                    type='ATTITUDE',
                    blocking=True,
                    timeout=0.1
                )

                if msg:
                    with self._lock:
                        self._roll = math.degrees(msg.roll)
                        self._pitch = math.degrees(msg.pitch)
                        self._yaw = math.degrees(msg.yaw)
                        # Convert yaw from -180..180 to 0..360 (compass heading)
                        if self._yaw < 0:
                            self._yaw += 360
                        self._last_update = time.time()

            except Exception as e:
                if self._running:
                    print(f"MAVLink read error: {e}")
                time.sleep(0.1)

    def get_orientation(self) -> IMUOrientation:
        """
        Get current orientation.

        Returns:
            IMUOrientation with roll, pitch, yaw in degrees
        """
        with self._lock:
            return IMUOrientation(
                roll=self._roll,
                pitch=self._pitch,
                yaw=self._yaw
            )

    @property
    def is_connected(self) -> bool:
        """Check if receiving data."""
        with self._lock:
            return (time.time() - self._last_update) < 1.0


class SimulatedIMU:
    """
    Simulated IMU for testing without hardware.

    Uses mouse movement to simulate orientation changes.
    """

    def __init__(self):
        self._yaw = 180.0     # Start facing South
        self._pitch = 30.0    # Looking up 30 degrees
        self._roll = 0.0

    def connect(self) -> bool:
        print("Using simulated IMU (mouse control)")
        print("  - Move mouse left/right: Change heading")
        print("  - Move mouse up/down: Change pitch")
        return True

    def start(self):
        pass

    def stop(self):
        pass

    def update_from_mouse(self, dx: int, dy: int, sensitivity: float = 0.2):
        """Update orientation from mouse delta."""
        self._yaw = (self._yaw + dx * sensitivity) % 360
        self._pitch = max(-89, min(89, self._pitch - dy * sensitivity))

    def get_orientation(self) -> IMUOrientation:
        return IMUOrientation(
            roll=self._roll,
            pitch=self._pitch,
            yaw=self._yaw
        )

    @property
    def is_connected(self) -> bool:
        return True


class PhotoPillsAR:
    """
    Main PhotoPills-like Night AR application.
    """

    def __init__(self, config: Config):
        """
        Initialize the application.

        Args:
            config: Application configuration
        """
        self.config = config

        # Create renderer
        self.renderer = create_renderer(
            latitude=config.latitude,
            longitude=config.longitude,
            camera_width=config.camera_width,
            camera_height=config.camera_height,
            hfov=config.camera_hfov,
            vfov=config.camera_vfov
        )

        # IMU
        self.imu: Optional[OrangeCubeIMU] = None
        self.simulated_imu: Optional[SimulatedIMU] = None
        self.use_simulated = False

        # Camera
        self.camera: Optional[cv2.VideoCapture] = None

        # State
        self.running = False
        self.show_help = False
        self.mouse_last_pos = None

    def setup_imu(self) -> bool:
        """
        Set up IMU connection.

        Returns:
            True if connected successfully
        """
        if MAVLINK_AVAILABLE:
            self.imu = OrangeCubeIMU(
                port=self.config.mavlink_port,
                baudrate=self.config.mavlink_baud
            )
            if self.imu.connect():
                self.imu.start()
                self.use_simulated = False
                return True
            else:
                print("Orange Cube not available, using simulated IMU")

        # Fall back to simulated IMU
        self.simulated_imu = SimulatedIMU()
        self.simulated_imu.connect()
        self.use_simulated = True
        return True

    def setup_camera(self) -> bool:
        """
        Set up camera capture.

        Returns:
            True if camera opened successfully
        """
        print(f"Opening camera {self.config.camera_index}...")

        # Try DirectShow backend on Windows
        self.camera = cv2.VideoCapture(
            self.config.camera_index,
            cv2.CAP_DSHOW
        )

        if not self.camera.isOpened():
            # Try default backend
            self.camera = cv2.VideoCapture(self.config.camera_index)

        if not self.camera.isOpened():
            print("Failed to open camera")
            return False

        # Set resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)

        actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_w}x{actual_h}")

        return True

    def get_orientation(self) -> IMUOrientation:
        """Get current IMU orientation."""
        if self.use_simulated:
            return self.simulated_imu.get_orientation()
        else:
            return self.imu.get_orientation()

    def handle_key(self, key: int) -> bool:
        """
        Handle keyboard input.

        Args:
            key: Key code from cv2.waitKey

        Returns:
            False if should quit, True otherwise
        """
        if key == -1:
            return True

        key = key & 0xFF

        # Quit
        if key == ord('q') or key == 27:  # Q or ESC
            return False

        # Help
        if key == ord('h'):
            self.show_help = not self.show_help
            return True

        # Toggle layers
        layer_keys = {
            ord('m'): OverlayLayer.MILKY_WAY,
            ord('g'): OverlayLayer.GALACTIC_CENTER,
            ord('s'): OverlayLayer.SUN,
            ord('l'): OverlayLayer.MOON,
            ord('e'): OverlayLayer.CELESTIAL_EQUATOR,
            ord('p'): OverlayLayer.CELESTIAL_POLES,
            ord('t'): OverlayLayer.STAR_TRAILS,
            ord('c'): OverlayLayer.COMPASS,
            ord('i'): OverlayLayer.INFO_PANEL,
        }

        if key in layer_keys:
            self.renderer.toggle_layer(layer_keys[key])
            return True

        # Time manipulation
        if key == ord('+') or key == ord('='):
            current = self.renderer.time_offset.total_seconds() / 3600
            self.renderer.set_time_offset(current + 1)
            print(f"Time offset: {current + 1:+.1f} hours")
        elif key == ord('-') or key == ord('_'):
            current = self.renderer.time_offset.total_seconds() / 3600
            self.renderer.set_time_offset(current - 1)
            print(f"Time offset: {current - 1:+.1f} hours")
        elif key == ord(']'):
            current = self.renderer.time_offset.total_seconds() / 3600
            self.renderer.set_time_offset(current + 0.25)
            print(f"Time offset: {current + 0.25:+.2f} hours")
        elif key == ord('['):
            current = self.renderer.time_offset.total_seconds() / 3600
            self.renderer.set_time_offset(current - 0.25)
            print(f"Time offset: {current - 0.25:+.2f} hours")
        elif key == ord('0'):
            self.renderer.set_time_offset(0)
            print("Time offset reset to now")

        # Save screenshot
        if key == ord('f'):
            filename = f"photopills_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            # Will be saved in main loop
            return True

        return True

    def handle_mouse(self, event, x, y, flags, param):
        """Handle mouse events for simulated IMU."""
        if not self.use_simulated:
            return

        if event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_last_pos is not None:
                dx = x - self.mouse_last_pos[0]
                dy = y - self.mouse_last_pos[1]
                self.simulated_imu.update_from_mouse(dx, dy)
            self.mouse_last_pos = (x, y)

    def run(self):
        """Main application loop."""
        print("=" * 60)
        print("PhotoPills Night AR")
        print("=" * 60)
        print(f"Location: {self.config.latitude:.4f}°N, {abs(self.config.longitude):.4f}°W")
        print()

        # Setup
        if not self.setup_imu():
            print("Failed to set up IMU")
            return

        if not self.setup_camera():
            print("Failed to set up camera")
            return

        # Create window
        cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
        if self.config.fullscreen:
            cv2.setWindowProperty(self.config.window_name,
                                 cv2.WND_PROP_FULLSCREEN,
                                 cv2.WINDOW_FULLSCREEN)

        cv2.setMouseCallback(self.config.window_name, self.handle_mouse)

        print("\nPress 'H' for help, 'Q' to quit")
        print("-" * 60)

        self.running = True
        frame_count = 0
        last_fps_time = time.time()
        fps = 0

        try:
            while self.running:
                # Read camera frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read frame")
                    time.sleep(0.1)
                    continue

                # Get orientation
                orientation = self.get_orientation()

                # Render AR overlay
                if self.show_help:
                    output = self.renderer.render_help(frame)
                else:
                    output = self.renderer.render(frame, orientation)

                # Add FPS counter
                frame_count += 1
                if time.time() - last_fps_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    last_fps_time = time.time()

                cv2.putText(output, f"FPS: {fps}", (output.shape[1] - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # IMU status
                if self.use_simulated:
                    status = "Simulated IMU (mouse)"
                else:
                    status = "Orange Cube" if self.imu.is_connected else "Orange Cube (no data)"
                cv2.putText(output, status, (output.shape[1] - 200, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # Display
                cv2.imshow(self.config.window_name, output)

                # Handle input
                key = cv2.waitKey(1)
                if not self.handle_key(key):
                    break

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.running = False

        if self.camera:
            self.camera.release()

        if self.imu:
            self.imu.stop()

        cv2.destroyAllWindows()
        print("Cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PhotoPills-like Night AR with Orange Cube + Harrier 10x"
    )

    # Location
    parser.add_argument("--lat", type=float, default=34.0522,
                       help="Observer latitude (degrees North)")
    parser.add_argument("--lon", type=float, default=-118.2437,
                       help="Observer longitude (degrees East, negative for West)")

    # Camera
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index")
    parser.add_argument("--width", type=int, default=1280,
                       help="Camera width")
    parser.add_argument("--height", type=int, default=720,
                       help="Camera height")
    parser.add_argument("--hfov", type=float, default=50.0,
                       help="Horizontal FOV (degrees)")
    parser.add_argument("--vfov", type=float, default=34.0,
                       help="Vertical FOV (degrees)")

    # MAVLink
    parser.add_argument("--port", type=str, default="COM6",
                       help="Orange Cube serial port")
    parser.add_argument("--baud", type=int, default=115200,
                       help="Serial baud rate")

    # Display
    parser.add_argument("--fullscreen", action="store_true",
                       help="Start in fullscreen mode")

    args = parser.parse_args()

    config = Config(
        latitude=args.lat,
        longitude=args.lon,
        camera_index=args.camera,
        camera_width=args.width,
        camera_height=args.height,
        camera_hfov=args.hfov,
        camera_vfov=args.vfov,
        mavlink_port=args.port,
        mavlink_baud=args.baud,
        fullscreen=args.fullscreen
    )

    app = PhotoPillsAR(config)
    app.run()


if __name__ == "__main__":
    main()

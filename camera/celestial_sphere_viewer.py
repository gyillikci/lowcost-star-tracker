#!/usr/bin/env python3
"""
Celestial Sphere Projection Viewer

Projects camera feed onto a celestial sphere based on Orange Cube attitude.
The camera's field of view covers a portion of the sphere, and the image
moves across the sphere as the attitude changes.

Visualizations:
1. Equirectangular projection (full sky view)
2. Orthographic projection (hemisphere view)
3. 3D sphere using OpenGL (optional)

Coordinate Systems:
- Altitude-Azimuth (local): Azimuth (0-360°), Altitude (-90° to +90°)
- Equatorial (celestial): Right Ascension (0-24h), Declination (-90° to +90°)

Requirements:
    pip install opencv-python numpy scipy

Usage:
    python celestial_sphere_viewer.py --camera 0 --port COM3
"""

import cv2
import numpy as np
import sys
import time
import threading
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
from scipy.spatial.transform import Rotation

# Import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mavlink.orange_cube_reader import OrangeCubeReader, AttitudeData
except ImportError:
    OrangeCubeReader = None
    AttitudeData = None

from src.imu_stabilizer import CameraIntrinsics


@dataclass
class SphereProjectionConfig:
    """Configuration for celestial sphere projection."""
    # Equirectangular map dimensions
    map_width: int = 1440  # 4 pixels per degree
    map_height: int = 720  # 2 pixels per degree

    # Grid settings
    show_grid: bool = True
    grid_spacing_deg: float = 15.0  # Grid lines every N degrees
    grid_color: Tuple[int, int, int] = (50, 50, 50)

    # Horizon and cardinal directions
    show_horizon: bool = True
    horizon_color: Tuple[int, int, int] = (0, 100, 0)
    show_cardinals: bool = True

    # FOV indicator
    fov_outline_color: Tuple[int, int, int] = (0, 255, 255)
    fov_outline_thickness: int = 2

    # Background
    background_color: Tuple[int, int, int] = (10, 10, 20)


class CelestialSphereViewer:
    """
    Projects camera feed onto a celestial sphere visualization.

    The camera image is mapped to the sphere based on:
    - Current attitude (roll, pitch, yaw) from Orange Cube
    - Camera intrinsics (FOV determines coverage area)
    """

    def __init__(self,
                 camera_id: int = 0,
                 camera_width: int = 640,
                 camera_height: int = 480,
                 fc_port: Optional[str] = None,
                 fc_baudrate: int = 115200,
                 fov_horizontal_deg: float = 60.0,
                 config: Optional[SphereProjectionConfig] = None):
        """
        Initialize the celestial sphere viewer.

        Args:
            camera_id: USB camera device ID
            camera_width: Camera capture width
            camera_height: Camera capture height
            fc_port: Flight controller serial port
            fc_baudrate: FC baud rate
            fov_horizontal_deg: Camera horizontal FOV
            config: Sphere projection configuration
        """
        self.camera_id = camera_id
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.fc_port = fc_port
        self.fc_baudrate = fc_baudrate

        # Camera intrinsics
        self.intrinsics = CameraIntrinsics.from_fov(
            camera_width, camera_height, fov_horizontal_deg
        )
        self.fov_h = fov_horizontal_deg
        self.fov_v = self.intrinsics.fov_vertical

        # Configuration
        self.config = config or SphereProjectionConfig()

        # Current attitude
        self.current_attitude = AttitudeData() if AttitudeData else None
        self.attitude_lock = threading.Lock()

        # Reference for accumulated yaw (since yaw wraps around)
        self.yaw_offset = 0.0

        # Hardware handles
        self.cap = None
        self.fc_reader = None

        # State
        self.running = False
        self.show_camera = True
        self.view_mode = 'equirectangular'  # or 'orthographic', 'fisheye'

        # Pre-create base map
        self.base_map = self._create_base_map()

    def _create_base_map(self) -> np.ndarray:
        """Create the base equirectangular map with grid and labels."""
        w, h = self.config.map_width, self.config.map_height

        # Create dark background (like night sky)
        base = np.zeros((h, w, 3), dtype=np.uint8)
        base[:] = self.config.background_color

        # Add some random stars for atmosphere
        n_stars = 500
        np.random.seed(42)
        star_x = np.random.randint(0, w, n_stars)
        star_y = np.random.randint(0, h, n_stars)
        star_brightness = np.random.randint(30, 150, n_stars)
        for x, y, b in zip(star_x, star_y, star_brightness):
            cv2.circle(base, (x, y), 1, (b, b, b), -1)

        if self.config.show_grid:
            spacing = self.config.grid_spacing_deg

            # Vertical lines (longitude/azimuth)
            for az in np.arange(0, 360, spacing):
                x = int(az / 360.0 * w)
                cv2.line(base, (x, 0), (x, h), self.config.grid_color, 1)

            # Horizontal lines (latitude/altitude)
            for alt in np.arange(-90, 91, spacing):
                y = int((90 - alt) / 180.0 * h)
                cv2.line(base, (0, y), (w, y), self.config.grid_color, 1)

        if self.config.show_horizon:
            # Horizon line (altitude = 0)
            y_horizon = h // 2
            cv2.line(base, (0, y_horizon), (w, y_horizon),
                    self.config.horizon_color, 2)

        if self.config.show_cardinals:
            # Cardinal direction labels
            cardinals = [
                (0, 'N'), (90, 'E'), (180, 'S'), (270, 'W')
            ]
            for az, label in cardinals:
                x = int(az / 360.0 * w)
                y = h // 2 + 20
                cv2.putText(base, label, (x - 10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Zenith and Nadir
            cv2.putText(base, "ZENITH", (w // 2 - 40, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(base, "NADIR", (w // 2 - 30, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return base

    def _attitude_to_sphere_coords(self, roll: float, pitch: float, yaw: float) -> Tuple[float, float]:
        """
        Convert attitude (roll, pitch, yaw) to sphere coordinates (azimuth, altitude).

        Assumes:
        - Pitch up = looking higher in sky (positive altitude)
        - Yaw right = looking east (positive azimuth)
        - Roll doesn't change center position (just rotates the view)

        Args:
            roll, pitch, yaw: Attitude in radians

        Returns:
            azimuth (0-360°), altitude (-90 to +90°)
        """
        # Convert yaw to azimuth (0-360)
        azimuth = np.degrees(yaw) % 360

        # Pitch directly maps to altitude
        altitude = np.degrees(pitch)
        altitude = np.clip(altitude, -90, 90)

        return azimuth, altitude

    def _sphere_to_pixel(self, azimuth: float, altitude: float) -> Tuple[int, int]:
        """
        Convert sphere coordinates to pixel position in equirectangular map.

        Args:
            azimuth: 0-360 degrees (east from north)
            altitude: -90 to +90 degrees (up from horizon)

        Returns:
            (x, y) pixel coordinates
        """
        w, h = self.config.map_width, self.config.map_height

        x = int((azimuth / 360.0) * w) % w
        y = int(((90 - altitude) / 180.0) * h)
        y = np.clip(y, 0, h - 1)

        return x, y

    def _get_fov_corners(self, azimuth: float, altitude: float, roll: float) -> List[Tuple[float, float]]:
        """
        Calculate the four corners of the camera FOV on the sphere.

        Args:
            azimuth, altitude: Center pointing direction
            roll: Camera roll angle

        Returns:
            List of (azimuth, altitude) for four corners
        """
        # Half FOV
        half_h = self.fov_h / 2
        half_v = self.fov_v / 2

        # Corner offsets before roll
        corners_local = [
            (-half_h, -half_v),  # Top-left
            (half_h, -half_v),   # Top-right
            (half_h, half_v),    # Bottom-right
            (-half_h, half_v),   # Bottom-left
        ]

        # Apply roll rotation to corners
        roll_rad = roll
        cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)

        corners_sphere = []
        for daz, dalt in corners_local:
            # Rotate by roll
            daz_rot = daz * cos_r - dalt * sin_r
            dalt_rot = daz * sin_r + dalt * cos_r

            # Add to center (simple approximation for small FOV)
            corner_az = (azimuth + daz_rot) % 360
            corner_alt = np.clip(altitude + dalt_rot, -90, 90)

            corners_sphere.append((corner_az, corner_alt))

        return corners_sphere

    def _project_camera_to_sphere(self,
                                   frame: np.ndarray,
                                   sphere_map: np.ndarray,
                                   azimuth: float,
                                   altitude: float,
                                   roll: float) -> np.ndarray:
        """
        Project camera frame onto the sphere map.

        Args:
            frame: Camera frame (BGR)
            sphere_map: Equirectangular sphere map to draw on
            azimuth, altitude: Center pointing direction
            roll: Camera roll angle

        Returns:
            Updated sphere map with camera image projected
        """
        result = sphere_map.copy()
        cam_h, cam_w = frame.shape[:2]
        map_h, map_w = result.shape[:2]

        # For each pixel in the camera frame, find its position on the sphere
        # This is computationally expensive, so we'll use a simplified approach

        # Create a mesh of camera pixel coordinates
        y_cam, x_cam = np.mgrid[0:cam_h, 0:cam_w]

        # Convert to normalized camera coordinates (-1 to 1)
        x_norm = (x_cam - cam_w / 2) / (cam_w / 2)  # -1 to 1
        y_norm = (y_cam - cam_h / 2) / (cam_h / 2)  # -1 to 1

        # Convert to angular offsets from center
        # (using FOV to scale)
        daz = x_norm * (self.fov_h / 2)  # degrees
        dalt = -y_norm * (self.fov_v / 2)  # degrees (invert y)

        # Apply roll rotation
        roll_rad = roll
        cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)
        daz_rot = daz * cos_r - dalt * sin_r
        dalt_rot = daz * sin_r + dalt * cos_r

        # Calculate absolute sphere coordinates
        sphere_az = (azimuth + daz_rot) % 360
        sphere_alt = np.clip(altitude + dalt_rot, -90, 90)

        # Convert to map pixel coordinates
        map_x = ((sphere_az / 360.0) * map_w).astype(np.int32) % map_w
        map_y = (((90 - sphere_alt) / 180.0) * map_h).astype(np.int32)
        map_y = np.clip(map_y, 0, map_h - 1)

        # Copy pixels from camera to sphere map
        # Flatten for indexing
        map_x_flat = map_x.flatten()
        map_y_flat = map_y.flatten()
        frame_flat = frame.reshape(-1, 3)

        # Direct assignment (may have overlaps, but fast)
        result[map_y_flat, map_x_flat] = frame_flat

        return result

    def _draw_fov_outline(self,
                          sphere_map: np.ndarray,
                          azimuth: float,
                          altitude: float,
                          roll: float) -> np.ndarray:
        """Draw FOV boundary on sphere map."""
        result = sphere_map.copy()

        # Get corners
        corners = self._get_fov_corners(azimuth, altitude, roll)

        # Convert to pixels
        corner_pixels = [self._sphere_to_pixel(az, alt) for az, alt in corners]

        # Handle wrap-around at azimuth 0/360
        # Check if corners span the wrap point
        azimuths = [c[0] for c in corners]
        if max(azimuths) - min(azimuths) > 180:
            # Wrap-around case - draw two polygons
            # This is a simplification; proper handling would split the polygon
            pass

        # Draw polygon outline
        pts = np.array(corner_pixels, dtype=np.int32)
        cv2.polylines(result, [pts], True,
                     self.config.fov_outline_color,
                     self.config.fov_outline_thickness)

        # Draw center crosshair
        cx, cy = self._sphere_to_pixel(azimuth, altitude)
        size = 15
        cv2.line(result, (cx - size, cy), (cx + size, cy), (0, 255, 0), 2)
        cv2.line(result, (cx, cy - size), (cx, cy + size), (0, 255, 0), 2)

        return result

    def _create_orthographic_view(self,
                                   sphere_map: np.ndarray,
                                   center_az: float,
                                   center_alt: float,
                                   view_radius_deg: float = 90) -> np.ndarray:
        """
        Create an orthographic (hemisphere) view of the sphere.

        Shows a circular view of the sky centered on the current pointing direction.

        Args:
            sphere_map: Equirectangular map
            center_az, center_alt: Center of view
            view_radius_deg: Radius of view in degrees

        Returns:
            Circular orthographic projection
        """
        size = 600  # Output size
        result = np.zeros((size, size, 3), dtype=np.uint8)
        result[:] = self.config.background_color

        map_h, map_w = sphere_map.shape[:2]

        # Create mesh for output image
        y_out, x_out = np.mgrid[0:size, 0:size]

        # Convert to normalized coordinates (-1 to 1)
        x_norm = (x_out - size / 2) / (size / 2)
        y_norm = (y_out - size / 2) / (size / 2)

        # Distance from center
        r = np.sqrt(x_norm**2 + y_norm**2)

        # Mask for valid points (inside circle)
        valid = r <= 1.0

        # Convert to angular offset from center
        # Using stereographic-like projection
        angle_from_center = r * view_radius_deg
        bearing = np.arctan2(x_norm, -y_norm)  # Bearing from north

        # Calculate sphere coordinates
        daz = angle_from_center * np.sin(bearing)
        dalt = angle_from_center * np.cos(bearing)

        sphere_az = (center_az + daz) % 360
        sphere_alt = np.clip(center_alt + dalt, -90, 90)

        # Convert to map pixel coordinates
        map_x = ((sphere_az / 360.0) * map_w).astype(np.int32) % map_w
        map_y = (((90 - sphere_alt) / 180.0) * map_h).astype(np.int32)
        map_y = np.clip(map_y, 0, map_h - 1)

        # Sample from sphere map
        result[valid] = sphere_map[map_y[valid], map_x[valid]]

        # Draw circle outline
        cv2.circle(result, (size // 2, size // 2), size // 2 - 2, (100, 100, 100), 2)

        # Draw crosshair at center
        cv2.line(result, (size // 2 - 20, size // 2), (size // 2 + 20, size // 2), (0, 255, 0), 1)
        cv2.line(result, (size // 2, size // 2 - 20), (size // 2, size // 2 + 20), (0, 255, 0), 1)

        return result

    def open_camera(self) -> bool:
        """Open USB camera."""
        print(f"Opening camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"WARNING: Cannot open camera {self.camera_id}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {actual_w}x{actual_h}")

        return True

    def connect_fc(self) -> bool:
        """Connect to flight controller."""
        if OrangeCubeReader is None:
            print("WARNING: Orange Cube reader not available")
            return False

        print("Connecting to flight controller...")
        self.fc_reader = OrangeCubeReader(port=self.fc_port, baudrate=self.fc_baudrate)

        if not self.fc_reader.connect():
            print("WARNING: FC not connected. Using simulated attitude.")
            self.fc_reader = None
            return False

        self.fc_reader.request_data_streams(rate_hz=50)
        return True

    def fc_reader_thread(self):
        """Background thread to read FC attitude."""
        while self.running and self.fc_reader:
            try:
                msg = self.fc_reader.read_message(timeout=0.01)
                if msg:
                    msg_type = self.fc_reader.process_message(msg)
                    if msg_type in ['ATTITUDE', 'ATTITUDE_QUATERNION']:
                        with self.attitude_lock:
                            self.current_attitude = self.fc_reader.attitude_data
            except:
                pass

    def get_current_attitude(self) -> Tuple[float, float, float]:
        """Get current attitude (thread-safe)."""
        if self.current_attitude is None:
            return 0.0, 0.0, 0.0

        with self.attitude_lock:
            return (self.current_attitude.roll,
                    self.current_attitude.pitch,
                    self.current_attitude.yaw)

    def run(self):
        """Main loop."""
        # Open camera (optional)
        camera_available = self.open_camera()

        # Connect to FC (optional)
        self.connect_fc()

        self.running = True

        # Start FC reader thread
        if self.fc_reader:
            fc_thread = threading.Thread(target=self.fc_reader_thread, daemon=True)
            fc_thread.start()

        # Create windows
        cv2.namedWindow('Celestial Sphere', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Hemisphere View', cv2.WINDOW_NORMAL)
        if camera_available:
            cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

        print("\n" + "=" * 60)
        print("CELESTIAL SPHERE VIEWER")
        print("=" * 60)
        print(f"Camera FOV: {self.fov_h:.1f}° x {self.fov_v:.1f}°")
        print("\nControls:")
        print("  C     - Toggle camera projection")
        print("  G     - Toggle grid")
        print("  Arrow keys - Manual attitude adjustment (no FC)")
        print("  R     - Reset view")
        print("  Q/ESC - Quit")
        print("=" * 60)

        # Simulated attitude (when no FC)
        sim_azimuth = 0.0
        sim_altitude = 45.0
        sim_roll = 0.0

        frame_count = 0
        start_time = time.time()

        try:
            while self.running:
                # Get attitude
                if self.fc_reader:
                    roll, pitch, yaw = self.get_current_attitude()
                    azimuth, altitude = self._attitude_to_sphere_coords(roll, pitch, yaw)
                else:
                    # Use simulated attitude
                    azimuth = sim_azimuth
                    altitude = sim_altitude
                    roll = sim_roll

                # Start with base map
                sphere_map = self.base_map.copy()

                # Get camera frame and project onto sphere
                if camera_available and self.show_camera:
                    ret, frame = self.cap.read()
                    if ret:
                        sphere_map = self._project_camera_to_sphere(
                            frame, sphere_map, azimuth, altitude, roll
                        )
                        cv2.imshow('Camera', frame)

                # Draw FOV outline
                sphere_map = self._draw_fov_outline(sphere_map, azimuth, altitude, roll)

                # Add info overlay
                info_y = 30
                cv2.rectangle(sphere_map, (5, 5), (350, 120), (0, 0, 0), -1)
                cv2.putText(sphere_map, f"Azimuth:  {azimuth:6.1f} deg", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                info_y += 25
                cv2.putText(sphere_map, f"Altitude: {altitude:+6.1f} deg", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                info_y += 25
                cv2.putText(sphere_map, f"Roll:     {np.degrees(roll):+6.1f} deg", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                info_y += 25

                fc_status = "CONNECTED" if self.fc_reader else "SIMULATED"
                fc_color = (0, 255, 0) if self.fc_reader else (0, 255, 255)
                cv2.putText(sphere_map, f"FC: {fc_status}", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, fc_color, 1)

                # Show equirectangular view
                cv2.imshow('Celestial Sphere', sphere_map)

                # Create and show orthographic (hemisphere) view
                ortho_view = self._create_orthographic_view(sphere_map, azimuth, altitude)
                cv2.imshow('Hemisphere View', ortho_view)

                frame_count += 1

                # Handle keyboard
                key = cv2.waitKey(30) & 0xFF

                if key == ord('q') or key == 27:
                    break
                elif key == ord('c') or key == ord('C'):
                    self.show_camera = not self.show_camera
                    print(f"Camera projection: {'ON' if self.show_camera else 'OFF'}")
                elif key == ord('g') or key == ord('G'):
                    self.config.show_grid = not self.config.show_grid
                    self.base_map = self._create_base_map()
                elif key == ord('r') or key == ord('R'):
                    sim_azimuth = 0.0
                    sim_altitude = 45.0
                    sim_roll = 0.0
                # Arrow keys for manual control (when no FC)
                elif key == 81 or key == ord('a'):  # Left
                    sim_azimuth = (sim_azimuth - 5) % 360
                elif key == 83 or key == ord('d'):  # Right
                    sim_azimuth = (sim_azimuth + 5) % 360
                elif key == 82 or key == ord('w'):  # Up
                    sim_altitude = min(90, sim_altitude + 5)
                elif key == 84 or key == ord('s'):  # Down
                    sim_altitude = max(-90, sim_altitude - 5)
                elif key == ord('e'):  # Roll CW
                    sim_roll += np.radians(5)
                elif key == ord('q'):  # Roll CCW
                    sim_roll -= np.radians(5)

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


def main():
    parser = argparse.ArgumentParser(
        description="Celestial Sphere Projection Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python celestial_sphere_viewer.py
  python celestial_sphere_viewer.py --camera 0 --port COM3
  python celestial_sphere_viewer.py --fov 90  # Wide angle lens
        """
    )

    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--port', '-p', type=str, default=None,
                       help='Flight controller serial port')
    parser.add_argument('--baud', '-b', type=int, default=115200,
                       help='FC baud rate (default: 115200)')
    parser.add_argument('--fov', type=float, default=60.0,
                       help='Camera horizontal FOV in degrees (default: 60)')
    parser.add_argument('--cam-res', type=str, default='640x480',
                       help='Camera resolution WxH (default: 640x480)')

    args = parser.parse_args()

    # Parse camera resolution
    try:
        cam_w, cam_h = map(int, args.cam_res.lower().split('x'))
    except:
        print(f"Invalid resolution: {args.cam_res}")
        return 1

    print("=" * 60)
    print("CELESTIAL SPHERE VIEWER")
    print("=" * 60)

    viewer = CelestialSphereViewer(
        camera_id=args.camera,
        camera_width=cam_w,
        camera_height=cam_h,
        fc_port=args.port,
        fc_baudrate=args.baud,
        fov_horizontal_deg=args.fov
    )

    viewer.run()

    return 0


if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python3
"""
3D OpenGL Celestial Sphere Viewer

Renders a 3D celestial sphere with camera FOV visualization using OpenGL.
The camera's field of view is shown as a highlighted region on the sphere
that updates based on Orange Cube attitude data.

Requirements:
    pip install PyOpenGL PyOpenGL_accelerate pygame numpy

Usage:
    python celestial_sphere_3d.py                    # Demo mode with simulated attitude
    python celestial_sphere_3d.py --port COM3        # With Orange Cube
    python celestial_sphere_3d.py --camera 0         # With USB camera feed

Controls:
    Left Mouse Drag  - Rotate view around sphere
    Scroll Wheel     - Zoom in/out
    R                - Reset view
    G                - Toggle grid lines
    C                - Toggle constellation lines
    F                - Toggle camera FOV display
    Space            - Pause/Resume attitude updates
    Q / ESC          - Quit
"""

import sys
import math
import time
import argparse
import threading
import numpy as np
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pygame
    from pygame.locals import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import glutInit, glutSolidSphere, glutWireSphere
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Install with: pip install PyOpenGL PyOpenGL_accelerate pygame")
    sys.exit(1)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available, camera feed disabled")


def load_calibration(filepath: str) -> dict:
    """Load camera calibration from JSON or YAML file."""
    import json
    try:
        import yaml
        YAML_AVAILABLE = True
    except ImportError:
        YAML_AVAILABLE = False

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Calibration file not found: {filepath}")

    if filepath.suffix.lower() in ['.yaml', '.yml']:
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for .yaml files: pip install pyyaml")
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    else:
        with open(filepath, 'r') as f:
            return json.load(f)


class CelestialSphere3D:
    """3D OpenGL renderer for celestial sphere with camera FOV visualization."""

    def __init__(self, width: int = 1280, height: int = 720,
                 fov_h: float = 60.0, fov_v: float = 45.0):
        """
        Initialize the 3D celestial sphere viewer.

        Args:
            width: Window width in pixels
            height: Window height in pixels
            fov_h: Camera horizontal field of view in degrees
            fov_v: Camera vertical field of view in degrees
        """
        self.width = width
        self.height = height
        self.fov_h = fov_h
        self.fov_v = fov_v

        # Camera attitude (from Orange Cube or simulation)
        self.azimuth = 0.0      # Yaw: 0-360 degrees
        self.altitude = 45.0    # Pitch: -90 to +90 degrees
        self.roll = 0.0         # Roll: -180 to +180 degrees
        
        # Yaw drift compensation
        self.yaw_offset = 0.0   # Offset to subtract from raw yaw (for drift correction)

        # View control
        self.view_rot_x = 30.0   # View elevation
        self.view_rot_y = 45.0   # View azimuth
        self.view_distance = 5.0 # Distance from sphere center

        # Mouse interaction
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)

        # Display options
        self.show_grid = True
        self.show_constellations = True
        self.show_fov = True
        self.paused = False

        # Sphere parameters
        self.sphere_radius = 2.0
        self.sphere_slices = 64
        self.sphere_stacks = 32

        # Star data (generated procedurally)
        self.stars = self._generate_stars(500)

        # Constellation lines (simplified)
        self.constellations = self._generate_constellations()

        # Threading for attitude updates
        self.attitude_lock = threading.Lock()
        self.running = True

        # Camera feed
        self.camera = None
        self.camera_frame = None
        self.camera_texture_id = None

        # Orange Cube reader
        self.imu_reader = None

        # Mock mode
        self.use_mock_camera = False
        self.mock_frame_counter = 0

        # Calibration data (loaded from file)
        self.calibration_data = None

        # Panorama stitching
        self.panorama_enabled = True  # Always show panorama on sphere
        self.panorama_width = 3600   # 10 pixels per degree (360° = 3600 px)
        self.panorama_height = 1800  # 10 pixels per degree (180° = 1800 px)
        self.panorama = None         # Equirectangular panorama image
        self.panorama_weight = None  # Weight map for blending
        self.panorama_texture_id = None
        self.panorama_dirty = False  # Flag to update texture
        self.frame_count = 0         # Number of captured frames
        
        # Optical flow for yaw tracking (primary source)
        self.prev_gray = None        # Previous grayscale frame
        self.prev_time = None        # Timestamp of previous frame
        self.optical_yaw_rate = 0.0  # Yaw rate from optical flow (deg/s)
        self.imu_yaw_rate = 0.0      # Yaw rate from IMU (deg/s) - for display only
        self.optical_flow_enabled = True
        self.yaw_rate_history = []   # History for smoothing
        self.optical_yaw_mode = True  # Use optical flow as primary yaw source
        
        # Optical flow integrated yaw tracking
        self.optical_yaw = 0.0       # Yaw from optical flow integration (degrees)
        self.optical_last_time = None
        
        # IMU-Camera calibration
        self.imu_calibration = None   # Calibration result dict
        self.R_cam_imu = None         # Rotation matrix from IMU to camera frame

    def _generate_stars(self, count: int) -> list:
        """Generate random star positions on the sphere."""
        stars = []
        np.random.seed(42)  # Reproducible star field

        for _ in range(count):
            # Random position on sphere using spherical coordinates
            az = np.random.uniform(0, 360)
            alt = np.random.uniform(-90, 90)

            # Random magnitude (brightness)
            magnitude = np.random.exponential(2.0)
            magnitude = min(magnitude, 6.0)  # Cap at mag 6

            # Size based on magnitude (brighter = larger)
            size = max(1.0, 4.0 - magnitude * 0.5)

            stars.append({
                'az': az,
                'alt': alt,
                'mag': magnitude,
                'size': size
            })

        return stars

    def _generate_constellations(self) -> list:
        """Generate simplified constellation line patterns."""
        # Define some recognizable constellation patterns
        constellations = []

        # Big Dipper (Ursa Major) - approximate positions
        big_dipper = [
            (165, 62), (178, 57), (193, 55), (207, 50),  # Bowl
            (207, 50), (220, 48), (235, 52), (248, 50)   # Handle
        ]
        for i in range(0, len(big_dipper) - 1, 2):
            if i + 1 < len(big_dipper):
                constellations.append((big_dipper[i], big_dipper[i + 1]))

        # Orion - simplified
        orion = [
            (85, 7), (88, -1),    # Betelgeuse to Bellatrix
            (85, -8), (82, -1),   # Belt area
            (85, 7), (82, -1),
            (88, -1), (85, -8),
            (82, -1), (78, -8),
            (85, -8), (88, -18)   # Rigel area
        ]
        for i in range(0, len(orion), 2):
            if i + 1 < len(orion):
                constellations.append((orion[i], orion[i + 1]))

        # Cassiopeia - W shape
        cassiopeia = [
            (10, 60), (15, 56), (20, 60), (25, 56), (30, 60)
        ]
        for i in range(len(cassiopeia) - 1):
            constellations.append((cassiopeia[i], cassiopeia[i + 1]))

        return constellations

    def _spherical_to_cartesian(self, az: float, alt: float, r: float = None) -> tuple:
        """Convert spherical coordinates to Cartesian (OpenGL coordinate system)."""
        if r is None:
            r = self.sphere_radius

        az_rad = math.radians(az)
        alt_rad = math.radians(alt)

        # OpenGL: Y is up, Z is towards viewer, X is right
        x = r * math.cos(alt_rad) * math.sin(az_rad)
        y = r * math.sin(alt_rad)
        z = r * math.cos(alt_rad) * math.cos(az_rad)

        return (x, y, z)

    def init_gl(self):
        """Initialize OpenGL settings."""
        glClearColor(0.02, 0.02, 0.08, 1.0)  # Dark blue-black background
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Lighting for sphere
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 5.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.4, 0.4, 0.5, 1.0])

        # Initialize GLUT for solid sphere rendering
        try:
            glutInit()
        except:
            pass  # May already be initialized

        # Create camera texture if camera is available
        if self.camera is not None:
            self.camera_texture_id = glGenTextures(1)

    def set_projection(self):
        """Set up the projection matrix."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def set_view(self):
        """Set up the view matrix based on user controls."""
        glLoadIdentity()

        # Position camera looking at sphere center
        cam_x = self.view_distance * math.cos(math.radians(self.view_rot_x)) * math.sin(math.radians(self.view_rot_y))
        cam_y = self.view_distance * math.sin(math.radians(self.view_rot_x))
        cam_z = self.view_distance * math.cos(math.radians(self.view_rot_x)) * math.cos(math.radians(self.view_rot_y))

        gluLookAt(cam_x, cam_y, cam_z,  # Camera position
                  0, 0, 0,               # Look at origin
                  0, 1, 0)               # Up vector

    def draw_sphere_wireframe(self):
        """Draw the celestial sphere as a wireframe (manual implementation)."""
        glDisable(GL_LIGHTING)
        glColor4f(0.15, 0.15, 0.25, 0.3)
        glLineWidth(1.0)

        # Draw sphere wireframe manually (avoids GLUT dependency issues on Windows)
        slices = 24
        stacks = 12
        r = self.sphere_radius

        # Draw latitude lines (horizontal circles)
        for i in range(stacks + 1):
            lat = math.pi * (-0.5 + float(i) / stacks)
            y = r * math.sin(lat)
            circle_r = r * math.cos(lat)
            
            glBegin(GL_LINE_LOOP)
            for j in range(slices):
                lon = 2 * math.pi * float(j) / slices
                x = circle_r * math.cos(lon)
                z = circle_r * math.sin(lon)
                glVertex3f(x, y, z)
            glEnd()

        # Draw longitude lines (vertical circles)
        for j in range(slices):
            lon = 2 * math.pi * float(j) / slices
            glBegin(GL_LINE_STRIP)
            for i in range(stacks + 1):
                lat = math.pi * (-0.5 + float(i) / stacks)
                x = r * math.cos(lat) * math.cos(lon)
                y = r * math.sin(lat)
                z = r * math.cos(lat) * math.sin(lon)
                glVertex3f(x, y, z)
            glEnd()

        glEnable(GL_LIGHTING)

    def draw_panorama_on_sphere(self):
        """Draw the stitched panorama as a texture on the celestial sphere."""
        if self.panorama is None:
            return
        
        # Create texture if needed
        if self.panorama_texture_id is None:
            self.panorama_texture_id = glGenTextures(1)
        
        # Update texture if panorama changed
        if self.panorama_dirty:
            pano_img = self.get_panorama_image()
            if pano_img is not None:
                # Convert BGR to RGB
                pano_rgb = cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB)
                
                # Create alpha channel - only show where we have data
                alpha = np.where(self.panorama_weight > 0, 200, 0).astype(np.uint8)  # Semi-transparent where data exists
                
                # Combine RGB + Alpha into RGBA
                pano_rgba = np.dstack((pano_rgb, alpha))
                
                glBindTexture(GL_TEXTURE_2D, self.panorama_texture_id)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                            pano_rgba.shape[1], pano_rgba.shape[0], 0,
                            GL_RGBA, GL_UNSIGNED_BYTE, pano_rgba)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                self.panorama_dirty = False
        
        # Draw textured sphere with blending
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, self.panorama_texture_id)
        glDisable(GL_LIGHTING)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        
        r = self.sphere_radius * 0.999  # Slightly inside the wireframe
        slices = 72
        stacks = 36
        
        # Draw sphere with equirectangular texture mapping
        # Panorama: y=0 is North Pole (alt=90), y=height is South Pole (alt=-90)
        for i in range(stacks):
            lat0 = math.pi * (-0.5 + float(i) / stacks)
            lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
            
            # Map latitude to texture V: lat=-90 -> v=1, lat=90 -> v=0
            v0 = 1.0 - float(i) / stacks
            v1 = 1.0 - float(i + 1) / stacks
            
            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                lon = 2 * math.pi * float(j) / slices
                u = float(j) / slices
                
                # Bottom vertex
                x0 = r * math.cos(lat0) * math.sin(lon)
                y0 = r * math.sin(lat0)
                z0 = r * math.cos(lat0) * math.cos(lon)
                glTexCoord2f(u, v0)
                glVertex3f(x0, y0, z0)
                
                # Top vertex
                x1 = r * math.cos(lat1) * math.sin(lon)
                y1 = r * math.sin(lat1)
                z1 = r * math.cos(lat1) * math.cos(lon)
                glTexCoord2f(u, v1)
                glVertex3f(x1, y1, z1)
            glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def draw_grid(self):
        """Draw coordinate grid lines on the sphere."""
        if not self.show_grid:
            return

        glDisable(GL_LIGHTING)
        glLineWidth(1.0)

        r = self.sphere_radius * 1.001  # Slightly outside sphere

        # Altitude lines (horizontal circles)
        for alt in range(-60, 90, 30):
            if alt == 0:
                glColor4f(0.8, 0.2, 0.2, 0.6)  # Red for horizon/equator
                glLineWidth(2.0)
            else:
                glColor4f(0.3, 0.3, 0.5, 0.4)
                glLineWidth(1.0)

            glBegin(GL_LINE_LOOP)
            circle_r = r * math.cos(math.radians(alt))
            y = r * math.sin(math.radians(alt))
            for az in range(0, 360, 5):
                x = circle_r * math.sin(math.radians(az))
                z = circle_r * math.cos(math.radians(az))
                glVertex3f(x, y, z)
            glEnd()

        # Azimuth lines (vertical great circles)
        for az in range(0, 360, 30):
            if az == 0:
                glColor4f(0.2, 0.8, 0.2, 0.6)  # Green for north
                glLineWidth(2.0)
            elif az == 90:
                glColor4f(0.2, 0.2, 0.8, 0.6)  # Blue for east
                glLineWidth(2.0)
            else:
                glColor4f(0.3, 0.3, 0.5, 0.4)
                glLineWidth(1.0)

            glBegin(GL_LINE_STRIP)
            for alt in range(-90, 91, 5):
                pos = self._spherical_to_cartesian(az, alt, r)
                glVertex3f(*pos)
            glEnd()

        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def draw_stars(self):
        """Draw stars on the celestial sphere."""
        glDisable(GL_LIGHTING)

        r = self.sphere_radius * 1.002

        for star in self.stars:
            # Brightness based on magnitude
            brightness = max(0.3, 1.0 - star['mag'] / 6.0)
            glColor4f(brightness, brightness, brightness * 1.1, 1.0)

            pos = self._spherical_to_cartesian(star['az'], star['alt'], r)

            glPointSize(star['size'])
            glBegin(GL_POINTS)
            glVertex3f(*pos)
            glEnd()

        glEnable(GL_LIGHTING)

    def draw_constellations(self):
        """Draw constellation lines."""
        if not self.show_constellations:
            return

        glDisable(GL_LIGHTING)
        glColor4f(0.3, 0.5, 0.7, 0.5)
        glLineWidth(1.5)

        r = self.sphere_radius * 1.003

        for start, end in self.constellations:
            pos1 = self._spherical_to_cartesian(start[0], start[1], r)
            pos2 = self._spherical_to_cartesian(end[0], end[1], r)

            glBegin(GL_LINES)
            glVertex3f(*pos1)
            glVertex3f(*pos2)
            glEnd()

        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def draw_camera_fov(self):
        """Draw the camera's field of view as a highlighted region on the sphere."""
        if not self.show_fov:
            return

        with self.attitude_lock:
            az = self.azimuth
            alt = self.altitude
            roll = self.roll

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)  # Always draw on top of everything

        r = self.sphere_radius * 1.005

        # Calculate FOV corners in camera frame, then transform to world
        half_fov_h = self.fov_h / 2
        half_fov_v = self.fov_v / 2

        # FOV boundary points with proper perspective projection
        num_points = 32
        fov_points = []
        
        az_rad = math.radians(az)
        alt_rad = math.radians(alt)
        roll_rad = math.radians(roll)

        # Generate FOV boundary as a rectangle using perspective projection
        for i in range(num_points):
            t = i / num_points
            if t < 0.25:
                # Top edge
                s = t / 0.25
                theta_x = math.radians((s * 2 - 1) * half_fov_h)
                theta_y = math.radians(half_fov_v)
            elif t < 0.5:
                # Right edge
                s = (t - 0.25) / 0.25
                theta_x = math.radians(half_fov_h)
                theta_y = math.radians((1 - s * 2) * half_fov_v)
            elif t < 0.75:
                # Bottom edge
                s = (t - 0.5) / 0.25
                theta_x = math.radians((1 - s * 2) * half_fov_h)
                theta_y = math.radians(-half_fov_v)
            else:
                # Left edge
                s = (t - 0.75) / 0.25
                theta_x = math.radians(-half_fov_h)
                theta_y = math.radians((s * 2 - 1) * half_fov_v)

            # Convert to unit vector in camera frame
            vec_cam = np.array([
                math.tan(theta_x),
                math.tan(theta_y),
                1.0
            ])
            vec_cam = vec_cam / np.linalg.norm(vec_cam)
            
            # Apply roll
            cos_roll, sin_roll = math.cos(roll_rad), math.sin(roll_rad)
            vec_rolled = np.array([
                vec_cam[0] * cos_roll - vec_cam[1] * sin_roll,
                vec_cam[0] * sin_roll + vec_cam[1] * cos_roll,
                vec_cam[2]
            ])
            
            # Transform to world frame
            cos_az, sin_az = math.cos(az_rad), math.sin(az_rad)
            cos_alt, sin_alt = math.cos(alt_rad), math.sin(alt_rad)
            
            vec_world = np.array([
                vec_rolled[0] * cos_az - vec_rolled[2] * sin_az * cos_alt - vec_rolled[1] * sin_az * sin_alt,
                vec_rolled[1] * cos_alt - vec_rolled[2] * sin_alt,
                vec_rolled[0] * sin_az + vec_rolled[2] * cos_az * cos_alt + vec_rolled[1] * cos_az * sin_alt
            ])
            
            # Convert to spherical
            world_alt_calc = math.degrees(math.asin(np.clip(vec_world[1], -1, 1)))
            world_az_calc = math.degrees(math.atan2(vec_world[0], vec_world[2])) % 360

            fov_points.append(self._spherical_to_cartesian(world_az_calc, world_alt_calc, r))

        # Draw FOV boundary
        glColor4f(0.0, 1.0, 0.5, 0.8)
        glLineWidth(3.0)
        glBegin(GL_LINE_LOOP)
        for point in fov_points:
            glVertex3f(*point)
        glEnd()

        # Draw FOV fill (semi-transparent) or camera texture
        if self.camera_frame is not None and self.camera_texture_id is not None:
            self._draw_camera_texture_on_fov(az, alt, roll, r)
        else:
            glColor4f(0.0, 0.8, 0.4, 0.15)
            glBegin(GL_POLYGON)
            for point in fov_points:
                glVertex3f(*point)
            glEnd()

        # Draw crosshair at center
        glColor4f(1.0, 1.0, 0.0, 1.0)
        glLineWidth(2.0)
        center = self._spherical_to_cartesian(az, alt, r)

        # Small cross
        cross_size = 0.05
        glBegin(GL_LINES)
        glVertex3f(center[0] - cross_size, center[1], center[2])
        glVertex3f(center[0] + cross_size, center[1], center[2])
        glVertex3f(center[0], center[1] - cross_size, center[2])
        glVertex3f(center[0], center[1] + cross_size, center[2])
        glEnd()

        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def _draw_camera_texture_on_fov(self, az: float, alt: float, roll: float, r: float):
        """Draw camera feed as texture on the FOV region of the sphere."""
        # Update texture with current camera frame
        glBindTexture(GL_TEXTURE_2D, self.camera_texture_id)

        # Convert BGR to RGB for OpenGL
        frame_rgb = cv2.cvtColor(self.camera_frame, cv2.COLOR_BGR2RGB)
        # Note: ROTATE_180 was already applied to camera_frame in main loop
        # OpenGL expects texture origin at bottom-left, so we flip vertically
        # This combined with ROTATE_180 gives correct orientation
        frame_rgb = cv2.flip(frame_rgb, 0)  # Flip vertically for OpenGL

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     frame_rgb.shape[1], frame_rgb.shape[0], 0,
                     GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glEnable(GL_TEXTURE_2D)
        glColor4f(1.0, 1.0, 1.0, 0.9)

        half_fov_h = self.fov_h / 2
        half_fov_v = self.fov_v / 2
        roll_rad = math.radians(roll)

        # Draw textured quad grid on sphere with proper spherical projection
        grid_size = 16
        glBegin(GL_QUADS)
        for i in range(grid_size):
            for j in range(grid_size):
                # Texture coordinates (0-1)
                u0, v0 = i / grid_size, j / grid_size
                u1, v1 = (i + 1) / grid_size, (j + 1) / grid_size

                # Angular offsets from center (-1 to 1 range)
                for u, v in [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]:
                    # Map texture coords to angular offsets in camera frame
                    # Use tangent plane projection to maintain rectangular FOV
                    theta_x = math.radians((u - 0.5) * self.fov_h)
                    theta_y = math.radians((0.5 - v) * self.fov_v)
                    
                    # Convert tangent plane to unit vector in camera frame
                    # Camera points along +Z, X is right, Y is up
                    vec_cam = np.array([
                        math.tan(theta_x),
                        math.tan(theta_y),
                        1.0
                    ])
                    vec_cam = vec_cam / np.linalg.norm(vec_cam)
                    
                    # Apply roll rotation around camera Z-axis
                    cos_roll, sin_roll = math.cos(roll_rad), math.sin(roll_rad)
                    vec_rolled = np.array([
                        vec_cam[0] * cos_roll - vec_cam[1] * sin_roll,
                        vec_cam[0] * sin_roll + vec_cam[1] * cos_roll,
                        vec_cam[2]
                    ])
                    
                    # Convert camera frame to world frame (spherical)
                    # Camera Z points to (az, alt)
                    az_rad = math.radians(az)
                    alt_rad = math.radians(alt)
                    
                    # Rotation matrices: first rotate to altitude, then to azimuth
                    # World frame: Y is up, X is east, Z is north
                    cos_az, sin_az = math.cos(az_rad), math.sin(az_rad)
                    cos_alt, sin_alt = math.cos(alt_rad), math.sin(alt_rad)
                    
                    # Transform from camera frame to world frame
                    vec_world = np.array([
                        vec_rolled[0] * cos_az - vec_rolled[2] * sin_az * cos_alt - vec_rolled[1] * sin_az * sin_alt,
                        vec_rolled[1] * cos_alt - vec_rolled[2] * sin_alt,
                        vec_rolled[0] * sin_az + vec_rolled[2] * cos_az * cos_alt + vec_rolled[1] * cos_az * sin_alt
                    ])
                    
                    # Convert to spherical coordinates
                    world_alt_calc = math.degrees(math.asin(np.clip(vec_world[1], -1, 1)))
                    world_az_calc = math.degrees(math.atan2(vec_world[0], vec_world[2])) % 360
                    
                    pos = self._spherical_to_cartesian(world_az_calc, world_alt_calc, r)
                    glTexCoord2f(u, v)
                    glVertex3f(*pos)

        glEnd()
        glDisable(GL_TEXTURE_2D)

    def draw_cardinal_directions(self):
        """Draw cardinal direction markers."""
        glDisable(GL_LIGHTING)

        r = self.sphere_radius * 1.15

        directions = [
            (0, 'N', (0.2, 1.0, 0.2)),     # North - Green
            (90, 'E', (0.2, 0.2, 1.0)),    # East - Blue
            (180, 'S', (1.0, 0.2, 0.2)),   # South - Red
            (270, 'W', (1.0, 1.0, 0.2))    # West - Yellow
        ]

        for az, label, color in directions:
            pos = self._spherical_to_cartesian(az, 0, r)

            # Draw marker
            glColor4f(*color, 1.0)
            glPointSize(10.0)
            glBegin(GL_POINTS)
            glVertex3f(*pos)
            glEnd()

        glEnable(GL_LIGHTING)

    def draw_attitude_indicator(self):
        """Draw attitude information as 2D overlay."""
        # Switch to 2D mode
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        with self.attitude_lock:
            az = self.azimuth
            alt = self.altitude
            roll = self.roll

        # Draw text background - larger to accommodate yaw rate info
        glColor4f(0.0, 0.0, 0.0, 0.6)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(280, 10)
        glVertex2f(280, 140)
        glVertex2f(10, 140)
        glEnd()

        # Note: For proper text rendering, you'd use pygame font
        # Here we'll just indicate with colored bars

        # Azimuth bar (0-360)
        glColor4f(0.2, 1.0, 0.2, 0.8)
        bar_width = (az / 360.0) * 200
        glBegin(GL_QUADS)
        glVertex2f(20, 25)
        glVertex2f(20 + bar_width, 25)
        glVertex2f(20 + bar_width, 35)
        glVertex2f(20, 35)
        glEnd()

        # Altitude bar (-90 to 90)
        glColor4f(0.2, 0.2, 1.0, 0.8)
        bar_width = ((alt + 90) / 180.0) * 200
        glBegin(GL_QUADS)
        glVertex2f(20, 50)
        glVertex2f(20 + bar_width, 50)
        glVertex2f(20 + bar_width, 60)
        glVertex2f(20, 60)
        glEnd()

        # Roll bar (-180 to 180)
        glColor4f(1.0, 0.2, 0.2, 0.8)
        bar_width = ((roll + 180) / 360.0) * 200
        glBegin(GL_QUADS)
        glVertex2f(20, 75)
        glVertex2f(20 + bar_width, 75)
        glVertex2f(20 + bar_width, 85)
        glVertex2f(20, 85)
        glEnd()

        # Yaw rate bar - Optical flow (primary source now)
        max_rate = 50.0
        center_x = 140
        
        # Optical flow yaw rate bar (cyan - this is what we're using)
        glColor4f(0.0, 1.0, 1.0, 0.8)  # Cyan for optical flow
        opt_bar = (self.optical_yaw_rate / max_rate) * 100
        opt_bar = max(-100, min(100, opt_bar))
        if opt_bar >= 0:
            glBegin(GL_QUADS)
            glVertex2f(center_x, 100)
            glVertex2f(center_x + opt_bar, 100)
            glVertex2f(center_x + opt_bar, 110)
            glVertex2f(center_x, 110)
            glEnd()
        else:
            glBegin(GL_QUADS)
            glVertex2f(center_x + opt_bar, 100)
            glVertex2f(center_x, 100)
            glVertex2f(center_x, 110)
            glVertex2f(center_x + opt_bar, 110)
            glEnd()
        
        # Center line marker
        glColor4f(1.0, 1.0, 1.0, 0.5)
        glBegin(GL_LINES)
        glVertex2f(center_x, 95)
        glVertex2f(center_x, 115)
        glEnd()
        
        # Mode indicator - show if using optical yaw
        if self.optical_yaw_mode:
            glColor4f(0.0, 1.0, 0.5, 0.8)  # Green for optical mode
        else:
            glColor4f(1.0, 0.5, 0.0, 0.8)  # Orange for IMU mode
        glBegin(GL_QUADS)
        glVertex2f(250, 20)
        glVertex2f(270, 20)
        glVertex2f(270, 40)
        glVertex2f(250, 40)
        glEnd()

        # Restore 3D mode
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def draw_help_text(self):
        """Draw help text overlay."""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        # Help background at bottom
        glColor4f(0.0, 0.0, 0.0, 0.5)
        glBegin(GL_QUADS)
        glVertex2f(0, self.height - 30)
        glVertex2f(self.width, self.height - 30)
        glVertex2f(self.width, self.height)
        glVertex2f(0, self.height)
        glEnd()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def render(self):
        """Render the complete scene."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.set_projection()
        self.set_view()

        # Draw panorama on sphere if available
        if self.panorama_enabled and self.panorama is not None and self.frame_count > 0:
            self.draw_panorama_on_sphere()

        # Draw celestial sphere elements
        self.draw_sphere_wireframe()
        self.draw_grid()
        self.draw_stars()
        self.draw_constellations()
        self.draw_cardinal_directions()
        self.draw_camera_fov()

        # Draw 2D overlays
        self.draw_attitude_indicator()
        self.draw_help_text()

        pygame.display.flip()

    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
                elif event.key == pygame.K_r:
                    # Reset view
                    self.view_rot_x = 30.0
                    self.view_rot_y = 45.0
                    self.view_distance = 5.0
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                elif event.key == pygame.K_c:
                    self.show_constellations = not self.show_constellations
                elif event.key == pygame.K_f:
                    self.show_fov = not self.show_fov
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_s:
                    # Capture current frame to panorama
                    self.capture_frame_to_panorama()
                elif event.key == pygame.K_w:
                    # Save panorama to file
                    self.save_panorama()
                elif event.key == pygame.K_x:
                    # Clear panorama
                    self.clear_panorama()
                elif event.key == pygame.K_p:
                    # Toggle panorama display on sphere
                    self.panorama_enabled = not self.panorama_enabled
                    print(f"Panorama display: {'ON' if self.panorama_enabled else 'OFF'}")
                elif event.key == pygame.K_z:
                    # Zero/reset yaw - reset optical yaw to 0
                    self.optical_yaw = 0.0
                    self.yaw_offset = 0.0
                    print(f"Yaw zeroed! Optical yaw reset to 0°")
                elif event.key == pygame.K_LEFT:
                    # Adjust yaw offset left (counter-clockwise)
                    self.yaw_offset = (self.yaw_offset - 5) % 360
                    print(f"Yaw offset: {self.yaw_offset:.1f}°")
                elif event.key == pygame.K_RIGHT:
                    # Adjust yaw offset right (clockwise)
                    self.yaw_offset = (self.yaw_offset + 5) % 360
                    print(f"Yaw offset: {self.yaw_offset:.1f}°")
                elif event.key == pygame.K_o:
                    # Toggle optical flow display/processing
                    self.optical_flow_enabled = not self.optical_flow_enabled
                    print(f"Optical flow: {'ON' if self.optical_flow_enabled else 'OFF'}")
                elif event.key == pygame.K_i:
                    # Toggle between optical yaw and IMU yaw
                    self.optical_yaw_mode = not self.optical_yaw_mode
                    print(f"Yaw source: {'OPTICAL FLOW' if self.optical_yaw_mode else 'IMU'}")
                elif event.key == pygame.K_d:
                    # Print current status
                    print(f"Yaw tracking status:")
                    print(f"  Mode: {'OPTICAL FLOW' if self.optical_yaw_mode else 'IMU'}")
                    print(f"  Optical yaw: {self.optical_yaw:.1f}°")
                    print(f"  Current azimuth: {self.azimuth:.1f}°")
                    print(f"  Yaw offset: {self.yaw_offset:.1f}°")
                    print(f"  IMU yaw rate: {self.imu_yaw_rate:.2f}°/s")
                    print(f"  Optical yaw rate: {self.optical_yaw_rate:.2f}°/s")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.mouse_dragging = True
                    self.last_mouse_pos = event.pos
                elif event.button == 4:  # Scroll up
                    self.view_distance = max(2.5, self.view_distance - 0.3)
                elif event.button == 5:  # Scroll down
                    self.view_distance = min(15.0, self.view_distance + 0.3)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]

                    self.view_rot_y += dx * 0.5
                    self.view_rot_x += dy * 0.5
                    self.view_rot_x = max(-89, min(89, self.view_rot_x))

                    self.last_mouse_pos = event.pos

        return True

    def update_attitude_simulation(self):
        """Update attitude with simulated motion (for demo mode)."""
        if self.paused:
            return

        t = time.time()

        with self.attitude_lock:
            # Slow drift in azimuth
            self.azimuth = (t * 5) % 360

            # Gentle oscillation in altitude
            self.altitude = 45 + 30 * math.sin(t * 0.3)

            # Small roll oscillation
            self.roll = 5 * math.sin(t * 0.7)

    def update_attitude_from_imu(self):
        """Update attitude from Orange Cube IMU data, applying R_cam_imu if calibrated."""
        if self.imu_reader is None or self.paused:
            return

        # Read latest attitude from reader's data
        attitude_data = self.imu_reader.attitude_data
        if attitude_data:
            with self.attitude_lock:
                if self.R_cam_imu is not None:
                    # Apply IMU-to-camera calibration rotation
                    # Build rotation matrix from IMU euler angles
                    R_imu = self._euler_to_rotation_matrix(
                        attitude_data.roll,
                        attitude_data.pitch,
                        attitude_data.yaw
                    )
                    
                    # Apply calibration (inverted): R_camera = R_imu @ R_cam_imu.T
                    R_camera = R_imu @ self.R_cam_imu.T
                    
                    # Extract euler angles from camera rotation matrix
                    roll, pitch, yaw = self._rotation_matrix_to_euler(R_camera)
                    
                    self.roll = math.degrees(roll)
                    self.altitude = -math.degrees(pitch)  # Negate for coordinate system
                    self.azimuth = math.degrees(yaw) % 360
                    
                    # Store IMU yaw rate for display
                    self.imu_yaw_rate = math.degrees(attitude_data.yawspeed)
                else:
                    # No calibration - use raw values (pitch/roll only)
                    self.roll = math.degrees(attitude_data.roll)
                    self.altitude = -math.degrees(attitude_data.pitch)
                    self.imu_yaw_rate = math.degrees(attitude_data.yawspeed)
                    # YAW DISABLED without calibration
    
    def _euler_to_rotation_matrix(self, roll, pitch, yaw):
        """Convert euler angles (rad) to 3x3 rotation matrix (ZYX convention)."""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
    
    def _quaternion_to_rotation_matrix(self, q):
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
    
    def _rotation_matrix_to_euler(self, R):
        """Extract euler angles (roll, pitch, yaw) from rotation matrix."""
        sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
        singular = sy < 1e-6
        
        if not singular:
            roll = math.atan2(R[2,1], R[2,2])
            pitch = math.atan2(-R[2,0], sy)
            yaw = math.atan2(R[1,0], R[0,0])
        else:
            roll = math.atan2(-R[1,2], R[1,1])
            pitch = math.atan2(-R[2,0], sy)
            yaw = 0
        
        return roll, pitch, yaw
    
    def load_imu_calibration(self, filepath: str):
        """Load IMU-camera calibration from JSON file."""
        import json
        try:
            with open(filepath, 'r') as f:
                self.imu_calibration = json.load(f)
            
            # Extract R_cam_imu matrix
            if 'R_cam_imu' in self.imu_calibration:
                self.R_cam_imu = np.array(self.imu_calibration['R_cam_imu'])
                euler = self.imu_calibration.get('euler_angles_deg', [0, 0, 0])
                print(f"\nLoaded IMU calibration:")
                print(f"  R_cam_imu euler: Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")
                print(f"  Time offset: {self.imu_calibration.get('time_offset', 0)*1000:.1f} ms")
                print(f"  Confidence: {self.imu_calibration.get('overall_confidence', 0):.3f}")
                return True
            else:
                print(f"Warning: No R_cam_imu in calibration file")
                return False
        except Exception as e:
            print(f"Error loading IMU calibration: {e}")
            return False

    def update_optical_yaw(self):
        """Integrate optical flow rate to get yaw position."""
        current_time = time.time()
        
        if self.optical_last_time is None:
            self.optical_last_time = current_time
            return
        
        dt = current_time - self.optical_last_time
        if dt > 0.001:  # Avoid division issues
            # Integrate yaw rate to get position
            # Negative because optical flow is opposite to rotation direction
            self.optical_yaw += self.optical_yaw_rate * dt
            self.optical_yaw = self.optical_yaw % 360
        
        self.optical_last_time = current_time

    def _imu_reader_thread(self):
        """Background thread to read IMU data."""
        while self.running and self.imu_reader:
            try:
                msg = self.imu_reader.read_message(timeout=0.1)
                if msg:
                    self.imu_reader.process_message(msg)
            except Exception as e:
                print(f"IMU read error: {e}")
                break

    def start_imu_reader(self, port: str, baudrate: int = 115200):
        """Start the Orange Cube IMU reader."""
        try:
            from mavlink.orange_cube_reader import OrangeCubeReader
            self.imu_reader = OrangeCubeReader(port=port, baudrate=baudrate)
            
            # Connect to the flight controller
            if self.imu_reader.connect():
                # Request attitude data stream
                self.imu_reader.request_data_streams(rate_hz=50)
                
                # Start background reader thread
                self._imu_thread = threading.Thread(target=self._imu_reader_thread, daemon=True)
                self._imu_thread.start()
                
                print(f"Started Orange Cube reader on {port}")
            else:
                print(f"Failed to connect to Orange Cube on {port}")
                self.imu_reader = None
        except Exception as e:
            print(f"Error starting IMU reader: {e}")
            self.imu_reader = None

    def start_camera(self, camera_index: int, resolution: tuple = None):
        """Start USB camera capture."""
        if not CV2_AVAILABLE:
            print("OpenCV not available for camera capture")
            return

        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            print(f"Could not open camera {camera_index}")
            self.camera = None
            return
        
        if resolution:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_w}x{actual_h}")

    def compute_optical_flow_yaw_rate(self, frame: np.ndarray) -> float:
        """
        Compute yaw rate from optical flow between consecutive frames.
        
        Yaw rotation causes horizontal motion across the image.
        Returns estimated yaw rate in degrees/second.
        """
        current_time = time.time()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Downsample for faster processing
        scale = 0.25
        small_gray = cv2.resize(gray, None, fx=scale, fy=scale)
        
        if self.prev_gray is None or self.prev_time is None:
            self.prev_gray = small_gray
            self.prev_time = current_time
            return 0.0
        
        # Calculate time delta
        dt = current_time - self.prev_time
        if dt < 0.001:  # Avoid division by zero
            return self.optical_yaw_rate
        
        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, small_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Extract horizontal flow component (x direction)
        flow_x = flow[:, :, 0]
        
        # For yaw rotation, the horizontal flow should be consistent across the image
        # Use the median to be robust against local motion
        h, w = flow_x.shape
        
        # Focus on the center region to avoid edge effects
        margin = int(w * 0.1)
        center_flow = flow_x[:, margin:w-margin]
        
        # Calculate median horizontal flow (pixels per frame)
        median_flow_x = np.median(center_flow)
        
        # Convert pixel flow to angular rate
        # pixels_per_degree = (image_width * scale) / FOV_H
        pixels_per_degree = (w) / self.fov_h
        
        # Angular rate = flow (pixels/frame) / pixels_per_degree / dt
        yaw_rate = median_flow_x / pixels_per_degree / dt
        
        # Apply smoothing using exponential moving average
        alpha = 0.3  # Smoothing factor
        self.optical_yaw_rate = alpha * yaw_rate + (1 - alpha) * self.optical_yaw_rate
        
        # Store history for analysis
        self.yaw_rate_history.append({
            'time': current_time,
            'optical': self.optical_yaw_rate,
            'imu': self.imu_yaw_rate
        })
        # Keep last 100 samples
        if len(self.yaw_rate_history) > 100:
            self.yaw_rate_history.pop(0)
        
        # Update previous frame
        self.prev_gray = small_gray
        self.prev_time = current_time
        
        return self.optical_yaw_rate

    def get_yaw_rate_discrepancy(self) -> float:
        """Get the difference between IMU and optical flow yaw rates."""
        return self.imu_yaw_rate - self.optical_yaw_rate

    def init_panorama(self):
        """Initialize the equirectangular panorama buffer."""
        self.panorama = np.zeros((self.panorama_height, self.panorama_width, 3), dtype=np.float32)
        self.panorama_weight = np.zeros((self.panorama_height, self.panorama_width), dtype=np.float32)
        self.panorama_enabled = True
        self.panorama_dirty = True
        self.frame_count = 0
        print(f"Panorama initialized: {self.panorama_width}x{self.panorama_height} (equirectangular)")
        print("  Press 'S' to capture frame at current attitude")
        print("  Press 'X' to clear/reset panorama")
        print("  Press 'W' to save panorama to file")

    def capture_frame_to_panorama(self):
        """Capture current frame and stitch to panorama at current attitude."""
        if self.camera_frame is None:
            print("No camera frame available")
            return
        
        if self.panorama is None:
            self.init_panorama()
        
        with self.attitude_lock:
            az = self.azimuth
            alt = self.altitude
            roll = self.roll
        
        self.stitch_frame_to_panorama(self.camera_frame, az, alt, roll)
        self.frame_count += 1
        coverage = np.sum(self.panorama_weight > 0) / (self.panorama_width * self.panorama_height) * 100
        print(f"Frame {self.frame_count} captured at Az={az:.1f}°, Alt={alt:.1f}°, Roll={roll:.1f}° | Coverage: {coverage:.1f}%")

    def stitch_frame_to_panorama(self, frame: np.ndarray, az: float, alt: float, roll: float):
        """
        Stitch a camera frame to the equirectangular panorama based on attitude.
        Uses full resolution with vectorized NumPy operations.
        
        Args:
            frame: Camera frame (BGR)
            az: Azimuth in degrees (0-360)
            alt: Altitude in degrees (-90 to +90)
            roll: Roll in degrees
        """
        if self.panorama is None:
            return
        
        frame_h, frame_w = frame.shape[:2]
        frame_float = frame.astype(np.float32)
        
        # Pre-compute rotation values
        az_rad = math.radians(az)
        alt_rad = math.radians(alt)
        roll_rad = math.radians(roll)
        
        cos_az, sin_az = math.cos(az_rad), math.sin(az_rad)
        cos_alt, sin_alt = math.cos(alt_rad), math.sin(alt_rad)
        cos_roll, sin_roll = math.cos(roll_rad), math.sin(roll_rad)
        
        # Create meshgrid of pixel coordinates
        u_coords = np.linspace(0, 1, frame_w, dtype=np.float32)
        v_coords = np.linspace(0, 1, frame_h, dtype=np.float32)
        uu, vv = np.meshgrid(u_coords, v_coords)
        
        # Map to angular offsets from camera center
        # Note: frame has been rotated 180° so top of image is physically up
        theta_x = np.radians((uu - 0.5) * self.fov_h)
        theta_y = np.radians((vv - 0.5) * self.fov_v)  # Flip vertical mapping
        
        # Convert to unit vectors in camera frame
        tan_x = np.tan(theta_x)
        tan_y = np.tan(theta_y)
        norm = np.sqrt(tan_x**2 + tan_y**2 + 1)
        
        vec_x = tan_x / norm
        vec_y = tan_y / norm
        vec_z = 1.0 / norm
        
        # Apply roll rotation
        vec_x_rolled = vec_x * cos_roll - vec_y * sin_roll
        vec_y_rolled = vec_x * sin_roll + vec_y * cos_roll
        vec_z_rolled = vec_z
        
        # Transform to world frame
        vec_world_x = vec_x_rolled * cos_az - vec_z_rolled * sin_az * cos_alt - vec_y_rolled * sin_az * sin_alt
        vec_world_y = vec_y_rolled * cos_alt - vec_z_rolled * sin_alt
        vec_world_z = vec_x_rolled * sin_az + vec_z_rolled * cos_az * cos_alt + vec_y_rolled * cos_az * sin_alt
        
        # Convert to spherical coordinates
        world_alt = np.degrees(np.arcsin(np.clip(vec_world_y, -1, 1)))
        world_az = np.degrees(np.arctan2(vec_world_x, vec_world_z)) % 360
        
        # Convert to panorama pixel coordinates
        pano_x = (world_az / 360.0 * self.panorama_width).astype(np.int32) % self.panorama_width
        pano_y = ((90 - world_alt) / 180.0 * self.panorama_height).astype(np.int32)
        pano_y = np.clip(pano_y, 0, self.panorama_height - 1)
        
        # Flatten arrays for vectorized indexing
        pano_x_flat = pano_x.ravel()
        pano_y_flat = pano_y.ravel()
        
        # Create weight mask (center pixels have higher weight)
        weight_x = 1.0 - 2.0 * np.abs(uu - 0.5)
        weight_y = 1.0 - 2.0 * np.abs(vv - 0.5)
        weights_flat = (weight_x * weight_y).ravel().astype(np.float32)
        
        # Flatten frame pixels
        frame_flat = frame_float.reshape(-1, 3)
        
        # Use np.add.at for fast accumulation (handles duplicate indices)
        weighted_pixels = frame_flat * weights_flat[:, np.newaxis]
        
        # Convert 2D indices to 1D for faster accumulation
        linear_idx = pano_y_flat * self.panorama_width + pano_x_flat
        
        # Accumulate weighted pixels and weights
        np.add.at(self.panorama.reshape(-1, 3), linear_idx, weighted_pixels)
        np.add.at(self.panorama_weight.ravel(), linear_idx, weights_flat)
        
        # Mark panorama as needing texture update
        self.panorama_dirty = True

    def get_panorama_image(self) -> np.ndarray:
        """Get the current panorama as a displayable image."""
        if self.panorama is None:
            return None
        
        # Normalize by weights
        result = np.zeros_like(self.panorama)
        mask = self.panorama_weight > 0
        for c in range(3):
            result[:, :, c][mask] = self.panorama[:, :, c][mask] / self.panorama_weight[mask]
        
        return result.astype(np.uint8)

    def save_panorama(self, filename: str = None):
        """Save the panorama to a file."""
        if self.panorama is None or self.frame_count == 0:
            print("No panorama to save")
            return
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"panorama_{timestamp}.png"
        
        pano_img = self.get_panorama_image()
        if pano_img is not None:
            cv2.imwrite(filename, pano_img)
            print(f"Panorama saved to: {filename}")
            
            # Also calculate coverage
            coverage = np.sum(self.panorama_weight > 0) / (self.panorama_width * self.panorama_height) * 100
            print(f"Sky coverage: {coverage:.1f}% ({self.frame_count} frames)")

    def clear_panorama(self):
        """Clear/reset the panorama."""
        if self.panorama is not None:
            self.panorama.fill(0)
            self.panorama_weight.fill(0)
            self.panorama_dirty = True
            self.frame_count = 0
            print("Panorama cleared")

    def start_mock_camera(self):
        """Start mock camera with synthetic star field."""
        self.use_mock_camera = True
        print("Mock camera enabled - generating synthetic star field")

    def _generate_mock_frame(self) -> np.ndarray:
        """Generate a synthetic star field frame."""
        self.mock_frame_counter += 1
        
        # Create dark background
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (5, 5, 15)  # Dark blue-ish background
        
        # Add some noise
        noise = np.random.randint(0, 10, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Generate consistent star positions (seeded by attitude)
        with self.attitude_lock:
            base_seed = int(self.azimuth * 10 + self.altitude * 5) % 10000
        
        np.random.seed(42)  # Fixed seed for consistent stars
        num_stars = 150
        
        for i in range(num_stars):
            # Star position with slight drift based on attitude
            x = int(np.random.uniform(20, 620))
            y = int(np.random.uniform(20, 460))
            
            # Star brightness
            brightness = int(np.random.exponential(80))
            brightness = min(255, max(50, brightness))
            
            # Star size
            radius = np.random.choice([1, 1, 1, 2, 2, 3])
            
            # Draw star with slight color variation
            color = (brightness, brightness, int(brightness * 0.9))
            cv2.circle(frame, (x, y), radius, color, -1)
            
            # Add glow for bright stars
            if brightness > 150:
                cv2.circle(frame, (x, y), radius + 2, 
                          (brightness//4, brightness//4, brightness//5), -1)
        
        # Add a "nebula" gradient region
        t = self.mock_frame_counter * 0.02
        nebula_x = int(320 + 100 * math.sin(t))
        nebula_y = int(240 + 80 * math.cos(t * 0.7))
        for r in range(60, 0, -5):
            alpha = (60 - r) / 60.0 * 0.15
            overlay = frame.copy()
            cv2.circle(overlay, (nebula_x, nebula_y), r, (30, 10, 50), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add crosshair in center
        cv2.line(frame, (310, 240), (330, 240), (0, 255, 0), 1)
        cv2.line(frame, (320, 230), (320, 250), (0, 255, 0), 1)
        
        # Add attitude text overlay
        with self.attitude_lock:
            text = f"Az:{self.azimuth:.1f} Alt:{self.altitude:.1f} Roll:{self.roll:.1f}"
        cv2.putText(frame, text, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 200, 0), 1)
        cv2.putText(frame, "MOCK CAMERA", (520, 470), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 150, 255), 1)
        
        return frame

    def run(self, demo_mode: bool = True):
        """Main rendering loop."""
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Celestial Sphere Viewer")

        self.init_gl()

        clock = pygame.time.Clock()

        print("\n" + "=" * 50)
        print("3D Celestial Sphere Viewer")
        print("=" * 50)
        print("Controls:")
        print("  Left Mouse Drag - Rotate view")
        print("  Scroll Wheel    - Zoom in/out")
        print("  R - Reset view")
        print("  G - Toggle grid")
        print("  C - Toggle constellations")
        print("  F - Toggle FOV display")
        print("  Space - Pause/Resume")
        print("  S - Capture frame to panorama")
        print("  W - Save panorama to file")
        print("  X - Clear/reset panorama")
        print("  P - Toggle panorama display")
        print("  Z - Zero yaw (reset optical yaw to 0)")
        print("  Left/Right Arrows - Adjust yaw offset (±5°)")
        print("  O - Toggle optical flow on/off")
        print("  I - Toggle yaw source (Optical/IMU)")
        print("  D - Print yaw tracking status")
        print("  Q/ESC - Quit")
        print("=" * 50)
        print("YAW TRACKING: Using OPTICAL FLOW (no IMU drift!)")
        print("  Pitch/Roll still from IMU, Yaw from camera")
        print("=" * 50)

        try:
            while self.running:
                if not self.handle_events():
                    break

                # Update attitude
                if self.imu_reader:
                    self.update_attitude_from_imu()
                elif demo_mode:
                    self.update_attitude_simulation()

                # Capture camera frame if available
                if self.camera is not None:
                    ret, frame = self.camera.read()
                    if ret:
                        # Camera may be mounted upside down - toggle with --flip-camera
                        # Currently: No rotation (was ROTATE_180 but removed for testing)
                        self.camera_frame = frame
                        
                        # Compute optical flow yaw rate
                        if self.optical_flow_enabled:
                            self.compute_optical_flow_yaw_rate(self.camera_frame)
                            # Integrate to get optical yaw position
                            self.update_optical_yaw()
                elif self.use_mock_camera:
                    self.camera_frame = self._generate_mock_frame()

                self.render()
                clock.tick(60)  # 60 FPS

        finally:
            self.running = False

            if self.imu_reader:
                if hasattr(self.imu_reader, 'stop'):
                    self.imu_reader.stop()
                elif hasattr(self.imu_reader, 'disconnect'):
                    self.imu_reader.disconnect()

            if self.camera:
                self.camera.release()

            pygame.quit()
            print("\nViewer closed.")


def main():
    parser = argparse.ArgumentParser(
        description="3D OpenGL Celestial Sphere Viewer"
    )
    parser.add_argument('--width', type=int, default=1280,
                        help='Window width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='Window height (default: 720)')
    parser.add_argument('--fov-h', type=float, default=None,
                        help='Camera horizontal FOV in degrees (default: 60, or from calibration)')
    parser.add_argument('--fov-v', type=float, default=None,
                        help='Camera vertical FOV in degrees (default: 45, or from calibration)')
    parser.add_argument('--calibration', '-cal', type=str, default=None,
                        help='Camera calibration file (JSON/YAML) to load FOV and distortion data')
    parser.add_argument('--port', type=str, default=None,
                        help='Orange Cube serial port (e.g., COM3, /dev/ttyUSB0)')
    parser.add_argument('--baudrate', type=int, default=115200,
                        help='Serial baudrate (default: 115200)')
    parser.add_argument('--camera', type=int, default=None,
                        help='USB camera index for live feed')
    parser.add_argument('--cam-res', type=str, default=None,
                        help='Camera resolution WxH (e.g., 1920x1080)')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock camera and simulated attitude (no hardware needed)')
    parser.add_argument('--imu-cal', type=str, default=None,
                        help='IMU-camera calibration file (from run_calibration.py)')

    args = parser.parse_args()

    # Determine FOV values (priority: command line > calibration file > defaults)
    fov_h = args.fov_h
    fov_v = args.fov_v
    calibration_data = None

    if args.calibration:
        try:
            calibration_data = load_calibration(args.calibration)
            print(f"\nLoaded calibration from: {args.calibration}")
            print(f"  Camera: {calibration_data.get('camera_name', 'Unknown')}")
            print(f"  Resolution: {calibration_data.get('image_width', '?')}x{calibration_data.get('image_height', '?')}")

            # Use FOV from calibration if not overridden by command line
            if fov_h is None:
                fov_h = calibration_data.get('fov_horizontal')
            if fov_v is None:
                fov_v = calibration_data.get('fov_vertical')

            if fov_h and fov_v:
                print(f"  FOV: {fov_h:.1f}° x {fov_v:.1f}° (from calibration)")

            # Show distortion info
            k1 = calibration_data.get('k1', 0)
            if abs(k1) > 1e-6:
                print(f"  Distortion: k1={k1:.4f} (barrel)" if k1 < 0 else f"  Distortion: k1={k1:.4f} (pincushion)")

        except Exception as e:
            print(f"Warning: Failed to load calibration: {e}")
            print("Using default FOV values")

    # Apply defaults if still not set
    if fov_h is None:
        fov_h = 60.0
    if fov_v is None:
        fov_v = 45.0

    print(f"\nUsing FOV: {fov_h:.1f}° horizontal x {fov_v:.1f}° vertical")

    viewer = CelestialSphere3D(
        width=args.width,
        height=args.height,
        fov_h=fov_h,
        fov_v=fov_v
    )

    # Store calibration data for potential future use (e.g., distortion correction)
    viewer.calibration_data = calibration_data

    # Load IMU-camera calibration if specified
    if args.imu_cal:
        viewer.load_imu_calibration(args.imu_cal)

    # Start Orange Cube reader if port specified
    if args.port:
        viewer.start_imu_reader(args.port, args.baudrate)

    # Parse camera resolution if specified
    cam_resolution = None
    if args.cam_res:
        try:
            cam_w, cam_h = map(int, args.cam_res.lower().split('x'))
            cam_resolution = (cam_w, cam_h)
        except:
            print(f"Invalid camera resolution: {args.cam_res}")

    # Start camera if specified, or mock if --mock flag
    if args.camera is not None:
        viewer.start_camera(args.camera, cam_resolution)
    elif args.mock:
        viewer.start_mock_camera()

    # Run in demo mode if no IMU connected (simulated attitude)
    demo_mode = args.port is None

    viewer.run(demo_mode=demo_mode)


if __name__ == '__main__':
    main()

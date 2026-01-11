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
        glEnable(GL_LIGHTING)

    def _draw_camera_texture_on_fov(self, az: float, alt: float, roll: float, r: float):
        """Draw camera feed as texture on the FOV region of the sphere."""
        # Update texture with current camera frame
        glBindTexture(GL_TEXTURE_2D, self.camera_texture_id)

        # Convert BGR to RGB for OpenGL
        frame_rgb = cv2.cvtColor(self.camera_frame, cv2.COLOR_BGR2RGB)
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

        # Draw text background
        glColor4f(0.0, 0.0, 0.0, 0.6)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(250, 10)
        glVertex2f(250, 100)
        glVertex2f(10, 100)
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
        """Update attitude from Orange Cube IMU data."""
        if self.imu_reader is None or self.paused:
            return

        # Read latest attitude from reader's data
        attitude_data = self.imu_reader.attitude_data
        if attitude_data:
            with self.attitude_lock:
                # Convert from radians to degrees
                self.roll = math.degrees(attitude_data.roll)
                # Negate pitch to match coordinate system (pitch up = view up)
                self.altitude = -math.degrees(attitude_data.pitch)
                # Yaw: positive rotation = counter-clockwise from above
                self.azimuth = math.degrees(attitude_data.yaw) % 360

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

    def start_camera(self, camera_index: int):
        """Start USB camera capture."""
        if not CV2_AVAILABLE:
            print("OpenCV not available for camera capture")
            return

        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            print(f"Could not open camera {camera_index}")
            self.camera = None

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
        print("  Q/ESC - Quit")
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
                        self.camera_frame = frame
                elif self.use_mock_camera:
                    self.camera_frame = self._generate_mock_frame()

                self.render()
                clock.tick(60)  # 60 FPS

        finally:
            self.running = False

            if self.imu_reader:
                self.imu_reader.stop()

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
    parser.add_argument('--mock', action='store_true',
                        help='Use mock camera and simulated attitude (no hardware needed)')

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

    # Start Orange Cube reader if port specified
    if args.port:
        viewer.start_imu_reader(args.port, args.baudrate)

    # Start camera if specified, or mock if --mock flag
    if args.camera is not None:
        viewer.start_camera(args.camera)
    elif args.mock:
        viewer.start_mock_camera()

    # Run in demo mode if no IMU connected (simulated attitude)
    demo_mode = args.port is None

    viewer.run(demo_mode=demo_mode)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Night AR Overlay Renderer for PhotoPills-like visualization.

Renders celestial objects as AR overlays on camera feed:
- Milky Way band with Galactic Center
- Sun and Moon positions with paths
- Celestial equator
- North/South Celestial Pole (star trails center)
- Constellation grid
- Compass directions

Designed for use with Harrier 10x camera + Orange Cube IMU.
"""

import math
import cv2
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

from .celestial import (
    CelestialCalculator, CelestialPosition, EquatorialPosition,
    ObserverLocation, create_calculator
)


class OverlayLayer(Enum):
    """AR overlay layers that can be toggled."""
    MILKY_WAY = "milky_way"
    GALACTIC_CENTER = "galactic_center"
    SUN = "sun"
    MOON = "moon"
    CELESTIAL_EQUATOR = "celestial_equator"
    CELESTIAL_POLES = "celestial_poles"
    STAR_TRAILS = "star_trails"
    COMPASS = "compass"
    INFO_PANEL = "info_panel"


@dataclass
class CameraParameters:
    """Camera intrinsic parameters for projection."""
    width: int
    height: int
    hfov: float         # Horizontal FOV in degrees
    vfov: float         # Vertical FOV in degrees

    @property
    def fx(self) -> float:
        """Focal length in pixels (x)."""
        return self.width / (2 * math.tan(math.radians(self.hfov / 2)))

    @property
    def fy(self) -> float:
        """Focal length in pixels (y)."""
        return self.height / (2 * math.tan(math.radians(self.vfov / 2)))

    @property
    def cx(self) -> float:
        """Principal point x."""
        return self.width / 2

    @property
    def cy(self) -> float:
        """Principal point y."""
        return self.height / 2


@dataclass
class IMUOrientation:
    """IMU orientation in Euler angles."""
    roll: float     # Degrees, rotation about forward axis
    pitch: float    # Degrees, nose up/down
    yaw: float      # Degrees, heading (0 = North, 90 = East)


class NightARRenderer:
    """
    Augmented Reality renderer for night sky visualization.

    Overlays celestial information on camera feed based on
    IMU orientation data.
    """

    # Colors (BGR format for OpenCV)
    COLOR_MILKY_WAY = (180, 150, 100)        # Light blue-ish
    COLOR_GALACTIC_CENTER = (0, 0, 255)      # Red
    COLOR_SUN = (0, 200, 255)                # Yellow-orange
    COLOR_MOON = (200, 200, 200)             # White-gray
    COLOR_CELESTIAL_EQUATOR = (0, 255, 255)  # Yellow
    COLOR_CELESTIAL_POLE = (255, 100, 100)   # Light blue
    COLOR_STAR_TRAILS = (255, 150, 50)       # Cyan
    COLOR_COMPASS = (0, 255, 0)              # Green
    COLOR_TEXT = (255, 255, 255)             # White
    COLOR_HORIZON = (100, 100, 100)          # Gray

    def __init__(self, calculator: CelestialCalculator,
                 camera: CameraParameters):
        """
        Initialize the Night AR renderer.

        Args:
            calculator: Celestial calculator for position calculations
            camera: Camera parameters for projection
        """
        self.calculator = calculator
        self.camera = camera

        # Active layers
        self.active_layers = {layer: True for layer in OverlayLayer}

        # Time offset for planning mode
        self.time_offset = timedelta(0)

        # Star trails duration (hours)
        self.star_trails_duration = 2.0

    def set_layer_visible(self, layer: OverlayLayer, visible: bool):
        """Set visibility of an overlay layer."""
        self.active_layers[layer] = visible

    def toggle_layer(self, layer: OverlayLayer):
        """Toggle visibility of an overlay layer."""
        self.active_layers[layer] = not self.active_layers[layer]

    def set_time_offset(self, hours: float):
        """Set time offset for planning mode."""
        self.time_offset = timedelta(hours=hours)

    def get_current_time(self) -> datetime:
        """Get current time with offset applied."""
        return datetime.now(timezone.utc) + self.time_offset

    def horizontal_to_pixel(self, pos: CelestialPosition,
                            orientation: IMUOrientation) -> Optional[Tuple[int, int]]:
        """
        Convert horizontal coordinates to pixel coordinates.

        Args:
            pos: Celestial position (azimuth, altitude)
            orientation: Camera orientation from IMU

        Returns:
            Pixel coordinates (x, y) or None if outside FOV
        """
        # Calculate relative azimuth and altitude from camera pointing
        rel_az = pos.azimuth - orientation.yaw
        rel_alt = pos.altitude - orientation.pitch

        # Normalize azimuth to -180 to 180
        while rel_az > 180:
            rel_az -= 360
        while rel_az < -180:
            rel_az += 360

        # Check if within FOV
        if abs(rel_az) > self.camera.hfov / 2 + 10:  # Small margin
            return None
        if abs(rel_alt) > self.camera.vfov / 2 + 10:
            return None

        # Project to pixel coordinates
        # Apply roll rotation
        roll_rad = math.radians(orientation.roll)

        # Convert angles to radians
        az_rad = math.radians(rel_az)
        alt_rad = math.radians(rel_alt)

        # Project using tangent (simple pinhole model)
        x_norm = math.tan(az_rad)
        y_norm = -math.tan(alt_rad)  # Negative because y increases downward

        # Apply roll rotation
        x_rot = x_norm * math.cos(roll_rad) - y_norm * math.sin(roll_rad)
        y_rot = x_norm * math.sin(roll_rad) + y_norm * math.cos(roll_rad)

        # Convert to pixels
        x = int(self.camera.cx + x_rot * self.camera.fx)
        y = int(self.camera.cy + y_rot * self.camera.fy)

        # Check bounds
        if x < -100 or x > self.camera.width + 100:
            return None
        if y < -100 or y > self.camera.height + 100:
            return None

        return (x, y)

    def render(self, frame: np.ndarray, orientation: IMUOrientation) -> np.ndarray:
        """
        Render all active AR overlays on the frame.

        Args:
            frame: Input camera frame (BGR)
            orientation: Current camera orientation

        Returns:
            Frame with AR overlays
        """
        output = frame.copy()
        dt = self.get_current_time()

        # Create overlay for transparency
        overlay = frame.copy()

        # Render each active layer
        if self.active_layers[OverlayLayer.CELESTIAL_EQUATOR]:
            self._render_celestial_equator(overlay, orientation, dt)

        if self.active_layers[OverlayLayer.MILKY_WAY]:
            self._render_milky_way(overlay, orientation, dt)

        if self.active_layers[OverlayLayer.GALACTIC_CENTER]:
            self._render_galactic_center(overlay, orientation, dt)

        if self.active_layers[OverlayLayer.SUN]:
            self._render_sun(overlay, orientation, dt)

        if self.active_layers[OverlayLayer.MOON]:
            self._render_moon(overlay, orientation, dt)

        if self.active_layers[OverlayLayer.CELESTIAL_POLES]:
            self._render_celestial_poles(overlay, orientation)

        if self.active_layers[OverlayLayer.STAR_TRAILS]:
            self._render_star_trails_preview(overlay, orientation, dt)

        if self.active_layers[OverlayLayer.COMPASS]:
            self._render_compass(overlay, orientation)

        # Blend overlay with original
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)

        # Info panel on top (not blended)
        if self.active_layers[OverlayLayer.INFO_PANEL]:
            self._render_info_panel(output, orientation, dt)

        return output

    def _render_milky_way(self, frame: np.ndarray,
                          orientation: IMUOrientation, dt: datetime):
        """Render Milky Way band."""
        # Get Milky Way positions
        mw_positions = self.calculator.milky_way_band(dt, n_points=72)

        # Convert to pixels
        pixels = []
        for pos in mw_positions:
            px = self.horizontal_to_pixel(pos, orientation)
            if px is not None:
                pixels.append(px)

        # Draw as thick semi-transparent band
        if len(pixels) >= 2:
            pts = np.array(pixels, dtype=np.int32)
            cv2.polylines(frame, [pts], False, self.COLOR_MILKY_WAY, 15)
            cv2.polylines(frame, [pts], False, (255, 220, 180), 5)

    def _render_galactic_center(self, frame: np.ndarray,
                                 orientation: IMUOrientation, dt: datetime):
        """Render Galactic Center marker."""
        gc = self.calculator.galactic_center_position(dt)
        px = self.horizontal_to_pixel(gc, orientation)

        if px is not None:
            x, y = px
            # Draw prominent marker
            cv2.circle(frame, (x, y), 25, self.COLOR_GALACTIC_CENTER, 3)
            cv2.circle(frame, (x, y), 15, self.COLOR_GALACTIC_CENTER, -1)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)

            # Label
            cv2.putText(frame, "Galactic Center", (x + 30, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_GALACTIC_CENTER, 2)
            cv2.putText(frame, f"Alt: {gc.altitude:.1f}°", (x + 30, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)

    def _render_sun(self, frame: np.ndarray,
                    orientation: IMUOrientation, dt: datetime):
        """Render Sun position."""
        sun_eq, sun_horiz = self.calculator.sun_position(dt)
        px = self.horizontal_to_pixel(sun_horiz, orientation)

        if px is not None:
            x, y = px
            # Draw sun symbol
            cv2.circle(frame, (x, y), 30, self.COLOR_SUN, 3)
            cv2.circle(frame, (x, y), 20, self.COLOR_SUN, -1)

            # Rays
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                x1 = int(x + 35 * math.cos(rad))
                y1 = int(y + 35 * math.sin(rad))
                x2 = int(x + 45 * math.cos(rad))
                y2 = int(y + 45 * math.sin(rad))
                cv2.line(frame, (x1, y1), (x2, y2), self.COLOR_SUN, 2)

            cv2.putText(frame, "Sun", (x + 50, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_SUN, 2)

    def _render_moon(self, frame: np.ndarray,
                     orientation: IMUOrientation, dt: datetime):
        """Render Moon position with phase."""
        moon_eq, moon_horiz, phase = self.calculator.moon_position(dt)
        px = self.horizontal_to_pixel(moon_horiz, orientation)

        if px is not None:
            x, y = px
            # Draw moon circle
            cv2.circle(frame, (x, y), 20, self.COLOR_MOON, 2)

            # Draw phase (simplified)
            if phase.illumination > 50:
                cv2.circle(frame, (x, y), 18, self.COLOR_MOON, -1)
            else:
                # Draw crescent
                cv2.ellipse(frame, (x, y), (18, 18), 0, -90, 90, self.COLOR_MOON, -1)

            cv2.putText(frame, f"Moon ({phase.name})", (x + 25, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_MOON, 1)
            cv2.putText(frame, f"{phase.illumination:.0f}%", (x + 25, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_MOON, 1)

    def _render_celestial_equator(self, frame: np.ndarray,
                                   orientation: IMUOrientation, dt: datetime):
        """Render celestial equator line."""
        eq_positions = self.calculator.celestial_equator(dt, n_points=72)

        pixels = []
        for pos in eq_positions:
            px = self.horizontal_to_pixel(pos, orientation)
            if px is not None:
                pixels.append(px)

        if len(pixels) >= 2:
            pts = np.array(pixels, dtype=np.int32)
            cv2.polylines(frame, [pts], False, self.COLOR_CELESTIAL_EQUATOR, 2)

    def _render_celestial_poles(self, frame: np.ndarray, orientation: IMUOrientation):
        """Render celestial pole markers."""
        # North Celestial Pole
        ncp = self.calculator.celestial_pole_position(north=True)
        px_ncp = self.horizontal_to_pixel(ncp, orientation)

        if px_ncp is not None:
            x, y = px_ncp
            # Draw concentric circles (star trails center)
            for r in [10, 20, 30]:
                cv2.circle(frame, (x, y), r, self.COLOR_CELESTIAL_POLE, 1)
            cv2.drawMarker(frame, (x, y), self.COLOR_CELESTIAL_POLE,
                          cv2.MARKER_CROSS, 15, 2)
            cv2.putText(frame, "NCP", (x + 35, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_CELESTIAL_POLE, 2)

        # South Celestial Pole
        scp = self.calculator.celestial_pole_position(north=False)
        px_scp = self.horizontal_to_pixel(scp, orientation)

        if px_scp is not None:
            x, y = px_scp
            for r in [10, 20, 30]:
                cv2.circle(frame, (x, y), r, self.COLOR_CELESTIAL_POLE, 1)
            cv2.drawMarker(frame, (x, y), self.COLOR_CELESTIAL_POLE,
                          cv2.MARKER_CROSS, 15, 2)
            cv2.putText(frame, "SCP", (x + 35, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_CELESTIAL_POLE, 2)

    def _render_star_trails_preview(self, frame: np.ndarray,
                                     orientation: IMUOrientation, dt: datetime):
        """Render star trails pattern preview."""
        trails = self.calculator.star_trail_pattern(
            duration_hours=self.star_trails_duration,
            n_samples=24
        )

        for dec, trail in trails:
            pixels = []
            for pos in trail:
                px = self.horizontal_to_pixel(pos, orientation)
                if px is not None:
                    pixels.append(px)

            if len(pixels) >= 2:
                pts = np.array(pixels, dtype=np.int32)
                # Draw curved trail
                cv2.polylines(frame, [pts], False, self.COLOR_STAR_TRAILS, 1)

    def _render_compass(self, frame: np.ndarray, orientation: IMUOrientation):
        """Render compass directions at horizon."""
        directions = [
            (0, "N"), (45, "NE"), (90, "E"), (135, "SE"),
            (180, "S"), (225, "SW"), (270, "W"), (315, "NW")
        ]

        for az, label in directions:
            pos = CelestialPosition(azimuth=az, altitude=0)
            px = self.horizontal_to_pixel(pos, orientation)

            if px is not None:
                x, y = px
                color = (0, 0, 255) if label == "N" else self.COLOR_COMPASS
                cv2.putText(frame, label, (x - 10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.line(frame, (x, y + 5), (x, y + 20), color, 2)

    def _render_info_panel(self, frame: np.ndarray,
                           orientation: IMUOrientation, dt: datetime):
        """Render information panel."""
        # Semi-transparent background
        panel_height = 180
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 10 + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = 35
        line_height = 22

        # Title
        cv2.putText(frame, "Night AR - PhotoPills Style", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += line_height + 5

        # Time (with offset indicator)
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        if self.time_offset.total_seconds() != 0:
            hours = self.time_offset.total_seconds() / 3600
            time_str += f" ({hours:+.1f}h)"
        cv2.putText(frame, f"Time: {time_str}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT, 1)
        y += line_height

        # Location
        loc = self.calculator.location
        cv2.putText(frame, f"Loc: {loc.latitude:.4f}N, {abs(loc.longitude):.4f}W",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT, 1)
        y += line_height

        # Camera orientation
        cv2.putText(frame, f"Heading: {orientation.yaw:.1f}°  Pitch: {orientation.pitch:.1f}°",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT, 1)
        y += line_height

        # Galactic Center
        gc = self.calculator.galactic_center_position(dt)
        cv2.putText(frame, f"Galactic Center: Az {gc.azimuth:.1f}°, Alt {gc.altitude:.1f}°",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_GALACTIC_CENTER, 1)
        y += line_height

        # Milky Way visibility
        visible, reason = self.calculator.is_milky_way_visible(dt)
        mw_color = (0, 255, 0) if visible else (0, 0, 255)
        cv2.putText(frame, f"MW Visible: {'Yes' if visible else 'No'} - {reason[:35]}",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, mw_color, 1)

    def render_help(self, frame: np.ndarray) -> np.ndarray:
        """Render help overlay with keyboard shortcuts."""
        output = frame.copy()

        # Semi-transparent background
        overlay = np.zeros_like(frame)
        cv2.rectangle(overlay, (50, 50), (frame.shape[1] - 50, frame.shape[0] - 50),
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, output, 0.2, 0, output)

        shortcuts = [
            ("Night AR - Keyboard Shortcuts", ""),
            ("", ""),
            ("M", "Toggle Milky Way"),
            ("G", "Toggle Galactic Center"),
            ("S", "Toggle Sun"),
            ("L", "Toggle Moon (Luna)"),
            ("E", "Toggle Celestial Equator"),
            ("P", "Toggle Celestial Poles"),
            ("T", "Toggle Star Trails Preview"),
            ("C", "Toggle Compass"),
            ("I", "Toggle Info Panel"),
            ("", ""),
            ("+/-", "Adjust time offset (+/- 1 hour)"),
            ("[ / ]", "Adjust time offset (+/- 15 min)"),
            ("0", "Reset time to now"),
            ("", ""),
            ("H", "Show/Hide this help"),
            ("Q/ESC", "Quit"),
        ]

        y = 100
        for key, desc in shortcuts:
            if key == "Night AR - Keyboard Shortcuts":
                cv2.putText(output, key, (80, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif key:
                cv2.putText(output, f"[{key}]", (80, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(output, desc, (180, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 28

        return output


def create_renderer(latitude: float, longitude: float,
                    camera_width: int = 1280, camera_height: int = 720,
                    hfov: float = 50.0, vfov: float = 34.0) -> NightARRenderer:
    """
    Create a Night AR renderer.

    Args:
        latitude: Observer latitude (degrees North)
        longitude: Observer longitude (degrees East, negative for West)
        camera_width: Camera frame width
        camera_height: Camera frame height
        hfov: Horizontal field of view (degrees)
        vfov: Vertical field of view (degrees)

    Returns:
        NightARRenderer instance
    """
    calculator = create_calculator(latitude, longitude)

    camera = CameraParameters(
        width=camera_width,
        height=camera_height,
        hfov=hfov,
        vfov=vfov
    )

    return NightARRenderer(calculator, camera)

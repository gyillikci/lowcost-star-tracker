"""
Fisheye Projection Models for AllSky Cameras

Implements various fisheye projection models commonly used in
AllSky cameras with 180° field of view.
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum


class ProjectionType(Enum):
    """Fisheye projection types."""
    EQUIDISTANT = "equidistant"      # r = f * theta (linear angle)
    EQUISOLID = "equisolid"          # r = 2f * sin(theta/2)
    STEREOGRAPHIC = "stereographic"  # r = 2f * tan(theta/2)
    ORTHOGRAPHIC = "orthographic"    # r = f * sin(theta)
    RECTILINEAR = "rectilinear"      # r = f * tan(theta) (standard)


@dataclass
class FisheyeProjection:
    """
    Fisheye lens projection model.

    Handles conversion between:
    - Image coordinates (pixels)
    - Normalized coordinates (unit sphere)
    - Alt/Az coordinates (horizon system)
    - RA/Dec coordinates (equatorial system)
    """
    # Camera parameters
    focal_length: float      # Effective focal length (pixels)
    cx: float               # Principal point X (pixels)
    cy: float               # Principal point Y (pixels)

    # Projection type
    projection: ProjectionType = ProjectionType.EQUIDISTANT

    # Field of view
    fov_degrees: float = 180.0

    def pixel_to_angle(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to angular coordinates.

        Args:
            x, y: Pixel coordinates

        Returns:
            (theta, phi) where:
                theta = angle from optical axis (0 to FOV/2)
                phi = azimuthal angle around optical axis
        """
        # Offset from principal point
        dx = x - self.cx
        dy = y - self.cy
        r = math.hypot(dx, dy)

        # Azimuthal angle
        phi = math.atan2(dy, dx)

        # Zenith angle depends on projection type
        if self.projection == ProjectionType.EQUIDISTANT:
            theta = r / self.focal_length
        elif self.projection == ProjectionType.EQUISOLID:
            theta = 2 * math.asin(np.clip(r / (2 * self.focal_length), -1, 1))
        elif self.projection == ProjectionType.STEREOGRAPHIC:
            theta = 2 * math.atan(r / (2 * self.focal_length))
        elif self.projection == ProjectionType.ORTHOGRAPHIC:
            theta = math.asin(np.clip(r / self.focal_length, -1, 1))
        else:  # RECTILINEAR
            theta = math.atan(r / self.focal_length)

        return theta, phi

    def angle_to_pixel(self, theta: float, phi: float) -> Tuple[float, float]:
        """
        Convert angular coordinates to pixel coordinates.

        Args:
            theta: Angle from optical axis (radians)
            phi: Azimuthal angle (radians)

        Returns:
            (x, y) pixel coordinates
        """
        # Radial distance depends on projection type
        if self.projection == ProjectionType.EQUIDISTANT:
            r = self.focal_length * theta
        elif self.projection == ProjectionType.EQUISOLID:
            r = 2 * self.focal_length * math.sin(theta / 2)
        elif self.projection == ProjectionType.STEREOGRAPHIC:
            r = 2 * self.focal_length * math.tan(theta / 2)
        elif self.projection == ProjectionType.ORTHOGRAPHIC:
            r = self.focal_length * math.sin(theta)
        else:  # RECTILINEAR
            r = self.focal_length * math.tan(theta)

        x = self.cx + r * math.cos(phi)
        y = self.cy + r * math.sin(phi)

        return x, y

    def pixel_to_unit_vector(self, x: float, y: float) -> np.ndarray:
        """
        Convert pixel to unit vector pointing to that direction.

        Args:
            x, y: Pixel coordinates

        Returns:
            3D unit vector [x, y, z] where z is along optical axis
        """
        theta, phi = self.pixel_to_angle(x, y)

        return np.array([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta)
        ])

    def unit_vector_to_pixel(self, v: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Convert unit vector to pixel coordinates.

        Args:
            v: 3D unit vector

        Returns:
            (x, y) pixel coordinates or None if behind camera
        """
        # Normalize
        v = v / np.linalg.norm(v)

        # Check if in front of camera
        if v[2] < 0:
            return None

        theta = math.acos(np.clip(v[2], -1, 1))
        phi = math.atan2(v[1], v[0])

        # Check FOV
        if theta > math.radians(self.fov_degrees / 2):
            return None

        return self.angle_to_pixel(theta, phi)


@dataclass
class AllSkyCamera:
    """
    Complete AllSky camera model with location and orientation.

    Combines fisheye projection with:
    - Observer location (lat/lon)
    - Camera orientation (azimuth offset, rotation)
    - Time-dependent coordinate transforms
    """
    # Camera projection
    projection: FisheyeProjection

    # Observer location
    latitude: float = 20.02    # degrees (default: Waimea, Hawaii)
    longitude: float = -155.67  # degrees

    # Camera orientation
    azimuth_offset: float = 0.0   # Rotation of camera from North (degrees)
    rotation: float = 0.0         # Image rotation (degrees, clockwise)

    # Image dimensions
    width: int = 1920
    height: int = 1080

    @classmethod
    def create_typical_allsky(cls, width: int = 1920, height: int = 1080,
                             fov: float = 180.0,
                             lat: float = 20.02, lon: float = -155.67) -> 'AllSkyCamera':
        """
        Create a typical AllSky camera setup.

        Args:
            width: Image width
            height: Image height
            fov: Field of view in degrees
            lat: Observer latitude
            lon: Observer longitude

        Returns:
            Configured AllSkyCamera
        """
        # Calculate focal length for equidistant projection
        # For 180° FOV: radius = focal_length * pi/2
        radius = min(width, height) / 2
        focal_length = radius / (math.radians(fov / 2))

        projection = FisheyeProjection(
            focal_length=focal_length,
            cx=width / 2,
            cy=height / 2,
            projection=ProjectionType.EQUIDISTANT,
            fov_degrees=fov
        )

        return cls(
            projection=projection,
            latitude=lat,
            longitude=lon,
            width=width,
            height=height
        )

    def pixel_to_altaz(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to altitude/azimuth.

        Assumes camera is pointed at zenith with North at top.

        Args:
            x, y: Pixel coordinates

        Returns:
            (altitude, azimuth) in degrees
        """
        theta, phi = self.projection.pixel_to_angle(x, y)

        # Apply camera rotation
        phi = phi - math.radians(self.rotation)

        # Altitude is 90° - zenith_angle
        altitude = 90.0 - math.degrees(theta)

        # Azimuth from North, clockwise
        # phi=0 is positive X direction, need to convert to compass bearing
        azimuth = (90.0 - math.degrees(phi) + self.azimuth_offset) % 360

        return altitude, azimuth

    def altaz_to_pixel(self, alt: float, az: float) -> Optional[Tuple[float, float]]:
        """
        Convert altitude/azimuth to pixel coordinates.

        Args:
            alt: Altitude in degrees (0 = horizon, 90 = zenith)
            az: Azimuth in degrees (0 = North, 90 = East)

        Returns:
            (x, y) pixel coordinates or None if below horizon
        """
        if alt < 0:
            return None

        theta = math.radians(90.0 - alt)
        phi = math.radians(90.0 - az + self.azimuth_offset)
        phi = phi + math.radians(self.rotation)

        if theta > math.radians(self.projection.fov_degrees / 2):
            return None

        return self.projection.angle_to_pixel(theta, phi)

    def pixel_to_radec(self, x: float, y: float,
                       lst_hours: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to RA/Dec.

        Args:
            x, y: Pixel coordinates
            lst_hours: Local sidereal time in hours

        Returns:
            (ra, dec) in degrees
        """
        alt, az = self.pixel_to_altaz(x, y)
        return self.altaz_to_radec(alt, az, lst_hours)

    def radec_to_pixel(self, ra: float, dec: float,
                       lst_hours: float) -> Optional[Tuple[float, float]]:
        """
        Convert RA/Dec to pixel coordinates.

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            lst_hours: Local sidereal time in hours

        Returns:
            (x, y) pixel coordinates or None if not visible
        """
        alt, az = self.radec_to_altaz(ra, dec, lst_hours)
        if alt is None:
            return None
        return self.altaz_to_pixel(alt, az)

    def altaz_to_radec(self, alt: float, az: float,
                       lst_hours: float) -> Tuple[float, float]:
        """
        Convert Alt/Az to RA/Dec.

        Args:
            alt: Altitude in degrees
            az: Azimuth in degrees
            lst_hours: Local sidereal time in hours

        Returns:
            (ra, dec) in degrees
        """
        alt_rad = math.radians(alt)
        az_rad = math.radians(az)
        lat_rad = math.radians(self.latitude)

        # Calculate declination
        sin_dec = (math.sin(alt_rad) * math.sin(lat_rad) +
                   math.cos(alt_rad) * math.cos(lat_rad) * math.cos(az_rad))
        dec = math.degrees(math.asin(np.clip(sin_dec, -1, 1)))

        # Calculate hour angle
        dec_rad = math.radians(dec)
        cos_ha = ((math.sin(alt_rad) - math.sin(lat_rad) * math.sin(dec_rad)) /
                  (math.cos(lat_rad) * math.cos(dec_rad) + 1e-10))
        ha = math.degrees(math.acos(np.clip(cos_ha, -1, 1)))

        if math.sin(az_rad) > 0:
            ha = -ha

        # Convert hour angle to RA
        ra = (lst_hours * 15 - ha) % 360

        return ra, dec

    def radec_to_altaz(self, ra: float, dec: float,
                       lst_hours: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Convert RA/Dec to Alt/Az.

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            lst_hours: Local sidereal time in hours

        Returns:
            (altitude, azimuth) in degrees, or (None, None) if below horizon
        """
        # Hour angle
        ha = (lst_hours * 15 - ra) % 360
        if ha > 180:
            ha -= 360

        ha_rad = math.radians(ha)
        dec_rad = math.radians(dec)
        lat_rad = math.radians(self.latitude)

        # Calculate altitude
        sin_alt = (math.sin(dec_rad) * math.sin(lat_rad) +
                   math.cos(dec_rad) * math.cos(lat_rad) * math.cos(ha_rad))
        alt = math.degrees(math.asin(np.clip(sin_alt, -1, 1)))

        if alt < 0:
            return None, None

        # Calculate azimuth
        cos_az = ((math.sin(dec_rad) - math.sin(lat_rad) * math.sin(math.radians(alt))) /
                  (math.cos(lat_rad) * math.cos(math.radians(alt)) + 1e-10))
        az = math.degrees(math.acos(np.clip(cos_az, -1, 1)))

        if math.sin(ha_rad) > 0:
            az = 360 - az

        return alt, az

    def get_visible_sky_mask(self, min_altitude: float = 0.0) -> np.ndarray:
        """
        Create a mask of valid sky pixels.

        Args:
            min_altitude: Minimum altitude to include

        Returns:
            Boolean mask array
        """
        mask = np.zeros((self.height, self.width), dtype=bool)

        for y in range(self.height):
            for x in range(self.width):
                try:
                    alt, _ = self.pixel_to_altaz(x, y)
                    if alt >= min_altitude:
                        mask[y, x] = True
                except:
                    pass

        return mask

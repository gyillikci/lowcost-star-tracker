#!/usr/bin/env python3
"""
Celestial Calculations Module for PhotoPills-like AR.

Provides calculations for:
- Sun position (azimuth, altitude)
- Moon position and phase
- Milky Way band and Galactic Center
- Celestial poles (Polaris, SCP)
- Celestial equator
- Star trails patterns

Based on astronomical algorithms from:
- Jean Meeus "Astronomical Algorithms"
- USNO calculations
"""

import math
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np


@dataclass
class CelestialPosition:
    """Position in horizontal coordinates."""
    azimuth: float      # Degrees from North (0-360)
    altitude: float     # Degrees above horizon (-90 to +90)

    def to_direction_vector(self) -> np.ndarray:
        """Convert to 3D unit vector (North-East-Up frame)."""
        az_rad = math.radians(self.azimuth)
        alt_rad = math.radians(self.altitude)

        x = math.cos(alt_rad) * math.cos(az_rad)  # North
        y = math.cos(alt_rad) * math.sin(az_rad)  # East
        z = math.sin(alt_rad)                      # Up

        return np.array([x, y, z])


@dataclass
class EquatorialPosition:
    """Position in equatorial coordinates."""
    ra: float           # Right Ascension in degrees (0-360)
    dec: float          # Declination in degrees (-90 to +90)

    @property
    def ra_hours(self) -> float:
        """RA in hours."""
        return self.ra / 15.0

    def to_unit_vector(self) -> np.ndarray:
        """Convert to 3D unit vector in celestial frame."""
        ra_rad = math.radians(self.ra)
        dec_rad = math.radians(self.dec)

        x = math.cos(dec_rad) * math.cos(ra_rad)
        y = math.cos(dec_rad) * math.sin(ra_rad)
        z = math.sin(dec_rad)

        return np.array([x, y, z])


@dataclass
class MoonPhase:
    """Moon phase information."""
    phase: float        # 0.0 = new, 0.5 = full, 1.0 = new again
    illumination: float # Percentage illuminated (0-100)
    name: str           # Phase name


@dataclass
class ObserverLocation:
    """Observer's location on Earth."""
    latitude: float     # Degrees North (positive) / South (negative)
    longitude: float    # Degrees East (positive) / West (negative)
    elevation: float = 0.0  # Meters above sea level


class CelestialCalculator:
    """
    Calculator for celestial object positions.

    Implements astronomical algorithms for calculating positions
    of Sun, Moon, planets, and stars from any location on Earth.
    """

    # Galactic Center coordinates (J2000)
    GALACTIC_CENTER_RA = 266.405  # degrees (17h 45m 37s)
    GALACTIC_CENTER_DEC = -29.008  # degrees

    # Galactic North Pole (J2000)
    GALACTIC_NORTH_POLE_RA = 192.859  # degrees
    GALACTIC_NORTH_POLE_DEC = 27.128  # degrees

    # Polaris position (approximate, J2000)
    POLARIS_RA = 37.954  # degrees (2h 31m 49s)
    POLARIS_DEC = 89.264  # degrees

    def __init__(self, location: ObserverLocation):
        """
        Initialize calculator for observer location.

        Args:
            location: Observer's geographic location
        """
        self.location = location

    def julian_date(self, dt: datetime) -> float:
        """
        Calculate Julian Date from datetime.

        Args:
            dt: UTC datetime

        Returns:
            Julian Date
        """
        # Ensure UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        year = dt.year
        month = dt.month
        day = dt.day + dt.hour/24.0 + dt.minute/1440.0 + dt.second/86400.0

        if month <= 2:
            year -= 1
            month += 12

        A = int(year / 100)
        B = 2 - A + int(A / 4)

        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5

        return jd

    def local_sidereal_time(self, dt: datetime) -> float:
        """
        Calculate Local Sidereal Time.

        Args:
            dt: UTC datetime

        Returns:
            LST in degrees (0-360)
        """
        jd = self.julian_date(dt)

        # Julian centuries from J2000.0
        T = (jd - 2451545.0) / 36525.0

        # Greenwich Mean Sidereal Time in degrees
        gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + \
               0.000387933 * T**2 - T**3 / 38710000.0

        # Local Sidereal Time
        lst = gmst + self.location.longitude

        # Normalize to 0-360
        lst = lst % 360.0
        if lst < 0:
            lst += 360.0

        return lst

    def equatorial_to_horizontal(self, eq: EquatorialPosition,
                                  dt: datetime) -> CelestialPosition:
        """
        Convert equatorial coordinates to horizontal (azimuth/altitude).

        Args:
            eq: Equatorial position (RA, Dec)
            dt: UTC datetime

        Returns:
            Horizontal position (azimuth, altitude)
        """
        lst = self.local_sidereal_time(dt)

        # Hour angle
        ha = lst - eq.ra
        ha_rad = math.radians(ha)
        dec_rad = math.radians(eq.dec)
        lat_rad = math.radians(self.location.latitude)

        # Altitude
        sin_alt = math.sin(dec_rad) * math.sin(lat_rad) + \
                  math.cos(dec_rad) * math.cos(lat_rad) * math.cos(ha_rad)
        alt = math.degrees(math.asin(max(-1, min(1, sin_alt))))

        # Azimuth
        cos_az = (math.sin(dec_rad) - math.sin(lat_rad) * sin_alt) / \
                 (math.cos(lat_rad) * math.cos(math.radians(alt)))
        cos_az = max(-1, min(1, cos_az))
        az = math.degrees(math.acos(cos_az))

        if math.sin(ha_rad) > 0:
            az = 360.0 - az

        return CelestialPosition(azimuth=az, altitude=alt)

    def sun_position(self, dt: datetime) -> Tuple[EquatorialPosition, CelestialPosition]:
        """
        Calculate Sun position.

        Args:
            dt: UTC datetime

        Returns:
            Tuple of (equatorial, horizontal) positions
        """
        jd = self.julian_date(dt)

        # Julian centuries from J2000.0
        T = (jd - 2451545.0) / 36525.0

        # Mean longitude of the Sun
        L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2
        L0 = L0 % 360.0

        # Mean anomaly of the Sun
        M = 357.52911 + 35999.05029 * T - 0.0001537 * T**2
        M_rad = math.radians(M % 360.0)

        # Equation of center
        C = (1.914602 - 0.004817 * T - 0.000014 * T**2) * math.sin(M_rad) + \
            (0.019993 - 0.000101 * T) * math.sin(2 * M_rad) + \
            0.000289 * math.sin(3 * M_rad)

        # Sun's true longitude
        sun_lon = L0 + C

        # Obliquity of ecliptic
        epsilon = 23.439291 - 0.0130042 * T
        epsilon_rad = math.radians(epsilon)

        # Sun's right ascension and declination
        sun_lon_rad = math.radians(sun_lon)

        ra = math.degrees(math.atan2(
            math.cos(epsilon_rad) * math.sin(sun_lon_rad),
            math.cos(sun_lon_rad)
        ))
        if ra < 0:
            ra += 360.0

        dec = math.degrees(math.asin(
            math.sin(epsilon_rad) * math.sin(sun_lon_rad)
        ))

        eq = EquatorialPosition(ra=ra, dec=dec)
        horiz = self.equatorial_to_horizontal(eq, dt)

        return eq, horiz

    def moon_position(self, dt: datetime) -> Tuple[EquatorialPosition, CelestialPosition, MoonPhase]:
        """
        Calculate Moon position and phase.

        Args:
            dt: UTC datetime

        Returns:
            Tuple of (equatorial, horizontal, phase)
        """
        jd = self.julian_date(dt)
        T = (jd - 2451545.0) / 36525.0

        # Moon's mean longitude
        L = 218.3164477 + 481267.88123421 * T - 0.0015786 * T**2
        L = L % 360.0

        # Moon's mean anomaly
        M = 134.9633964 + 477198.8675055 * T + 0.0087414 * T**2
        M_rad = math.radians(M % 360.0)

        # Moon's mean elongation
        D = 297.8501921 + 445267.1114034 * T - 0.0018819 * T**2
        D_rad = math.radians(D % 360.0)

        # Sun's mean anomaly
        M_sun = 357.5291092 + 35999.0502909 * T - 0.0001536 * T**2
        M_sun_rad = math.radians(M_sun % 360.0)

        # Moon's argument of latitude
        F = 93.2720950 + 483202.0175233 * T - 0.0036539 * T**2
        F_rad = math.radians(F % 360.0)

        # Simplified longitude correction
        delta_lon = 6.289 * math.sin(M_rad) - 1.274 * math.sin(2*D_rad - M_rad) \
                   + 0.658 * math.sin(2*D_rad) - 0.214 * math.sin(2*M_rad)

        # Simplified latitude
        beta = 5.128 * math.sin(F_rad)

        # Ecliptic longitude and latitude
        lambda_moon = L + delta_lon
        beta_rad = math.radians(beta)
        lambda_rad = math.radians(lambda_moon)

        # Obliquity
        epsilon = 23.439291 - 0.0130042 * T
        epsilon_rad = math.radians(epsilon)

        # Convert to equatorial
        ra = math.degrees(math.atan2(
            math.sin(lambda_rad) * math.cos(epsilon_rad) - math.tan(beta_rad) * math.sin(epsilon_rad),
            math.cos(lambda_rad)
        ))
        if ra < 0:
            ra += 360.0

        dec = math.degrees(math.asin(
            math.sin(beta_rad) * math.cos(epsilon_rad) +
            math.cos(beta_rad) * math.sin(epsilon_rad) * math.sin(lambda_rad)
        ))

        eq = EquatorialPosition(ra=ra, dec=dec)
        horiz = self.equatorial_to_horizontal(eq, dt)

        # Moon phase
        phase_angle = D % 360.0
        phase = phase_angle / 360.0
        illumination = (1 - math.cos(math.radians(phase_angle))) / 2 * 100

        # Phase name
        if phase < 0.03 or phase > 0.97:
            phase_name = "New Moon"
        elif phase < 0.22:
            phase_name = "Waxing Crescent"
        elif phase < 0.28:
            phase_name = "First Quarter"
        elif phase < 0.47:
            phase_name = "Waxing Gibbous"
        elif phase < 0.53:
            phase_name = "Full Moon"
        elif phase < 0.72:
            phase_name = "Waning Gibbous"
        elif phase < 0.78:
            phase_name = "Last Quarter"
        else:
            phase_name = "Waning Crescent"

        moon_phase = MoonPhase(phase=phase, illumination=illumination, name=phase_name)

        return eq, horiz, moon_phase

    def galactic_center_position(self, dt: datetime) -> CelestialPosition:
        """
        Calculate Galactic Center position in horizontal coordinates.

        Args:
            dt: UTC datetime

        Returns:
            Horizontal position of Galactic Center
        """
        gc = EquatorialPosition(
            ra=self.GALACTIC_CENTER_RA,
            dec=self.GALACTIC_CENTER_DEC
        )
        return self.equatorial_to_horizontal(gc, dt)

    def polaris_position(self, dt: datetime) -> CelestialPosition:
        """
        Calculate Polaris (North Star) position.

        Args:
            dt: UTC datetime

        Returns:
            Horizontal position of Polaris
        """
        polaris = EquatorialPosition(
            ra=self.POLARIS_RA,
            dec=self.POLARIS_DEC
        )
        return self.equatorial_to_horizontal(polaris, dt)

    def celestial_pole_position(self, north: bool = True) -> CelestialPosition:
        """
        Calculate celestial pole position (constant for given location).

        The celestial poles appear at fixed positions in the sky
        based on observer latitude.

        Args:
            north: True for NCP, False for SCP

        Returns:
            Horizontal position of celestial pole
        """
        if north:
            # North Celestial Pole
            alt = self.location.latitude
            az = 0.0  # Due North
        else:
            # South Celestial Pole
            alt = -self.location.latitude
            az = 180.0  # Due South

        return CelestialPosition(azimuth=az, altitude=alt)

    def milky_way_band(self, dt: datetime, n_points: int = 36) -> List[CelestialPosition]:
        """
        Calculate positions along the Milky Way band (galactic equator).

        Args:
            dt: UTC datetime
            n_points: Number of points to sample

        Returns:
            List of horizontal positions along the Milky Way
        """
        positions = []

        for i in range(n_points):
            # Sample galactic longitude (0-360)
            gal_lon = i * 360.0 / n_points
            gal_lat = 0.0  # Galactic equator

            # Convert galactic to equatorial
            eq = self._galactic_to_equatorial(gal_lon, gal_lat)

            # Convert to horizontal
            horiz = self.equatorial_to_horizontal(eq, dt)
            positions.append(horiz)

        return positions

    def _galactic_to_equatorial(self, gal_lon: float, gal_lat: float) -> EquatorialPosition:
        """
        Convert galactic coordinates to equatorial (J2000).

        Args:
            gal_lon: Galactic longitude in degrees
            gal_lat: Galactic latitude in degrees

        Returns:
            Equatorial position
        """
        # Galactic coordinate system parameters (J2000)
        l_ncp = 122.932  # Galactic longitude of NCP
        ra_ngp = 192.859  # RA of North Galactic Pole
        dec_ngp = 27.128  # Dec of North Galactic Pole

        l_rad = math.radians(gal_lon)
        b_rad = math.radians(gal_lat)
        l_ncp_rad = math.radians(l_ncp)
        ra_ngp_rad = math.radians(ra_ngp)
        dec_ngp_rad = math.radians(dec_ngp)

        # Calculate declination
        sin_dec = math.sin(b_rad) * math.sin(dec_ngp_rad) + \
                  math.cos(b_rad) * math.cos(dec_ngp_rad) * math.sin(l_rad - l_ncp_rad)
        dec = math.degrees(math.asin(max(-1, min(1, sin_dec))))

        # Calculate right ascension
        y = math.cos(b_rad) * math.cos(l_rad - l_ncp_rad)
        x = math.sin(b_rad) * math.cos(dec_ngp_rad) - \
            math.cos(b_rad) * math.sin(dec_ngp_rad) * math.sin(l_rad - l_ncp_rad)

        ra = math.degrees(math.atan2(y, x)) + math.degrees(ra_ngp_rad)

        if ra < 0:
            ra += 360.0
        elif ra >= 360:
            ra -= 360.0

        return EquatorialPosition(ra=ra, dec=dec)

    def celestial_equator(self, dt: datetime, n_points: int = 36) -> List[CelestialPosition]:
        """
        Calculate positions along the celestial equator.

        Args:
            dt: UTC datetime
            n_points: Number of points to sample

        Returns:
            List of horizontal positions along celestial equator
        """
        positions = []

        for i in range(n_points):
            ra = i * 360.0 / n_points
            eq = EquatorialPosition(ra=ra, dec=0.0)
            horiz = self.equatorial_to_horizontal(eq, dt)
            positions.append(horiz)

        return positions

    def star_trail_center(self) -> CelestialPosition:
        """
        Get the center point for star trails (celestial pole).

        All stars appear to rotate around this point.

        Returns:
            Horizontal position of rotation center
        """
        if self.location.latitude >= 0:
            return self.celestial_pole_position(north=True)
        else:
            return self.celestial_pole_position(north=False)

    def star_trail_pattern(self, duration_hours: float = 2.0,
                          n_samples: int = 12) -> List[Tuple[float, List[CelestialPosition]]]:
        """
        Generate star trail pattern preview.

        Shows how stars at different declinations will trail.

        Args:
            duration_hours: Trail duration in hours
            n_samples: Number of time samples

        Returns:
            List of (declination, trail_positions) for different declinations
        """
        now = datetime.now(timezone.utc)
        trails = []

        # Sample stars at different declinations
        declinations = [85, 60, 30, 0, -30, -60]

        for dec in declinations:
            if self.location.latitude >= 0:
                # Northern hemisphere - use RA that's currently visible
                ra_start = self.local_sidereal_time(now)
            else:
                ra_start = self.local_sidereal_time(now) + 180

            trail = []
            for i in range(n_samples + 1):
                # Time offset
                dt_offset = timedelta(hours=duration_hours * i / n_samples)
                dt = now + dt_offset

                # Star position (RA advances with time)
                eq = EquatorialPosition(ra=ra_start, dec=dec)
                horiz = self.equatorial_to_horizontal(eq, dt)
                trail.append(horiz)

            trails.append((dec, trail))

        return trails

    def is_milky_way_visible(self, dt: datetime) -> Tuple[bool, str]:
        """
        Check if the Milky Way core is visible.

        Args:
            dt: UTC datetime

        Returns:
            Tuple of (is_visible, reason)
        """
        # Get Galactic Center position
        gc = self.galactic_center_position(dt)

        # Get Sun position
        _, sun_horiz = self.sun_position(dt)

        # Get Moon position and phase
        _, moon_horiz, moon_phase = self.moon_position(dt)

        # Check conditions
        if sun_horiz.altitude > -12:
            return False, "Sun too high (not astronomical twilight)"

        if gc.altitude < 10:
            return False, f"Galactic Center below horizon ({gc.altitude:.1f}°)"

        if moon_horiz.altitude > 0 and moon_phase.illumination > 50:
            return False, f"Bright moon above horizon ({moon_phase.illumination:.0f}% illuminated)"

        return True, f"Galactic Center at {gc.altitude:.1f}° altitude"


def create_calculator(latitude: float, longitude: float,
                      elevation: float = 0.0) -> CelestialCalculator:
    """
    Create a celestial calculator for a location.

    Args:
        latitude: Degrees North (positive) / South (negative)
        longitude: Degrees East (positive) / West (negative)
        elevation: Meters above sea level

    Returns:
        CelestialCalculator instance
    """
    location = ObserverLocation(
        latitude=latitude,
        longitude=longitude,
        elevation=elevation
    )
    return CelestialCalculator(location)


if __name__ == "__main__":
    # Test with a sample location (e.g., Los Angeles)
    calc = create_calculator(34.0522, -118.2437)

    now = datetime.now(timezone.utc)

    print("=" * 60)
    print(f"Celestial Calculations for {now.isoformat()}")
    print(f"Location: {calc.location.latitude:.4f}°N, {calc.location.longitude:.4f}°E")
    print("=" * 60)

    # Sun
    sun_eq, sun_horiz = calc.sun_position(now)
    print(f"\nSun:")
    print(f"  RA: {sun_eq.ra:.2f}°, Dec: {sun_eq.dec:.2f}°")
    print(f"  Az: {sun_horiz.azimuth:.2f}°, Alt: {sun_horiz.altitude:.2f}°")

    # Moon
    moon_eq, moon_horiz, moon_phase = calc.moon_position(now)
    print(f"\nMoon ({moon_phase.name}, {moon_phase.illumination:.0f}% illuminated):")
    print(f"  RA: {moon_eq.ra:.2f}°, Dec: {moon_eq.dec:.2f}°")
    print(f"  Az: {moon_horiz.azimuth:.2f}°, Alt: {moon_horiz.altitude:.2f}°")

    # Galactic Center
    gc = calc.galactic_center_position(now)
    print(f"\nGalactic Center:")
    print(f"  Az: {gc.azimuth:.2f}°, Alt: {gc.altitude:.2f}°")

    # Polaris
    polaris = calc.polaris_position(now)
    print(f"\nPolaris:")
    print(f"  Az: {polaris.azimuth:.2f}°, Alt: {polaris.altitude:.2f}°")

    # Celestial Pole
    ncp = calc.celestial_pole_position(north=True)
    print(f"\nNorth Celestial Pole:")
    print(f"  Az: {ncp.azimuth:.2f}°, Alt: {ncp.altitude:.2f}°")

    # Milky Way visibility
    visible, reason = calc.is_milky_way_visible(now)
    print(f"\nMilky Way Core Visible: {visible}")
    print(f"  Reason: {reason}")

    print("\n" + "=" * 60)

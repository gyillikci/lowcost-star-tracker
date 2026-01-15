"""
AllSky Camera Configuration for McAlister Observatory

Camera setup specifications:
- Observer: R. McAlister
- Location: Kamuela (Waimea), Hawaii
- Camera: ZWO ASI224MC
- Lens: 150° Wide Angle
- Computer: Raspberry Pi 4B 8GB

ZWO ASI224MC Specs (IMX224 sensor):
- Resolution: 1304 x 976 pixels
- Pixel Size: 3.75 μm
- Sensor Size: 4.8mm x 3.6mm (6.09mm diagonal)
- ADC: 12-bit
- Read Noise: 1.5e
- QE Peak: >80%
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from .fisheye_model import FisheyeProjection, ProjectionType, AllSkyCamera


# Observer Location
KAMUELA_LAT = 20.02    # degrees North
KAMUELA_LON = -155.67  # degrees West (negative for West)
KAMUELA_ALT = 800      # approximate elevation in meters


# ZWO ASI224MC Camera Specs
ASI224MC_SPECS = {
    'sensor': 'IMX224',
    'resolution': (1304, 976),        # width x height in pixels
    'pixel_size_um': 3.75,            # micrometers
    'sensor_size_mm': (4.8, 3.6),     # width x height in mm
    'diagonal_mm': 6.09,
    'adc_bits': 12,
    'read_noise_e': 1.5,
    'qe_peak': 0.80,                  # 80%
}


# 150° Wide Angle Lens
LENS_150_SPECS = {
    'fov_degrees': 150.0,
    'projection': ProjectionType.EQUIDISTANT,  # Most common for AllSky
    # Focal length calculated from FOV and sensor
    # For equidistant: r = f * theta
    # At edge: r = diagonal/2, theta = FOV/2 (in radians)
}


def create_mcalister_allsky_camera(
    width: int = 1304,
    height: int = 976,
    fov: float = 150.0
) -> AllSkyCamera:
    """
    Create camera model for McAlister AllSky setup.

    Args:
        width: Image width (default: 1304 for ASI224MC)
        height: Image height (default: 976 for ASI224MC)
        fov: Field of view in degrees (default: 150°)

    Returns:
        Configured AllSkyCamera instance
    """
    # Calculate focal length from FOV
    # For equidistant projection: r = f * theta
    # radius_pixels = f * (FOV/2 in radians)
    radius = min(width, height) / 2
    fov_rad = math.radians(fov / 2)
    focal_length = radius / fov_rad

    projection = FisheyeProjection(
        focal_length=focal_length,
        cx=width / 2,
        cy=height / 2,
        projection=ProjectionType.EQUIDISTANT,
        fov_degrees=fov
    )

    return AllSkyCamera(
        projection=projection,
        latitude=KAMUELA_LAT,
        longitude=KAMUELA_LON,
        width=width,
        height=height,
        azimuth_offset=0.0,   # Adjust if camera is rotated from North
        rotation=0.0
    )


def pixel_scale_arcsec(width: int = 1304, fov_degrees: float = 150.0) -> float:
    """
    Calculate pixel scale in arcseconds per pixel.

    For a 150° FOV across 1304 pixels:
    150° = 540,000 arcsec
    Scale = 540000 / 1304 ≈ 414 arcsec/pixel at center

    Note: For fisheye, pixel scale varies across the image.
    """
    fov_arcsec = fov_degrees * 3600
    return fov_arcsec / width


@dataclass
class McAlisterAllSkyConfig:
    """Configuration for McAlister Observatory AllSky camera."""

    # Observer
    observer: str = "R. McAlister"
    location: str = "Kamuela, Hawaii"
    latitude: float = KAMUELA_LAT
    longitude: float = KAMUELA_LON
    elevation_m: float = KAMUELA_ALT

    # Camera
    camera_model: str = "ZWO ASI224MC"
    sensor: str = "Sony IMX224"
    resolution: Tuple[int, int] = (1304, 976)
    pixel_size_um: float = 3.75

    # Lens
    lens_fov: float = 150.0  # degrees

    # Processing
    computer: str = "Raspberry Pi 4B 8GB"

    def get_camera(self) -> AllSkyCamera:
        """Create AllSkyCamera instance from this config."""
        return create_mcalister_allsky_camera(
            width=self.resolution[0],
            height=self.resolution[1],
            fov=self.lens_fov
        )

    def summary(self) -> str:
        """Get configuration summary string."""
        return f"""
McAlister AllSky Observatory Configuration
==========================================
Observer: {self.observer}
Location: {self.location}
  Latitude:  {self.latitude:.4f}°N
  Longitude: {abs(self.longitude):.4f}°W
  Elevation: {self.elevation_m:.0f}m

Camera: {self.camera_model}
  Sensor: {self.sensor}
  Resolution: {self.resolution[0]} x {self.resolution[1]}
  Pixel Size: {self.pixel_size_um} μm

Lens: {self.lens_fov}° Wide Angle
  Pixel Scale: ~{pixel_scale_arcsec(self.resolution[0], self.lens_fov):.0f} arcsec/pixel (center)

Processing: {self.computer}
"""


# Default configuration instance
DEFAULT_CONFIG = McAlisterAllSkyConfig()


def print_config():
    """Print the current configuration."""
    print(DEFAULT_CONFIG.summary())


if __name__ == '__main__':
    print_config()

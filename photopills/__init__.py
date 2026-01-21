"""
PhotoPills-like Night AR Module.

Provides augmented reality visualization of celestial objects
using camera + IMU sensor fusion.

Features:
- Milky Way band and Galactic Center position
- Sun and Moon positions with phase
- Celestial equator and poles
- Star trails pattern preview
- Time manipulation for planning

Hardware Support:
- Harrier 10x AF Zoom Camera
- Orange Cube Flight Controller (MAVLink)

Usage:
    from photopills import PhotoPillsAR, Config

    config = Config(latitude=34.05, longitude=-118.24)
    app = PhotoPillsAR(config)
    app.run()

Or run from command line:
    python -m photopills --lat 34.05 --lon -118.24
"""

from .celestial import (
    CelestialCalculator,
    CelestialPosition,
    EquatorialPosition,
    MoonPhase,
    ObserverLocation,
    create_calculator,
)

from .night_ar import (
    NightARRenderer,
    CameraParameters,
    IMUOrientation,
    OverlayLayer,
    create_renderer,
)

from .photopills_ar import (
    PhotoPillsAR,
    Config,
    OrangeCubeIMU,
    SimulatedIMU,
)

__all__ = [
    # Celestial calculations
    'CelestialCalculator',
    'CelestialPosition',
    'EquatorialPosition',
    'MoonPhase',
    'ObserverLocation',
    'create_calculator',
    # Night AR rendering
    'NightARRenderer',
    'CameraParameters',
    'IMUOrientation',
    'OverlayLayer',
    'create_renderer',
    # Main application
    'PhotoPillsAR',
    'Config',
    'OrangeCubeIMU',
    'SimulatedIMU',
]

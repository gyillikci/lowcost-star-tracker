# Plate Solving Module for Star Tracker
#
# Provides astrometric plate solving using the tetra3 library from ESA.
# Tetra3 is included locally in external/tetra3/ with numpy compatibility patches.
#
# Basic usage:
#     from src.plate_solving import Tetra3Solver, solve_image
#
#     # Quick solve
#     solution = solve_image("starfield.jpg", fov_estimate=10.0)
#
#     # Or use solver directly
#     solver = Tetra3Solver()
#     solution = solver.solve(image, fov_estimate=10.0)
#     print(f"RA: {solution.ra_hms}, Dec: {solution.dec_dms}")
#
# McAlister AllSky Configuration:
#     from src.plate_solving import McAlisterAllSkyConfig
#     config = McAlisterAllSkyConfig()
#     camera = config.get_camera()

from .tetra3_solver import (
    Tetra3Solver,
    AllSkySolver,
    PlateSolution,
    solve_image,
    check_dependencies,
    TETRA3_AVAILABLE,
)

# Fisheye models for AllSky cameras
from .fisheye_model import (
    FisheyeProjection,
    AllSkyCamera,
    ProjectionType,
)

# McAlister Observatory AllSky configuration
from .allsky_config import (
    McAlisterAllSkyConfig,
    create_mcalister_allsky_camera,
    KAMUELA_LAT,
    KAMUELA_LON,
    ASI224MC_SPECS,
    DEFAULT_CONFIG,
)

__all__ = [
    # Main solver (tetra3-based)
    'Tetra3Solver',
    'AllSkySolver',
    'PlateSolution',
    'solve_image',
    'check_dependencies',
    'TETRA3_AVAILABLE',
    # Fisheye models
    'FisheyeProjection',
    'AllSkyCamera',
    'ProjectionType',
    # McAlister AllSky config
    'McAlisterAllSkyConfig',
    'create_mcalister_allsky_camera',
    'KAMUELA_LAT',
    'KAMUELA_LON',
    'ASI224MC_SPECS',
    'DEFAULT_CONFIG',
]

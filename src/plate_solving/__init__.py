# Plate Solving Module for Star Tracker
#
# Provides astrometric plate solving using the tetra3 library from ESA.
#
# Installation:
#     pip install tetra3 opencv-python astropy
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

from .tetra3_solver import (
    Tetra3Solver,
    AllSkySolver,
    PlateSolution,
    solve_image,
    check_dependencies,
    TETRA3_AVAILABLE,
)

# Also export fisheye model for AllSky cameras
from .fisheye_model import (
    FisheyeProjection,
    AllSkyCamera,
    ProjectionType,
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
]

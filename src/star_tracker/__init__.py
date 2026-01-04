"""
Low-Cost Star Tracker
=====================

A Python library for astrophotography using consumer action cameras
with gyroscope-based motion compensation and frame stacking.

Main components:
- gyro_extractor: Extract gyroscope data from GoPro video files
- motion_compensator: Apply gyro-based frame stabilization
- frame_extractor: Extract frames from stabilized video
- star_detector: Detect stars in frames
- quality_assessor: Assess frame quality for stacking
- frame_aligner: Sub-pixel frame alignment using star positions
- stacker: Combine aligned frames using various algorithms
- pipeline: Orchestrate the complete processing workflow
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .config import Config
from .pipeline import Pipeline

__all__ = [
    "Config",
    "Pipeline",
    "__version__",
]

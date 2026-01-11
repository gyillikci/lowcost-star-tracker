"""
Low-Cost Star Tracker
=====================

A Python library for astrophotography using consumer action cameras
with gyroscope-based motion compensation and frame stacking.

Main components:
- gyro_extractor: Extract gyroscope data from GoPro video files (with VQF sensor fusion)
- vqf_integrator: VQF algorithm for gyro+accel sensor fusion
- motion_compensator: Apply gyro-based frame stabilization (with rolling shutter correction)
- lens_profile: Lens profile loading and fisheye distortion handling
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
from .gyro_extractor import GyroExtractor, GyroData
from .motion_compensator import MotionCompensator, CameraIntrinsics
from .vqf_integrator import VQFIntegrator, VQFParams, vqf_offline
from .lens_profile import LensProfile, load_lens_profile, find_lens_profile, FisheyeDistortion

__all__ = [
    "Config",
    "Pipeline",
    "GyroExtractor",
    "GyroData",
    "MotionCompensator",
    "CameraIntrinsics",
    "VQFIntegrator",
    "VQFParams",
    "vqf_offline",
    "LensProfile",
    "load_lens_profile",
    "find_lens_profile",
    "FisheyeDistortion",
    "__version__",
]

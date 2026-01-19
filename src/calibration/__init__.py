"""
Camera-IMU Calibration Toolbox for Star Tracker.

A Python library for joint calibration of camera and IMU sensors,
inspired by CRISP but tailored for star tracker applications.

Features:
- Temporal calibration: Time offset between camera and IMU
- Spatial calibration: Rotation alignment between coordinate frames
- Gyroscope bias estimation
- Rolling shutter compensation
- Star-based calibration refinement

Main Classes:
- GyroStream: IMU data handling and preprocessing
- VideoStream: Video frame management
- CameraModel: Camera intrinsics and distortion
- TemporalCalibrator: Time synchronization
- SpatialCalibrator: Rotation alignment
- AutoCalibrator: Full calibration pipeline
"""

from .gyro_stream import GyroStream, IMUData
from .video_stream import VideoStream, FrameData
from .camera_model import CameraModel, PinholeCamera, FisheyeCamera, CameraIntrinsics
from .temporal_calibrator import TemporalCalibrator
from .spatial_calibrator import SpatialCalibrator
from .auto_calibrator import AutoCalibrator

__version__ = "1.0.0"
__all__ = [
    "GyroStream",
    "IMUData",
    "VideoStream",
    "FrameData",
    "CameraModel",
    "CameraIntrinsics",
    "PinholeCamera",
    "FisheyeCamera",
    "TemporalCalibrator",
    "SpatialCalibrator",
    "AutoCalibrator",
]

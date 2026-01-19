"""
Motion Deblur Module for Star Tracker Images.

This module provides tools for compensating motion blur in star field
images using IMU attitude data. It handles the challenging case of
star trail overlap when frame shifts significantly during exposure.

Key Components:
- synthetic_data: Generate synthetic star fields and IMU motion data
- psf_generator: Create PSFs from IMU quaternion trajectories
- motion_deblur: Core deblurring algorithms with overlap handling
- demo: Demonstration and testing utilities

Usage Example:
    from motion_deblur import MotionDeblur, IMUData, DeblurParams

    # Load your IMU data
    imu_data = IMUData(timestamps, quaternions)

    # Create deblur processor
    deblur = MotionDeblur(width=1920, height=1080, focal_length_px=1200)

    # Deblur image with overlap handling
    params = DeblurParams(method='richardson_lucy', iterations=30)
    result, metadata = deblur.deblur(blurred_image, imu_data, params)
"""

from .synthetic_data import (
    Star,
    IMUData,
    SyntheticStarField,
    IMUMotionSimulator,
    MotionBlurRenderer,
    generate_test_dataset
)

from .psf_generator import (
    PSFParams,
    MotionPSFGenerator,
    OverlapAwarePSFGenerator,
    estimate_psf_from_imu_file,
    visualize_psf
)

from .motion_deblur import (
    DeblurParams,
    MotionDeblur,
    compute_quality_metrics
)

__all__ = [
    # Data classes
    'Star',
    'IMUData',
    'PSFParams',
    'DeblurParams',

    # Synthetic data generation
    'SyntheticStarField',
    'IMUMotionSimulator',
    'MotionBlurRenderer',
    'generate_test_dataset',

    # PSF generation
    'MotionPSFGenerator',
    'OverlapAwarePSFGenerator',
    'estimate_psf_from_imu_file',
    'visualize_psf',

    # Deblurring
    'MotionDeblur',
    'compute_quality_metrics',
]

__version__ = '1.0.0'

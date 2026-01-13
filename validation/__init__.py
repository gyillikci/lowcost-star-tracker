"""
Validation module for Low-Cost Star Tracker.

This module provides validation tests and benchmarks for the star tracker
algorithms using synthetic data.

Usage:
    python -m validation.validation_framework  # Run all tests
    python -m validation.generate_validation_plots  # Generate plots
"""

from .validation_framework import (
    ValidationResult,
    Star,
    SyntheticStarField,
    CentroidDetector,
    validate_centroid_accuracy,
    validate_snr_scaling,
    validate_processing_performance,
    validate_motion_compensation,
    run_all_validations
)

__all__ = [
    'ValidationResult',
    'Star',
    'SyntheticStarField',
    'CentroidDetector',
    'validate_centroid_accuracy',
    'validate_snr_scaling',
    'validate_processing_performance',
    'validate_motion_compensation',
    'run_all_validations'
]

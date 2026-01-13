"""
Low-Cost Star Tracker Algorithm Modules.

Phase 2 Algorithm Enhancements:
- drift_compensation: Gyroscope drift compensation with star-aided correction
- optical_calibration: Dark frame, flat field, and bad pixel calibration
- triangle_matching: Robust triangle-based star identification
"""

from .drift_compensation import (
    DriftCompensator,
    GyroParameters,
    StarMatch as DriftStarMatch,
    AdaptiveBiasEstimator
)

from .optical_calibration import (
    OpticalCalibrator,
    DarkFrameCalibrator,
    FlatFieldCalibrator,
    BadPixelCorrector,
    CalibrationConfig,
    CalibrationData
)

from .triangle_matching import (
    TriangleMatcher,
    FalseStarFilter,
    SparseFieldMatcher,
    ConfidenceMetrics,
    DetectedStar,
    CatalogStar,
    StarMatch,
    MatchResult
)

__all__ = [
    # Drift compensation
    'DriftCompensator',
    'GyroParameters',
    'DriftStarMatch',
    'AdaptiveBiasEstimator',

    # Optical calibration
    'OpticalCalibrator',
    'DarkFrameCalibrator',
    'FlatFieldCalibrator',
    'BadPixelCorrector',
    'CalibrationConfig',
    'CalibrationData',

    # Triangle matching
    'TriangleMatcher',
    'FalseStarFilter',
    'SparseFieldMatcher',
    'ConfidenceMetrics',
    'DetectedStar',
    'CatalogStar',
    'StarMatch',
    'MatchResult'
]

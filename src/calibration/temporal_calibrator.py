#!/usr/bin/env python3
"""
Temporal Calibrator Module.

Estimates the time offset between camera and IMU data streams.
This is critical for accurate sensor fusion as even small timing
errors (tens of milliseconds) can significantly degrade performance.

Methods:
- Cross-correlation of angular velocities
- Phase correlation in frequency domain
- Exhaustive search with rotation error metric
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.signal import correlate, correlation_lags
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import interp1d


@dataclass
class TemporalCalibrationResult:
    """Result of temporal calibration."""
    time_offset: float  # Camera time = IMU time + offset (seconds)
    confidence: float  # Calibration confidence (0-1)
    correlation: float  # Peak cross-correlation value
    search_range: Tuple[float, float]  # Search range used
    method: str  # Calibration method used


class TemporalCalibrator:
    """
    Temporal calibration between camera and IMU.

    Determines the time offset between the two sensors by
    correlating their angular velocity signals.
    """

    def __init__(self,
                 search_range: Tuple[float, float] = (-0.5, 0.5),
                 search_resolution: float = 0.001):
        """
        Initialize temporal calibrator.

        Args:
            search_range: Time offset search range in seconds (min, max)
            search_resolution: Initial search resolution in seconds
        """
        self.search_range = search_range
        self.search_resolution = search_resolution

        # Results
        self.result: Optional[TemporalCalibrationResult] = None

    def calibrate_cross_correlation(self,
                                     gyro_times: np.ndarray,
                                     gyro_data: np.ndarray,
                                     video_times: np.ndarray,
                                     video_angular_velocity: np.ndarray) -> TemporalCalibrationResult:
        """
        Calibrate using cross-correlation of angular velocities.

        Args:
            gyro_times: Gyroscope timestamps
            gyro_data: Gyroscope angular velocity (N, 3)
            video_times: Video-derived timestamps
            video_angular_velocity: Video-derived angular velocity (M, 3)

        Returns:
            TemporalCalibrationResult with time offset
        """
        # Resample both signals to common rate
        common_rate = 100.0  # Hz

        # Find common time range
        t_start = max(gyro_times[0], video_times[0])
        t_end = min(gyro_times[-1], video_times[-1])

        if t_end <= t_start:
            raise ValueError("No overlapping time range between gyro and video")

        # Create common time base
        n_samples = int((t_end - t_start) * common_rate)
        common_times = np.linspace(t_start, t_end, n_samples)

        # Interpolate gyro
        gyro_interp = interp1d(gyro_times, gyro_data, axis=0,
                               bounds_error=False, fill_value='extrapolate')
        gyro_resampled = gyro_interp(common_times)

        # Interpolate video
        video_interp = interp1d(video_times, video_angular_velocity, axis=0,
                                bounds_error=False, fill_value='extrapolate')
        video_resampled = video_interp(common_times)

        # Compute cross-correlation for each axis
        correlations = []
        lags_samples = []

        for axis in range(3):
            # Normalize signals
            g = gyro_resampled[:, axis]
            v = video_resampled[:, axis]

            g = (g - np.mean(g)) / (np.std(g) + 1e-10)
            v = (v - np.mean(v)) / (np.std(v) + 1e-10)

            # Cross-correlation
            corr = correlate(g, v, mode='full')
            lags = correlation_lags(len(g), len(v), mode='full')

            correlations.append(corr)
            lags_samples.append(lags)

        # Sum correlations across axes
        total_corr = sum(correlations)
        lags_time = lags_samples[0] / common_rate

        # Find peak within search range
        valid_mask = (lags_time >= self.search_range[0]) & (lags_time <= self.search_range[1])

        if not np.any(valid_mask):
            raise ValueError("Search range contains no valid lags")

        valid_corr = total_corr.copy()
        valid_corr[~valid_mask] = -np.inf

        peak_idx = np.argmax(valid_corr)
        time_offset = lags_time[peak_idx]
        peak_correlation = total_corr[peak_idx] / len(common_times)

        # Compute confidence based on peak sharpness
        confidence = self._compute_correlation_confidence(total_corr, peak_idx)

        self.result = TemporalCalibrationResult(
            time_offset=time_offset,
            confidence=confidence,
            correlation=peak_correlation,
            search_range=self.search_range,
            method="cross_correlation"
        )

        return self.result

    def calibrate_exhaustive_search(self,
                                     gyro_stream,  # GyroStream
                                     video_stream,  # VideoStream
                                     camera_model=None) -> TemporalCalibrationResult:
        """
        Calibrate by exhaustive search minimizing rotation error.

        For each candidate offset, compute rotation from video features
        and compare to integrated gyro. Find offset that minimizes error.

        Args:
            gyro_stream: GyroStream with IMU data
            video_stream: VideoStream with tracked features
            camera_model: Camera model for projection (optional)

        Returns:
            TemporalCalibrationResult with optimal time offset
        """
        # Generate candidate offsets
        n_candidates = int((self.search_range[1] - self.search_range[0]) / self.search_resolution)
        candidate_offsets = np.linspace(
            self.search_range[0], self.search_range[1], n_candidates
        )

        # Compute video-derived angular velocity
        video_times, video_omega = video_stream.get_angular_velocity_from_video()

        if len(video_omega) < 5:
            raise ValueError("Insufficient video angular velocity data")

        # Search for best offset
        errors = []

        for offset in candidate_offsets:
            error = self._compute_rotation_error(
                gyro_stream, video_times, video_omega, offset
            )
            errors.append(error)

        errors = np.array(errors)

        # Find minimum
        best_idx = np.argmin(errors)
        coarse_offset = candidate_offsets[best_idx]

        # Refine with optimization
        def objective(offset):
            return self._compute_rotation_error(
                gyro_stream, video_times, video_omega, offset
            )

        result = minimize_scalar(
            objective,
            bounds=(coarse_offset - 0.05, coarse_offset + 0.05),
            method='bounded'
        )

        time_offset = result.x
        min_error = result.fun

        # Compute confidence
        error_range = np.max(errors) - np.min(errors)
        confidence = 1.0 - (min_error / (error_range + 1e-10))
        confidence = np.clip(confidence, 0, 1)

        self.result = TemporalCalibrationResult(
            time_offset=time_offset,
            confidence=confidence,
            correlation=1.0 - min_error,  # Inverse of error
            search_range=self.search_range,
            method="exhaustive_search"
        )

        return self.result

    def calibrate_phase_correlation(self,
                                     gyro_times: np.ndarray,
                                     gyro_data: np.ndarray,
                                     video_times: np.ndarray,
                                     video_angular_velocity: np.ndarray) -> TemporalCalibrationResult:
        """
        Calibrate using phase correlation in frequency domain.

        More robust to amplitude differences between signals.
        """
        # Resample to common rate
        common_rate = 100.0
        t_start = max(gyro_times[0], video_times[0])
        t_end = min(gyro_times[-1], video_times[-1])
        n_samples = int((t_end - t_start) * common_rate)

        # Pad to power of 2 for FFT efficiency
        n_fft = 2 ** int(np.ceil(np.log2(n_samples)))

        common_times = np.linspace(t_start, t_end, n_samples)

        # Interpolate
        gyro_interp = interp1d(gyro_times, gyro_data, axis=0,
                               bounds_error=False, fill_value=0)
        video_interp = interp1d(video_times, video_angular_velocity, axis=0,
                                bounds_error=False, fill_value=0)

        gyro_resampled = gyro_interp(common_times)
        video_resampled = video_interp(common_times)

        # Compute phase correlation for each axis
        phase_correlations = []

        for axis in range(3):
            g = np.zeros(n_fft)
            v = np.zeros(n_fft)

            g[:n_samples] = gyro_resampled[:, axis]
            v[:n_samples] = video_resampled[:, axis]

            # FFT
            G = np.fft.fft(g)
            V = np.fft.fft(v)

            # Normalized cross-power spectrum
            cross_power = G * np.conj(V)
            cross_power_norm = cross_power / (np.abs(cross_power) + 1e-10)

            # Inverse FFT gives phase correlation
            phase_corr = np.fft.ifft(cross_power_norm).real

            phase_correlations.append(phase_corr)

        # Sum across axes
        total_phase_corr = sum(phase_correlations)

        # Find peak
        peak_idx = np.argmax(total_phase_corr[:n_fft // 2])

        # Convert to time offset
        if peak_idx < n_fft // 2:
            time_offset = peak_idx / common_rate
        else:
            time_offset = (peak_idx - n_fft) / common_rate

        # Check if within search range
        if time_offset < self.search_range[0] or time_offset > self.search_range[1]:
            # Search within valid range
            valid_range_samples = (
                int(self.search_range[0] * common_rate),
                int(self.search_range[1] * common_rate)
            )
            # Handle negative indices
            valid_indices = np.concatenate([
                np.arange(0, valid_range_samples[1]),
                np.arange(n_fft + valid_range_samples[0], n_fft)
            ])
            valid_indices = valid_indices[valid_indices < n_fft]

            peak_idx = valid_indices[np.argmax(total_phase_corr[valid_indices])]
            time_offset = peak_idx / common_rate if peak_idx < n_fft // 2 else (peak_idx - n_fft) / common_rate

        peak_value = total_phase_corr[peak_idx % n_fft]
        confidence = peak_value / 3.0  # Normalize by number of axes

        self.result = TemporalCalibrationResult(
            time_offset=time_offset,
            confidence=np.clip(confidence, 0, 1),
            correlation=peak_value,
            search_range=self.search_range,
            method="phase_correlation"
        )

        return self.result

    def _compute_rotation_error(self,
                                 gyro_stream,
                                 video_times: np.ndarray,
                                 video_omega: np.ndarray,
                                 offset: float) -> float:
        """
        Compute rotation error for a given time offset.

        Args:
            gyro_stream: GyroStream with IMU data
            video_times: Video timestamps
            video_omega: Video-derived angular velocity
            offset: Time offset to test (video_time = gyro_time + offset)

        Returns:
            Mean squared rotation error
        """
        # Shift video times by offset
        shifted_times = video_times - offset

        # Get gyro data at shifted times
        try:
            gyro_omega = gyro_stream.get_corrected_gyro(shifted_times)
        except Exception:
            return np.inf

        # Compute error
        diff = video_omega - gyro_omega
        error = np.mean(np.sum(diff**2, axis=1))

        return error

    def _compute_correlation_confidence(self,
                                         correlation: np.ndarray,
                                         peak_idx: int) -> float:
        """
        Compute confidence based on peak prominence.

        Higher confidence when peak is sharp and well-defined.
        """
        peak_value = correlation[peak_idx]

        # Compute mean and std of correlation (excluding peak region)
        window = max(10, len(correlation) // 50)
        mask = np.ones(len(correlation), dtype=bool)
        mask[max(0, peak_idx - window):min(len(correlation), peak_idx + window)] = False

        if np.any(mask):
            mean_corr = np.mean(correlation[mask])
            std_corr = np.std(correlation[mask])
        else:
            mean_corr = np.mean(correlation)
            std_corr = np.std(correlation)

        # Confidence based on peak prominence
        if std_corr > 0:
            prominence = (peak_value - mean_corr) / std_corr
            confidence = 1.0 - np.exp(-prominence / 5.0)
        else:
            confidence = 0.5

        return np.clip(confidence, 0, 1)


def demonstrate_temporal_calibration():
    """Demonstrate temporal calibration with synthetic data."""
    print("=" * 60)
    print("Temporal Calibration Demonstration")
    print("=" * 60)

    # Generate synthetic gyro data
    np.random.seed(42)
    duration = 10.0
    gyro_rate = 200.0
    video_rate = 30.0

    # True time offset (video lags gyro by 0.05 seconds)
    true_offset = 0.05

    gyro_times = np.arange(0, duration, 1.0 / gyro_rate)

    # Synthetic angular velocity (sinusoids + noise)
    gyro_data = np.column_stack([
        0.5 * np.sin(2 * np.pi * 0.5 * gyro_times),
        0.3 * np.cos(2 * np.pi * 0.7 * gyro_times),
        0.2 * np.sin(2 * np.pi * 1.0 * gyro_times)
    ])
    gyro_data += np.random.normal(0, 0.02, gyro_data.shape)

    # Video-derived angular velocity (same signal, shifted and noisier)
    video_times = np.arange(0, duration, 1.0 / video_rate)

    # Apply true offset
    video_times_shifted = video_times + true_offset

    gyro_interp = interp1d(gyro_times, gyro_data, axis=0,
                           bounds_error=False, fill_value='extrapolate')
    video_omega = gyro_interp(video_times_shifted)
    video_omega += np.random.normal(0, 0.05, video_omega.shape)

    # Calibrate
    calibrator = TemporalCalibrator(search_range=(-0.2, 0.2))

    print("\n1. Cross-correlation method:")
    result1 = calibrator.calibrate_cross_correlation(
        gyro_times, gyro_data, video_times, video_omega
    )
    print(f"   Estimated offset: {result1.time_offset*1000:.2f} ms")
    print(f"   True offset: {true_offset*1000:.2f} ms")
    print(f"   Error: {abs(result1.time_offset - true_offset)*1000:.2f} ms")
    print(f"   Confidence: {result1.confidence:.3f}")

    print("\n2. Phase correlation method:")
    result2 = calibrator.calibrate_phase_correlation(
        gyro_times, gyro_data, video_times, video_omega
    )
    print(f"   Estimated offset: {result2.time_offset*1000:.2f} ms")
    print(f"   Error: {abs(result2.time_offset - true_offset)*1000:.2f} ms")
    print(f"   Confidence: {result2.confidence:.3f}")

    return calibrator


if __name__ == "__main__":
    demonstrate_temporal_calibration()

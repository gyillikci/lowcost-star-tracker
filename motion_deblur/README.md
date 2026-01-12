# Motion Deblur Module for Star Tracker Images

This module provides motion blur compensation for star field images using IMU attitude data from Orange Cube or similar flight controllers.

## Overview

During long-exposure astrophotography, camera motion causes star trails (motion blur). This module uses synchronized IMU data to:

1. **Generate PSFs** (Point Spread Functions) from quaternion trajectories
2. **Deconvolve** the blurred image using Richardson-Lucy or Wiener deconvolution
3. **Handle star trail overlap** when frame shifts significantly

## The Overlap Problem

When the camera moves significantly during exposure (e.g., >30% of frame width), a critical corner case occurs:

```
Frame shift to the right:
+----------------------------------+
|  <---Star A (exits left)         |
|       ============>              |
|                                  |
|                  Star B (enters from right)-->
|                  ============>   |
+----------------------------------+

The END of Star A's trail can overlap with the BEGINNING of Star B's trail!
```

This module detects and handles this overlap by:
- Segmenting the exposure into time windows
- Processing overlapping regions with time-windowed PSFs
- Blending results smoothly

## Installation

```bash
# Requires these dependencies:
pip install numpy scipy opencv-python matplotlib
```

## Quick Start

```python
from motion_deblur import (
    MotionDeblur,
    IMUData,
    DeblurParams,
    SyntheticStarField,
    IMUMotionSimulator
)

# Load IMU data from Orange Cube (or use synthetic)
motion_sim = IMUMotionSimulator(duration=5.0, sample_rate=200)
imu_data = motion_sim.generate_combined_motion(
    drift_rate=0.3,  # deg/s
    vib_frequencies=[5.0, 12.0],
    vib_amplitudes=[0.05, 0.02]
)

# Create deblur processor
deblur = MotionDeblur(
    image_width=1920,
    image_height=1080,
    focal_length_px=1200
)

# Deblur with overlap handling
params = DeblurParams(
    method='richardson_lucy',
    iterations=25,
    handle_overlap=True
)
result, metadata = deblur.deblur(blurred_image, imu_data, params)
```

## Module Structure

```
motion_deblur/
├── __init__.py          # Package exports
├── synthetic_data.py    # Star field and IMU motion generation
├── psf_generator.py     # PSF from quaternion trajectory
├── motion_deblur.py     # Core deblurring algorithms
├── demo.py              # Demo and visualization
└── README.md            # This file
```

## Components

### 1. Synthetic Data Generation (`synthetic_data.py`)

Generate test data for algorithm development:

```python
from motion_deblur import SyntheticStarField, IMUMotionSimulator, MotionBlurRenderer

# Create star field
star_field = SyntheticStarField(
    width=1920, height=1080,
    num_stars=200,
    min_magnitude=1.0,
    max_magnitude=8.0,
    star_fwhm=2.5
)

# Simulate IMU motion
motion_sim = IMUMotionSimulator(duration=10.0, sample_rate=200)

# Motion patterns:
imu_drift = motion_sim.generate_drift_motion(drift_rate=0.5)
imu_vibration = motion_sim.generate_vibration_motion()
imu_combined = motion_sim.generate_combined_motion()

# Render blurred image
renderer = MotionBlurRenderer(1920, 1080, focal_length_px=1200)
blurred, metadata = renderer.render_blurred_image(star_field, imu_data)
```

### 2. PSF Generation (`psf_generator.py`)

Create motion blur PSFs from IMU quaternion data:

```python
from motion_deblur import MotionPSFGenerator, OverlapAwarePSFGenerator, PSFParams

# Basic PSF generation
psf_gen = MotionPSFGenerator(1920, 1080, focal_length_px=1200)
params = PSFParams(kernel_size=101, star_fwhm=2.5)

# Center PSF (assumes uniform blur)
psf = psf_gen.generate_uniform_psf(imu_data, params)

# Spatially-varying PSF grid
psfs, x_pos, y_pos = psf_gen.generate_psf_grid(imu_data, grid_size=(3, 3))

# Overlap-aware PSF with analysis
overlap_gen = OverlapAwarePSFGenerator(1920, 1080, 1200)
psf, overlap_info = overlap_gen.generate_overlap_aware_psf((100, 540), imu_data)
```

### 3. Motion Deblur (`motion_deblur.py`)

Core deblurring algorithms:

```python
from motion_deblur import MotionDeblur, DeblurParams

deblur = MotionDeblur(1920, 1080, 1200)

# Richardson-Lucy (iterative, handles noise well)
params = DeblurParams(
    method='richardson_lucy',
    iterations=30,
    handle_overlap=True
)
result, meta = deblur.deblur(image, imu_data, params)

# Wiener filter (fast, frequency domain)
params = DeblurParams(
    method='wiener',
    wiener_k=0.01,
    handle_overlap=True
)
result, meta = deblur.deblur(image, imu_data, params)

# Spatially-varying (for wide-field)
params = DeblurParams(
    method='spatially_varying',
    iterations=20
)
result, meta = deblur.deblur(image, imu_data, params)
```

## Loading Orange Cube IMU Data

To use real IMU data from Orange Cube:

```python
import numpy as np
from motion_deblur import IMUData

# Load from MAVLink log (example format)
data = np.load('imu_log.npz')
imu_data = IMUData(
    timestamps=data['timestamps'],      # Seconds from exposure start
    quaternions=data['quaternions'],    # [w, x, y, z] format
    angular_velocity=data['gyro']       # Optional: [wx, wy, wz] rad/s
)

# Ensure timestamps start at 0
imu_data.timestamps = imu_data.timestamps - imu_data.timestamps[0]
```

## Parameters

### DeblurParams

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | 'richardson_lucy' | 'richardson_lucy', 'wiener', or 'spatially_varying' |
| `iterations` | 30 | Richardson-Lucy iterations |
| `wiener_k` | 0.01 | Wiener regularization (noise/signal ratio) |
| `handle_overlap` | True | Enable overlap region handling |
| `overlap_blend_width` | 50 | Overlap blending transition width (pixels) |
| `psf_kernel_size` | 51 | PSF kernel size (odd number) |

### PSFParams

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel_size` | 51 | PSF kernel size |
| `star_fwhm` | 2.5 | Star point spread FWHM (pixels) |
| `subsampling` | 3 | Anti-aliasing subpixel factor |
| `normalize` | True | Normalize PSF sum to 1 |

## Running the Demo

```bash
# Run all demos
python -m motion_deblur.demo --output ./demo_output

# Specific demos
python -m motion_deblur.demo --demo basic
python -m motion_deblur.demo --demo overlap
python -m motion_deblur.demo --demo psf
python -m motion_deblur.demo --demo spatial
```

## Quality Metrics

```python
from motion_deblur import compute_quality_metrics

metrics = compute_quality_metrics(sharp, blurred, deblurred)
print(f"PSNR improvement: {metrics['psnr_improvement']:.2f} dB")
print(f"SSIM improvement: {metrics['ssim_improvement']:.4f}")
```

## Algorithm Details

### Richardson-Lucy Deconvolution

Iterative algorithm that maximizes the likelihood of the observed image:

```
estimate(n+1) = estimate(n) * conv(image / conv(estimate, PSF), PSF_flipped)
```

Advantages:
- Handles Poisson noise well (star photon counting)
- Preserves positivity
- Converges to maximum likelihood solution

### Wiener Filter

Frequency domain deconvolution with regularization:

```
H_wiener = H* / (|H|^2 + K)
```

Where K is the noise-to-signal ratio estimate.

Advantages:
- Fast (single FFT operation)
- Explicit noise handling

### Overlap Handling

For significant frame shifts:

1. Detect overlap based on total frame displacement
2. Segment image into time-windowed regions
3. Generate time-windowed PSFs for each region
4. Deconvolve each region independently
5. Blend results with soft transitions

## References

- Richardson, W.H. (1972). "Bayesian-Based Iterative Method of Image Restoration"
- Lucy, L.B. (1974). "An iterative technique for the rectification of observed distributions"
- Wiener, N. (1949). "Extrapolation, Interpolation, and Smoothing of Stationary Time Series"

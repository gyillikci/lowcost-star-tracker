# Low-Cost Star Tracker

A Python library for astrophotography using consumer action cameras (GoPro Hero 7 Black) with gyroscope-based motion compensation and frame stacking.

## Overview

This project provides a software-based alternative to expensive equatorial tracking mounts for astrophotography. By leveraging:

1. **Gyroscope data** embedded in GoPro video files (GPMF format)
2. **Motion compensation** algorithms similar to Gyroflow
3. **Frame stacking** techniques from traditional astrophotography

We can capture sharp star field images using affordable consumer hardware.

## Features

- Extract and process gyroscope telemetry from GoPro videos
- Apply frame-by-frame motion compensation using quaternion-based transformations
- Detect and catalog stars using source extraction algorithms
- Assess frame quality and filter unsuitable exposures
- Sub-pixel frame alignment using star position matching
- Multiple stacking methods: mean, median, sigma-clipping, winsorized mean
- Command-line interface for easy processing

## Installation

```bash
# Clone or extract the project
cd lowcost-star-tracker

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Dependencies

- Python 3.10+
- NumPy, SciPy
- OpenCV
- Astropy
- sep (Source Extractor as Python)
- FFmpeg (for video processing)

## Quick Start

### Process a video

```bash
star-tracker process video.mp4 -o output.tiff
```

### With options

```bash
star-tracker process video.mp4 \
    --method sigma_clip \
    --min-stars 15 \
    --max-fwhm 6.0 \
    --reject-fraction 0.3 \
    --verbose
```

### Analyze gyroscope data

```bash
star-tracker analyze-gyro video.mp4
```

### Analyze stars in an image

```bash
star-tracker analyze-stars image.png
```

## Python API

```python
from pathlib import Path
from star_tracker import Pipeline, Config

# Create pipeline with default config
pipeline = Pipeline()

# Or with custom config
config = Config()
config.stacking.method = "sigma_clip"
config.quality.min_stars = 15
pipeline = Pipeline(config)

# Process video
result = pipeline.process(Path("video.mp4"))

print(f"Stacked {result.num_stacked_frames} frames")
print(f"Output: {result.stacked_image_path}")
```

## Project Structure

```
lowcost-star-tracker/
├── pyproject.toml                    # Project configuration
├── README.md                         # Main documentation
├── asd                               # Temporary file
│
├── Root-level scripts/
│   ├── compare_stabilization.py      # Stabilization comparison tool
│   ├── compare_videos.py            # Video comparison utility
│   ├── convert_md_to_docx.py        # Documentation converter
│   ├── debug_stabilization.py       # Stabilization debugging
│   ├── gyro_stabilizer.py           # Gyroscope stabilization
│   ├── live_simple_star_solve.py    # Simple star solving (live)
│   ├── live_tetra3_solve.py         # Tetra3 star solving (live)
│   ├── plot_gyro.py                 # Gyroscope data visualization
│   ├── stabilize_video.py           # Video stabilization tool
│   ├── stellarium_config.py         # Stellarium configuration
│   ├── stellarium_shake.py          # Stellarium shake simulator
│   ├── stellarium_toggle_labels.py  # Stellarium UI control
│   └── test_witmotion_pywitmotion.py # IMU testing
│
├── src/
│   ├── algorithms/                  # Core algorithms
│   ├── calibration/                 # Calibration modules
│   ├── plate_solving/               # Plate solving algorithms
│   └── star_tracker/                # Main star tracker package
│
├── camera/                          # Camera and visualization tools
│   ├── celestial_sphere_3d.py       # 3D celestial sphere visualization
│   ├── celestial_sphere_viewer.py   # Celestial sphere viewer
│   ├── integrated_stabilizer.py     # Integrated stabilization system
│   └── usb_camera_viewer.py         # USB camera interface
│
├── calibration/                     # Calibration data and scripts
│
├── imu/                             # IMU integration
│   ├── __init__.py                  # IMU package initialization
│   ├── find_witmotion_windows.py    # Windows IMU detection
│   ├── pywitmotion_adapter.py       # Pywitmotion adapter
│   └── witmotion_reader.py          # Witmotion IMU reader
│
├── mavlink/                         # MAVLink integration
│   └── orange_cube_reader.py        # Orange Cube flight controller
│
├── wfb-stabilizer/                  # WFB stabilizer variants
│   ├── README.md                    # WFB documentation
│   ├── STABILIZATION_NOTES.md       # Stabilization notes
│   ├── ejo_wfb_stabilizer.py        # EJO WFB stabilizer
│   └── run_camera1_*.py             # Various camera stabilizer configs
│
├── validation/                      # Validation framework
│   ├── VALIDATION_REPORT.md         # Validation report
│   ├── __init__.py                  # Validation package
│   ├── generate_validation_plots.py # Plot generation
│   ├── validation_framework.py      # Validation framework
│   └── results/                     # Validation results
│
├── experiments/                     # Experimental code
├── motion_deblur/                   # Motion deblur algorithms
├── output/                          # Output files
│
├── external/                        # External dependencies
│   ├── pywitmotion/                 # Pywitmotion library
│   └── tetra3/                      # Tetra3 star matching
│
├── data/
│   └── lens_profiles/               # Camera calibration profiles
│
├── examples/                        # Example videos and data
│   ├── GL*.LRV                      # GoPro low-res videos
│   └── GX*.THM                      # GoPro thumbnails
│
└── docs/                            # Documentation
    ├── Development_Roadmap.md       # Project roadmap
    ├── Technical_Paper_Evaluation.md # Paper evaluation
    ├── LowCost_StarTracker_Technical_Paper.md # Technical paper
    ├── LowCost_StarTracker_Technical_Paper.docx # Paper (Word)
    ├── LowCost_StarTracker_Technical_Paper_with_images.docx
    ├── Star_Tracker_Technical_Paper (1).pdf # Paper (PDF)
    └── images/                      # Documentation images
```

## Camera Settings (GoPro Hero 7 Black)

For best results, use these Protune settings:

- **ISO Min**: 800
- **ISO Max**: 6400
- **Shutter**: 1/30s (for 30fps) or 1/24s (for 24fps)
- **White Balance**: 5500K (Native)
- **Color**: Flat
- **Lens**: Linear (recommended) or Wide

## Processing Pipeline

1. **Gyro Extraction**: Extract GPMF telemetry from video
2. **Motion Compensation**: Apply gyro-based frame stabilization
3. **Frame Extraction**: Extract individual frames
4. **Star Detection**: Detect stars in each frame
5. **Quality Assessment**: Filter unsuitable frames
6. **Frame Alignment**: Sub-pixel registration using star positions
7. **Stacking**: Combine frames using statistical methods

## Expected Results

- **SNR Improvement**: ~√n (30× for 900 frames from 30s video)
- **Detection Depth**: 3-4 magnitudes fainter than single frames
- **Angular Resolution**: 1-2 arcminutes (limited by optics/seeing)

## Limitations

- Maximum single-frame exposure limited by video frame rate
- Consumer sensor has lower quantum efficiency than astronomical CCDs
- Wide-angle lens introduces edge distortion
- Light pollution remains the primary constraint on results

## Future Work

- Support for additional cameras (DJI, smartphones)
- Real-time preview mode
- Plate-solving integration
- Machine learning quality assessment
- Dark/flat calibration support
- Web interface

## License

MIT License

## Acknowledgments

- [Gyroflow](https://gyroflow.xyz/) - Inspiration for motion compensation
- [Astropy](https://www.astropy.org/) - Astronomical computing
- [OpenCV](https://opencv.org/) - Computer vision
- GoPro GPMF format specification

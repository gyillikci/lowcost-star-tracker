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
├── pyproject.toml          # Project configuration
├── README.md               # This file
├── src/star_tracker/
│   ├── __init__.py         # Package initialization
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── gyro_extractor.py   # Gyroscope data extraction
│   ├── motion_compensator.py # Frame stabilization
│   ├── frame_extractor.py  # Video frame extraction
│   ├── star_detector.py    # Star detection algorithms
│   ├── quality_assessor.py # Frame quality assessment
│   ├── frame_aligner.py    # Sub-pixel alignment
│   ├── stacker.py          # Image stacking algorithms
│   └── pipeline.py         # Main processing pipeline
├── tests/                  # Unit tests
├── examples/               # Example scripts
├── docs/                   # Documentation
└── data/
    └── lens_profiles/      # Camera calibration profiles
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

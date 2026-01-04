"""
Command-line interface for the star tracker.
"""

import sys
from pathlib import Path
from typing import Optional
import logging

import click

from .config import Config
from .pipeline import Pipeline


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Low-Cost Star Tracker - Motion-compensated astrophotography from action cameras."""
    pass


@main.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output",
    type=click.Path(path_type=Path),
    help="Output image path (default: <video>_stacked.tiff)"
)
@click.option(
    "-c", "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration YAML file"
)
@click.option(
    "--method",
    type=click.Choice(["mean", "median", "sigma_clip", "winsorized"]),
    default="sigma_clip",
    help="Stacking method"
)
@click.option(
    "--min-stars",
    type=int,
    default=10,
    help="Minimum stars required per frame"
)
@click.option(
    "--max-fwhm",
    type=float,
    default=8.0,
    help="Maximum FWHM in pixels"
)
@click.option(
    "--reject-fraction",
    type=float,
    default=0.2,
    help="Fraction of worst frames to reject (0-1)"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output"
)
@click.option(
    "--gpu/--no-gpu",
    default=False,
    help="Enable GPU acceleration (if available)"
)
def process(
    video_path: Path,
    output: Optional[Path],
    config: Optional[Path],
    method: str,
    min_stars: int,
    max_fwhm: float,
    reject_fraction: float,
    verbose: bool,
    gpu: bool,
):
    """
    Process a GoPro video to create a stacked star field image.
    
    VIDEO_PATH: Path to the input video file (MP4 from GoPro)
    """
    # Load or create config
    if config:
        cfg = Config.from_yaml(config)
    else:
        cfg = Config()
    
    # Apply command-line overrides
    cfg.stacking.method = method
    cfg.quality.min_stars = min_stars
    cfg.quality.max_fwhm_pixels = max_fwhm
    cfg.quality.reject_fraction = reject_fraction
    cfg.verbose = verbose
    cfg.use_gpu = gpu
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create pipeline and process
    click.echo(f"Processing: {video_path}")
    
    pipeline = Pipeline(cfg)
    
    def progress_callback(stage: str, progress: float):
        if verbose:
            click.echo(f"  {stage}: {progress*100:.0f}%", nl=False)
            click.echo("\r", nl=False)
    
    try:
        result = pipeline.process(
            video_path,
            output_path=output,
            progress_callback=progress_callback if verbose else None,
        )
        
        click.echo()
        
        if result.success:
            click.echo(click.style("✓ Processing complete!", fg="green"))
            click.echo(f"  Input frames: {result.num_input_frames}")
            click.echo(f"  Stacked frames: {result.num_stacked_frames}")
            click.echo(f"  Processing time: {result.processing_time_seconds:.1f}s")
            click.echo(f"  Output: {result.stacked_image_path}")
        else:
            click.echo(click.style("✗ Processing failed!", fg="red"))
            click.echo("  No frames passed quality filtering.")
            click.echo()
            click.echo("Quality Report:")
            click.echo(result.quality_report)
            sys.exit(1)
            
    except Exception as e:
        click.echo(click.style(f"✗ Error: {e}", fg="red"))
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output-dir",
    type=click.Path(path_type=Path),
    default="frames",
    help="Output directory for frames"
)
def extract_frames(video_path: Path, output_dir: Path):
    """
    Extract individual frames from a video file.
    
    Useful for debugging or manual inspection.
    """
    from .frame_extractor import FrameExtractor
    
    extractor = FrameExtractor()
    info = extractor.get_video_info(video_path)
    
    click.echo(f"Video info:")
    click.echo(f"  Resolution: {info['width']}x{info['height']}")
    click.echo(f"  FPS: {info['fps']}")
    click.echo(f"  Duration: {info['duration']:.1f}s")
    click.echo(f"  Total frames: {info['total_frames']}")
    click.echo()
    
    click.echo(f"Extracting frames to: {output_dir}")
    frames = extractor.extract_all_frames(video_path, output_dir)
    click.echo(f"Extracted {len(frames)} frames")


@main.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
def analyze_gyro(video_path: Path):
    """
    Analyze gyroscope data from a GoPro video.
    
    Shows statistics about camera motion during recording.
    """
    from .gyro_extractor import GyroExtractor
    import numpy as np
    
    extractor = GyroExtractor()
    gyro_data = extractor.extract(video_path)
    
    click.echo(f"Gyroscope Analysis: {video_path}")
    click.echo(f"  Duration: {gyro_data.duration:.1f}s")
    click.echo(f"  Sample rate: {gyro_data.sample_rate:.1f} Hz")
    click.echo(f"  Total samples: {gyro_data.num_samples}")
    click.echo()
    
    # Angular velocity statistics
    omega = gyro_data.angular_velocity
    click.echo("Angular velocity (rad/s):")
    click.echo(f"  X: mean={np.mean(omega[:,0]):.4f}, std={np.std(omega[:,0]):.4f}")
    click.echo(f"  Y: mean={np.mean(omega[:,1]):.4f}, std={np.std(omega[:,1]):.4f}")
    click.echo(f"  Z: mean={np.mean(omega[:,2]):.4f}, std={np.std(omega[:,2]):.4f}")
    
    # Total rotation estimate
    total_rotation = np.degrees(np.sqrt(np.sum(omega**2, axis=1)).sum() / gyro_data.sample_rate)
    click.echo(f"\n  Estimated total rotation: {total_rotation:.1f}°")


@main.command()
@click.argument("output_path", type=click.Path(path_type=Path))
def init_config(output_path: Path):
    """
    Create a default configuration file.
    """
    cfg = Config()
    cfg.to_yaml(output_path)
    click.echo(f"Created configuration file: {output_path}")


@main.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
def analyze_stars(image_path: Path):
    """
    Analyze stars in a single image.
    
    Useful for testing star detection parameters.
    """
    import cv2
    from .star_detector import StarDetector
    
    image = cv2.imread(str(image_path))
    if image is None:
        click.echo(click.style(f"Could not read image: {image_path}", fg="red"))
        sys.exit(1)
    
    detector = StarDetector()
    star_field = detector.detect(image)
    
    click.echo(f"Star Analysis: {image_path}")
    click.echo(f"  Image size: {image.shape[1]}x{image.shape[0]}")
    click.echo(f"  Stars detected: {star_field.num_stars}")
    click.echo(f"  Median FWHM: {star_field.median_fwhm:.2f} pixels")
    click.echo(f"  Background mean: {star_field.background_mean:.1f}")
    click.echo(f"  Background std: {star_field.background_std:.1f}")
    
    if star_field.num_stars > 0:
        click.echo("\nTop 10 brightest stars:")
        sorted_stars = sorted(star_field.stars, key=lambda s: s.flux, reverse=True)[:10]
        for i, star in enumerate(sorted_stars, 1):
            click.echo(f"  {i}. pos=({star.x:.1f}, {star.y:.1f}), "
                      f"flux={star.flux:.0f}, fwhm={star.fwhm:.2f}, snr={star.snr:.1f}")


if __name__ == "__main__":
    main()

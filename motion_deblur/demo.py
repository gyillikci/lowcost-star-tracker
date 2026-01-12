#!/usr/bin/env python3
"""
Motion Deblur Demo Script.

Demonstrates motion blur compensation for star field images using
IMU attitude data, with special handling for star trail overlap.

This script shows:
1. Synthetic star field generation
2. IMU-based motion blur simulation
3. PSF generation from quaternion trajectory
4. Motion deblurring with overlap handling
5. Quality metric evaluation

The overlap corner case occurs when frame shifts significantly:
- Star trails from one side of the image can overlap with trails
  from stars entering from the opposite side
- Example: Frame shifts right -> left-side stars exit left, but their
  ending trails overlap with starting trails of stars entering from right
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os

from synthetic_data import (
    SyntheticStarField,
    IMUMotionSimulator,
    MotionBlurRenderer,
    IMUData
)
from psf_generator import (
    MotionPSFGenerator,
    OverlapAwarePSFGenerator,
    PSFParams
)
from motion_deblur import (
    MotionDeblur,
    DeblurParams,
    compute_quality_metrics
)


def demo_basic_deblur(output_dir: str = None):
    """
    Basic deblurring demo with moderate motion.

    Shows standard motion blur compensation without significant overlap.
    """
    print("=" * 60)
    print("Demo 1: Basic Motion Deblur")
    print("=" * 60)

    # Create star field
    star_field = SyntheticStarField(
        width=1920, height=1080,
        num_stars=150,
        star_fwhm=2.5,
        seed=42
    )

    # Generate moderate motion (drift + vibration)
    motion_sim = IMUMotionSimulator(duration=5.0, sample_rate=200)
    imu_data = motion_sim.generate_combined_motion(
        drift_rate=0.3,  # 0.3 deg/s drift
        vib_frequencies=[5.0, 15.0],
        vib_amplitudes=[0.05, 0.02]
    )

    # Render images
    sharp_image = star_field.render_sharp_image()
    sharp_noisy = star_field.add_noise(sharp_image)

    renderer = MotionBlurRenderer(
        width=star_field.width,
        height=star_field.height,
        focal_length_px=1200
    )

    blurred_image, blur_meta = renderer.render_blurred_image(
        star_field, imu_data,
        num_subframes=150
    )
    blurred_noisy = star_field.add_noise(blurred_image)

    print(f"Star field: {star_field.num_stars} stars")
    print(f"Motion: drift=0.3 deg/s + vibration")
    print(f"Max motion: {blur_meta['max_motion_pixels']:.1f} pixels")
    print(f"Overlap events: {blur_meta['overlap_events']}")

    # Deblur
    deblur = MotionDeblur(1920, 1080, 1200)

    params = DeblurParams(
        method='richardson_lucy',
        iterations=25,
        handle_overlap=True,
        psf_kernel_size=61
    )

    result, meta = deblur.deblur(blurred_noisy, imu_data, params)

    # Compute metrics
    metrics = compute_quality_metrics(sharp_noisy, blurred_noisy, result)
    print(f"\nQuality Metrics:")
    print(f"  PSNR blurred: {metrics['psnr_blurred']:.2f} dB")
    print(f"  PSNR deblurred: {metrics['psnr_deblurred']:.2f} dB")
    print(f"  PSNR improvement: {metrics['psnr_improvement']:.2f} dB")

    # Save/display results
    if output_dir:
        save_comparison(output_dir, "basic",
                        sharp_noisy, blurred_noisy, result,
                        "Basic Deblur (Moderate Motion)")

    return {
        'sharp': sharp_noisy,
        'blurred': blurred_noisy,
        'deblurred': result,
        'metrics': metrics
    }


def demo_overlap_handling(output_dir: str = None):
    """
    Demo showing star trail overlap handling.

    This demonstrates the corner case where significant frame shift
    causes star trails to overlap with trails from opposite side of image.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Star Trail Overlap Handling")
    print("=" * 60)

    # Create star field
    star_field = SyntheticStarField(
        width=1920, height=1080,
        num_stars=100,
        star_fwhm=2.5,
        seed=123
    )

    # Generate large drift motion (will cause overlap)
    motion_sim = IMUMotionSimulator(duration=15.0, sample_rate=200)
    imu_data = motion_sim.generate_drift_motion(
        drift_rate=1.5,  # 1.5 deg/s - significant drift
        drift_axis=np.array([0.0, 1.0, 0.0])  # Horizontal drift
    )

    # Render images
    sharp_image = star_field.render_sharp_image()
    sharp_noisy = star_field.add_noise(sharp_image)

    renderer = MotionBlurRenderer(
        width=star_field.width,
        height=star_field.height,
        focal_length_px=1200
    )

    blurred_image, blur_meta = renderer.render_blurred_image(
        star_field, imu_data,
        num_subframes=450,  # High subframe count for long exposure
        handle_wraparound=True
    )
    blurred_noisy = star_field.add_noise(blurred_image)

    print(f"Star field: {star_field.num_stars} stars")
    print(f"Motion: 1.5 deg/s horizontal drift, 15s exposure")
    print(f"Max motion: {blur_meta['max_motion_pixels']:.1f} pixels")
    print(f"Overlap events: {blur_meta['overlap_events']}")

    # Check for overlap detection
    overlap_gen = OverlapAwarePSFGenerator(1920, 1080, 1200)
    overlap_info = overlap_gen.detect_overlap_regions(imu_data)
    print(f"\nOverlap Analysis:")
    print(f"  Frame shift: ({overlap_info['frame_shift'][0]:.1f}, {overlap_info['frame_shift'][1]:.1f}) pixels")
    print(f"  Horizontal overlap: {overlap_info['has_horizontal_overlap']}")
    print(f"  Vertical overlap: {overlap_info['has_vertical_overlap']}")

    # Deblur WITHOUT overlap handling
    deblur = MotionDeblur(1920, 1080, 1200)

    params_no_overlap = DeblurParams(
        method='richardson_lucy',
        iterations=25,
        handle_overlap=False,  # Disabled
        psf_kernel_size=101
    )

    result_no_overlap, _ = deblur.deblur(blurred_noisy, imu_data, params_no_overlap)

    # Deblur WITH overlap handling
    params_overlap = DeblurParams(
        method='richardson_lucy',
        iterations=25,
        handle_overlap=True,  # Enabled
        psf_kernel_size=101,
        overlap_blend_width=60
    )

    result_overlap, meta = deblur.deblur(blurred_noisy, imu_data, params_overlap)

    print(f"\nDeblur with overlap handling:")
    print(f"  Regions processed: {meta.get('regions_processed', 1)}")
    print(f"  Overlap detected: {meta.get('overlap_detected', False)}")

    # Compare quality
    metrics_no_overlap = compute_quality_metrics(sharp_noisy, blurred_noisy, result_no_overlap)
    metrics_overlap = compute_quality_metrics(sharp_noisy, blurred_noisy, result_overlap)

    print(f"\nQuality Comparison:")
    print(f"  Without overlap handling:")
    print(f"    PSNR: {metrics_no_overlap['psnr_deblurred']:.2f} dB (improvement: {metrics_no_overlap['psnr_improvement']:.2f} dB)")
    print(f"  With overlap handling:")
    print(f"    PSNR: {metrics_overlap['psnr_deblurred']:.2f} dB (improvement: {metrics_overlap['psnr_improvement']:.2f} dB)")

    if output_dir:
        save_overlap_comparison(output_dir,
                                 sharp_noisy, blurred_noisy,
                                 result_no_overlap, result_overlap,
                                 "Star Trail Overlap Handling")

    return {
        'sharp': sharp_noisy,
        'blurred': blurred_noisy,
        'deblurred_no_overlap': result_no_overlap,
        'deblurred_overlap': result_overlap,
        'metrics_no_overlap': metrics_no_overlap,
        'metrics_overlap': metrics_overlap
    }


def demo_psf_visualization(output_dir: str = None):
    """
    Demo showing PSF generation and visualization.
    """
    print("\n" + "=" * 60)
    print("Demo 3: PSF Visualization")
    print("=" * 60)

    # Generate different motion scenarios
    motion_sim = IMUMotionSimulator(duration=5.0, sample_rate=200)

    scenarios = [
        ("Drift Only", motion_sim.generate_drift_motion(drift_rate=0.5)),
        ("Vibration Only", motion_sim.generate_vibration_motion(
            frequencies=[5.0, 12.0, 25.0],
            amplitudes=[0.1, 0.05, 0.02]
        )),
        ("Combined", motion_sim.generate_combined_motion(
            drift_rate=0.3,
            vib_frequencies=[8.0, 15.0],
            vib_amplitudes=[0.08, 0.03]
        ))
    ]

    psf_gen = MotionPSFGenerator(1920, 1080, 1200)
    params = PSFParams(kernel_size=101, star_fwhm=2.5, subsampling=2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, (name, imu_data) in enumerate(scenarios):
        # Generate PSF
        psf = psf_gen.generate_uniform_psf(imu_data, params)

        # Compute trajectory
        trajectory = psf_gen.compute_motion_trajectory(
            (960, 540), imu_data, num_samples=500
        )

        # Linear PSF
        axes[0, idx].imshow(psf, cmap='hot', interpolation='nearest')
        axes[0, idx].set_title(f"{name}\n(Linear Scale)")
        axes[0, idx].set_xlabel("X (pixels)")
        axes[0, idx].set_ylabel("Y (pixels)")

        # Trajectory
        if len(trajectory) > 0:
            axes[1, idx].plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=0.5)
            axes[1, idx].scatter(trajectory[0, 0], trajectory[0, 1], c='g', s=50, label='Start')
            axes[1, idx].scatter(trajectory[-1, 0], trajectory[-1, 1], c='r', s=50, label='End')
            axes[1, idx].set_title(f"Motion Trajectory")
            axes[1, idx].set_xlabel("X offset (pixels)")
            axes[1, idx].set_ylabel("Y offset (pixels)")
            axes[1, idx].legend()
            axes[1, idx].axis('equal')
            axes[1, idx].grid(True, alpha=0.3)

        print(f"\n{name}:")
        print(f"  PSF shape: {psf.shape}")
        print(f"  PSF max: {psf.max():.6f}")
        motion_length = np.linalg.norm(trajectory[-1]) if len(trajectory) > 1 else 0
        print(f"  Motion length: {motion_length:.1f} pixels")

    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "psf_visualization.png"), dpi=150)
        print(f"\nSaved PSF visualization to {output_dir}/psf_visualization.png")

    plt.show()


def demo_spatially_varying(output_dir: str = None):
    """
    Demo showing spatially-varying PSF deconvolution.

    For wide-field cameras, motion blur varies across the image.
    """
    print("\n" + "=" * 60)
    print("Demo 4: Spatially-Varying Deconvolution")
    print("=" * 60)

    # Create star field with more stars
    star_field = SyntheticStarField(
        width=1920, height=1080,
        num_stars=200,
        star_fwhm=2.5,
        seed=789
    )

    # Generate rotation motion (varies across field)
    motion_sim = IMUMotionSimulator(duration=5.0, sample_rate=200)
    imu_data = motion_sim.generate_drift_motion(
        drift_rate=0.4,
        drift_axis=np.array([0.0, 0.0, 1.0])  # Rotation around Z
    )

    # Render
    sharp_image = star_field.render_sharp_image()
    sharp_noisy = star_field.add_noise(sharp_image)

    renderer = MotionBlurRenderer(
        width=star_field.width,
        height=star_field.height,
        focal_length_px=800  # Wider field
    )

    blurred_image, blur_meta = renderer.render_blurred_image(
        star_field, imu_data,
        num_subframes=150
    )
    blurred_noisy = star_field.add_noise(blurred_image)

    print(f"Motion: rotation around optical axis")
    print(f"Max motion: {blur_meta['max_motion_pixels']:.1f} pixels")

    # Compare uniform vs spatially-varying
    deblur = MotionDeblur(1920, 1080, 800)

    # Uniform PSF
    params_uniform = DeblurParams(
        method='richardson_lucy',
        iterations=20,
        psf_kernel_size=61
    )
    result_uniform, _ = deblur.deblur(blurred_noisy, imu_data, params_uniform)

    # Spatially-varying
    params_sv = DeblurParams(
        method='spatially_varying',
        iterations=20,
        psf_kernel_size=61
    )
    result_sv, meta_sv = deblur.deblur(blurred_noisy, imu_data, params_sv)

    print(f"\nSpatially-varying: {meta_sv['tiles_processed']} tiles processed")

    # Compare quality
    metrics_uniform = compute_quality_metrics(sharp_noisy, blurred_noisy, result_uniform)
    metrics_sv = compute_quality_metrics(sharp_noisy, blurred_noisy, result_sv)

    print(f"\nQuality Comparison:")
    print(f"  Uniform PSF:")
    print(f"    PSNR improvement: {metrics_uniform['psnr_improvement']:.2f} dB")
    print(f"  Spatially-varying:")
    print(f"    PSNR improvement: {metrics_sv['psnr_improvement']:.2f} dB")

    if output_dir:
        save_comparison(output_dir, "spatially_varying",
                        sharp_noisy, blurred_noisy, result_sv,
                        "Spatially-Varying Deconvolution")

    return {
        'sharp': sharp_noisy,
        'blurred': blurred_noisy,
        'deblurred_uniform': result_uniform,
        'deblurred_sv': result_sv
    }


def save_comparison(output_dir: str, name: str,
                     sharp: np.ndarray, blurred: np.ndarray, deblurred: np.ndarray,
                     title: str):
    """Save comparison figure."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Normalize for display
    vmax = max(sharp.max(), blurred.max(), deblurred.max()) * 0.3

    axes[0].imshow(sharp, cmap='gray', vmin=0, vmax=vmax)
    axes[0].set_title("Sharp (Ground Truth)")
    axes[0].axis('off')

    axes[1].imshow(blurred, cmap='gray', vmin=0, vmax=vmax)
    axes[1].set_title("Motion Blurred")
    axes[1].axis('off')

    axes[2].imshow(deblurred, cmap='gray', vmin=0, vmax=vmax)
    axes[2].set_title("Deblurred")
    axes[2].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved to {output_dir}/{name}_comparison.png")


def save_overlap_comparison(output_dir: str,
                             sharp: np.ndarray, blurred: np.ndarray,
                             no_overlap: np.ndarray, with_overlap: np.ndarray,
                             title: str):
    """Save overlap handling comparison figure."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    vmax = max(sharp.max(), blurred.max()) * 0.3

    axes[0, 0].imshow(sharp, cmap='gray', vmin=0, vmax=vmax)
    axes[0, 0].set_title("Sharp (Ground Truth)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(blurred, cmap='gray', vmin=0, vmax=vmax)
    axes[0, 1].set_title("Motion Blurred (Large Drift)")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(no_overlap, cmap='gray', vmin=0, vmax=vmax)
    axes[1, 0].set_title("Deblurred (No Overlap Handling)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(with_overlap, cmap='gray', vmin=0, vmax=vmax)
    axes[1, 1].set_title("Deblurred (With Overlap Handling)")
    axes[1, 1].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overlap_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved to {output_dir}/overlap_comparison.png")


def run_all_demos(output_dir: str = None):
    """Run all demonstration functions."""
    print("\n" + "=" * 70)
    print("MOTION DEBLUR DEMO - Star Tracker IMU Motion Compensation")
    print("=" * 70)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")

    # Run demos
    demo_basic_deblur(output_dir)
    demo_overlap_handling(output_dir)
    demo_psf_visualization(output_dir)
    demo_spatially_varying(output_dir)

    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Motion Deblur Demo for Star Tracker Images"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='motion_deblur/demo_output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--demo',
        type=str,
        choices=['basic', 'overlap', 'psf', 'spatial', 'all'],
        default='all',
        help='Which demo to run'
    )

    args = parser.parse_args()

    if args.demo == 'basic':
        demo_basic_deblur(args.output)
    elif args.demo == 'overlap':
        demo_overlap_handling(args.output)
    elif args.demo == 'psf':
        demo_psf_visualization(args.output)
    elif args.demo == 'spatial':
        demo_spatially_varying(args.output)
    else:
        run_all_demos(args.output)

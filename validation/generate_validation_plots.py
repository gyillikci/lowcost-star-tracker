#!/usr/bin/env python3
"""
Generate validation result visualizations.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(results_file: str = "validation/results/validation_results.json"):
    """Load validation results from JSON."""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_snr_scaling(results: dict, output_dir: Path):
    """Plot SNR vs frame count showing sqrt(N) scaling."""
    snr_test = next((t for t in results['tests'] if t['name'] == 'snr_scaling'), None)
    if not snr_test:
        return

    metrics = snr_test['metrics']
    frame_counts = np.array(metrics['frame_counts'])
    measured_snr = np.array(metrics['measured_snr'])

    # Theoretical sqrt(N) curve
    theoretical = measured_snr[0] * np.sqrt(frame_counts)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(frame_counts, measured_snr, 'bo-', markersize=10, linewidth=2, label='Measured SNR')
    ax.plot(frame_counts, theoretical, 'r--', linewidth=2, label='Theoretical (√N scaling)')

    ax.set_xlabel('Number of Stacked Frames', fontsize=12)
    ax.set_ylabel('Signal-to-Noise Ratio', fontsize=12)
    ax.set_title('SNR Improvement with Frame Stacking\n(Validates √N theoretical scaling)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add correlation annotation
    corr = metrics['fit_correlation']
    ax.annotate(f'Correlation with √N model: {corr:.4f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'snr_scaling_validation.png', dpi=150)
    plt.close()
    print(f"  Saved: snr_scaling_validation.png")


def plot_centroid_accuracy(results: dict, output_dir: Path):
    """Plot centroid accuracy results."""
    centroid_test = next((t for t in results['tests'] if t['name'] == 'centroid_accuracy'), None)
    if not centroid_test:
        return

    metrics = centroid_test['metrics']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of error metrics
    error_metrics = ['mean_error_pixels', 'rms_error_pixels', 'median_error_pixels', 'max_error_pixels']
    error_labels = ['Mean', 'RMS', 'Median', 'Max']
    error_values = [metrics[m] for m in error_metrics]

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    bars = axes[0].bar(error_labels, error_values, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Error (pixels)', fontsize=12)
    axes[0].set_title('Centroid Position Errors', fontsize=12, fontweight='bold')
    axes[0].axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Target RMS (<0.5 px)')
    axes[0].legend()

    # Add value labels on bars
    for bar, val in zip(bars, error_values):
        axes[0].annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Detection metrics
    det_metrics = ['detection_rate', 'false_positive_rate']
    det_labels = ['Detection Rate', 'False Positive Rate']
    det_values = [metrics[m] * 100 for m in det_metrics]

    colors2 = ['#2ecc71', '#e74c3c']
    bars2 = axes[1].bar(det_labels, det_values, color=colors2, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1].set_title('Detection Performance', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 110)
    axes[1].axhline(y=90, color='green', linestyle='--', linewidth=2, label='Target Detection (>90%)')
    axes[1].axhline(y=10, color='red', linestyle='--', linewidth=2, label='Target FP (<10%)')
    axes[1].legend(loc='upper right')

    for bar, val in zip(bars2, det_values):
        axes[1].annotate(f'{val:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Centroid Accuracy Validation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'centroid_accuracy_validation.png', dpi=150)
    plt.close()
    print(f"  Saved: centroid_accuracy_validation.png")


def plot_processing_performance(results: dict, output_dir: Path):
    """Plot processing benchmark results."""
    perf_test = next((t for t in results['tests'] if t['name'] == 'processing_performance'), None)
    if not perf_test:
        return

    metrics = perf_test['metrics']
    benchmarks = metrics['benchmarks']

    resolutions = list(benchmarks.keys())
    detection_times = [benchmarks[r]['detection_time_ms'] for r in resolutions]
    stacking_times = [benchmarks[r]['stacking_10_frames_ms'] for r in resolutions]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Detection time
    x = np.arange(len(resolutions))
    width = 0.35

    bars1 = axes[0].bar(x, detection_times, width, label='Star Detection', color='#3498db', edgecolor='black')
    axes[0].set_xlabel('Resolution', fontsize=12)
    axes[0].set_ylabel('Time (ms)', fontsize=12)
    axes[0].set_title('Star Detection Time', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(resolutions)
    axes[0].axhline(y=500, color='red', linestyle='--', linewidth=2, label='Target (<500ms)')
    axes[0].legend()

    for bar, val in zip(bars1, detection_times):
        axes[0].annotate(f'{val:.0f}ms',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Stacking time
    bars2 = axes[1].bar(x, stacking_times, width, label='10-Frame Stack', color='#2ecc71', edgecolor='black')
    axes[1].set_xlabel('Resolution', fontsize=12)
    axes[1].set_ylabel('Time (ms)', fontsize=12)
    axes[1].set_title('Frame Stacking Time (10 frames)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(resolutions)

    for bar, val in zip(bars2, stacking_times):
        axes[1].annotate(f'{val:.0f}ms',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Processing Performance Benchmarks\n(Python/NumPy, single-threaded)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'processing_performance_validation.png', dpi=150)
    plt.close()
    print(f"  Saved: processing_performance_validation.png")


def plot_motion_compensation(results: dict, output_dir: Path):
    """Plot motion compensation effectiveness."""
    motion_test = next((t for t in results['tests'] if t['name'] == 'motion_compensation'), None)
    if not motion_test:
        return

    metrics = motion_test['metrics']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # FWHM comparison
    fwhm_labels = ['Reference\n(No Motion)', 'Blurred\n(15px Motion)', 'Compensated\n(Gyro-Based)']
    fwhm_values = [metrics['fwhm_reference'], metrics['fwhm_blurred'], metrics['fwhm_compensated']]
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    bars1 = axes[0].bar(fwhm_labels, fwhm_values, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('FWHM (pixels)', fontsize=12)
    axes[0].set_title('Star FWHM (Sharpness)', fontsize=12, fontweight='bold')

    for bar, val in zip(bars1, fwhm_values):
        axes[0].annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Detection count comparison
    det_labels = ['Reference', 'Blurred', 'Compensated']
    det_values = [metrics['detections_reference'], metrics['detections_blurred'], metrics['detections_compensated']]

    bars2 = axes[1].bar(det_labels, det_values, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Stars Detected', fontsize=12)
    axes[1].set_title('Detection Count', fontsize=12, fontweight='bold')

    recovery_rate = metrics['detection_recovery_rate'] * 100
    axes[1].annotate(f'Recovery Rate: {recovery_rate:.1f}%',
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', va='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    for bar, val in zip(bars2, det_values):
        axes[1].annotate(f'{val}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('Motion Compensation Effectiveness\n(15-pixel simulated camera motion)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'motion_compensation_validation.png', dpi=150)
    plt.close()
    print(f"  Saved: motion_compensation_validation.png")


def plot_summary(results: dict, output_dir: Path):
    """Plot overall validation summary."""
    tests = results['tests']
    summary = results['summary']

    fig, ax = plt.subplots(figsize=(10, 6))

    test_names = [t['name'].replace('_', '\n') for t in tests]
    passed = [1 if t['passed'] else 0 for t in tests]
    colors = ['#2ecc71' if p else '#e74c3c' for p in passed]

    bars = ax.barh(test_names, [1]*len(tests), color=colors, edgecolor='black', linewidth=1.5)

    # Add pass/fail labels
    for i, (bar, p) in enumerate(zip(bars, passed)):
        label = '✓ PASS' if p else '✗ FAIL'
        ax.annotate(label,
                   xy=(0.5, bar.get_y() + bar.get_height()/2),
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   color='white')

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title(f'Validation Summary: {summary["passed"]}/{summary["total"]} Tests Passed ({summary["pass_rate"]*100:.0f}%)',
                fontsize=14, fontweight='bold')

    # Add timestamp
    ax.annotate(f'Generated: {summary["timestamp"]}',
               xy=(0.99, 0.01), xycoords='axes fraction',
               ha='right', va='bottom', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'validation_summary.png', dpi=150)
    plt.close()
    print(f"  Saved: validation_summary.png")


def generate_all_plots():
    """Generate all validation plots."""
    print("Generating validation plots...")

    output_dir = Path("validation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results()

    plot_snr_scaling(results, output_dir)
    plot_centroid_accuracy(results, output_dir)
    plot_processing_performance(results, output_dir)
    plot_motion_compensation(results, output_dir)
    plot_summary(results, output_dir)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    generate_all_plots()

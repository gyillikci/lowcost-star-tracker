#!/usr/bin/env python3
"""
Generate visualization plots for Phase 2 validation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def generate_drift_compensation_plot(results: dict, output_dir: Path):
    """Generate drift compensation comparison plot."""
    metrics = results.get("metrics", {})

    errors_with = metrics.get("errors_with_comp", [])
    errors_without = metrics.get("errors_without_comp", [])

    if not errors_with or not errors_without:
        print("No drift compensation data to plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Time axis
    duration = metrics.get("duration_s", 120)
    t = np.linspace(0, duration, len(errors_with))

    # Error over time
    ax1 = axes[0]
    ax1.plot(t, errors_without, 'r-', label='Without Compensation', linewidth=2)
    ax1.plot(t, errors_with, 'b-', label='With Star-Aided Compensation', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Attitude Error (degrees)')
    ax1.set_title('Gyroscope Drift Compensation Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Improvement factor
    ax2 = axes[1]
    improvement = [w / max(c, 0.001) for w, c in zip(errors_without, errors_with)]
    ax2.plot(t, improvement, 'g-', linewidth=2)
    ax2.axhline(y=metrics.get("improvement_factor", 10), color='k', linestyle='--',
                label=f'Final: {metrics.get("improvement_factor", 0):.1f}x')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Improvement Factor')
    ax2.set_title('Error Reduction Factor Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'phase2_drift_compensation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'phase2_drift_compensation.png'}")


def generate_calibration_plot(results: dict, output_dir: Path):
    """Generate optical calibration comparison plot."""
    metrics = results.get("metrics", {})

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Vignetting correction
    ax1 = axes[0, 0]
    categories = ['Before\nCalibration', 'After\nCalibration', 'Ideal']
    values = [
        metrics.get("vignetting_ratio_before", 1.3),
        metrics.get("vignetting_ratio_after", 1.0),
        1.0
    ]
    colors = ['red', 'green', 'blue']
    bars = ax1.bar(categories, values, color=colors, alpha=0.7)
    ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Center/Corner Ratio')
    ax1.set_title('Vignetting Correction')
    ax1.set_ylim(0.9, 1.4)

    # Hot pixel detection
    ax2 = axes[0, 1]
    detected = metrics.get("hot_pixels_detected", 90)
    total = metrics.get("hot_pixels_true", 100)
    missed = max(0, total - detected)  # Ensure non-negative
    if missed > 0:
        ax2.pie([detected, missed], labels=[f'Detected\n({detected})', f'Missed\n({missed})'],
                colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
    else:
        ax2.pie([detected], labels=[f'Detected\n({detected})'],
                colors=['green'], autopct='%1.1f%%', startangle=90)
    ax2.set_title('Hot Pixel Detection')

    # Hot pixel intensity reduction
    ax3 = axes[1, 0]
    categories = ['Before', 'After']
    values = [
        metrics.get("hot_pixel_mean_before", 1500),
        metrics.get("hot_pixel_mean_after", 500)
    ]
    colors = ['red', 'green']
    ax3.bar(categories, values, color=colors, alpha=0.7)
    ax3.set_ylabel('Mean Intensity (ADU)')
    ax3.set_title('Hot Pixel Correction')

    # Summary metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
Optical Calibration Summary
===========================

Image Size: {metrics.get('image_size', [0,0])}
Dark Frames: {metrics.get('n_dark_frames', 0)}
Flat Frames: {metrics.get('n_flat_frames', 0)}

Vignetting:
  Before: {metrics.get('vignetting_ratio_before', 0):.3f}
  After: {metrics.get('vignetting_ratio_after', 0):.3f}
  Accuracy: {metrics.get('vignetting_correction_accuracy', 0)*100:.1f}%

Hot Pixels:
  Detection Rate: {metrics.get('hot_pixel_detection_rate', 0)*100:.1f}%
  Reduction: {metrics.get('hot_pixel_reduction', 0)*100:.1f}%
"""
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontfamily='monospace', fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(output_dir / 'phase2_optical_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'phase2_optical_calibration.png'}")


def generate_false_star_plot(results: dict, output_dir: Path):
    """Generate false star rejection plot."""
    metrics = results.get("metrics", {})

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Confusion matrix style
    ax1 = axes[0]
    tp = metrics.get("true_positives", 0)
    fp = metrics.get("false_positives", 0)
    tn = metrics.get("true_negatives", 0)
    fn = metrics.get("false_negatives", 0)

    matrix = np.array([[tp, fn], [fp, tn]])
    im = ax1.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=max(tp, tn))

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f'{matrix[i, j]}',
                     ha='center', va='center', fontsize=20, fontweight='bold')

    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Accepted', 'Rejected'])
    ax1.set_yticklabels(['Real Stars', 'False\nDetections'])
    ax1.set_title('Detection Classification')

    # Precision/Recall/F1
    ax2 = axes[1]
    metrics_names = ['Precision', 'Recall', 'F1 Score']
    metrics_values = [
        metrics.get("precision", 0),
        metrics.get("recall", 0),
        metrics.get("f1_score", 0)
    ]
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    bars = ax2.bar(metrics_names, metrics_values, color=colors)
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Score')
    ax2.set_title('Classification Metrics')

    for bar, val in zip(bars, metrics_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Detection breakdown
    ax3 = axes[2]
    categories = ['Real Stars', 'Cosmic Rays', 'Noise', 'Satellites']
    values = [
        metrics.get("n_real_stars", 30),
        metrics.get("n_cosmic_rays", 10),
        metrics.get("n_noise_spikes", 8),
        metrics.get("n_satellites", 5)
    ]
    colors = ['#27ae60', '#e74c3c', '#f39c12', '#8e44ad']
    ax3.pie(values, labels=categories, colors=colors, autopct='%1.0f%%',
            startangle=90, explode=[0.05, 0, 0, 0])
    ax3.set_title('Detection Types in Test Set')

    plt.tight_layout()
    plt.savefig(output_dir / 'phase2_false_star_rejection.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'phase2_false_star_rejection.png'}")


def generate_confidence_metrics_plot(results: dict, output_dir: Path):
    """Generate confidence metrics comparison plot."""
    metrics = results.get("metrics", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Photometric consistency
    ax1 = axes[0]
    categories = ['Consistent\nMatches', 'Inconsistent\nMatches']
    values = [
        metrics.get("photometric_consistent_score", 1.0),
        metrics.get("photometric_inconsistent_score", 0.5)
    ]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax1.bar(categories, values, color=colors, alpha=0.8)
    ax1.set_ylim(0, 1.2)
    ax1.set_ylabel('Photometric Consistency Score')
    ax1.set_title('Photometric Consistency Metric')

    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Coverage score
    ax2 = axes[1]
    categories = ['Spread\nDetections', 'Clustered\nDetections']
    values = [
        metrics.get("coverage_spread_score", 0.8),
        metrics.get("coverage_clustered_score", 0.2)
    ]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax2.bar(categories, values, color=colors, alpha=0.8)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('Coverage Score')
    ax2.set_title('Spatial Coverage Metric')

    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'phase2_confidence_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'phase2_confidence_metrics.png'}")


def generate_summary_plot(all_results: dict, output_dir: Path):
    """Generate overall Phase 2 summary plot."""
    tests = all_results.get("tests", [])

    fig, ax = plt.subplots(figsize=(10, 6))

    test_names = [t.get("test", "Unknown").replace("_", "\n") for t in tests]
    passed = [1 if t.get("passed", False) else 0 for t in tests]
    colors = ['#2ecc71' if p else '#e74c3c' for p in passed]

    bars = ax.barh(test_names, [1]*len(tests), color=colors, alpha=0.8)

    # Add pass/fail labels
    for i, (bar, p) in enumerate(zip(bars, passed)):
        label = 'PASS' if p else 'FAIL'
        ax.text(0.5, i, label, ha='center', va='center',
                fontsize=16, fontweight='bold', color='white')

    ax.set_xlim(0, 1)
    ax.set_xlabel('')
    ax.set_title('Phase 2: Algorithm Enhancements - Validation Summary', fontsize=14)
    ax.set_xticks([])

    # Add summary text
    summary = all_results.get("summary", {})
    summary_text = f"Total: {summary.get('passed', 0)}/{summary.get('total_tests', 0)} tests passed"
    ax.text(0.5, -0.15, summary_text, transform=ax.transAxes,
            ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'phase2_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'phase2_summary.png'}")


def main():
    """Generate all Phase 2 plots."""
    results_file = Path("validation/results/phase2_validation_results.json")
    output_dir = Path("docs/images/phase2")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Run phase2_validation.py first")
        return

    with open(results_file, 'r') as f:
        all_results = json.load(f)

    print("Generating Phase 2 visualization plots...")

    # Generate individual test plots
    for test in all_results.get("tests", []):
        test_name = test.get("test", "")

        if test_name == "drift_compensation":
            generate_drift_compensation_plot(test, output_dir)
        elif test_name == "optical_calibration":
            generate_calibration_plot(test, output_dir)
        elif test_name == "false_star_rejection":
            generate_false_star_plot(test, output_dir)
        elif test_name == "confidence_metrics":
            generate_confidence_metrics_plot(test, output_dir)

    # Generate summary plot
    generate_summary_plot(all_results, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

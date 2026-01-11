"""
Step 2: Compare OpenCV Motion Detection vs Gyro Telemetry

Plots the motion detected by OpenCV (phase correlation) against
the gyroscope data embedded in the GoPro video file (GPMF format).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.star_tracker.gyro_extractor import GyroExtractor


def load_opencv_motion(npz_path: str) -> tuple:
    """Load OpenCV motion data from saved file."""
    # Try different file formats
    if npz_path.endswith('.npy'):
        transforms = np.load(npz_path, allow_pickle=True)
        return transforms, None
    else:
        data = np.load(npz_path, allow_pickle=True)
        if 'transforms' in data:
            return data['transforms'], data.get('fps', 60)
        else:
            return data, None


def extract_gyro_data(video_path: str) -> tuple:
    """Extract gyroscope data from GoPro video."""
    extractor = GyroExtractor()
    gyro_data = extractor.extract(video_path)
    
    timestamps = gyro_data.timestamps
    angular_velocity = gyro_data.angular_velocity  # rad/s
    
    return timestamps, angular_velocity


def integrate_gyro_to_angles(timestamps: np.ndarray, angular_velocity: np.ndarray) -> np.ndarray:
    """
    Integrate angular velocity to get rotation angles.
    Uses trapezoidal integration.
    """
    dt = np.diff(timestamps)
    
    # Trapezoidal integration
    angles = np.zeros_like(angular_velocity)
    for i in range(1, len(timestamps)):
        angles[i] = angles[i-1] + (angular_velocity[i] + angular_velocity[i-1]) / 2 * dt[i-1]
    
    return angles  # in radians


def opencv_transforms_to_angles(transforms: list, fps: float) -> tuple:
    """
    Convert OpenCV frame-to-frame transforms to cumulative angles.
    Returns timestamps and angles.
    """
    n = len(transforms)
    timestamps = np.arange(n) / fps
    
    # Cumulative sum of translations and rotations
    cum_dx = np.cumsum([t[0] for t in transforms])
    cum_dy = np.cumsum([t[1] for t in transforms])
    cum_da = np.cumsum([t[2] for t in transforms])  # rotation in radians
    
    return timestamps, cum_dx, cum_dy, cum_da


def resample_to_video_frames(gyro_timestamps: np.ndarray, gyro_angles: np.ndarray,
                               video_timestamps: np.ndarray) -> np.ndarray:
    """Resample gyro data to match video frame timestamps."""
    resampled = np.zeros((len(video_timestamps), 3))
    for i in range(3):
        resampled[:, i] = np.interp(video_timestamps, gyro_timestamps, gyro_angles[:, i])
    return resampled


def plot_comparison(video_path: str, opencv_data_path: str, output_path: str):
    """
    Create side-by-side comparison plot of OpenCV motion vs gyro data.
    """
    print(f"Loading OpenCV motion data from: {opencv_data_path}")
    
    # Load OpenCV data
    transforms = np.load(opencv_data_path, allow_pickle=True)
    
    # Get video FPS
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Video FPS: {fps}")
    
    # Convert to cumulative motion
    n_frames = len(transforms)
    opencv_time = np.arange(n_frames) / fps
    
    opencv_dx = np.array([t[0] for t in transforms])
    opencv_dy = np.array([t[1] for t in transforms])
    opencv_da = np.array([t[2] for t in transforms])  # radians
    
    # Cumulative
    opencv_cum_dx = np.cumsum(opencv_dx)
    opencv_cum_dy = np.cumsum(opencv_dy)
    opencv_cum_da = np.cumsum(opencv_da)
    
    print(f"OpenCV: {n_frames} frames, {opencv_time[-1]:.2f}s duration")
    
    # Extract gyro data
    print(f"\nExtracting gyro data from: {video_path}")
    gyro_timestamps, gyro_angular_vel = extract_gyro_data(video_path)
    
    # Convert to degrees/s for display
    gyro_angular_vel_deg = np.degrees(gyro_angular_vel)
    
    # Integrate to get angles
    gyro_angles = integrate_gyro_to_angles(gyro_timestamps, gyro_angular_vel)
    gyro_angles_deg = np.degrees(gyro_angles)
    
    print(f"Gyro: {len(gyro_timestamps)} samples, {gyro_timestamps[-1]:.2f}s duration")
    
    # Resample gyro to video frame rate for direct comparison
    gyro_angles_resampled = resample_to_video_frames(gyro_timestamps, gyro_angles_deg, opencv_time)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'OpenCV Motion Detection vs Gyro Telemetry\n{Path(video_path).name}', fontsize=14)
    
    # === Left column: Angular velocity (frame-to-frame changes) ===
    
    # Convert OpenCV to angular velocity (approx)
    # Note: OpenCV gives us translation in pixels, rotation in radians per frame
    opencv_angular_vel_deg = np.degrees(opencv_da[1:]) * fps  # Convert to deg/s
    
    # X axis (Roll)
    ax = axes[0, 0]
    ax.plot(gyro_timestamps, gyro_angular_vel_deg[:, 0], 'b-', alpha=0.7, label='Gyro X (roll)', linewidth=0.5)
    ax.set_ylabel('Angular Velocity (deg/s)')
    ax.set_title('X-Axis (Roll) - Angular Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Y axis (Pitch)  
    ax = axes[1, 0]
    ax.plot(gyro_timestamps, gyro_angular_vel_deg[:, 1], 'g-', alpha=0.7, label='Gyro Y (pitch)', linewidth=0.5)
    ax.set_ylabel('Angular Velocity (deg/s)')
    ax.set_title('Y-Axis (Pitch) - Angular Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Z axis (Yaw) - This should correlate with OpenCV rotation
    ax = axes[2, 0]
    ax.plot(gyro_timestamps, gyro_angular_vel_deg[:, 2], 'r-', alpha=0.7, label='Gyro Z (yaw)', linewidth=0.5)
    ax.plot(opencv_time[1:], opencv_angular_vel_deg, 'k-', alpha=0.7, label='OpenCV rotation rate', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Velocity (deg/s)')
    ax.set_title('Z-Axis (Yaw) - Angular Velocity (OpenCV rotation should match)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # === Right column: Cumulative angles (integrated motion) ===
    
    # X axis cumulative
    ax = axes[0, 1]
    ax.plot(gyro_timestamps, gyro_angles_deg[:, 0], 'b-', alpha=0.7, label='Gyro X (roll)', linewidth=1)
    ax.set_ylabel('Cumulative Angle (deg)')
    ax.set_title('X-Axis (Roll) - Cumulative Rotation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Y axis cumulative
    ax = axes[1, 1]
    ax.plot(gyro_timestamps, gyro_angles_deg[:, 1], 'g-', alpha=0.7, label='Gyro Y (pitch)', linewidth=1)
    ax.set_ylabel('Cumulative Angle (deg)')
    ax.set_title('Y-Axis (Pitch) - Cumulative Rotation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Z axis cumulative - compare with OpenCV
    ax = axes[2, 1]
    ax.plot(gyro_timestamps, gyro_angles_deg[:, 2], 'r-', alpha=0.7, label='Gyro Z (yaw)', linewidth=1)
    ax.plot(opencv_time, np.degrees(opencv_cum_da), 'k-', alpha=0.7, label='OpenCV cumulative rotation', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Angle (deg)')
    ax.set_title('Z-Axis (Yaw) - Cumulative Rotation Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    plt.show()
    
    # Also create a translation comparison plot
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))
    fig2.suptitle(f'OpenCV Translation Detection\n{Path(video_path).name}', fontsize=14)
    
    # X translation
    ax = axes2[0]
    ax.plot(opencv_time, opencv_cum_dx, 'b-', linewidth=1, label='Cumulative X translation')
    ax.set_ylabel('Translation X (pixels)')
    ax.set_title('Horizontal Translation (should correlate with Gyro Y - pitch)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Y translation
    ax = axes2[1]
    ax.plot(opencv_time, opencv_cum_dy, 'g-', linewidth=1, label='Cumulative Y translation')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Translation Y (pixels)')
    ax.set_title('Vertical Translation (should correlate with Gyro X - roll)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    translation_path = output_path.replace('.png', '_translation.png')
    plt.savefig(translation_path, dpi=150, bbox_inches='tight')
    print(f"Translation plot saved: {translation_path}")
    plt.show()
    
    # Print correlation analysis
    print("\n" + "=" * 50)
    print("CORRELATION ANALYSIS")
    print("=" * 50)
    
    # Resample gyro to match OpenCV frames for correlation
    gyro_z_resampled = np.interp(opencv_time, gyro_timestamps, gyro_angles_deg[:, 2])
    opencv_rotation_deg = np.degrees(opencv_cum_da)
    
    # Compute correlation
    corr_rotation = np.corrcoef(gyro_z_resampled, opencv_rotation_deg)[0, 1]
    print(f"Correlation (Gyro Z vs OpenCV rotation): {corr_rotation:.4f}")
    
    # Scale factor estimation
    if np.std(gyro_z_resampled) > 0:
        scale = np.std(opencv_rotation_deg) / np.std(gyro_z_resampled)
        print(f"Scale factor (OpenCV / Gyro): {scale:.4f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python step2_compare_motion.py <video_path>")
        print("Expects OpenCV motion data in experiments/output/<video_name>_cumulative_transforms.npy")
        sys.exit(1)
    
    video_path = sys.argv[1]
    video_name = Path(video_path).stem
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    # Find OpenCV data file
    opencv_data_path = f"experiments/output/{video_name}_cumulative_transforms.npy"
    if not os.path.exists(opencv_data_path):
        print(f"Error: OpenCV motion data not found: {opencv_data_path}")
        print("Run step1_opencv_stabilization_v3.py first!")
        sys.exit(1)
    
    output_dir = Path("experiments/output")
    output_path = str(output_dir / f"{video_name}_motion_comparison.png")
    
    print("=" * 50)
    print("Step 2: OpenCV vs Gyro Motion Comparison")
    print("=" * 50)
    
    plot_comparison(video_path, opencv_data_path, output_path)


if __name__ == "__main__":
    main()

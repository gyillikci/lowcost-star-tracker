"""
Step 3: Gyro-based Stabilization vs OpenCV Stabilization

This script:
1. Uses gyro telemetry to compute stabilization transforms
2. Applies both OpenCV and Gyro stabilization to the same video
3. Creates a side-by-side comparison to validate the correlation

If gyro-based stabilization matches OpenCV stabilization, it proves
we can rely solely on gyro data for future stabilization.
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm
import imageio
from scipy.spatial.transform import Rotation
from scipy import signal

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.star_tracker.gyro_extractor import GyroExtractor


def enhance_stars(gray: np.ndarray) -> np.ndarray:
    """Enhance star visibility with high-pass filter."""
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    enhanced = cv2.subtract(gray, blurred * 0.8)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced


def extract_opencv_motion(video_path: str) -> tuple:
    """Extract frame-to-frame motion using phase correlation."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    proc_scale = 0.25
    proc_width = int(width * proc_scale)
    proc_height = int(height * proc_scale)
    
    ret, ref_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read video")
    
    ref_small = cv2.resize(ref_frame, (proc_width, proc_height))
    ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref_enhanced = enhance_stars(ref_gray)
    
    hann = cv2.createHanningWindow((proc_width, proc_height), cv2.CV_32F)
    
    frames = [ref_frame]
    transforms = [(0.0, 0.0, 0.0)]
    
    prev_gray = ref_enhanced
    cum_dx, cum_dy, cum_da = 0.0, 0.0, 0.0
    
    print("Extracting OpenCV motion...")
    for i in tqdm(range(frame_count - 1), desc="OpenCV"):
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        
        curr_small = cv2.resize(frame, (proc_width, proc_height))
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        curr_enhanced = enhance_stars(curr_gray)
        
        shift, response = cv2.phaseCorrelate(prev_gray, curr_enhanced, hann)
        dx, dy = shift[0] / proc_scale, shift[1] / proc_scale
        
        # Estimate rotation
        da = 0.0
        if response > 0.05:
            try:
                h, w = prev_gray.shape
                M_trans = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
                curr_aligned = cv2.warpAffine(curr_enhanced, M_trans, (w, h))
                
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
                
                margin = int(min(h, w) * 0.2)
                prev_crop = prev_gray[margin:-margin, margin:-margin]
                curr_crop = curr_aligned[margin:-margin, margin:-margin]
                
                prev_norm = cv2.normalize(prev_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                curr_norm = cv2.normalize(curr_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                _, warp_matrix = cv2.findTransformECC(
                    prev_norm, curr_norm, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
                )
                da = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])
            except cv2.error:
                da = 0.0
        
        cum_dx += dx
        cum_dy += dy
        cum_da += da
        
        transforms.append((cum_dx, cum_dy, cum_da))
        prev_gray = curr_enhanced
    
    cap.release()
    return frames, transforms, fps, (width, height)


def extract_gyro_motion(video_path: str, video_fps: float, num_frames: int) -> list:
    """
    Extract gyro data and convert to per-frame cumulative transforms.
    
    Returns cumulative (dx, dy, da) for each video frame, aligned to frame 0.
    """
    extractor = GyroExtractor()
    gyro_data = extractor.extract(video_path)
    
    gyro_timestamps = gyro_data.timestamps
    angular_velocity = gyro_data.angular_velocity  # rad/s, shape (N, 3)
    
    print(f"Gyro: {len(gyro_timestamps)} samples over {gyro_timestamps[-1]:.2f}s")
    
    # Integrate angular velocity to get cumulative angles
    # Using trapezoidal integration
    dt = np.diff(gyro_timestamps)
    cumulative_angles = np.zeros_like(angular_velocity)
    
    for i in range(1, len(gyro_timestamps)):
        cumulative_angles[i] = cumulative_angles[i-1] + \
            (angular_velocity[i] + angular_velocity[i-1]) / 2 * dt[i-1]
    
    # Cumulative angles are now in radians for X (roll), Y (pitch), Z (yaw)
    
    # Create video frame timestamps
    video_timestamps = np.arange(num_frames) / video_fps
    
    # Resample gyro angles to video frame rate
    gyro_angles_at_frames = np.zeros((num_frames, 3))
    for axis in range(3):
        gyro_angles_at_frames[:, axis] = np.interp(
            video_timestamps, 
            gyro_timestamps, 
            cumulative_angles[:, axis]
        )
    
    # Normalize to start at 0 (first frame is reference)
    gyro_angles_at_frames -= gyro_angles_at_frames[0]
    
    # Convert gyro angles to image-space transforms
    # This is where we need to understand the camera-gyro relationship
    #
    # For a camera looking forward:
    # - Gyro X (roll) -> image rotation around center
    # - Gyro Y (pitch) -> vertical translation (after focal length scaling)
    # - Gyro Z (yaw) -> horizontal translation (after focal length scaling)
    #
    # But for astrophotography, the camera might be pointing up, so axes may differ
    
    # Get camera intrinsics for angle-to-pixel conversion
    # Using GoPro Hero 7 wide mode: ~118° FOV
    # focal_length = (width/2) / tan(hfov/2)
    # For 2704 width, 118° FOV: f ≈ 812 pixels
    
    # We'll estimate the relationship empirically by comparing with OpenCV
    # For now, use theoretical values and let the comparison reveal the mapping
    
    transforms = []
    for i in range(num_frames):
        # Gyro gives us rotation angles
        # For 2D image stabilization:
        # - Z rotation (yaw) causes horizontal shift
        # - Y rotation (pitch) causes vertical shift
        # - X rotation (roll) causes image rotation
        
        roll = gyro_angles_at_frames[i, 0]   # X axis
        pitch = gyro_angles_at_frames[i, 1]  # Y axis
        yaw = gyro_angles_at_frames[i, 2]    # Z axis
        
        # For now, store raw angles - we'll calibrate later
        transforms.append((roll, pitch, yaw))
    
    return transforms, gyro_angles_at_frames


def calibrate_gyro_to_opencv(opencv_transforms: list, gyro_transforms: list) -> dict:
    """
    Find the mapping between gyro angles and OpenCV-detected motion.
    
    This determines:
    1. Which gyro axis corresponds to which image motion (dx, dy, da)
    2. The scale factor (focal length effect)
    3. Any axis inversions
    """
    opencv_arr = np.array(opencv_transforms)  # (N, 3) -> dx, dy, da
    gyro_arr = np.array(gyro_transforms)      # (N, 3) -> roll, pitch, yaw
    
    n = min(len(opencv_arr), len(gyro_arr))
    opencv_arr = opencv_arr[:n]
    gyro_arr = gyro_arr[:n]
    
    # Compute correlations between all pairs
    print("\n=== AXIS CORRELATION MATRIX ===")
    print("            Gyro X(roll)  Gyro Y(pitch)  Gyro Z(yaw)")
    
    correlations = np.zeros((3, 3))
    for i, cv_name in enumerate(['OpenCV dx', 'OpenCV dy', 'OpenCV da']):
        corr_row = []
        for j in range(3):
            if np.std(gyro_arr[:, j]) > 1e-10 and np.std(opencv_arr[:, i]) > 1e-10:
                corr = np.corrcoef(opencv_arr[:, i], gyro_arr[:, j])[0, 1]
            else:
                corr = 0
            correlations[i, j] = corr
            corr_row.append(f"{corr:+.3f}")
        print(f"{cv_name:12s}  {'  '.join(corr_row)}")
    
    # Find best axis mapping
    # For each OpenCV output, find the gyro axis with highest |correlation|
    mapping = {}
    
    # OpenCV da (rotation) should map to one of the gyro axes
    da_corrs = correlations[2, :]
    best_da_axis = np.argmax(np.abs(da_corrs))
    mapping['da'] = {
        'gyro_axis': best_da_axis,
        'axis_name': ['roll', 'pitch', 'yaw'][best_da_axis],
        'correlation': da_corrs[best_da_axis],
        'scale': 1.0 if da_corrs[best_da_axis] > 0 else -1.0
    }
    
    # For translation, we need to consider focal length
    # dx should correlate with yaw or pitch depending on camera orientation
    dx_corrs = correlations[0, :]
    best_dx_axis = np.argmax(np.abs(dx_corrs))
    
    # Compute scale factor: how many pixels per radian
    gyro_range = np.max(gyro_arr[:, best_dx_axis]) - np.min(gyro_arr[:, best_dx_axis])
    opencv_range = np.max(opencv_arr[:, 0]) - np.min(opencv_arr[:, 0])
    dx_scale = opencv_range / gyro_range if gyro_range > 1e-10 else 0
    
    mapping['dx'] = {
        'gyro_axis': best_dx_axis,
        'axis_name': ['roll', 'pitch', 'yaw'][best_dx_axis],
        'correlation': dx_corrs[best_dx_axis],
        'scale': dx_scale if dx_corrs[best_dx_axis] > 0 else -dx_scale
    }
    
    # dy mapping
    dy_corrs = correlations[1, :]
    best_dy_axis = np.argmax(np.abs(dy_corrs))
    
    gyro_range = np.max(gyro_arr[:, best_dy_axis]) - np.min(gyro_arr[:, best_dy_axis])
    opencv_range = np.max(opencv_arr[:, 1]) - np.min(opencv_arr[:, 1])
    dy_scale = opencv_range / gyro_range if gyro_range > 1e-10 else 0
    
    mapping['dy'] = {
        'gyro_axis': best_dy_axis,
        'axis_name': ['roll', 'pitch', 'yaw'][best_dy_axis],
        'correlation': dy_corrs[best_dy_axis],
        'scale': dy_scale if dy_corrs[best_dy_axis] > 0 else -dy_scale
    }
    
    print("\n=== CALIBRATION RESULT ===")
    for output, m in mapping.items():
        print(f"{output}: gyro {m['axis_name']} (axis {m['gyro_axis']}), "
              f"corr={m['correlation']:.3f}, scale={m['scale']:.2f}")
    
    return mapping


def apply_gyro_stabilization(frames: list, gyro_transforms: list, 
                              calibration: dict, size: tuple) -> list:
    """
    Apply stabilization using gyro data with calibrated mapping.
    """
    width, height = size
    stabilized = []
    
    for i, frame in enumerate(frames):
        if i >= len(gyro_transforms):
            stabilized.append(frame)
            continue
        
        gyro = gyro_transforms[i]  # (roll, pitch, yaw)
        
        # Map gyro to image motion using calibration
        dx_axis = calibration['dx']['gyro_axis']
        dy_axis = calibration['dy']['gyro_axis']
        da_axis = calibration['da']['gyro_axis']
        
        # Compute correction (negative of motion to stabilize)
        corr_dx = -gyro[dx_axis] * calibration['dx']['scale']
        corr_dy = -gyro[dy_axis] * calibration['dy']['scale']
        corr_da = -gyro[da_axis] * calibration['da']['scale']
        
        # Build transform matrix
        cx, cy = width / 2, height / 2
        cos_a = np.cos(corr_da)
        sin_a = np.sin(corr_da)
        
        M = np.array([
            [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy + corr_dx],
            [sin_a,  cos_a, -sin_a * cx + (1 - cos_a) * cy + corr_dy]
        ], dtype=np.float32)
        
        stab = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        stabilized.append(stab)
    
    return stabilized


def apply_opencv_stabilization(frames: list, opencv_transforms: list, size: tuple) -> list:
    """Apply stabilization using OpenCV-detected motion."""
    width, height = size
    stabilized = []
    
    for i, frame in enumerate(frames):
        if i >= len(opencv_transforms):
            stabilized.append(frame)
            continue
        
        cum_dx, cum_dy, cum_da = opencv_transforms[i]
        
        # Correction is negative of cumulative motion
        corr_dx = -cum_dx
        corr_dy = -cum_dy
        corr_da = -cum_da
        
        cx, cy = width / 2, height / 2
        cos_a = np.cos(corr_da)
        sin_a = np.sin(corr_da)
        
        M = np.array([
            [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy + corr_dx],
            [sin_a,  cos_a, -sin_a * cx + (1 - cos_a) * cy + corr_dy]
        ], dtype=np.float32)
        
        stab = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        stabilized.append(stab)
    
    return stabilized


def create_comparison_video(frames_original: list, frames_opencv: list, 
                            frames_gyro: list, output_path: str, fps: float):
    """
    Create a 3-way comparison video: Original | OpenCV Stabilized | Gyro Stabilized
    """
    if not frames_original:
        return
    
    height, width = frames_original[0].shape[:2]
    
    # Downscale for output - ensure even dimensions for H.264
    out_scale = 0.4
    out_width = int(width * out_scale)
    out_height = int(height * out_scale)
    # Make dimensions even
    out_width = out_width - (out_width % 2)
    out_height = out_height - (out_height % 2)
    
    print("\nCreating 3-way comparison video...")
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=7,
                                 macro_block_size=1)
    
    n = min(len(frames_original), len(frames_opencv), len(frames_gyro))
    
    for i in tqdm(range(n), desc="Writing"):
        orig = cv2.resize(frames_original[i], (out_width, out_height))
        opencv = cv2.resize(frames_opencv[i], (out_width, out_height))
        gyro = cv2.resize(frames_gyro[i], (out_width, out_height))
        
        # Add labels
        cv2.putText(orig, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(opencv, "OPENCV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(gyro, "GYRO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Combine horizontally
        combined = np.hstack([orig, opencv, gyro])
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        writer.append_data(combined_rgb)
    
    writer.close()
    print(f"Saved: {output_path}")


def plot_comparison(opencv_transforms: list, gyro_transforms: list, 
                    calibration: dict, output_path: str, fps: float):
    """
    Plot the calibrated gyro motion overlaid on OpenCV motion to show how well they match.
    """
    import matplotlib.pyplot as plt
    
    opencv_arr = np.array(opencv_transforms)
    gyro_arr = np.array(gyro_transforms)
    
    n = min(len(opencv_arr), len(gyro_arr))
    time = np.arange(n) / fps
    
    # Apply calibration to gyro data
    gyro_calibrated_dx = gyro_arr[:n, calibration['dx']['gyro_axis']] * calibration['dx']['scale']
    gyro_calibrated_dy = gyro_arr[:n, calibration['dy']['gyro_axis']] * calibration['dy']['scale']
    gyro_calibrated_da = gyro_arr[:n, calibration['da']['gyro_axis']] * calibration['da']['scale']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('OpenCV vs Calibrated Gyro Motion', fontsize=14)
    
    # dx comparison
    ax = axes[0]
    ax.plot(time, opencv_arr[:n, 0], 'b-', label='OpenCV dx', linewidth=1.5)
    ax.plot(time, gyro_calibrated_dx, 'r--', label='Gyro (calibrated)', linewidth=1.5)
    ax.set_ylabel('Translation X (px)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"X Translation - Gyro axis: {calibration['dx']['axis_name']}, corr: {calibration['dx']['correlation']:.3f}")
    
    # dy comparison
    ax = axes[1]
    ax.plot(time, opencv_arr[:n, 1], 'b-', label='OpenCV dy', linewidth=1.5)
    ax.plot(time, gyro_calibrated_dy, 'r--', label='Gyro (calibrated)', linewidth=1.5)
    ax.set_ylabel('Translation Y (px)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Y Translation - Gyro axis: {calibration['dy']['axis_name']}, corr: {calibration['dy']['correlation']:.3f}")
    
    # da comparison
    ax = axes[2]
    ax.plot(time, np.degrees(opencv_arr[:n, 2]), 'b-', label='OpenCV da', linewidth=1.5)
    ax.plot(time, np.degrees(gyro_calibrated_da), 'r--', label='Gyro (calibrated)', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rotation (deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Rotation - Gyro axis: {calibration['da']['axis_name']}, corr: {calibration['da']['correlation']:.3f}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Calibration plot saved: {output_path}")
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python step3_gyro_vs_opencv.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    video_name = Path(video_path).stem
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    output_dir = Path("experiments/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Step 3: Gyro-based Stabilization vs OpenCV Stabilization")
    print("=" * 60)
    
    # 1. Extract OpenCV motion
    frames, opencv_transforms, fps, size = extract_opencv_motion(video_path)
    print(f"\nExtracted {len(frames)} frames at {fps:.2f} fps")
    
    # 2. Extract gyro motion
    gyro_transforms, gyro_angles = extract_gyro_motion(video_path, fps, len(frames))
    
    # 3. Calibrate gyro to match OpenCV
    print("\nCalibrating gyro-to-image mapping...")
    calibration = calibrate_gyro_to_opencv(opencv_transforms, gyro_transforms)
    
    # 4. Apply both stabilizations
    print("\nApplying stabilizations...")
    frames_opencv = apply_opencv_stabilization(frames, opencv_transforms, size)
    frames_gyro = apply_gyro_stabilization(frames, gyro_transforms, calibration, size)
    
    # 5. Create comparison video
    output_video = str(output_dir / f"{video_name}_opencv_vs_gyro.mp4")
    create_comparison_video(frames, frames_opencv, frames_gyro, output_video, fps)
    
    # 6. Create calibration plot
    plot_path = str(output_dir / f"{video_name}_calibration.png")
    plot_comparison(opencv_transforms, gyro_transforms, calibration, plot_path, fps)
    
    # 7. Save calibration for future use
    np.save(str(output_dir / f"{video_name}_calibration.npy"), calibration)
    
    # Open video
    os.startfile(output_video)


if __name__ == "__main__":
    main()

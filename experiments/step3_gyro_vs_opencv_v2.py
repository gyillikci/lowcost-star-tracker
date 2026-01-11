"""
Step 3 v2: Gyro-based Stabilization vs OpenCV Stabilization

Fixed version with proper physical camera-gyro mapping.
For a camera pointing at the sky:
- Gyro X (roll) -> image rotation around optical axis
- Gyro Y (pitch) -> vertical shift (tilt up/down)
- Gyro Z (yaw) -> horizontal shift (pan left/right)

The key is to use the camera's focal length to convert angular motion to pixels.
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm
import imageio

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.star_tracker.gyro_extractor import GyroExtractor


def enhance_stars(gray: np.ndarray) -> np.ndarray:
    """Enhance star visibility with high-pass filter."""
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    enhanced = cv2.subtract(gray, blurred * 0.8)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced


def get_gopro_focal_length(width: int, fov_mode: str = 'wide') -> float:
    """
    Get GoPro focal length in pixels based on FOV mode.
    
    For GoPro Hero 7 Black:
    - Wide: ~118° horizontal FOV
    - Linear: ~86° horizontal FOV
    - Narrow: ~70° horizontal FOV
    """
    fov_degrees = {
        'wide': 118,
        'linear': 86,
        'narrow': 70,
        'superview': 118,
    }
    
    hfov = fov_degrees.get(fov_mode, 118)
    hfov_rad = np.radians(hfov)
    focal_length = (width / 2) / np.tan(hfov_rad / 2)
    
    return focal_length


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


def extract_gyro_cumulative_angles(video_path: str, video_fps: float, num_frames: int) -> np.ndarray:
    """
    Extract gyro data and compute cumulative rotation angles for each video frame.
    
    Returns:
        cumulative_angles: (N, 3) array of cumulative angles in radians [roll, pitch, yaw]
    """
    extractor = GyroExtractor()
    gyro_data = extractor.extract(video_path)
    
    gyro_timestamps = gyro_data.timestamps
    angular_velocity = gyro_data.angular_velocity  # rad/s, shape (N, 3)
    
    print(f"Gyro: {len(gyro_timestamps)} samples over {gyro_timestamps[-1]:.2f}s")
    print(f"Gyro sample rate: {len(gyro_timestamps) / gyro_timestamps[-1]:.1f} Hz")
    
    # Integrate angular velocity to get cumulative angles using trapezoidal rule
    dt = np.diff(gyro_timestamps)
    cumulative_angles = np.zeros_like(angular_velocity)
    
    for i in range(1, len(gyro_timestamps)):
        cumulative_angles[i] = cumulative_angles[i-1] + \
            (angular_velocity[i] + angular_velocity[i-1]) / 2 * dt[i-1]
    
    # Resample to video frame timestamps
    video_timestamps = np.arange(num_frames) / video_fps
    
    # Clip to available gyro data range
    max_gyro_time = gyro_timestamps[-1]
    video_timestamps = np.clip(video_timestamps, 0, max_gyro_time)
    
    gyro_at_frames = np.zeros((num_frames, 3))
    for axis in range(3):
        gyro_at_frames[:, axis] = np.interp(
            video_timestamps, 
            gyro_timestamps, 
            cumulative_angles[:, axis]
        )
    
    # Normalize to start at 0
    gyro_at_frames -= gyro_at_frames[0]
    
    return gyro_at_frames


def gyro_angles_to_pixel_motion(gyro_angles: np.ndarray, focal_length: float, 
                                  width: int, height: int) -> list:
    """
    Convert gyro angles to pixel-space motion.
    
    For small angles and a camera pointing at the sky:
    - Roll (X): rotation around optical axis -> image rotation
    - Pitch (Y): tilt up/down -> vertical pixel shift = focal_length * tan(angle)
    - Yaw (Z): pan left/right -> horizontal pixel shift = focal_length * tan(angle)
    
    Returns:
        transforms: list of (dx, dy, da) in pixels/radians
    """
    transforms = []
    
    for i in range(len(gyro_angles)):
        roll = gyro_angles[i, 0]   # X axis - rotation
        pitch = gyro_angles[i, 1]  # Y axis - vertical
        yaw = gyro_angles[i, 2]    # Z axis - horizontal
        
        # Convert angles to pixel displacement
        # For small angles: tan(angle) ≈ angle
        # But use exact formula for accuracy
        dx = focal_length * np.tan(yaw)    # Yaw causes horizontal shift
        dy = focal_length * np.tan(pitch)  # Pitch causes vertical shift
        da = roll                          # Roll causes image rotation
        
        transforms.append((dx, dy, da))
    
    return transforms


def apply_stabilization(frames: list, transforms: list, size: tuple) -> list:
    """Apply stabilization using cumulative transforms."""
    width, height = size
    stabilized = []
    
    for i, frame in enumerate(frames):
        if i >= len(transforms):
            stabilized.append(frame)
            continue
        
        cum_dx, cum_dy, cum_da = transforms[i]
        
        # Correction = negative of cumulative motion
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


def find_best_axis_mapping(opencv_transforms: list, gyro_angles: np.ndarray, focal_length: float):
    """
    Try different axis mappings and find the one that best matches OpenCV motion.
    
    GoPro gyro axes might not align with camera axes due to mounting orientation.
    Test all permutations and sign flips.
    """
    opencv_arr = np.array(opencv_transforms)
    n = min(len(opencv_arr), len(gyro_angles))
    
    opencv_dx = opencv_arr[:n, 0]
    opencv_dy = opencv_arr[:n, 1]
    opencv_da = opencv_arr[:n, 2]
    
    # Try all axis permutations and sign combinations
    best_score = -np.inf
    best_mapping = None
    
    axis_names = ['X (roll)', 'Y (pitch)', 'Z (yaw)']
    
    print("\n=== SEARCHING FOR BEST AXIS MAPPING ===")
    
    for da_axis in range(3):  # Which gyro axis maps to image rotation
        for da_sign in [-1, 1]:
            for dx_axis in range(3):  # Which gyro axis maps to horizontal shift
                for dx_sign in [-1, 1]:
                    for dy_axis in range(3):  # Which gyro axis maps to vertical shift
                        for dy_sign in [-1, 1]:
                            # Compute predicted motion
                            pred_da = da_sign * gyro_angles[:n, da_axis]
                            pred_dx = dx_sign * focal_length * np.tan(gyro_angles[:n, dx_axis])
                            pred_dy = dy_sign * focal_length * np.tan(gyro_angles[:n, dy_axis])
                            
                            # Compute correlation for each
                            corr_da = np.corrcoef(opencv_da, pred_da)[0, 1] if np.std(pred_da) > 1e-10 else 0
                            corr_dx = np.corrcoef(opencv_dx, pred_dx)[0, 1] if np.std(pred_dx) > 1e-10 else 0
                            corr_dy = np.corrcoef(opencv_dy, pred_dy)[0, 1] if np.std(pred_dy) > 1e-10 else 0
                            
                            # Handle NaN
                            if np.isnan(corr_da): corr_da = 0
                            if np.isnan(corr_dx): corr_dx = 0
                            if np.isnan(corr_dy): corr_dy = 0
                            
                            # Score: sum of correlations (we want all to be positive and high)
                            score = corr_da + corr_dx + corr_dy
                            
                            if score > best_score:
                                best_score = score
                                best_mapping = {
                                    'da_axis': da_axis,
                                    'da_sign': da_sign,
                                    'dx_axis': dx_axis,
                                    'dx_sign': dx_sign,
                                    'dy_axis': dy_axis,
                                    'dy_sign': dy_sign,
                                    'corr_da': corr_da,
                                    'corr_dx': corr_dx,
                                    'corr_dy': corr_dy,
                                    'score': score
                                }
    
    print(f"\nBest mapping (score={best_mapping['score']:.3f}):")
    print(f"  da (rotation): gyro {axis_names[best_mapping['da_axis']]} × {best_mapping['da_sign']:+d}, corr={best_mapping['corr_da']:.3f}")
    print(f"  dx (horizontal): gyro {axis_names[best_mapping['dx_axis']]} × {best_mapping['dx_sign']:+d}, corr={best_mapping['corr_dx']:.3f}")
    print(f"  dy (vertical): gyro {axis_names[best_mapping['dy_axis']]} × {best_mapping['dy_sign']:+d}, corr={best_mapping['corr_dy']:.3f}")
    
    return best_mapping


def apply_gyro_stabilization_with_mapping(frames: list, gyro_angles: np.ndarray,
                                           mapping: dict, focal_length: float, size: tuple) -> list:
    """Apply stabilization using gyro with the discovered axis mapping."""
    width, height = size
    stabilized = []
    
    for i, frame in enumerate(frames):
        if i >= len(gyro_angles):
            stabilized.append(frame)
            continue
        
        # Get gyro angles for this frame
        angles = gyro_angles[i]
        
        # Apply mapping to get motion
        da = mapping['da_sign'] * angles[mapping['da_axis']]
        dx = mapping['dx_sign'] * focal_length * np.tan(angles[mapping['dx_axis']])
        dy = mapping['dy_sign'] * focal_length * np.tan(angles[mapping['dy_axis']])
        
        # Correction = negative of motion
        corr_dx = -dx
        corr_dy = -dy
        corr_da = -da
        
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
    """Create 3-way comparison: Original | OpenCV | Gyro"""
    if not frames_original:
        return
    
    height, width = frames_original[0].shape[:2]
    
    out_scale = 0.4
    out_width = int(width * out_scale)
    out_height = int(height * out_scale)
    out_width = out_width - (out_width % 2)
    out_height = out_height - (out_height % 2)
    
    print("\nCreating 3-way comparison video...")
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=7,
                                 macro_block_size=1)
    
    n = min(len(frames_original), len(frames_opencv), len(frames_gyro))
    
    for i in tqdm(range(n), desc="Writing"):
        orig = cv2.resize(frames_original[i], (out_width, out_height))
        opencv_stab = cv2.resize(frames_opencv[i], (out_width, out_height))
        gyro_stab = cv2.resize(frames_gyro[i], (out_width, out_height))
        
        cv2.putText(orig, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(opencv_stab, "OPENCV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(gyro_stab, "GYRO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        combined = np.hstack([orig, opencv_stab, gyro_stab])
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        writer.append_data(combined_rgb)
    
    writer.close()
    print(f"Saved: {output_path}")


def plot_motion_comparison(opencv_transforms: list, gyro_angles: np.ndarray,
                            mapping: dict, focal_length: float, fps: float, output_path: str):
    """Plot OpenCV motion vs mapped gyro motion."""
    import matplotlib.pyplot as plt
    
    opencv_arr = np.array(opencv_transforms)
    n = min(len(opencv_arr), len(gyro_angles))
    time = np.arange(n) / fps
    
    # Apply mapping to gyro
    gyro_da = mapping['da_sign'] * gyro_angles[:n, mapping['da_axis']]
    gyro_dx = mapping['dx_sign'] * focal_length * np.tan(gyro_angles[:n, mapping['dx_axis']])
    gyro_dy = mapping['dy_sign'] * focal_length * np.tan(gyro_angles[:n, mapping['dy_axis']])
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('OpenCV Motion vs Calibrated Gyro Motion', fontsize=14)
    
    # dx
    ax = axes[0]
    ax.plot(time, opencv_arr[:n, 0], 'b-', label='OpenCV dx', linewidth=1.5)
    ax.plot(time, gyro_dx, 'r--', label='Gyro dx', linewidth=1.5)
    ax.set_ylabel('Horizontal (px)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Horizontal Motion (corr={mapping['corr_dx']:.3f})")
    
    # dy
    ax = axes[1]
    ax.plot(time, opencv_arr[:n, 1], 'b-', label='OpenCV dy', linewidth=1.5)
    ax.plot(time, gyro_dy, 'r--', label='Gyro dy', linewidth=1.5)
    ax.set_ylabel('Vertical (px)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Vertical Motion (corr={mapping['corr_dy']:.3f})")
    
    # da
    ax = axes[2]
    ax.plot(time, np.degrees(opencv_arr[:n, 2]), 'b-', label='OpenCV da', linewidth=1.5)
    ax.plot(time, np.degrees(gyro_da), 'r--', label='Gyro da', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rotation (deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Rotation (corr={mapping['corr_da']:.3f})")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {output_path}")
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python step3_gyro_vs_opencv_v2.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    video_name = Path(video_path).stem
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    output_dir = Path("experiments/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Step 3 v2: Gyro vs OpenCV with Proper Axis Mapping")
    print("=" * 60)
    
    # 1. Extract OpenCV motion
    frames, opencv_transforms, fps, size = extract_opencv_motion(video_path)
    width, height = size
    print(f"\nExtracted {len(frames)} frames at {fps:.2f} fps, {width}x{height}")
    
    # 2. Get focal length
    focal_length = get_gopro_focal_length(width, 'wide')
    print(f"Focal length (wide mode): {focal_length:.1f} px")
    
    # 3. Extract gyro cumulative angles
    gyro_angles = extract_gyro_cumulative_angles(video_path, fps, len(frames))
    
    # Print gyro motion range
    print(f"\nGyro angle ranges:")
    print(f"  X (roll):  {np.degrees(gyro_angles[:, 0].min()):.2f} to {np.degrees(gyro_angles[:, 0].max()):.2f} deg")
    print(f"  Y (pitch): {np.degrees(gyro_angles[:, 1].min()):.2f} to {np.degrees(gyro_angles[:, 1].max()):.2f} deg")
    print(f"  Z (yaw):   {np.degrees(gyro_angles[:, 2].min()):.2f} to {np.degrees(gyro_angles[:, 2].max()):.2f} deg")
    
    # Print OpenCV motion range
    opencv_arr = np.array(opencv_transforms)
    print(f"\nOpenCV motion ranges:")
    print(f"  dx: {opencv_arr[:, 0].min():.1f} to {opencv_arr[:, 0].max():.1f} px")
    print(f"  dy: {opencv_arr[:, 1].min():.1f} to {opencv_arr[:, 1].max():.1f} px")
    print(f"  da: {np.degrees(opencv_arr[:, 2].min()):.2f} to {np.degrees(opencv_arr[:, 2].max()):.2f} deg")
    
    # 4. Find best axis mapping
    mapping = find_best_axis_mapping(opencv_transforms, gyro_angles, focal_length)
    
    # 5. Apply both stabilizations
    print("\nApplying stabilizations...")
    frames_opencv = apply_stabilization(frames, opencv_transforms, size)
    frames_gyro = apply_gyro_stabilization_with_mapping(frames, gyro_angles, mapping, focal_length, size)
    
    # 6. Create comparison video
    output_video = str(output_dir / f"{video_name}_opencv_vs_gyro_v2.mp4")
    create_comparison_video(frames, frames_opencv, frames_gyro, output_video, fps)
    
    # 7. Create comparison plot
    plot_path = str(output_dir / f"{video_name}_mapping_v2.png")
    plot_motion_comparison(opencv_transforms, gyro_angles, mapping, focal_length, fps, plot_path)
    
    # 8. Save mapping
    np.save(str(output_dir / f"{video_name}_mapping_v2.npy"), mapping)
    
    os.startfile(output_video)


if __name__ == "__main__":
    main()

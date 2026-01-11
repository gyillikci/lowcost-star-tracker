"""
Step 3 v3: Direct frame-to-frame comparison

Instead of cumulative integration, compare frame-to-frame motion directly:
- OpenCV frame-to-frame delta (dx, dy, da per frame)
- Gyro angular velocity integrated over one frame period

This avoids cumulative drift issues.
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm
import imageio

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.star_tracker.gyro_extractor import GyroExtractor


def enhance_stars(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    enhanced = cv2.subtract(gray, blurred * 0.8)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced


def get_gopro_focal_length(width: int, fov_mode: str = 'wide') -> float:
    fov_degrees = {'wide': 118, 'linear': 86, 'narrow': 70}
    hfov = fov_degrees.get(fov_mode, 118)
    hfov_rad = np.radians(hfov)
    return (width / 2) / np.tan(hfov_rad / 2)


def extract_opencv_frame_deltas(video_path: str) -> tuple:
    """Extract frame-to-frame motion deltas using phase correlation."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    proc_scale = 0.25
    proc_width = int(width * proc_scale)
    proc_height = int(height * proc_scale)
    
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read video")
    
    frames = [prev_frame]
    
    prev_small = cv2.resize(prev_frame, (proc_width, proc_height))
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    prev_enhanced = enhance_stars(prev_gray)
    
    hann = cv2.createHanningWindow((proc_width, proc_height), cv2.CV_32F)
    
    # Store frame-to-frame deltas (not cumulative)
    deltas = []  # Each entry: (dx, dy, da) for this frame transition
    
    print("Extracting OpenCV frame-to-frame deltas...")
    for i in tqdm(range(frame_count - 1), desc="OpenCV"):
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        frames.append(curr_frame)
        
        curr_small = cv2.resize(curr_frame, (proc_width, proc_height))
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        curr_enhanced = enhance_stars(curr_gray)
        
        shift, response = cv2.phaseCorrelate(prev_enhanced, curr_enhanced, hann)
        dx, dy = shift[0] / proc_scale, shift[1] / proc_scale
        
        da = 0.0
        if response > 0.05:
            try:
                h, w = prev_enhanced.shape
                M_trans = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
                curr_aligned = cv2.warpAffine(curr_enhanced, M_trans, (w, h))
                
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
                
                margin = int(min(h, w) * 0.2)
                prev_crop = prev_enhanced[margin:-margin, margin:-margin]
                curr_crop = curr_aligned[margin:-margin, margin:-margin]
                
                prev_norm = cv2.normalize(prev_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                curr_norm = cv2.normalize(curr_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                _, warp_matrix = cv2.findTransformECC(
                    prev_norm, curr_norm, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
                )
                da = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])
            except cv2.error:
                da = 0.0
        
        deltas.append((dx, dy, da))
        prev_enhanced = curr_enhanced
    
    cap.release()
    return frames, deltas, fps, (width, height)


def extract_gyro_frame_deltas(video_path: str, fps: float, num_frames: int) -> np.ndarray:
    """
    Get gyro angular change per video frame.
    
    Returns:
        frame_deltas: (N-1, 3) array of angular changes [droll, dpitch, dyaw] in radians per frame
    """
    extractor = GyroExtractor()
    gyro_data = extractor.extract(video_path)
    
    gyro_timestamps = gyro_data.timestamps
    angular_velocity = gyro_data.angular_velocity  # rad/s
    
    print(f"Gyro: {len(gyro_timestamps)} samples, {gyro_timestamps[-1]:.2f}s, ~{len(gyro_timestamps)/gyro_timestamps[-1]:.0f} Hz")
    
    # For each frame transition, integrate gyro over that time period
    frame_period = 1.0 / fps
    frame_deltas = []
    
    for frame_idx in range(num_frames - 1):
        t_start = frame_idx / fps
        t_end = (frame_idx + 1) / fps
        
        # Find gyro samples in this time window
        mask = (gyro_timestamps >= t_start) & (gyro_timestamps < t_end)
        
        if np.sum(mask) > 0:
            # Integrate angular velocity over this period
            # Simple average * dt
            avg_omega = np.mean(angular_velocity[mask], axis=0)
            delta_angle = avg_omega * frame_period
        else:
            # Interpolate if no samples in window
            delta_angle = np.zeros(3)
            for axis in range(3):
                omega_interp = np.interp((t_start + t_end) / 2, gyro_timestamps, angular_velocity[:, axis])
                delta_angle[axis] = omega_interp * frame_period
        
        frame_deltas.append(delta_angle)
    
    return np.array(frame_deltas)


def find_axis_mapping_from_deltas(opencv_deltas: list, gyro_deltas: np.ndarray, focal_length: float):
    """
    Find mapping between gyro axes and OpenCV-detected motion using frame deltas.
    """
    opencv_arr = np.array(opencv_deltas)
    n = min(len(opencv_arr), len(gyro_deltas))
    
    opencv_dx = opencv_arr[:n, 0]
    opencv_dy = opencv_arr[:n, 1]
    opencv_da = opencv_arr[:n, 2]
    
    print("\n=== FRAME-TO-FRAME MOTION STATISTICS ===")
    print(f"OpenCV dx: std={np.std(opencv_dx):.2f} px, range=[{opencv_dx.min():.1f}, {opencv_dx.max():.1f}]")
    print(f"OpenCV dy: std={np.std(opencv_dy):.2f} px, range=[{opencv_dy.min():.1f}, {opencv_dy.max():.1f}]")
    print(f"OpenCV da: std={np.degrees(np.std(opencv_da)):.4f} deg, range=[{np.degrees(opencv_da.min()):.4f}, {np.degrees(opencv_da.max()):.4f}]")
    
    print(f"\nGyro dX: std={np.degrees(np.std(gyro_deltas[:n, 0])):.4f} deg")
    print(f"Gyro dY: std={np.degrees(np.std(gyro_deltas[:n, 1])):.4f} deg")
    print(f"Gyro dZ: std={np.degrees(np.std(gyro_deltas[:n, 2])):.4f} deg")
    
    # Search for best mapping
    best_score = -np.inf
    best_mapping = None
    
    for da_axis in range(3):
        for da_sign in [-1, 1]:
            for dx_axis in range(3):
                for dx_sign in [-1, 1]:
                    for dy_axis in range(3):
                        for dy_sign in [-1, 1]:
                            pred_da = da_sign * gyro_deltas[:n, da_axis]
                            pred_dx = dx_sign * focal_length * gyro_deltas[:n, dx_axis]
                            pred_dy = dy_sign * focal_length * gyro_deltas[:n, dy_axis]
                            
                            # Use correlation for matching
                            def safe_corr(a, b):
                                if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                                    return 0
                                c = np.corrcoef(a, b)[0, 1]
                                return c if not np.isnan(c) else 0
                            
                            corr_da = safe_corr(opencv_da, pred_da)
                            corr_dx = safe_corr(opencv_dx, pred_dx)
                            corr_dy = safe_corr(opencv_dy, pred_dy)
                            
                            score = corr_da + corr_dx + corr_dy
                            
                            if score > best_score:
                                best_score = score
                                best_mapping = {
                                    'da_axis': da_axis, 'da_sign': da_sign, 'corr_da': corr_da,
                                    'dx_axis': dx_axis, 'dx_sign': dx_sign, 'corr_dx': corr_dx,
                                    'dy_axis': dy_axis, 'dy_sign': dy_sign, 'corr_dy': corr_dy,
                                    'score': score
                                }
    
    axis_names = ['X(roll)', 'Y(pitch)', 'Z(yaw)']
    print(f"\n=== BEST AXIS MAPPING (score={best_mapping['score']:.3f}) ===")
    print(f"  da: gyro {axis_names[best_mapping['da_axis']]} × {best_mapping['da_sign']:+d}, corr={best_mapping['corr_da']:.3f}")
    print(f"  dx: gyro {axis_names[best_mapping['dx_axis']]} × {best_mapping['dx_sign']:+d}, corr={best_mapping['corr_dx']:.3f}")
    print(f"  dy: gyro {axis_names[best_mapping['dy_axis']]} × {best_mapping['dy_sign']:+d}, corr={best_mapping['corr_dy']:.3f}")
    
    return best_mapping


def stabilize_with_deltas(frames: list, deltas: list, size: tuple) -> list:
    """Stabilize using cumulative sum of frame deltas."""
    width, height = size
    stabilized = [frames[0].copy()]  # First frame unchanged
    
    cum_dx, cum_dy, cum_da = 0.0, 0.0, 0.0
    
    for i, frame in enumerate(frames[1:]):
        if i < len(deltas):
            dx, dy, da = deltas[i]
            cum_dx += dx
            cum_dy += dy
            cum_da += da
        
        # Correction = negative of cumulative motion
        cx, cy = width / 2, height / 2
        cos_a = np.cos(-cum_da)
        sin_a = np.sin(-cum_da)
        
        M = np.array([
            [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy - cum_dx],
            [sin_a,  cos_a, -sin_a * cx + (1 - cos_a) * cy - cum_dy]
        ], dtype=np.float32)
        
        stab = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        stabilized.append(stab)
    
    return stabilized


def gyro_deltas_to_image_deltas(gyro_deltas: np.ndarray, mapping: dict, focal_length: float) -> list:
    """Convert gyro frame deltas to image motion deltas using mapping."""
    deltas = []
    for i in range(len(gyro_deltas)):
        da = mapping['da_sign'] * gyro_deltas[i, mapping['da_axis']]
        dx = mapping['dx_sign'] * focal_length * gyro_deltas[i, mapping['dx_axis']]
        dy = mapping['dy_sign'] * focal_length * gyro_deltas[i, mapping['dy_axis']]
        deltas.append((dx, dy, da))
    return deltas


def create_comparison_video(frames_orig: list, frames_opencv: list, 
                            frames_gyro: list, output_path: str, fps: float):
    if not frames_orig:
        return
    
    height, width = frames_orig[0].shape[:2]
    out_scale = 0.4
    out_w = int(width * out_scale) // 2 * 2
    out_h = int(height * out_scale) // 2 * 2
    
    print("\nCreating comparison video...")
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=7, macro_block_size=1)
    
    n = min(len(frames_orig), len(frames_opencv), len(frames_gyro))
    for i in tqdm(range(n), desc="Writing"):
        orig = cv2.resize(frames_orig[i], (out_w, out_h))
        ocv = cv2.resize(frames_opencv[i], (out_w, out_h))
        gyro = cv2.resize(frames_gyro[i], (out_w, out_h))
        
        cv2.putText(orig, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(ocv, "OPENCV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(gyro, "GYRO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        combined = np.hstack([orig, ocv, gyro])
        writer.append_data(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    
    writer.close()
    print(f"Saved: {output_path}")


def plot_delta_comparison(opencv_deltas: list, gyro_deltas: np.ndarray, 
                           mapping: dict, focal_length: float, fps: float, output_path: str):
    import matplotlib.pyplot as plt
    
    opencv_arr = np.array(opencv_deltas)
    n = min(len(opencv_arr), len(gyro_deltas))
    time = np.arange(n) / fps
    
    # Convert gyro to image space
    gyro_da = mapping['da_sign'] * gyro_deltas[:n, mapping['da_axis']]
    gyro_dx = mapping['dx_sign'] * focal_length * gyro_deltas[:n, mapping['dx_axis']]
    gyro_dy = mapping['dy_sign'] * focal_length * gyro_deltas[:n, mapping['dy_axis']]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Frame-to-Frame Motion: OpenCV vs Gyro', fontsize=14)
    
    ax = axes[0]
    ax.plot(time, opencv_arr[:n, 0], 'b-', alpha=0.7, label='OpenCV dx')
    ax.plot(time, gyro_dx, 'r-', alpha=0.7, label='Gyro dx')
    ax.set_ylabel('dx (px/frame)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Horizontal: corr={mapping['corr_dx']:.3f}")
    
    ax = axes[1]
    ax.plot(time, opencv_arr[:n, 1], 'b-', alpha=0.7, label='OpenCV dy')
    ax.plot(time, gyro_dy, 'r-', alpha=0.7, label='Gyro dy')
    ax.set_ylabel('dy (px/frame)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Vertical: corr={mapping['corr_dy']:.3f}")
    
    ax = axes[2]
    ax.plot(time, np.degrees(opencv_arr[:n, 2]), 'b-', alpha=0.7, label='OpenCV da')
    ax.plot(time, np.degrees(gyro_da), 'r-', alpha=0.7, label='Gyro da')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('da (deg/frame)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Rotation: corr={mapping['corr_da']:.3f}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved: {output_path}")
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python step3_gyro_vs_opencv_v3.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    video_name = Path(video_path).stem
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    output_dir = Path("experiments/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Step 3 v3: Frame-to-Frame Delta Comparison")
    print("=" * 60)
    
    # 1. Extract OpenCV frame deltas
    frames, opencv_deltas, fps, size = extract_opencv_frame_deltas(video_path)
    width, height = size
    print(f"\n{len(frames)} frames, {fps:.2f} fps, {width}x{height}")
    
    focal_length = get_gopro_focal_length(width, 'wide')
    print(f"Focal length: {focal_length:.1f} px")
    
    # 2. Extract gyro frame deltas
    gyro_deltas = extract_gyro_frame_deltas(video_path, fps, len(frames))
    
    # 3. Find axis mapping
    mapping = find_axis_mapping_from_deltas(opencv_deltas, gyro_deltas, focal_length)
    
    # 4. Convert gyro deltas to image deltas
    gyro_image_deltas = gyro_deltas_to_image_deltas(gyro_deltas, mapping, focal_length)
    
    # 5. Apply stabilization
    print("\nApplying stabilizations...")
    frames_opencv = stabilize_with_deltas(frames, opencv_deltas, size)
    frames_gyro = stabilize_with_deltas(frames, gyro_image_deltas, size)
    
    # 6. Create comparison video
    output_video = str(output_dir / f"{video_name}_opencv_vs_gyro_v3.mp4")
    create_comparison_video(frames, frames_opencv, frames_gyro, output_video, fps)
    
    # 7. Plot comparison
    plot_path = str(output_dir / f"{video_name}_deltas_v3.png")
    plot_delta_comparison(opencv_deltas, gyro_deltas, mapping, focal_length, fps, plot_path)
    
    os.startfile(output_video)


if __name__ == "__main__":
    main()

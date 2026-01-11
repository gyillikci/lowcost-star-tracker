"""
Step 1: OpenCV-based Video Stabilization (V2 - Optimized for Night Sky)

Uses phase correlation for global motion estimation, which works better
for dark scenes with stars. Also tries ECC (Enhanced Correlation Coefficient)
for sub-pixel accuracy.
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm
import imageio


def extract_motion_phase_correlation(video_path: str, use_ecc_refinement: bool = True) -> tuple:
    """
    Extract frame-to-frame motion using phase correlation.
    Works better for night sky videos than optical flow.
    
    Returns:
        transforms: List of (dx, dy, da) for each frame transition
        frame_count: Total number of frames
        fps: Video FPS
        frame_size: (width, height)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use smaller resolution for faster processing
    proc_scale = 0.25
    proc_width = int(width * proc_scale)
    proc_height = int(height * proc_scale)
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read video")
    
    prev_small = cv2.resize(prev_frame, (proc_width, proc_height))
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Apply high-pass filter to enhance stars
    prev_gray = enhance_stars(prev_gray)
    
    transforms = []
    
    # Hanning window for phase correlation
    hann = cv2.createHanningWindow((proc_width, proc_height), cv2.CV_32F)
    
    print("Extracting motion with phase correlation...")
    for i in tqdm(range(frame_count - 1), desc="Analyzing"):
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        curr_small = cv2.resize(curr_frame, (proc_width, proc_height))
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        curr_gray = enhance_stars(curr_gray)
        
        # Phase correlation for translation
        shift, response = cv2.phaseCorrelate(prev_gray, curr_gray, hann)
        dx, dy = shift[0], shift[1]
        
        # Estimate rotation using ECC if enabled
        da = 0.0
        if use_ecc_refinement and response > 0.1:
            da = estimate_rotation_ecc(prev_gray, curr_gray, dx, dy)
        
        # Scale back to full resolution
        dx = dx / proc_scale
        dy = dy / proc_scale
        
        transforms.append((dx, dy, da))
        prev_gray = curr_gray
    
    cap.release()
    return transforms, frame_count, fps, (width, height)


def enhance_stars(gray: np.ndarray) -> np.ndarray:
    """
    Enhance star visibility by applying high-pass filter and contrast stretch.
    """
    # Subtract blurred version (high-pass filter)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    enhanced = cv2.subtract(gray, blurred * 0.8)
    
    # Normalize
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    
    return enhanced


def estimate_rotation_ecc(prev: np.ndarray, curr: np.ndarray, dx: float, dy: float) -> float:
    """
    Estimate rotation angle using ECC algorithm after compensating for translation.
    """
    h, w = prev.shape
    
    # First compensate for translation
    M_trans = np.float32([[1, 0, -dx], [0, 1, -dy]])
    curr_aligned = cv2.warpAffine(curr, M_trans, (w, h))
    
    # Now estimate rotation using ECC with Euclidean motion model
    # Euclidean = rotation + translation (but we already removed translation)
    try:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        
        # Use only central region for rotation estimation
        margin = int(min(h, w) * 0.2)
        prev_crop = prev[margin:-margin, margin:-margin]
        curr_crop = curr_aligned[margin:-margin, margin:-margin]
        
        # Normalize to 0-255 range for ECC
        prev_norm = cv2.normalize(prev_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        curr_norm = cv2.normalize(curr_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        _, warp_matrix = cv2.findTransformECC(
            prev_norm, curr_norm, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
        )
        
        # Extract rotation angle from matrix
        da = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])
        return da
        
    except cv2.error:
        return 0.0


def smooth_trajectory(transforms: list, smoothing_radius: int = 30) -> tuple:
    """
    Smooth the cumulative trajectory using a moving average filter.
    """
    trajectory = []
    x, y, a = 0, 0, 0
    for dx, dy, da in transforms:
        x += dx
        y += dy
        a += da
        trajectory.append((x, y, a))
    
    trajectory = np.array(trajectory)
    smoothed_trajectory = np.copy(trajectory)
    
    for i in range(3):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], smoothing_radius)
    
    return trajectory, smoothed_trajectory


def moving_average(data: np.ndarray, radius: int) -> np.ndarray:
    """Apply moving average filter with edge handling."""
    window_size = 2 * radius + 1
    kernel = np.ones(window_size) / window_size
    padded = np.pad(data, radius, mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def stabilize_and_compare(video_path: str, output_path: str, smoothing_radius: int = 30):
    """
    Stabilize video and create side-by-side comparison.
    """
    # Extract motion
    transforms, frame_count, fps, (width, height) = extract_motion_phase_correlation(video_path)
    
    # Smooth trajectory
    trajectory, smoothed_trajectory = smooth_trajectory(transforms, smoothing_radius)
    
    # Calculate corrections
    corrections = smoothed_trajectory - trajectory
    
    # Print motion statistics
    print(f"\n=== MOTION DETECTED ===")
    print(f"Translation X: {trajectory[-1, 0]:.1f} px total, max delta = {max(abs(t[0]) for t in transforms):.2f} px/frame")
    print(f"Translation Y: {trajectory[-1, 1]:.1f} px total, max delta = {max(abs(t[1]) for t in transforms):.2f} px/frame")
    print(f"Rotation: {np.degrees(trajectory[-1, 2]):.2f} deg total, max delta = {np.degrees(max(abs(t[2]) for t in transforms)):.4f} deg/frame")
    
    # Open video for reading
    cap = cv2.VideoCapture(video_path)
    
    # Prepare output with imageio for better codec support
    print("\nApplying stabilization...")
    
    frames_original = []
    frames_stabilized = []
    
    ret, first_frame = cap.read()
    if ret:
        frames_original.append(first_frame)
        frames_stabilized.append(first_frame.copy())
    
    for i in tqdm(range(len(transforms)), desc="Stabilizing"):
        ret, frame = cap.read()
        if not ret:
            break
        
        frames_original.append(frame)
        
        # Get correction for this frame
        corr_dx = corrections[i, 0]
        corr_dy = corrections[i, 1]
        corr_da = corrections[i, 2]
        
        # Build transform matrix
        cx, cy = width / 2, height / 2
        cos_a = np.cos(corr_da)
        sin_a = np.sin(corr_da)
        
        M = np.array([
            [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy + corr_dx],
            [sin_a,  cos_a, -sin_a * cx + (1 - cos_a) * cy + corr_dy]
        ], dtype=np.float32)
        
        stabilized = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        frames_stabilized.append(stabilized)
    
    cap.release()
    
    # Create side-by-side comparison video
    print("\nCreating comparison video...")
    
    # Downscale for reasonable file size
    out_scale = 0.5
    out_width = int(width * out_scale)
    out_height = int(height * out_scale)
    
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=7,
                                 macro_block_size=1)  # Avoid resizing
    
    for i, (orig, stab) in enumerate(tqdm(zip(frames_original, frames_stabilized), 
                                           total=len(frames_original), desc="Writing")):
        # Resize
        orig_small = cv2.resize(orig, (out_width, out_height))
        stab_small = cv2.resize(stab, (out_width, out_height))
        
        # Add labels
        cv2.putText(orig_small, "ORIGINAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(stab_small, "STABILIZED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Combine side by side
        combined = np.hstack([orig_small, stab_small])
        
        # Convert BGR to RGB for imageio
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        writer.append_data(combined_rgb)
    
    writer.close()
    print(f"\nSaved: {output_path}")
    
    return transforms, trajectory, smoothed_trajectory, fps


def main():
    if len(sys.argv) < 2:
        print("Usage: python step1_opencv_stabilization_v2.py <video_path> [smoothing_radius]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    smoothing_radius = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    output_dir = Path("experiments/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_name = Path(video_path).stem
    output_path = str(output_dir / f"{video_name}_stabilized_v2.mp4")
    
    print(f"Processing: {video_path}")
    print(f"Smoothing radius: {smoothing_radius} frames")
    print("=" * 50)
    
    transforms, trajectory, smoothed_trajectory, fps = stabilize_and_compare(
        video_path, output_path, smoothing_radius
    )
    
    # Save motion data
    np.savez(
        str(output_dir / f"{video_name}_motion_v2.npz"),
        transforms=transforms,
        trajectory=trajectory,
        smoothed_trajectory=smoothed_trajectory,
        fps=fps
    )
    
    # Open video
    os.startfile(output_path)


if __name__ == "__main__":
    main()

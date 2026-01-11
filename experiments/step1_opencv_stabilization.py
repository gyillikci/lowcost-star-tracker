"""
Step 1: OpenCV-based Video Stabilization

Uses optical flow (Lucas-Kanade) to track features between frames,
estimates affine/homography transforms, and applies smoothing for stabilization.
Outputs a side-by-side comparison video.
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm


def extract_motion_opencv(video_path: str, max_corners: int = 200, scale: float = 0.5) -> tuple:
    """
    Extract frame-to-frame motion using optical flow.
    
    Args:
        video_path: Path to video file
        max_corners: Max features to track
        scale: Downscale factor for faster processing (0.5 = half resolution)
    
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
    
    # Scaled dimensions for processing
    proc_width = int(width * scale)
    proc_height = int(height * scale)
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read video")
    
    # Downscale for faster processing
    prev_small = cv2.resize(prev_frame, (proc_width, proc_height))
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    
    # Store transforms: dx, dy, da (translation x, y, rotation angle)
    transforms = []
    
    # Lucas-Kanade optical flow parameters
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
    # Feature detection parameters
    feature_params = dict(
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=30,
        blockSize=3
    )
    
    print("Extracting motion with OpenCV optical flow...")
    for i in tqdm(range(frame_count - 1), desc="Analyzing"):
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        # Downscale for faster processing
        curr_small = cv2.resize(curr_frame, (proc_width, proc_height))
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
        
        # Detect features in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
        
        if prev_pts is None or len(prev_pts) < 10:
            # Not enough features, assume no motion
            transforms.append((0, 0, 0))
            prev_gray = curr_gray
            continue
        
        # Track features to current frame
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
        
        # Filter valid points
        valid_prev = prev_pts[status.flatten() == 1]
        valid_curr = curr_pts[status.flatten() == 1]
        
        if len(valid_prev) < 4:
            transforms.append((0, 0, 0))
            prev_gray = curr_gray
            continue
        
        # Estimate affine transform (rotation + translation)
        # Use RANSAC for robustness
        m, inliers = cv2.estimateAffinePartial2D(valid_prev, valid_curr)
        
        if m is None:
            transforms.append((0, 0, 0))
        else:
            # Extract translation and rotation from affine matrix
            # m = [[cos(a)*s, -sin(a)*s, tx],
            #      [sin(a)*s,  cos(a)*s, ty]]
            # Scale translations back to full resolution
            dx = m[0, 2] / scale
            dy = m[1, 2] / scale
            da = np.arctan2(m[1, 0], m[0, 0])  # rotation angle in radians
            transforms.append((dx, dy, da))
        
        prev_gray = curr_gray
    
    cap.release()
    return transforms, frame_count, fps, (width, height)


def smooth_trajectory(transforms: list, smoothing_radius: int = 30) -> list:
    """
    Smooth the cumulative trajectory using a moving average filter.
    
    Args:
        transforms: List of (dx, dy, da) frame-to-frame transforms
        smoothing_radius: Window size for moving average
        
    Returns:
        smooth_transforms: Smoothed transforms to apply
    """
    # Calculate cumulative trajectory
    trajectory = []
    x, y, a = 0, 0, 0
    for dx, dy, da in transforms:
        x += dx
        y += dy
        a += da
        trajectory.append((x, y, a))
    
    trajectory = np.array(trajectory)
    
    # Apply moving average filter
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):  # x, y, a
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], smoothing_radius)
    
    # Calculate difference between smoothed and original trajectory
    # This gives us the correction to apply
    diff = smoothed_trajectory - trajectory
    
    # Convert back to frame-to-frame transforms with correction
    smooth_transforms = []
    for i, (dx, dy, da) in enumerate(transforms):
        smooth_transforms.append((
            dx + diff[i, 0],
            dy + diff[i, 1],
            da + diff[i, 2]
        ))
    
    return smooth_transforms, trajectory, smoothed_trajectory


def moving_average(data: np.ndarray, radius: int) -> np.ndarray:
    """Apply moving average filter with edge handling."""
    window_size = 2 * radius + 1
    kernel = np.ones(window_size) / window_size
    
    # Pad the data to handle edges
    padded = np.pad(data, radius, mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    
    return smoothed


def stabilize_video_opencv(video_path: str, output_path: str, smoothing_radius: int = 30):
    """
    Stabilize video using OpenCV optical flow and output side-by-side comparison.
    """
    # Step 1: Extract motion
    transforms, frame_count, fps, (width, height) = extract_motion_opencv(video_path)
    
    # Step 2: Smooth trajectory
    smooth_transforms, trajectory, smoothed_trajectory = smooth_trajectory(transforms, smoothing_radius)
    
    # Step 3: Apply stabilization and create side-by-side video
    cap = cv2.VideoCapture(video_path)
    
    # Output video: side by side (2x width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    print("\nApplying stabilization and creating comparison video...")
    
    # Read first frame (no transform needed)
    ret, first_frame = cap.read()
    if ret:
        combined = np.hstack([first_frame, first_frame])
        # Add labels
        cv2.putText(combined, "ORIGINAL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(combined, "STABILIZED (OpenCV)", (width + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        out.write(combined)
    
    # Cumulative transform for stabilization
    cum_dx, cum_dy, cum_da = 0, 0, 0
    smooth_cum_dx, smooth_cum_dy, smooth_cum_da = 0, 0, 0
    
    for i in tqdm(range(len(smooth_transforms)), desc="Stabilizing"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Accumulate original trajectory
        dx, dy, da = transforms[i]
        cum_dx += dx
        cum_dy += dy
        cum_da += da
        
        # Accumulate smoothed trajectory
        s_dx, s_dy, s_da = smooth_transforms[i]
        smooth_cum_dx += s_dx
        smooth_cum_dy += s_dy
        smooth_cum_da += s_da
        
        # Calculate correction (difference between smooth and original)
        corr_dx = smooth_cum_dx - cum_dx
        corr_dy = smooth_cum_dy - cum_dy
        corr_da = smooth_cum_da - cum_da
        
        # Build affine transform matrix for correction
        cos_a = np.cos(corr_da)
        sin_a = np.sin(corr_da)
        
        # Transform around center
        cx, cy = width / 2, height / 2
        
        # Affine matrix: rotate around center, then translate
        m = np.array([
            [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy + corr_dx],
            [sin_a,  cos_a, -sin_a * cx + (1 - cos_a) * cy + corr_dy]
        ], dtype=np.float32)
        
        # Apply transform
        stabilized = cv2.warpAffine(frame, m, (width, height), borderMode=cv2.BORDER_REPLICATE)
        
        # Combine side by side
        combined = np.hstack([frame, stabilized])
        
        # Add labels
        cv2.putText(combined, "ORIGINAL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(combined, "STABILIZED (OpenCV)", (width + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Add frame number and correction info
        info = f"Frame {i+1} | Corr: dx={corr_dx:.1f}px, dy={corr_dy:.1f}px, da={np.degrees(corr_da):.2f}deg"
        cv2.putText(combined, info, (50, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(combined)
    
    cap.release()
    out.release()
    
    print(f"\nOutput saved: {output_path}")
    
    # Return trajectory data for later analysis
    return transforms, trajectory, smoothed_trajectory, fps


def main():
    if len(sys.argv) < 2:
        print("Usage: python step1_opencv_stabilization.py <video_path> [smoothing_radius]")
        print("Example: python step1_opencv_stabilization.py examples/GX010911.MP4 30")
        sys.exit(1)
    
    video_path = sys.argv[1]
    smoothing_radius = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path("experiments/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output filename
    video_name = Path(video_path).stem
    output_path = str(output_dir / f"{video_name}_opencv_stabilized.mp4")
    
    print(f"Processing: {video_path}")
    print(f"Smoothing radius: {smoothing_radius} frames")
    print("=" * 50)
    
    transforms, trajectory, smoothed_trajectory, fps = stabilize_video_opencv(
        video_path, output_path, smoothing_radius
    )
    
    # Save trajectory data for step 2
    np.savez(
        str(output_dir / f"{video_name}_opencv_motion.npz"),
        transforms=transforms,
        trajectory=trajectory,
        smoothed_trajectory=smoothed_trajectory,
        fps=fps
    )
    print(f"Motion data saved: {output_dir / f'{video_name}_opencv_motion.npz'}")
    
    # Open the video
    os.startfile(output_path)


if __name__ == "__main__":
    main()

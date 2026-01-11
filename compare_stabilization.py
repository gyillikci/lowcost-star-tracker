"""
Video Stabilization Comparison: OpenCV vs IMU-based

This script creates a side-by-side comparison video showing:
- Left: OpenCV optical flow based stabilization
- Right: IMU/Gyroscope based stabilization

Usage:
    python compare_stabilization.py input_video.mp4 -o comparison_output.mp4
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional
import time

import numpy as np
import cv2

# Import our gyro stabilizer
from gyro_stabilizer import GyroStabilizer, IMUData, slerp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenCVStabilizer:
    """
    OpenCV-based video stabilization using optical flow.
    
    This uses feature tracking to estimate frame-to-frame motion
    and then smooths the camera trajectory.
    """
    
    def __init__(
        self,
        smoothing_radius: int = 30,
        crop_ratio: float = 0.9,
    ):
        self.smoothing_radius = smoothing_radius
        self.crop_ratio = crop_ratio
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3
        )
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    
    def analyze_motion(self, video_path: Path) -> Tuple[np.ndarray, int, int, float]:
        """
        Analyze video motion and compute cumulative transforms.
        
        Returns:
            (transforms, width, height, fps)
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Analyzing motion in {n_frames} frames...")
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read video")
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Store transforms (dx, dy, da)
        transforms = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect features in previous frame
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, **self.feature_params)
            
            if prev_pts is not None and len(prev_pts) > 0:
                # Track features
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_pts, None, **self.lk_params
                )
                
                # Filter good points
                idx = np.where(status == 1)[0]
                if len(idx) >= 4:
                    prev_good = prev_pts[idx]
                    curr_good = curr_pts[idx]
                    
                    # Estimate affine transform
                    m, _ = cv2.estimateAffinePartial2D(prev_good, curr_good)
                    
                    if m is not None:
                        dx = m[0, 2]
                        dy = m[1, 2]
                        da = np.arctan2(m[1, 0], m[0, 0])
                        transforms.append([dx, dy, da])
                    else:
                        transforms.append([0, 0, 0])
                else:
                    transforms.append([0, 0, 0])
            else:
                transforms.append([0, 0, 0])
            
            prev_gray = gray
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logger.info(f"Analyzed {frame_idx}/{n_frames} frames")
        
        cap.release()
        
        return np.array(transforms), width, height, fps
    
    def smooth_trajectory(self, transforms: np.ndarray) -> np.ndarray:
        """Smooth the camera trajectory using moving average."""
        # Compute cumulative trajectory
        trajectory = np.cumsum(transforms, axis=0)
        
        # Smooth trajectory
        smoothed = np.zeros_like(trajectory)
        for i in range(3):
            smoothed[:, i] = self._moving_average(trajectory[:, i], self.smoothing_radius)
        
        # Compute smoothed transforms
        smooth_transforms = transforms + (smoothed - trajectory)
        
        return smooth_transforms
    
    def _moving_average(self, data: np.ndarray, radius: int) -> np.ndarray:
        """Apply moving average filter."""
        ret = np.cumsum(data, dtype=float)
        ret[radius:] = ret[radius:] - ret[:-radius]
        
        result = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - radius)
            end = min(len(data), i + radius + 1)
            result[i] = np.mean(data[start:end])
        
        return result
    
    def stabilize(self, video_path: Path, output_path: Path) -> Path:
        """Stabilize video using optical flow."""
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        # Analyze motion
        transforms, width, height, fps = self.analyze_motion(video_path)
        
        # Smooth trajectory
        smooth_transforms = self.smooth_trajectory(transforms)
        
        # Apply stabilization
        cap = cv2.VideoCapture(str(video_path))
        
        out_width = int(width * self.crop_ratio)
        out_height = int(height * self.crop_ratio)
        out_width = out_width - (out_width % 2)
        out_height = out_height - (out_height % 2)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
        
        n_frames = len(smooth_transforms) + 1
        logger.info(f"Applying stabilization to {n_frames} frames...")
        
        # First frame (no transform)
        ret, frame = cap.read()
        if ret:
            x = (width - out_width) // 2
            y = (height - out_height) // 2
            cropped = frame[y:y+out_height, x:x+out_width]
            writer.write(cropped)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= len(smooth_transforms):
                break
            
            dx, dy, da = smooth_transforms[frame_idx]
            
            # Build transform matrix
            m = np.array([
                [np.cos(da), -np.sin(da), dx],
                [np.sin(da), np.cos(da), dy]
            ], dtype=np.float32)
            
            # Apply transform
            stabilized = cv2.warpAffine(frame, m, (width, height))
            
            # Center crop
            x = (width - out_width) // 2
            y = (height - out_height) // 2
            cropped = stabilized[y:y+out_height, x:x+out_width]
            
            writer.write(cropped)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logger.info(f"Stabilized {frame_idx}/{n_frames} frames")
        
        cap.release()
        writer.release()
        
        logger.info(f"Saved OpenCV stabilized video to {output_path}")
        return output_path


def create_comparison_video(
    original_path: Path,
    opencv_path: Path,
    imu_path: Path,
    output_path: Path,
    include_original: bool = True
) -> Path:
    """
    Create side-by-side comparison video.
    
    Layout (if include_original):
        [Original | OpenCV | IMU]
    
    Layout (without original):
        [OpenCV | IMU]
    """
    cap_orig = cv2.VideoCapture(str(original_path))
    cap_opencv = cv2.VideoCapture(str(opencv_path))
    cap_imu = cv2.VideoCapture(str(imu_path))
    
    fps = cap_opencv.get(cv2.CAP_PROP_FPS)
    opencv_w = int(cap_opencv.get(cv2.CAP_PROP_FRAME_WIDTH))
    opencv_h = int(cap_opencv.get(cv2.CAP_PROP_FRAME_HEIGHT))
    imu_w = int(cap_imu.get(cv2.CAP_PROP_FRAME_WIDTH))
    imu_h = int(cap_imu.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use the smaller dimensions
    panel_w = min(opencv_w, imu_w)
    panel_h = min(opencv_h, imu_h)
    
    # Scale down for reasonable output size
    if panel_w > 960:
        scale = 960 / panel_w
        panel_w = 960
        panel_h = int(panel_h * scale)
    
    panel_h = panel_h - (panel_h % 2)
    
    if include_original:
        out_w = panel_w * 3
        num_panels = 3
    else:
        out_w = panel_w * 2
        num_panels = 2
    
    out_h = panel_h
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    
    # Label settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    
    frame_count = 0
    logger.info("Creating comparison video...")
    
    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_opencv, frame_opencv = cap_opencv.read()
        ret_imu, frame_imu = cap_imu.read()
        
        if not ret_opencv or not ret_imu:
            break
        
        # Resize frames
        frame_opencv = cv2.resize(frame_opencv, (panel_w, panel_h))
        frame_imu = cv2.resize(frame_imu, (panel_w, panel_h))
        
        # Add labels
        cv2.putText(frame_opencv, "OpenCV", (10, 30), font, font_scale, (0, 255, 0), font_thickness)
        cv2.putText(frame_imu, "IMU/Gyro", (10, 30), font, font_scale, (0, 255, 255), font_thickness)
        
        if include_original and ret_orig:
            frame_orig = cv2.resize(frame_orig, (panel_w, panel_h))
            cv2.putText(frame_orig, "Original", (10, 30), font, font_scale, (255, 255, 255), font_thickness)
            combined = np.hstack([frame_orig, frame_opencv, frame_imu])
        else:
            combined = np.hstack([frame_opencv, frame_imu])
        
        writer.write(combined)
        frame_count += 1
        
        if frame_count % 100 == 0:
            logger.info(f"Combined {frame_count} frames")
    
    cap_orig.release()
    cap_opencv.release()
    cap_imu.release()
    writer.release()
    
    logger.info(f"Saved comparison video to {output_path} ({frame_count} frames)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Compare OpenCV vs IMU video stabilization")
    parser.add_argument("input", type=Path, help="Input video file")
    parser.add_argument("-o", "--output", type=Path, help="Output comparison video")
    parser.add_argument("--crop", type=float, default=0.9, help="Crop ratio (default: 0.9)")
    parser.add_argument("--smoothing", type=int, default=30, help="OpenCV smoothing radius")
    parser.add_argument("--no-original", action="store_true", help="Don't include original in comparison")
    parser.add_argument("--no-rolling-shutter", action="store_true", help="Disable IMU rolling shutter correction")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    # Setup output paths
    stem = input_path.stem
    parent = input_path.parent
    
    opencv_output = parent / f"{stem}_opencv.mp4"
    imu_output = parent / f"{stem}_imu.mp4"
    comparison_output = args.output or parent / f"{stem}_comparison.mp4"
    
    # Run OpenCV stabilization
    logger.info("=" * 60)
    logger.info("Running OpenCV optical flow stabilization...")
    logger.info("=" * 60)
    start = time.time()
    
    opencv_stab = OpenCVStabilizer(smoothing_radius=args.smoothing, crop_ratio=args.crop)
    opencv_stab.stabilize(input_path, opencv_output)
    
    opencv_time = time.time() - start
    logger.info(f"OpenCV stabilization completed in {opencv_time:.1f}s")
    
    # Run IMU stabilization
    logger.info("=" * 60)
    logger.info("Running IMU/Gyroscope stabilization...")
    logger.info("=" * 60)
    start = time.time()
    
    imu_stab = GyroStabilizer(
        rolling_shutter=not args.no_rolling_shutter,
        velocity_smoothing=True,
        use_vqf=True,
    )
    imu_stab.stabilize(input_path, imu_output, crop_ratio=args.crop)
    
    imu_time = time.time() - start
    logger.info(f"IMU stabilization completed in {imu_time:.1f}s")
    
    # Create comparison video
    logger.info("=" * 60)
    logger.info("Creating comparison video...")
    logger.info("=" * 60)
    
    create_comparison_video(
        input_path,
        opencv_output,
        imu_output,
        comparison_output,
        include_original=not args.no_original
    )
    
    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Input video:       {input_path}")
    logger.info(f"OpenCV output:     {opencv_output} ({opencv_time:.1f}s)")
    logger.info(f"IMU output:        {imu_output} ({imu_time:.1f}s)")
    logger.info(f"Comparison output: {comparison_output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

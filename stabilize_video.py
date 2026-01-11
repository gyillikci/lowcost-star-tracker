"""
Simple video stabilization using gyroscope data.
Outputs a stabilized video and a simple mean-stacked image.
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from star_tracker.gyro_extractor import GyroExtractor
from star_tracker.frame_extractor import FrameExtractor
from star_tracker.motion_compensator import MotionCompensator, CameraIntrinsics


def stabilize_video(input_path: str, output_video: str = None, output_stack: str = None):
    """
    Stabilize a GoPro video using embedded gyroscope data.
    
    Args:
        input_path: Path to input video
        output_video: Path for stabilized video output (optional)
        output_stack: Path for stacked image output (optional)
    """
    input_path = Path(input_path)
    
    if output_video is None:
        output_video = f"output/{input_path.stem}_stabilized.mp4"
    if output_stack is None:
        output_stack = f"output/{input_path.stem}_stacked.tiff"
    
    Path("output").mkdir(exist_ok=True)
    
    print(f"Processing: {input_path}")
    
    # Step 1: Extract gyroscope data
    print("Step 1: Extracting gyroscope data...")
    gyro_extractor = GyroExtractor()
    gyro_data = gyro_extractor.extract(input_path)
    print(f"  - Extracted {gyro_data.num_samples} gyro samples over {gyro_data.duration:.1f}s")
    
    # Step 2: Get video info
    print("Step 2: Reading video info...")
    frame_extractor = FrameExtractor()
    video_info = frame_extractor.get_video_info(input_path)
    width, height = video_info["width"], video_info["height"]
    fps = video_info["fps"]
    total_frames = video_info["total_frames"]
    print(f"  - Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")
    
    # Step 3: Setup motion compensator
    print("Step 3: Setting up motion compensator...")
    camera_intrinsics = CameraIntrinsics.from_gopro_hero7((width, height))
    motion_compensator = MotionCompensator(
        camera_intrinsics=camera_intrinsics,
        target_orientation="mean",
        interpolation="linear",
        crop_black_borders=True,
        crop_margin_percent=5.0,
    )
    
    # Compute target orientation (average of all orientations)
    target_orientation = motion_compensator.compute_target_orientation(gyro_data)
    print(f"  - Target orientation computed")
    
    # Step 4: Setup video writer
    print("Step 4: Processing frames...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Accumulator for stacking
    frame_sum = None
    frame_count = 0
    
    # Process each frame
    for idx, timestamp, frame in tqdm(
        frame_extractor.iterate_frames(input_path),
        total=total_frames,
        desc="Stabilizing"
    ):
        # Get orientation at this timestamp
        frame_orientation = gyro_extractor.get_orientation_at_time(gyro_data, timestamp)
        
        # Apply motion compensation (stabilization)
        stabilized = motion_compensator.compensate_frame(
            frame, frame_orientation, target_orientation
        )
        
        # Write to video
        out.write(stabilized)
        
        # Accumulate for stacking
        if frame_sum is None:
            frame_sum = stabilized.astype(np.float64)
        else:
            frame_sum += stabilized.astype(np.float64)
        frame_count += 1
    
    out.release()
    print(f"  - Stabilized video saved: {output_video}")
    
    # Step 5: Create stacked image (simple mean)
    print("Step 5: Stacking frames...")
    stacked = (frame_sum / frame_count).astype(np.uint16)
    # Scale to 16-bit range
    stacked = (stacked * 256).astype(np.uint16)
    
    cv2.imwrite(output_stack, stacked)
    print(f"  - Stacked image saved: {output_stack}")
    
    print("\nDone!")
    print(f"  Stabilized video: {output_video}")
    print(f"  Stacked image: {output_stack}")
    
    return output_video, output_stack


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python stabilize_video.py <input_video> [output_video] [output_stack]")
        print("Example: python stabilize_video.py examples/GX010925.MP4")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else None
    output_stack = sys.argv[3] if len(sys.argv) > 3 else None
    
    stabilize_video(input_video, output_video, output_stack)

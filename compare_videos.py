"""
Create a side-by-side comparison of original vs stabilized video.
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from star_tracker.gyro_extractor import GyroExtractor
from star_tracker.frame_extractor import FrameExtractor
from star_tracker.motion_compensator import MotionCompensator, CameraIntrinsics


def create_comparison(input_path: str, output_path: str = None):
    """
    Create a side-by-side comparison video: Original | Stabilized
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = f"output/{input_path.stem}_comparison.mp4"
    
    Path("output").mkdir(exist_ok=True)
    
    print(f"Processing: {input_path}")
    
    # Step 1: Extract gyroscope data
    print("Extracting gyroscope data...")
    gyro_extractor = GyroExtractor()
    gyro_data = gyro_extractor.extract(input_path)
    print(f"  - {gyro_data.num_samples} gyro samples extracted")
    
    # Step 2: Get video info
    frame_extractor = FrameExtractor()
    video_info = frame_extractor.get_video_info(input_path)
    width, height = video_info["width"], video_info["height"]
    fps = video_info["fps"]
    total_frames = video_info["total_frames"]
    
    # Resize for comparison (half size each)
    comp_width = width // 2
    comp_height = height // 2
    
    print(f"  - Original: {width}x{height}")
    print(f"  - Comparison output: {comp_width*2}x{comp_height}")
    
    # Step 3: Setup motion compensator
    camera_intrinsics = CameraIntrinsics.from_gopro_hero7((width, height))
    motion_compensator = MotionCompensator(
        camera_intrinsics=camera_intrinsics,
        target_orientation="mean",
        interpolation="linear",
        crop_black_borders=True,
        crop_margin_percent=5.0,
    )
    
    target_orientation = motion_compensator.compute_target_orientation(gyro_data)
    
    # Step 4: Create comparison video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (comp_width * 2, comp_height))
    
    print("Creating comparison video...")
    for idx, timestamp, frame in tqdm(
        frame_extractor.iterate_frames(input_path),
        total=total_frames,
        desc="Processing"
    ):
        # Get stabilized frame
        frame_orientation = gyro_extractor.get_orientation_at_time(gyro_data, timestamp)
        stabilized = motion_compensator.compensate_frame(
            frame, frame_orientation, target_orientation
        )
        
        # Resize both for comparison
        original_small = cv2.resize(frame, (comp_width, comp_height))
        stabilized_small = cv2.resize(stabilized, (comp_width, comp_height))
        
        # Add labels
        cv2.putText(original_small, "ORIGINAL", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(stabilized_small, "STABILIZED (Gyro)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Combine side by side
        combined = np.hstack([original_small, stabilized_small])
        
        out.write(combined)
    
    out.release()
    print(f"\nComparison video saved: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python compare_videos.py <input_video> [output_video]")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = create_comparison(input_video, output_video)
    
    # Open the result
    import os
    os.startfile(result)

"""
Debug video stabilization - show rotation being applied on each frame.
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from star_tracker.gyro_extractor import GyroExtractor
from star_tracker.frame_extractor import FrameExtractor
from star_tracker.motion_compensator import MotionCompensator, CameraIntrinsics
from scipy.spatial.transform import Rotation


def debug_stabilization(input_path: str, output_path: str = None):
    """
    Create debug video showing rotation magnitude being applied.
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = f"output/{input_path.stem}_debug.mp4"
    
    Path("output").mkdir(exist_ok=True)
    
    print(f"Processing: {input_path}")
    
    # Extract gyro data
    print("Extracting gyroscope data...")
    gyro_extractor = GyroExtractor()
    gyro_data = gyro_extractor.extract(input_path)
    
    # Analyze gyro data
    print("\n=== GYRO DATA ANALYSIS ===")
    print(f"Duration: {gyro_data.duration:.2f}s, Samples: {gyro_data.num_samples}")
    
    # Angular velocity stats
    ang_vel_deg = np.rad2deg(gyro_data.angular_velocity)
    print(f"\nAngular velocity (deg/s):")
    print(f"  X: min={ang_vel_deg[:,0].min():.3f}, max={ang_vel_deg[:,0].max():.3f}, std={ang_vel_deg[:,0].std():.3f}")
    print(f"  Y: min={ang_vel_deg[:,1].min():.3f}, max={ang_vel_deg[:,1].max():.3f}, std={ang_vel_deg[:,1].std():.3f}")
    print(f"  Z: min={ang_vel_deg[:,2].min():.3f}, max={ang_vel_deg[:,2].max():.3f}, std={ang_vel_deg[:,2].std():.3f}")
    
    # Orientation change
    orientations = gyro_data.orientations
    first_rot = Rotation.from_quat([orientations[0,1], orientations[0,2], orientations[0,3], orientations[0,0]])
    last_rot = Rotation.from_quat([orientations[-1,1], orientations[-1,2], orientations[-1,3], orientations[-1,0]])
    total_rotation = (first_rot.inv() * last_rot).as_euler('xyz', degrees=True)
    print(f"\nTotal rotation from first to last frame:")
    print(f"  X (roll):  {total_rotation[0]:.2f} deg")
    print(f"  Y (pitch): {total_rotation[1]:.2f} deg")
    print(f"  Z (yaw):   {total_rotation[2]:.2f} deg")
    print("=" * 30)
    
    # Get video info
    frame_extractor = FrameExtractor()
    video_info = frame_extractor.get_video_info(input_path)
    width, height = video_info["width"], video_info["height"]
    fps = video_info["fps"]
    total_frames = video_info["total_frames"]
    
    # Setup motion compensator
    camera_intrinsics = CameraIntrinsics.from_gopro_hero7((width, height))
    motion_compensator = MotionCompensator(
        camera_intrinsics=camera_intrinsics,
        target_orientation="mean",
        interpolation="linear",
        crop_black_borders=True,
        crop_margin_percent=5.0,
    )
    
    target_orientation = motion_compensator.compute_target_orientation(gyro_data)
    target_rot = Rotation.from_quat([target_orientation[1], target_orientation[2], 
                                      target_orientation[3], target_orientation[0]])
    
    # Create video with debug overlay
    out_width, out_height = width // 2, height // 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width * 2, out_height))
    
    rotation_history = []
    
    print("\nCreating debug video...")
    for idx, timestamp, frame in tqdm(
        frame_extractor.iterate_frames(input_path),
        total=total_frames,
        desc="Processing"
    ):
        # Get frame orientation
        frame_orientation = gyro_extractor.get_orientation_at_time(gyro_data, timestamp)
        frame_rot = Rotation.from_quat([frame_orientation[1], frame_orientation[2], 
                                        frame_orientation[3], frame_orientation[0]])
        
        # Compute rotation from target
        rel_rot = (target_rot.inv() * frame_rot).as_euler('xyz', degrees=True)
        rotation_history.append(rel_rot)
        
        # Apply stabilization
        stabilized = motion_compensator.compensate_frame(
            frame, frame_orientation, target_orientation
        )
        
        # Resize for display
        original_small = cv2.resize(frame, (out_width, out_height))
        stabilized_small = cv2.resize(stabilized, (out_width, out_height))
        
        # Add debug overlay
        cv2.putText(original_small, "ORIGINAL", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(stabilized_small, "STABILIZED", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show rotation being corrected
        cv2.putText(stabilized_small, f"Rotation from target:", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(stabilized_small, f"  X: {rel_rot[0]:+.2f} deg", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(stabilized_small, f"  Y: {rel_rot[1]:+.2f} deg", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(stabilized_small, f"  Z: {rel_rot[2]:+.2f} deg", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.putText(original_small, f"Frame: {idx}/{total_frames}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(original_small, f"Time: {timestamp:.2f}s", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Combine
        combined = np.hstack([original_small, stabilized_small])
        out.write(combined)
    
    out.release()
    
    # Print rotation statistics
    rotation_history = np.array(rotation_history)
    print(f"\n=== ROTATION CORRECTIONS APPLIED ===")
    print(f"X (roll):  range [{rotation_history[:,0].min():.2f}, {rotation_history[:,0].max():.2f}] deg")
    print(f"Y (pitch): range [{rotation_history[:,1].min():.2f}, {rotation_history[:,1].max():.2f}] deg")
    print(f"Z (yaw):   range [{rotation_history[:,2].min():.2f}, {rotation_history[:,2].max():.2f}] deg")
    
    print(f"\nDebug video saved: {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python debug_stabilization.py <input_video>")
        sys.exit(1)
    
    result = debug_stabilization(sys.argv[1])
    
    import os
    os.startfile(result)

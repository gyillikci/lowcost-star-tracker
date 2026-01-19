#!/usr/bin/env python3
"""
Run Camera-IMU Calibration on Captured Data.

Takes the captured video + IMU data and computes:
1. Time offset between camera and IMU
2. Rotation matrix R_cam_imu (spatial alignment)
3. Gyro bias estimation

Usage:
    python run_calibration.py calibration_data/session_20260115_010500
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration import (
    GyroStream, VideoStream,
    PinholeCamera, CameraIntrinsics,
    AutoCalibrator
)


def run_calibration(session_dir: Path, fov_h: float = 57.0, fov_v: float = 34.0):
    """Run the calibration pipeline using AutoCalibrator."""
    print(f"\nLoading data from: {session_dir}")
    
    # Load frame info for resolution
    frames_path = session_dir / "frames.json"
    with open(frames_path) as f:
        frames_data = json.load(f)
    width = frames_data['width']
    height = frames_data['height']
    
    # Create camera model
    intrinsics = CameraIntrinsics(
        fx=width / (2 * np.tan(np.radians(fov_h/2))),
        fy=height / (2 * np.tan(np.radians(fov_v/2))),
        cx=width / 2,
        cy=height / 2,
        width=width,
        height=height
    )
    camera = PinholeCamera(intrinsics)
    
    print(f"Camera model: {width}x{height}, FOV {fov_h}°x{fov_v}°")
    
    # Create AutoCalibrator
    calibrator = AutoCalibrator(camera)
    
    # Load data
    video_path = str(session_dir / "video.mp4")
    imu_path = str(session_dir / "imu_data.csv")
    
    calibrator.load_data(
        video_path=video_path,
        imu_data_path=imu_path,
        imu_format="csv",
        max_frames=500
    )
    
    # Run calibration
    print("\n" + "="*60)
    print("CAMERA-IMU CALIBRATION")
    print("="*60)
    
    results = calibrator.calibrate(method="full", refine_bias=True)
    
    # Convert results to dict for saving
    result_dict = results.to_dict()
    result_dict['fov_h'] = fov_h
    result_dict['fov_v'] = fov_v
    
    return result_dict


def save_calibration(results: dict, output_path: Path):
    """Save calibration results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Calibration saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run camera-IMU calibration on captured data"
    )
    parser.add_argument('session_dir', type=str,
                       help='Path to captured session directory')
    parser.add_argument('--fov-h', type=float, default=57.0,
                       help='Camera horizontal FOV in degrees (default: 57)')
    parser.add_argument('--fov-v', type=float, default=34.0,
                       help='Camera vertical FOV in degrees (default: 34)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output calibration file (default: session_dir/calibration_result.json)')
    
    args = parser.parse_args()
    
    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        print(f"✗ Session directory not found: {session_dir}")
        return 1
        
    try:
        # Run calibration
        results = run_calibration(session_dir, fov_h=args.fov_h, fov_v=args.fov_v)
        
        # Save results
        output_path = Path(args.output) if args.output else session_dir / "calibration_result.json"
        save_calibration(results, output_path)
        
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE")
        print("="*60)
        print(f"\nResults summary:")
        print(f"  Time offset: {results['time_offset']*1000:.1f} ms")
        print(f"  Gyro bias: {results['gyro_bias']}")
        print(f"  R_cam_imu euler (deg): {results['euler_angles_deg']}")
        print(f"  Confidence: {results['overall_confidence']:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

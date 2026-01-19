#!/usr/bin/env python3
"""
Capture Calibration Data for Camera-IMU System.

Records synchronized video and IMU data from Orange Cube + Harrier camera
for later calibration of spatial and temporal alignment.

Usage:
    python capture_calibration_data.py --camera 1 --port COM6 --duration 60

The script will:
1. Connect to Orange Cube IMU
2. Open camera
3. Display live preview
4. Record video + IMU data with timestamps
5. Save to output folder for calibration

Move the camera+IMU assembly through various orientations during capture:
- Slow rotations (not too fast)
- Cover pitch, roll, and yaw axes
- Avoid pure translations (rotation is what we need)
- Keep some static periods for bias estimation
"""

import cv2
import numpy as np
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List
import argparse
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from mavlink.orange_cube_reader import OrangeCubeReader, AttitudeData, IMUData


@dataclass
class CalibrationFrame:
    """Single frame with timestamp."""
    timestamp: float  # seconds since start
    frame_number: int
    
    
@dataclass
class CalibrationIMU:
    """IMU sample with timestamp."""
    timestamp: float  # seconds since start
    gyro: List[float]  # [wx, wy, wz] in rad/s
    accel: List[float]  # [ax, ay, az] in m/s²
    quaternion: List[float]  # [w, x, y, z]
    euler: List[float]  # [roll, pitch, yaw] in radians
    

class CalibrationDataCapture:
    """Captures synchronized video and IMU data."""
    
    def __init__(self,
                 camera_id: int = 0,
                 port: str = "COM6",
                 baudrate: int = 115200,
                 width: int = 1280,
                 height: int = 720,
                 output_dir: str = "calibration_data"):
        """
        Initialize capture system.
        
        Args:
            camera_id: Camera device index
            port: Serial port for Orange Cube
            baudrate: Serial baudrate
            width: Video width
            height: Video height
            output_dir: Directory to save captured data
        """
        self.camera_id = camera_id
        self.port = port
        self.baudrate = baudrate
        self.width = width
        self.height = height
        self.output_dir = Path(output_dir)
        
        # Data storage
        self.video_frames: List[CalibrationFrame] = []
        self.imu_samples: List[CalibrationIMU] = []
        
        # Devices
        self.camera: Optional[cv2.VideoCapture] = None
        self.imu_reader: Optional[OrangeCubeReader] = None
        
        # Timing
        self.start_time: Optional[float] = None
        self.last_imu_time: float = 0
        
        # Video writer
        self.video_writer: Optional[cv2.VideoWriter] = None
        
    def open_camera(self) -> bool:
        """Open camera."""
        print(f"Opening camera {self.camera_id}...")
        self.camera = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        
        if not self.camera.isOpened():
            print(f"✗ Failed to open camera {self.camera_id}")
            return False
            
        # Set resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Verify resolution
        actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        
        print(f"✓ Camera opened: {actual_w}x{actual_h} @ {actual_fps:.1f} fps")
        
        return True
        
    def connect_imu(self) -> bool:
        """Connect to Orange Cube IMU."""
        print(f"Connecting to Orange Cube on {self.port} at {self.baudrate} baud...")
        
        try:
            self.imu_reader = OrangeCubeReader(
                port=self.port,
                baudrate=self.baudrate
            )
            
            if not self.imu_reader.connect():
                print("✗ Failed to connect to Orange Cube")
                return False
                
            print("✓ Orange Cube connected")
            
            # Request data streams
            self.imu_reader.request_data_streams(rate_hz=50)
            time.sleep(0.5)  # Let it collect some data
            
            return True
            
        except Exception as e:
            print(f"✗ Error connecting to IMU: {e}")
            return False
            
    def capture_imu_sample(self):
        """Capture current IMU data."""
        if not self.imu_reader or not self.start_time:
            return
            
        # Read latest messages (non-blocking, quick)
        msg = self.imu_reader.read_message(timeout=0.01)
        if msg:
            self.imu_reader.process_message(msg)
            
        # Get the latest data
        imu_data = self.imu_reader.imu_data
        attitude_data = self.imu_reader.attitude_data
        
        if imu_data is None or attitude_data is None:
            return
            
        current_time = time.time() - self.start_time
        
        # Avoid duplicate timestamps (only record if new data)
        if current_time <= self.last_imu_time + 0.001:  # 1ms threshold
            return
            
        self.last_imu_time = current_time
        
        # Get quaternion and convert to euler
        q = attitude_data.quaternion  # [w, x, y, z]
        roll, pitch, yaw = attitude_data.roll, attitude_data.pitch, attitude_data.yaw
        
        sample = CalibrationIMU(
            timestamp=current_time,
            gyro=[imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z],
            accel=[imu_data.accel_x, imu_data.accel_y, imu_data.accel_z],
            quaternion=[q[0], q[1], q[2], q[3]],  # [w, x, y, z]
            euler=[roll, pitch, yaw]
        )
        
        self.imu_samples.append(sample)
        
    @staticmethod
    def quaternion_to_euler(q):
        """Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw] in radians."""
        w, x, y, z = q
        
        # Roll (x-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
        
    def start_recording(self, duration: float = 60.0):
        """
        Start recording video and IMU data.
        
        Args:
            duration: Recording duration in seconds
        """
        if not self.camera or not self.imu_reader:
            print("✗ Camera or IMU not initialized")
            return False
            
        # Create output directory with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.output_dir / f"session_{timestamp_str}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"CALIBRATION DATA CAPTURE")
        print(f"{'='*60}")
        print(f"Output directory: {session_dir}")
        print(f"Duration: {duration:.1f} seconds")
        print(f"\nINSTRUCTIONS:")
        print("  1. Slowly rotate the camera+IMU assembly")
        print("  2. Cover all axes: pitch, roll, yaw")
        print("  3. Include some static periods (for bias estimation)")
        print("  4. Avoid pure translations")
        print("  5. Press 'Q' to stop early")
        print(f"{'='*60}\n")
        
        # Initialize video writer
        video_path = session_dir / "video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            30.0,
            (self.width, self.height)
        )
        
        if not self.video_writer.isOpened():
            print("✗ Failed to create video writer")
            return False
            
        print("Recording will start in 3 seconds...")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        print("RECORDING!")
        
        self.start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                elapsed = time.time() - self.start_time
                
                # Check if duration exceeded
                if elapsed >= duration:
                    print(f"\n✓ Recording complete ({duration:.1f}s)")
                    break
                    
                # Capture IMU sample
                self.capture_imu_sample()
                
                # Capture video frame
                ret, frame = self.camera.read()
                if not ret:
                    print("✗ Failed to read frame")
                    break
                    
                # Rotate frame 180° (camera is upside down)
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                
                # Record frame metadata
                frame_meta = CalibrationFrame(
                    timestamp=elapsed,
                    frame_number=frame_count
                )
                self.video_frames.append(frame_meta)
                
                # Write frame to video
                self.video_writer.write(frame)
                frame_count += 1
                
                # Display with overlay
                display_frame = frame.copy()
                
                # Add status text
                cv2.putText(display_frame, f"Time: {elapsed:.1f}s / {duration:.1f}s",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Frames: {frame_count}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"IMU samples: {len(self.imu_samples)}",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add current attitude if available
                if self.imu_samples:
                    last_imu = self.imu_samples[-1]
                    roll_deg = np.degrees(last_imu.euler[0])
                    pitch_deg = np.degrees(last_imu.euler[1])
                    yaw_deg = np.degrees(last_imu.euler[2])
                    cv2.putText(display_frame, f"R:{roll_deg:+6.1f} P:{pitch_deg:+6.1f} Y:{yaw_deg:+6.1f}",
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Show instructions
                cv2.putText(display_frame, "Rotate camera slowly in all axes",
                           (10, self.height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, "Press 'Q' to stop",
                           (10, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("Calibration Capture", display_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                    print(f"\n✓ Recording stopped by user ({elapsed:.1f}s)")
                    break
                    
        except KeyboardInterrupt:
            print("\n✓ Recording interrupted by user")
            
        finally:
            # Save data
            print("\nSaving data...")
            self.save_data(session_dir)
            
            # Cleanup
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()
            
            print(f"\n{'='*60}")
            print(f"CAPTURE COMPLETE")
            print(f"{'='*60}")
            print(f"  Video frames: {len(self.video_frames)}")
            print(f"  IMU samples: {len(self.imu_samples)}")
            print(f"  Duration: {elapsed:.2f}s")
            print(f"  Saved to: {session_dir}")
            print(f"{'='*60}\n")
            
        return True
        
    def save_data(self, output_dir: Path):
        """Save all captured data."""
        # Save frame timestamps
        frame_data = {
            "frames": [asdict(f) for f in self.video_frames],
            "fps": 30.0,
            "width": self.width,
            "height": self.height
        }
        with open(output_dir / "frames.json", 'w') as f:
            json.dump(frame_data, f, indent=2)
            
        # Save IMU data in multiple formats
        
        # 1. JSON format (human readable)
        imu_data = {
            "samples": [asdict(s) for s in self.imu_samples],
            "rate_hz": len(self.imu_samples) / (self.imu_samples[-1].timestamp if self.imu_samples else 1),
            "camera_port": self.camera_id,
            "imu_port": self.port,
            "capture_info": {
                "timestamp": datetime.now().isoformat(),
                "camera_id": self.camera_id,
                "resolution": f"{self.width}x{self.height}",
                "imu_port": self.port,
                "imu_baudrate": self.baudrate
            }
        }
        with open(output_dir / "imu_data.json", 'w') as f:
            json.dump(imu_data, f, indent=2)
            
        # 2. NumPy format (for calibration toolbox)
        if self.imu_samples:
            timestamps = np.array([s.timestamp for s in self.imu_samples])
            gyro = np.array([s.gyro for s in self.imu_samples])
            accel = np.array([s.accel for s in self.imu_samples])
            quaternions = np.array([s.quaternion for s in self.imu_samples])
            euler = np.array([s.euler for s in self.imu_samples])
            
            np.savez(
                output_dir / "imu_data.npz",
                timestamps=timestamps,
                gyro=gyro,
                accel=accel,
                quaternions=quaternions,
                euler=euler
            )
            
        # 3. CSV format (for external tools)
        with open(output_dir / "imu_data.csv", 'w') as f:
            f.write("timestamp,gx,gy,gz,ax,ay,az,qw,qx,qy,qz,roll,pitch,yaw\n")
            for s in self.imu_samples:
                f.write(f"{s.timestamp:.6f},")
                f.write(f"{s.gyro[0]:.6f},{s.gyro[1]:.6f},{s.gyro[2]:.6f},")
                f.write(f"{s.accel[0]:.6f},{s.accel[1]:.6f},{s.accel[2]:.6f},")
                f.write(f"{s.quaternion[0]:.6f},{s.quaternion[1]:.6f},{s.quaternion[2]:.6f},{s.quaternion[3]:.6f},")
                f.write(f"{s.euler[0]:.6f},{s.euler[1]:.6f},{s.euler[2]:.6f}\n")
                
        print(f"  ✓ Saved frames.json")
        print(f"  ✓ Saved imu_data.json")
        print(f"  ✓ Saved imu_data.npz")
        print(f"  ✓ Saved imu_data.csv")
        print(f"  ✓ Saved video.mp4")
        
    def cleanup(self):
        """Clean up resources."""
        if self.camera:
            self.camera.release()
        if self.imu_reader:
            self.imu_reader.close()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Capture calibration data from Orange Cube + Camera"
    )
    parser.add_argument('--camera', '-c', type=int, default=1,
                       help='Camera device ID (default: 1)')
    parser.add_argument('--port', '-p', type=str, default='COM6',
                       help='Orange Cube serial port (default: COM6)')
    parser.add_argument('--baudrate', '-b', type=int, default=115200,
                       help='Serial baudrate (default: 115200)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Video width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Video height (default: 720)')
    parser.add_argument('--duration', '-d', type=float, default=60.0,
                       help='Recording duration in seconds (default: 60)')
    parser.add_argument('--output', '-o', type=str, default='calibration_data',
                       help='Output directory (default: calibration_data)')
    
    args = parser.parse_args()
    
    # Create capture system
    capture = CalibrationDataCapture(
        camera_id=args.camera,
        port=args.port,
        baudrate=args.baudrate,
        width=args.width,
        height=args.height,
        output_dir=args.output
    )
    
    try:
        # Open devices
        if not capture.open_camera():
            return 1
            
        if not capture.connect_imu():
            capture.cleanup()
            return 1
            
        # Start recording
        capture.start_recording(duration=args.duration)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        capture.cleanup()
        
    return 0


if __name__ == '__main__':
    sys.exit(main())

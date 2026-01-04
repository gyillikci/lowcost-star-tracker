"""
Motion compensation using gyroscope-derived orientations.

This module implements frame-by-frame geometric correction based on
gyroscope data to stabilize video for astrophotography.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from .gyro_extractor import GyroData


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    
    focal_length_x: float
    focal_length_y: float
    principal_point_x: float
    principal_point_y: float
    distortion_coeffs: Optional[np.ndarray] = None
    
    @property
    def matrix(self) -> np.ndarray:
        """Get 3x3 camera intrinsic matrix."""
        return np.array([
            [self.focal_length_x, 0, self.principal_point_x],
            [0, self.focal_length_y, self.principal_point_y],
            [0, 0, 1]
        ], dtype=np.float64)
    
    @classmethod
    def from_gopro_hero7(cls, resolution: tuple[int, int] = (3840, 2160)) -> "CameraIntrinsics":
        """Create intrinsics for GoPro Hero 7 Black."""
        width, height = resolution
        
        # Approximate values for GoPro Hero 7 Black in Linear mode
        # These should be calibrated for best results
        focal_length = width * 0.8  # Approximate focal length in pixels
        
        return cls(
            focal_length_x=focal_length,
            focal_length_y=focal_length,
            principal_point_x=width / 2,
            principal_point_y=height / 2,
            distortion_coeffs=np.array([-0.1, 0.01, 0, 0, 0])  # Approximate distortion
        )


class MotionCompensator:
    """
    Apply gyroscope-based motion compensation to video frames.
    """
    
    def __init__(
        self,
        camera_intrinsics: CameraIntrinsics,
        target_orientation: Literal["mean", "median", "first", "custom"] = "mean",
        interpolation: Literal["nearest", "linear", "cubic"] = "cubic",
        crop_black_borders: bool = True,
        crop_margin_percent: float = 5.0,
    ):
        self.camera = camera_intrinsics
        self.target_orientation_mode = target_orientation
        self.interpolation = interpolation
        self.crop_black_borders = crop_black_borders
        self.crop_margin_percent = crop_margin_percent
        
        # Interpolation flags for OpenCV
        self._interp_flags = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
        }
    
    def compute_target_orientation(self, gyro_data: GyroData) -> np.ndarray:
        """
        Compute the target orientation for stabilization.
        
        Args:
            gyro_data: GyroData containing orientation quaternions
            
        Returns:
            Target quaternion [w, x, y, z]
        """
        orientations = gyro_data.orientations
        
        if self.target_orientation_mode == "first":
            return orientations[0]
        elif self.target_orientation_mode == "mean":
            # Average quaternions (simple mean, works for small variations)
            mean_q = np.mean(orientations, axis=0)
            return mean_q / np.linalg.norm(mean_q)
        elif self.target_orientation_mode == "median":
            # Use median index orientation
            median_idx = len(orientations) // 2
            return orientations[median_idx]
        else:
            raise ValueError(f"Unknown target orientation mode: {self.target_orientation_mode}")
    
    def compute_homography(
        self,
        source_orientation: np.ndarray,
        target_orientation: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the homography matrix to transform from source to target orientation.
        
        Args:
            source_orientation: Source quaternion [w, x, y, z]
            target_orientation: Target quaternion [w, x, y, z]
            
        Returns:
            3x3 homography matrix
        """
        # Convert quaternions to rotation matrices
        R_source = Rotation.from_quat([
            source_orientation[1], source_orientation[2], 
            source_orientation[3], source_orientation[0]
        ]).as_matrix()
        
        R_target = Rotation.from_quat([
            target_orientation[1], target_orientation[2],
            target_orientation[3], target_orientation[0]
        ]).as_matrix()
        
        # Relative rotation from source to target
        R_relative = R_target @ R_source.T
        
        # Compute homography: H = K * R * K^-1
        K = self.camera.matrix
        K_inv = np.linalg.inv(K)
        H = K @ R_relative @ K_inv
        
        return H
    
    def compensate_frame(
        self,
        frame: np.ndarray,
        source_orientation: np.ndarray,
        target_orientation: np.ndarray,
        output_size: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Apply motion compensation to a single frame.
        
        Args:
            frame: Input image (H, W, C)
            source_orientation: Frame's orientation quaternion
            target_orientation: Target orientation quaternion
            output_size: Optional output size (width, height)
            
        Returns:
            Stabilized frame
        """
        H = self.compute_homography(source_orientation, target_orientation)
        
        if output_size is None:
            output_size = (frame.shape[1], frame.shape[0])
        
        # Undistort if distortion coefficients are available
        if self.camera.distortion_coeffs is not None:
            frame = cv2.undistort(
                frame, 
                self.camera.matrix, 
                self.camera.distortion_coeffs
            )
        
        # Apply homography
        stabilized = cv2.warpPerspective(
            frame, H, output_size,
            flags=self._interp_flags[self.interpolation],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return stabilized
    
    def compute_crop_region(
        self,
        frame_size: tuple[int, int],
        gyro_data: GyroData,
        target_orientation: np.ndarray,
    ) -> tuple[int, int, int, int]:
        """
        Compute the largest valid crop region that contains no black borders.
        
        Args:
            frame_size: (width, height) of frames
            gyro_data: GyroData for all frames
            target_orientation: Target orientation
            
        Returns:
            Crop region (x, y, width, height)
        """
        width, height = frame_size
        
        # Sample orientations to find maximum displacement
        sample_indices = np.linspace(0, len(gyro_data.orientations) - 1, 100).astype(int)
        
        min_x, min_y = 0, 0
        max_x, max_y = width, height
        
        for idx in sample_indices:
            orientation = gyro_data.orientations[idx]
            H = self.compute_homography(orientation, target_orientation)
            
            # Transform corner points
            corners = np.array([
                [0, 0], [width, 0], [width, height], [0, height]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            transformed = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
            
            # Update bounds
            min_x = max(min_x, transformed[:, 0].min())
            min_y = max(min_y, transformed[:, 1].min())
            max_x = min(max_x, transformed[:, 0].max())
            max_y = min(max_y, transformed[:, 1].max())
        
        # Add margin
        margin_x = (max_x - min_x) * self.crop_margin_percent / 100
        margin_y = (max_y - min_y) * self.crop_margin_percent / 100
        
        crop_x = int(min_x + margin_x)
        crop_y = int(min_y + margin_y)
        crop_w = int(max_x - min_x - 2 * margin_x)
        crop_h = int(max_y - min_y - 2 * margin_y)
        
        return crop_x, crop_y, crop_w, crop_h
    
    def process_video(
        self,
        video_path: Path,
        gyro_data: GyroData,
        output_dir: Path,
        frame_rate: Optional[float] = None,
    ) -> list[Path]:
        """
        Process entire video with motion compensation.
        
        Args:
            video_path: Path to input video
            gyro_data: GyroData for the video
            output_dir: Directory to save stabilized frames
            frame_rate: Override frame rate (default: from video)
            
        Returns:
            List of paths to saved frames
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = frame_rate or cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Compute target orientation
        target_orientation = self.compute_target_orientation(gyro_data)
        
        # Compute crop region if needed
        if self.crop_black_borders:
            crop_region = self.compute_crop_region(
                (width, height), gyro_data, target_orientation
            )
        else:
            crop_region = None
        
        saved_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get frame timestamp
            timestamp = frame_idx / fps
            
            # Get orientation at this timestamp
            frame_orientation = self._interpolate_orientation(gyro_data, timestamp)
            
            # Apply motion compensation
            stabilized = self.compensate_frame(
                frame, frame_orientation, target_orientation
            )
            
            # Apply crop if needed
            if crop_region is not None:
                x, y, w, h = crop_region
                stabilized = stabilized[y:y+h, x:x+w]
            
            # Save frame
            frame_path = output_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(frame_path), stabilized)
            saved_frames.append(frame_path)
            
            frame_idx += 1
        
        cap.release()
        
        return saved_frames
    
    def _interpolate_orientation(
        self, 
        gyro_data: GyroData, 
        timestamp: float
    ) -> np.ndarray:
        """Interpolate orientation at a specific timestamp."""
        idx = np.searchsorted(gyro_data.timestamps, timestamp)
        
        if idx == 0:
            return gyro_data.orientations[0]
        if idx >= len(gyro_data.timestamps):
            return gyro_data.orientations[-1]
        
        # Linear interpolation (SLERP for better accuracy)
        t0 = gyro_data.timestamps[idx - 1]
        t1 = gyro_data.timestamps[idx]
        alpha = (timestamp - t0) / (t1 - t0)
        
        q0 = gyro_data.orientations[idx - 1]
        q1 = gyro_data.orientations[idx]
        
        # Simple linear interpolation (SLERP would be better)
        q = (1 - alpha) * q0 + alpha * q1
        return q / np.linalg.norm(q)

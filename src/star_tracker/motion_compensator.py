"""
Motion compensation using gyroscope-derived orientations.

This module implements frame-by-frame geometric correction based on
gyroscope data to stabilize video for astrophotography.

Features:
- Homography-based frame warping
- Rolling shutter correction (per-row rotation)
- Lens distortion handling
- GPU-accelerated pixel remapping (optional)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Tuple, List

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from .gyro_extractor import GyroData
from .lens_profile import LensProfile, FisheyeDistortion, load_lens_profile


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
    def from_gopro_hero7(
        cls, 
        resolution: tuple[int, int] = (3840, 2160),
        fov_mode: str = "wide"
    ) -> "CameraIntrinsics":
        """
        Create intrinsics for GoPro Hero 7 Black.
        
        Args:
            resolution: Video resolution (width, height)
            fov_mode: FOV mode - "wide" (~118°), "linear" (~86°), "narrow" (~70°)
            
        GoPro Hero 7 Black sensor: 1/2.3" CMOS
        Focal length: 3mm (physical)
        
        Calibration values based on typical GoPro measurements.
        """
        width, height = resolution
        
        # FOV-based focal length calculation
        # f = (width/2) / tan(hfov/2)
        # These are empirically calibrated values for GoPro Hero 7
        fov_params = {
            # mode: (horizontal_fov_degrees, k1_distortion)
            "superview": (170, -0.35),
            "wide": (118, -0.25),      # Default wide angle mode
            "linear": (86, -0.05),     # Lens distortion corrected
            "narrow": (70, -0.02),
        }
        
        if fov_mode not in fov_params:
            fov_mode = "wide"
        
        hfov_deg, k1 = fov_params[fov_mode]
        hfov_rad = np.deg2rad(hfov_deg)
        
        # Calculate focal length from horizontal FOV
        focal_length = (width / 2) / np.tan(hfov_rad / 2)
        
        return cls(
            focal_length_x=focal_length,
            focal_length_y=focal_length,
            principal_point_x=width / 2,
            principal_point_y=height / 2,
            distortion_coeffs=np.array([k1, 0.05, 0, 0, 0])
        )


class MotionCompensator:
    """
    Apply gyroscope-based motion compensation to video frames.
    
    Supports:
    - Simple homography-based stabilization
    - Per-row rolling shutter correction
    - Lens distortion compensation
    """
    
    def __init__(
        self,
        camera_intrinsics: CameraIntrinsics,
        target_orientation: Literal["mean", "median", "first", "custom"] = "mean",
        interpolation: Literal["nearest", "linear", "cubic"] = "cubic",
        crop_black_borders: bool = True,
        crop_margin_percent: float = 5.0,
        rolling_shutter_correction: bool = True,
        frame_readout_time_ms: float = 8.3,  # 4K60 GoPro
        lens_profile: Optional[LensProfile] = None,
        use_smoothed_orientations: bool = True,
    ):
        self.camera = camera_intrinsics
        self.target_orientation_mode = target_orientation
        self.interpolation = interpolation
        self.crop_black_borders = crop_black_borders
        self.crop_margin_percent = crop_margin_percent
        self.rolling_shutter_correction = rolling_shutter_correction
        self.frame_readout_time_ms = frame_readout_time_ms
        self.lens_profile = lens_profile
        self.use_smoothed_orientations = use_smoothed_orientations
        
        # Initialize lens distortion handler if profile provided
        self.fisheye = None
        if lens_profile is not None:
            self.fisheye = FisheyeDistortion(lens_profile)
            # Update camera intrinsics from profile
            self.camera = CameraIntrinsics(
                focal_length_x=lens_profile.fx,
                focal_length_y=lens_profile.fy,
                principal_point_x=lens_profile.cx,
                principal_point_y=lens_profile.cy,
                distortion_coeffs=lens_profile.distortion_coeffs,
            )
            self.frame_readout_time_ms = lens_profile.frame_readout_time
        
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
    
    def compensate_frame_rolling_shutter(
        self,
        frame: np.ndarray,
        gyro_data: GyroData,
        frame_timestamp: float,
        target_orientation: np.ndarray,
        output_size: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Apply motion compensation with per-row rolling shutter correction.
        
        Each row of the sensor is exposed at a slightly different time.
        This method computes a separate rotation for each row based on
        the gyro orientation at that row's exposure time.
        
        Args:
            frame: Input image (H, W, C)
            gyro_data: GyroData with orientation timeline
            frame_timestamp: Start timestamp of frame exposure (seconds)
            target_orientation: Target orientation quaternion
            output_size: Optional output size (width, height)
            
        Returns:
            Stabilized frame with rolling shutter correction
        """
        height, width = frame.shape[:2]
        if output_size is None:
            output_size = (width, height)
        
        # Time per row in seconds
        readout_time_sec = self.frame_readout_time_ms / 1000.0
        row_time = readout_time_sec / height
        
        # Get orientations to use (smoothed if available)
        orientations = gyro_data.orientations
        if self.use_smoothed_orientations and gyro_data.smoothed_orientations is not None:
            orientations = gyro_data.smoothed_orientations
        
        # Compute per-row rotation matrices
        row_matrices = self._compute_row_rotation_matrices(
            gyro_data.timestamps,
            orientations,
            frame_timestamp,
            row_time,
            height,
            target_orientation
        )
        
        # Apply per-row remapping
        stabilized = self._apply_rolling_shutter_remap(
            frame, row_matrices, output_size
        )
        
        return stabilized
    
    def _compute_row_rotation_matrices(
        self,
        timestamps: np.ndarray,
        orientations: np.ndarray,
        frame_start: float,
        row_time: float,
        num_rows: int,
        target_orientation: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Compute rotation matrix for each row based on its exposure time.
        
        Args:
            timestamps: Gyro timestamps
            orientations: Orientation quaternions
            frame_start: Frame start timestamp
            row_time: Time between rows
            num_rows: Number of rows
            target_orientation: Target orientation
            
        Returns:
            List of 3x3 rotation matrices (one per row)
        """
        K = self.camera.matrix
        K_inv = np.linalg.inv(K)
        
        # Convert target to rotation matrix
        R_target = Rotation.from_quat([
            target_orientation[1], target_orientation[2],
            target_orientation[3], target_orientation[0]
        ]).as_matrix()
        
        matrices = []
        
        for row in range(num_rows):
            # Timestamp for this row
            row_timestamp = frame_start + row * row_time
            
            # Interpolate orientation at this time
            row_quat = self._interpolate_orientation(timestamps, orientations, row_timestamp)
            
            # Convert to rotation matrix
            R_row = Rotation.from_quat([
                row_quat[1], row_quat[2], row_quat[3], row_quat[0]
            ]).as_matrix()
            
            # Relative rotation for this row
            R_relative = R_target @ R_row.T
            
            # Homography for this row
            H_row = K @ R_relative @ K_inv
            matrices.append(H_row)
        
        return matrices
    
    def _interpolate_orientation(
        self,
        timestamps: np.ndarray,
        orientations: np.ndarray,
        time: float
    ) -> np.ndarray:
        """SLERP interpolation of orientation at a specific time."""
        idx = np.searchsorted(timestamps, time)
        
        if idx == 0:
            return orientations[0]
        if idx >= len(timestamps):
            return orientations[-1]
        
        t0 = timestamps[idx - 1]
        t1 = timestamps[idx]
        alpha = (time - t0) / (t1 - t0)
        
        q0 = orientations[idx - 1]
        q1 = orientations[idx]
        
        return self._slerp(q0, q1, alpha)
    
    @staticmethod
    def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between quaternions."""
        dot = np.dot(q0, q1)
        if dot < 0:
            q1 = -q1
            dot = -dot
        
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)
        
        theta_0 = np.arccos(np.clip(dot, -1, 1))
        theta = theta_0 * t
        
        q2 = q1 - q0 * dot
        q2 = q2 / np.linalg.norm(q2)
        
        return q0 * np.cos(theta) + q2 * np.sin(theta)
    
    def _apply_rolling_shutter_remap(
        self,
        frame: np.ndarray,
        row_matrices: List[np.ndarray],
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Apply per-row homography using pixel remapping.
        
        This creates a remap table where each output pixel maps to
        an input pixel based on the homography for its row.
        
        Args:
            frame: Input frame
            row_matrices: List of 3x3 homography matrices (one per row)
            output_size: Output (width, height)
            
        Returns:
            Remapped frame
        """
        out_width, out_height = output_size
        in_height, in_width = frame.shape[:2]
        
        # Create output coordinate grids
        x = np.arange(out_width, dtype=np.float32)
        y = np.arange(out_height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        
        # Initialize remap arrays
        map_x = np.zeros((out_height, out_width), dtype=np.float32)
        map_y = np.zeros((out_height, out_width), dtype=np.float32)
        
        # Apply per-row homography
        for row in range(out_height):
            # Get homography for this row
            H = row_matrices[min(row, len(row_matrices) - 1)]
            H_inv = np.linalg.inv(H)
            
            # Transform coordinates for this row
            # [x', y', w'] = H^-1 @ [x, y, 1]
            row_x = xx[row, :]
            row_y = np.full_like(row_x, row)
            ones = np.ones_like(row_x)
            
            # Stack as homogeneous coordinates
            coords = np.stack([row_x, row_y, ones], axis=0)  # (3, W)
            
            # Apply inverse homography
            src_coords = H_inv @ coords
            src_coords = src_coords / src_coords[2:3, :]  # Normalize
            
            map_x[row, :] = src_coords[0, :]
            map_y[row, :] = src_coords[1, :]
        
        # Apply remap
        stabilized = cv2.remap(
            frame, map_x, map_y,
            interpolation=self._interp_flags[self.interpolation],
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
        use_rolling_shutter: Optional[bool] = None,
    ) -> list[Path]:
        """
        Process entire video with motion compensation.
        
        Args:
            video_path: Path to input video
            gyro_data: GyroData for the video
            output_dir: Directory to save stabilized frames
            frame_rate: Override frame rate (default: from video)
            use_rolling_shutter: Override rolling shutter setting
            
        Returns:
            List of paths to saved frames
        """
        import logging
        logger = logging.getLogger(__name__)
        
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
        
        # Determine which orientations to use
        orientations = gyro_data.orientations
        if self.use_smoothed_orientations and gyro_data.smoothed_orientations is not None:
            orientations = gyro_data.smoothed_orientations
            logger.info("Using velocity-adaptive smoothed orientations")
        
        # Compute target orientation
        target_orientation = self.compute_target_orientation(gyro_data)
        
        # Compute crop region if needed
        if self.crop_black_borders:
            crop_region = self.compute_crop_region(
                (width, height), gyro_data, target_orientation
            )
        else:
            crop_region = None
        
        # Determine if using rolling shutter correction
        rs_enabled = use_rolling_shutter if use_rolling_shutter is not None else self.rolling_shutter_correction
        if rs_enabled:
            logger.info(f"Rolling shutter correction enabled (readout time: {self.frame_readout_time_ms:.1f}ms)")
        
        saved_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get frame timestamp
            timestamp = frame_idx / fps
            
            if rs_enabled:
                # Apply rolling shutter correction
                stabilized = self.compensate_frame_rolling_shutter(
                    frame, gyro_data, timestamp, target_orientation
                )
            else:
                # Simple homography-based stabilization
                frame_orientation = self._interpolate_orientation(
                    gyro_data.timestamps, orientations, timestamp
                )
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
            
            # Progress logging
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        logger.info(f"Completed processing {frame_idx} frames")
        
        return saved_frames
    
    def _interpolate_orientation_legacy(
        self, 
        gyro_data: GyroData, 
        timestamp: float
    ) -> np.ndarray:
        """Legacy method for backward compatibility."""
        orientations = gyro_data.orientations
        if self.use_smoothed_orientations and gyro_data.smoothed_orientations is not None:
            orientations = gyro_data.smoothed_orientations
        return self._interpolate_orientation(gyro_data.timestamps, orientations, timestamp)

"""
Lens profile loading and distortion handling.

This module provides lens profile management and fisheye distortion/undistortion
functions compatible with Gyroflow lens profiles.

Supports:
- opencv_fisheye distortion model
- Newton-Raphson iterative undistortion
- Resolution scaling for different video modes
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LensProfile:
    """Lens profile container matching Gyroflow format."""
    
    camera_brand: str
    camera_model: str
    camera_setting: str
    
    # Calibration resolution
    calib_width: int
    calib_height: int
    
    # Distortion model
    distortion_model: str  # "opencv_fisheye", "opencv_standard", etc.
    
    # Camera matrix (3x3)
    camera_matrix: np.ndarray  # [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    
    # Distortion coefficients (k1, k2, k3, k4 for fisheye)
    distortion_coeffs: np.ndarray
    
    # Rolling shutter
    frame_readout_time: float  # milliseconds
    
    # Optional: gyro low-pass filter
    gyro_lpf: float = 50.0
    
    @property
    def fx(self) -> float:
        return self.camera_matrix[0, 0]
    
    @property
    def fy(self) -> float:
        return self.camera_matrix[1, 1]
    
    @property
    def cx(self) -> float:
        return self.camera_matrix[0, 2]
    
    @property
    def cy(self) -> float:
        return self.camera_matrix[1, 2]
    
    @property
    def k1(self) -> float:
        return self.distortion_coeffs[0] if len(self.distortion_coeffs) > 0 else 0.0
    
    @property
    def k2(self) -> float:
        return self.distortion_coeffs[1] if len(self.distortion_coeffs) > 1 else 0.0
    
    @property
    def k3(self) -> float:
        return self.distortion_coeffs[2] if len(self.distortion_coeffs) > 2 else 0.0
    
    @property
    def k4(self) -> float:
        return self.distortion_coeffs[3] if len(self.distortion_coeffs) > 3 else 0.0
    
    def scale_to_resolution(self, width: int, height: int) -> "LensProfile":
        """
        Scale lens profile to a different resolution.
        
        Focal lengths and principal points scale with resolution.
        Distortion coefficients are dimensionless and stay the same.
        
        Args:
            width: Target width
            height: Target height
            
        Returns:
            New LensProfile scaled to target resolution
        """
        scale_x = width / self.calib_width
        scale_y = height / self.calib_height
        
        new_matrix = self.camera_matrix.copy()
        new_matrix[0, 0] *= scale_x  # fx
        new_matrix[0, 2] *= scale_x  # cx
        new_matrix[1, 1] *= scale_y  # fy
        new_matrix[1, 2] *= scale_y  # cy
        
        # Adjust rolling shutter time based on height ratio
        # More rows = more time (approximately)
        new_readout = self.frame_readout_time * scale_y
        
        return LensProfile(
            camera_brand=self.camera_brand,
            camera_model=self.camera_model,
            camera_setting=self.camera_setting,
            calib_width=width,
            calib_height=height,
            distortion_model=self.distortion_model,
            camera_matrix=new_matrix,
            distortion_coeffs=self.distortion_coeffs.copy(),
            frame_readout_time=new_readout,
            gyro_lpf=self.gyro_lpf,
        )


def load_lens_profile(path: Path) -> LensProfile:
    """
    Load lens profile from JSON file.
    
    Args:
        path: Path to lens profile JSON
        
    Returns:
        LensProfile object
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Parse camera matrix
    if "fisheye_params" in data:
        matrix = np.array(data["fisheye_params"]["camera_matrix"], dtype=np.float64)
        coeffs = np.array(data["fisheye_params"]["distortion_coeffs"], dtype=np.float64)
    else:
        # Fallback format
        matrix = np.array(data.get("camera_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        coeffs = np.array(data.get("distortion_coeffs", [0, 0, 0, 0]))
    
    return LensProfile(
        camera_brand=data.get("camera_brand", "Unknown"),
        camera_model=data.get("camera_model", "Unknown"),
        camera_setting=data.get("camera_setting", ""),
        calib_width=data.get("calib_dimension", {}).get("w", 1920),
        calib_height=data.get("calib_dimension", {}).get("h", 1080),
        distortion_model=data.get("distortion_model", "opencv_fisheye"),
        camera_matrix=matrix,
        distortion_coeffs=coeffs,
        frame_readout_time=data.get("frame_readout_time", 16.0),
        gyro_lpf=data.get("gyro_lpf", 50.0),
    )


def find_lens_profile(
    profiles_dir: Path,
    camera_model: str,
    resolution: Tuple[int, int],
    setting: Optional[str] = None
) -> Optional[LensProfile]:
    """
    Find matching lens profile for camera and resolution.
    
    Args:
        profiles_dir: Directory containing lens profile JSONs
        camera_model: Camera model name (e.g., "Hero7 Black")
        resolution: (width, height) tuple
        setting: Optional camera setting (e.g., "Wide")
        
    Returns:
        Matching LensProfile or None
    """
    if not profiles_dir.exists():
        return None
    
    best_match = None
    best_score = -1
    
    for json_file in profiles_dir.glob("*.json"):
        try:
            profile = load_lens_profile(json_file)
            
            # Score based on matching criteria
            score = 0
            
            # Camera model match
            if camera_model.lower() in profile.camera_model.lower():
                score += 10
            
            # Setting match
            if setting and setting.lower() in profile.camera_setting.lower():
                score += 5
            
            # Resolution match (exact or scalable)
            if (profile.calib_width == resolution[0] and 
                profile.calib_height == resolution[1]):
                score += 20  # Exact match
            elif (profile.calib_width / profile.calib_height == 
                  resolution[0] / resolution[1]):
                score += 10  # Same aspect ratio (scalable)
            
            if score > best_score:
                best_score = score
                best_match = profile
                
        except Exception as e:
            logger.warning(f"Failed to load profile {json_file}: {e}")
    
    if best_match is not None:
        # Scale to target resolution if needed
        if (best_match.calib_width != resolution[0] or 
            best_match.calib_height != resolution[1]):
            best_match = best_match.scale_to_resolution(resolution[0], resolution[1])
            logger.info(f"Scaled lens profile to {resolution[0]}x{resolution[1]}")
    
    return best_match


class FisheyeDistortion:
    """
    OpenCV fisheye distortion model.
    
    Implements distortion and undistortion for fisheye lenses using
    the equidistant projection model.
    
    Distortion model:
        θ_d = θ(1 + k1*θ² + k2*θ⁴ + k3*θ⁶ + k4*θ⁸)
    
    Where θ is the angle from optical axis and θ_d is the distorted angle.
    """
    
    EPS = 1e-10
    MAX_ITERATIONS = 10
    
    def __init__(self, profile: LensProfile):
        self.profile = profile
        self.k = profile.distortion_coeffs
    
    def distort_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Apply fisheye distortion to a normalized point.
        
        Args:
            x, y: Normalized coordinates (relative to principal point, divided by f)
            
        Returns:
            Distorted normalized coordinates
        """
        r = np.sqrt(x*x + y*y)
        if r < self.EPS:
            return x, y
        
        theta = np.arctan(r)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        
        # Apply distortion polynomial
        theta_d = theta * (1 + self.k[0]*theta2 + self.k[1]*theta4 + 
                          self.k[2]*theta6 + self.k[3]*theta8)
        
        scale = theta_d / r if r > self.EPS else 1.0
        return x * scale, y * scale
    
    def undistort_point(self, x_d: float, y_d: float) -> Optional[Tuple[float, float]]:
        """
        Remove fisheye distortion from a normalized point using Newton-Raphson.
        
        Args:
            x_d, y_d: Distorted normalized coordinates
            
        Returns:
            Undistorted normalized coordinates, or None if fails
        """
        r_d = np.sqrt(x_d*x_d + y_d*y_d)
        if r_d < self.EPS:
            return x_d, y_d
        
        # Clamp to valid range
        theta_d = np.clip(r_d, -np.pi/2, np.pi/2)
        
        # Newton-Raphson iteration to find theta from theta_d
        theta = theta_d
        
        for _ in range(self.MAX_ITERATIONS):
            theta2 = theta * theta
            theta4 = theta2 * theta2
            theta6 = theta4 * theta2
            theta8 = theta4 * theta4
            
            # f(theta) = theta * (1 + k0*θ² + k1*θ⁴ + k2*θ⁶ + k3*θ⁸) - theta_d
            f = theta * (1 + self.k[0]*theta2 + self.k[1]*theta4 + 
                        self.k[2]*theta6 + self.k[3]*theta8) - theta_d
            
            # f'(theta) = 1 + 3*k0*θ² + 5*k1*θ⁴ + 7*k2*θ⁶ + 9*k3*θ⁸
            f_prime = (1 + 3*self.k[0]*theta2 + 5*self.k[1]*theta4 + 
                      7*self.k[2]*theta6 + 9*self.k[3]*theta8)
            
            if abs(f_prime) < self.EPS:
                return None
            
            theta_fix = f / f_prime
            theta -= theta_fix
            
            if abs(theta_fix) < self.EPS:
                break
        
        # Convert back to Cartesian
        r = np.tan(theta)
        scale = r / r_d if r_d > self.EPS else 1.0
        
        return x_d * scale, y_d * scale
    
    def distort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Apply distortion to array of pixel coordinates.
        
        Args:
            points: (N, 2) array of pixel coordinates
            
        Returns:
            (N, 2) array of distorted pixel coordinates
        """
        # Convert to normalized coordinates
        fx, fy = self.profile.fx, self.profile.fy
        cx, cy = self.profile.cx, self.profile.cy
        
        normalized = np.zeros_like(points)
        normalized[:, 0] = (points[:, 0] - cx) / fx
        normalized[:, 1] = (points[:, 1] - cy) / fy
        
        # Apply distortion
        distorted = np.zeros_like(normalized)
        for i in range(len(points)):
            d = self.distort_point(normalized[i, 0], normalized[i, 1])
            distorted[i] = d
        
        # Convert back to pixel coordinates
        result = np.zeros_like(points)
        result[:, 0] = distorted[:, 0] * fx + cx
        result[:, 1] = distorted[:, 1] * fy + cy
        
        return result
    
    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Remove distortion from array of pixel coordinates.
        
        Args:
            points: (N, 2) array of distorted pixel coordinates
            
        Returns:
            (N, 2) array of undistorted pixel coordinates
        """
        fx, fy = self.profile.fx, self.profile.fy
        cx, cy = self.profile.cx, self.profile.cy
        
        # Convert to normalized coordinates
        normalized = np.zeros_like(points)
        normalized[:, 0] = (points[:, 0] - cx) / fx
        normalized[:, 1] = (points[:, 1] - cy) / fy
        
        # Remove distortion
        undistorted = np.zeros_like(normalized)
        for i in range(len(points)):
            u = self.undistort_point(normalized[i, 0], normalized[i, 1])
            if u is not None:
                undistorted[i] = u
            else:
                undistorted[i] = normalized[i]
        
        # Convert back to pixel coordinates
        result = np.zeros_like(points)
        result[:, 0] = undistorted[:, 0] * fx + cx
        result[:, 1] = undistorted[:, 1] * fy + cy
        
        return result
    
    def create_undistort_map(
        self, 
        width: int, 
        height: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create undistortion remap arrays for cv2.remap().
        
        Args:
            width: Output width
            height: Output height
            
        Returns:
            Tuple of (map_x, map_y) arrays for cv2.remap()
        """
        fx, fy = self.profile.fx, self.profile.fy
        cx, cy = self.profile.cx, self.profile.cy
        
        # Create coordinate grids
        u = np.arange(width)
        v = np.arange(height)
        u_grid, v_grid = np.meshgrid(u, v)
        
        # Normalize
        x_norm = (u_grid - cx) / fx
        y_norm = (v_grid - cy) / fy
        
        # Apply distortion (we want to map undistorted → distorted for remap)
        r = np.sqrt(x_norm**2 + y_norm**2)
        theta = np.arctan(r)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        
        theta_d = theta * (1 + self.k[0]*theta2 + self.k[1]*theta4 + 
                          self.k[2]*theta6 + self.k[3]*theta8)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where(r > self.EPS, theta_d / r, 1.0)
        
        x_d = x_norm * scale
        y_d = y_norm * scale
        
        # Convert back to pixel coordinates
        map_x = (x_d * fx + cx).astype(np.float32)
        map_y = (y_d * fy + cy).astype(np.float32)
        
        return map_x, map_y

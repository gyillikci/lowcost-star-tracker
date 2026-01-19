#!/usr/bin/env python3
"""
Camera Model Module.

Implements various camera projection models for star tracker calibration:
- Pinhole (standard) camera model with Brown-Conrady distortion
- Fisheye (equidistant) model for wide-angle lenses
- Division model for extreme wide-angle

These models handle projection from 3D world coordinates to 2D image
coordinates and vice versa.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Optional, Union
from scipy.optimize import minimize


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    # Focal length
    fx: float  # pixels
    fy: float  # pixels

    # Principal point
    cx: float  # pixels
    cy: float  # pixels

    # Image size
    width: int
    height: int

    # Distortion coefficients (Brown-Conrady / Fisheye)
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))

    # Rolling shutter readout time (seconds from first to last row)
    readout_time: float = 0.0

    @property
    def K(self) -> np.ndarray:
        """Get 3x3 camera matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

    @property
    def fov_horizontal(self) -> float:
        """Horizontal field of view in radians."""
        return 2 * np.arctan(self.width / (2 * self.fx))

    @property
    def fov_vertical(self) -> float:
        """Vertical field of view in radians."""
        return 2 * np.arctan(self.height / (2 * self.fy))

    @property
    def fov_diagonal(self) -> float:
        """Diagonal field of view in radians."""
        diag = np.sqrt(self.width**2 + self.height**2)
        f = (self.fx + self.fy) / 2
        return 2 * np.arctan(diag / (2 * f))


class CameraModel(ABC):
    """
    Abstract base class for camera projection models.

    Defines the interface for projecting 3D points to 2D image coordinates
    and unprojecting 2D points back to 3D rays.
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        """
        Initialize camera model.

        Args:
            intrinsics: Camera intrinsic parameters
        """
        self.intrinsics = intrinsics

    @abstractmethod
    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates.

        Args:
            points_3d: 3D points in camera frame, shape (N, 3) or (3,)

        Returns:
            2D image coordinates, shape (N, 2) or (2,)
        """
        pass

    @abstractmethod
    def unproject(self, points_2d: np.ndarray) -> np.ndarray:
        """
        Unproject 2D image points to 3D unit vectors.

        Args:
            points_2d: 2D image coordinates, shape (N, 2) or (2,)

        Returns:
            3D unit vectors in camera frame, shape (N, 3) or (3,)
        """
        pass

    @abstractmethod
    def undistort_point(self, point: np.ndarray) -> np.ndarray:
        """
        Remove distortion from a 2D point.

        Args:
            point: Distorted 2D coordinates

        Returns:
            Undistorted 2D coordinates
        """
        pass

    @abstractmethod
    def distort_point(self, point: np.ndarray) -> np.ndarray:
        """
        Apply distortion to a 2D point.

        Args:
            point: Undistorted 2D coordinates

        Returns:
            Distorted 2D coordinates
        """
        pass

    def is_in_image(self, points_2d: np.ndarray, margin: float = 0) -> np.ndarray:
        """
        Check if 2D points are within image bounds.

        Args:
            points_2d: 2D image coordinates, shape (N, 2) or (2,)
            margin: Margin from image edge

        Returns:
            Boolean array indicating valid points
        """
        points_2d = np.atleast_2d(points_2d)
        w, h = self.intrinsics.width, self.intrinsics.height

        valid = (
            (points_2d[:, 0] >= margin) &
            (points_2d[:, 0] < w - margin) &
            (points_2d[:, 1] >= margin) &
            (points_2d[:, 1] < h - margin)
        )
        return valid.squeeze() if points_2d.shape[0] == 1 else valid

    def get_rolling_shutter_time(self, row: float) -> float:
        """
        Get exposure time offset for a given image row.

        Args:
            row: Image row (0 = top)

        Returns:
            Time offset in seconds from frame start
        """
        if self.intrinsics.readout_time == 0:
            return 0.0
        return (row / self.intrinsics.height) * self.intrinsics.readout_time


class PinholeCamera(CameraModel):
    """
    Standard pinhole camera model with Brown-Conrady distortion.

    Distortion model:
        x' = x * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r² + 2*x²)
        y' = y * (1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y²) + 2*p2*x*y

    where:
        r² = x² + y²
        k1, k2, k3: radial distortion coefficients
        p1, p2: tangential distortion coefficients
    """

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D image coordinates."""
        points_3d = np.atleast_2d(points_3d)
        single_point = points_3d.shape[0] == 1

        # Handle points behind camera
        z = points_3d[:, 2]
        valid = z > 0

        # Normalize to z=1 plane
        x = np.zeros(len(points_3d))
        y = np.zeros(len(points_3d))

        x[valid] = points_3d[valid, 0] / z[valid]
        y[valid] = points_3d[valid, 1] / z[valid]

        # Apply distortion
        distorted = self._apply_distortion(x, y)

        # Apply camera matrix
        u = self.intrinsics.fx * distorted[:, 0] + self.intrinsics.cx
        v = self.intrinsics.fy * distorted[:, 1] + self.intrinsics.cy

        # Mark invalid points
        u[~valid] = np.nan
        v[~valid] = np.nan

        result = np.column_stack([u, v])
        return result[0] if single_point else result

    def unproject(self, points_2d: np.ndarray) -> np.ndarray:
        """Unproject 2D image points to 3D unit vectors."""
        points_2d = np.atleast_2d(points_2d)
        single_point = points_2d.shape[0] == 1

        # Remove camera matrix
        x = (points_2d[:, 0] - self.intrinsics.cx) / self.intrinsics.fx
        y = (points_2d[:, 1] - self.intrinsics.cy) / self.intrinsics.fy

        # Remove distortion
        undistorted = self._remove_distortion(x, y)

        # Create unit vectors
        vectors = np.column_stack([
            undistorted[:, 0],
            undistorted[:, 1],
            np.ones(len(points_2d))
        ])

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        return vectors[0] if single_point else vectors

    def undistort_point(self, point: np.ndarray) -> np.ndarray:
        """Remove distortion from a 2D point."""
        point = np.atleast_1d(point)

        # Normalize
        x = (point[0] - self.intrinsics.cx) / self.intrinsics.fx
        y = (point[1] - self.intrinsics.cy) / self.intrinsics.fy

        # Undistort
        undist = self._remove_distortion(np.array([x]), np.array([y]))

        # Back to pixels
        u = undist[0, 0] * self.intrinsics.fx + self.intrinsics.cx
        v = undist[0, 1] * self.intrinsics.fy + self.intrinsics.cy

        return np.array([u, v])

    def distort_point(self, point: np.ndarray) -> np.ndarray:
        """Apply distortion to a 2D point."""
        point = np.atleast_1d(point)

        # Normalize
        x = (point[0] - self.intrinsics.cx) / self.intrinsics.fx
        y = (point[1] - self.intrinsics.cy) / self.intrinsics.fy

        # Distort
        dist = self._apply_distortion(np.array([x]), np.array([y]))

        # Back to pixels
        u = dist[0, 0] * self.intrinsics.fx + self.intrinsics.cx
        v = dist[0, 1] * self.intrinsics.fy + self.intrinsics.cy

        return np.array([u, v])

    def _apply_distortion(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply Brown-Conrady distortion."""
        d = self.intrinsics.distortion
        if len(d) < 5:
            d = np.concatenate([d, np.zeros(5 - len(d))])

        k1, k2, p1, p2, k3 = d[:5]

        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r2**3

        # Radial distortion
        radial = 1 + k1*r2 + k2*r4 + k3*r6

        # Tangential distortion
        x_dist = x * radial + 2*p1*x*y + p2*(r2 + 2*x**2)
        y_dist = y * radial + p1*(r2 + 2*y**2) + 2*p2*x*y

        return np.column_stack([x_dist, y_dist])

    def _remove_distortion(self, x: np.ndarray, y: np.ndarray,
                           max_iterations: int = 10) -> np.ndarray:
        """Remove distortion iteratively."""
        # Initial guess = distorted coordinates
        x_undist = x.copy()
        y_undist = y.copy()

        for _ in range(max_iterations):
            dist = self._apply_distortion(x_undist, y_undist)
            x_undist = x - (dist[:, 0] - x_undist)
            y_undist = y - (dist[:, 1] - y_undist)

        return np.column_stack([x_undist, y_undist])


class FisheyeCamera(CameraModel):
    """
    Fisheye camera model (equidistant projection).

    Projection model:
        r = f * theta
        where theta is the angle from the optical axis

    Suitable for wide-angle lenses (>120° FOV).
    Uses polynomial distortion on the angle.
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        """
        Initialize fisheye camera.

        Distortion coefficients represent polynomial on angle:
            theta_d = theta * (1 + k1*theta² + k2*theta⁴ + k3*theta⁶ + k4*theta⁸)
        """
        super().__init__(intrinsics)

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points using equidistant fisheye model."""
        points_3d = np.atleast_2d(points_3d)
        single_point = points_3d.shape[0] == 1

        # Angle from optical axis
        xy_norm = np.sqrt(points_3d[:, 0]**2 + points_3d[:, 1]**2)
        theta = np.arctan2(xy_norm, points_3d[:, 2])

        # Apply distortion to angle
        theta_d = self._distort_angle(theta)

        # Project
        r = theta_d  # r = f * theta_d, but normalized coords

        # Handle zero angles
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where(xy_norm > 1e-10, r / xy_norm, 0)

        x = points_3d[:, 0] * scale
        y = points_3d[:, 1] * scale

        # To pixels
        u = self.intrinsics.fx * x + self.intrinsics.cx
        v = self.intrinsics.fy * y + self.intrinsics.cy

        # Mark points behind camera (theta > pi/2)
        behind = theta > np.pi / 2
        u[behind] = np.nan
        v[behind] = np.nan

        result = np.column_stack([u, v])
        return result[0] if single_point else result

    def unproject(self, points_2d: np.ndarray) -> np.ndarray:
        """Unproject 2D points to 3D unit vectors."""
        points_2d = np.atleast_2d(points_2d)
        single_point = points_2d.shape[0] == 1

        # To normalized coordinates
        x = (points_2d[:, 0] - self.intrinsics.cx) / self.intrinsics.fx
        y = (points_2d[:, 1] - self.intrinsics.cy) / self.intrinsics.fy

        # Distorted angle
        r = np.sqrt(x**2 + y**2)
        theta_d = r  # Since we used r = theta_d in projection

        # Undistort angle
        theta = self._undistort_angle(theta_d)

        # Create 3D vectors
        xy_scale = np.where(r > 1e-10, np.sin(theta) / r, 0)

        vectors = np.column_stack([
            x * xy_scale,
            y * xy_scale,
            np.cos(theta)
        ])

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        vectors = vectors / norms

        return vectors[0] if single_point else vectors

    def undistort_point(self, point: np.ndarray) -> np.ndarray:
        """Remove fisheye distortion from a 2D point."""
        point = np.atleast_1d(point)

        # Unproject to get direction
        direction = self.unproject(point)

        # Project with pinhole model to get undistorted coords
        if direction[2] > 0:
            u = direction[0] / direction[2] * self.intrinsics.fx + self.intrinsics.cx
            v = direction[1] / direction[2] * self.intrinsics.fy + self.intrinsics.cy
            return np.array([u, v])
        else:
            return np.array([np.nan, np.nan])

    def distort_point(self, point: np.ndarray) -> np.ndarray:
        """Apply fisheye distortion to a 2D point."""
        point = np.atleast_1d(point)

        # Assume point is undistorted (pinhole projection)
        x = (point[0] - self.intrinsics.cx) / self.intrinsics.fx
        y = (point[1] - self.intrinsics.cy) / self.intrinsics.fy

        # Convert to 3D direction
        direction = np.array([x, y, 1.0])
        direction = direction / np.linalg.norm(direction)

        # Project with fisheye model
        projected = self.project(direction)
        return projected

    def _distort_angle(self, theta: np.ndarray) -> np.ndarray:
        """Apply distortion to angle."""
        d = self.intrinsics.distortion
        if len(d) < 4:
            d = np.concatenate([d, np.zeros(4 - len(d))])

        k1, k2, k3, k4 = d[:4]

        theta2 = theta**2
        theta4 = theta2**2
        theta6 = theta2 * theta4
        theta8 = theta4**2

        return theta * (1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8)

    def _undistort_angle(self, theta_d: np.ndarray,
                         max_iterations: int = 10) -> np.ndarray:
        """Remove distortion from angle iteratively."""
        theta = theta_d.copy()

        for _ in range(max_iterations):
            theta_d_est = self._distort_angle(theta)
            theta = theta - (theta_d_est - theta_d)

        return theta


class DivisionModel(CameraModel):
    """
    Division model for extreme wide-angle lenses.

    Simpler than polynomial models for very wide FOV:
        r_undist = r_dist / (1 + k * r_dist²)
    """

    def __init__(self, intrinsics: CameraIntrinsics, division_k: float = 0.0):
        """
        Initialize division model.

        Args:
            intrinsics: Camera intrinsics
            division_k: Single distortion parameter
        """
        super().__init__(intrinsics)
        self.division_k = division_k

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points with division model."""
        points_3d = np.atleast_2d(points_3d)
        single_point = points_3d.shape[0] == 1

        # Normalize
        z = points_3d[:, 2]
        valid = z > 0

        x = np.zeros(len(points_3d))
        y = np.zeros(len(points_3d))

        x[valid] = points_3d[valid, 0] / z[valid]
        y[valid] = points_3d[valid, 1] / z[valid]

        # Apply division distortion
        r2 = x**2 + y**2
        scale = 1 + self.division_k * r2

        x_dist = x * scale
        y_dist = y * scale

        # To pixels
        u = self.intrinsics.fx * x_dist + self.intrinsics.cx
        v = self.intrinsics.fy * y_dist + self.intrinsics.cy

        u[~valid] = np.nan
        v[~valid] = np.nan

        result = np.column_stack([u, v])
        return result[0] if single_point else result

    def unproject(self, points_2d: np.ndarray) -> np.ndarray:
        """Unproject with division model."""
        points_2d = np.atleast_2d(points_2d)
        single_point = points_2d.shape[0] == 1

        # Normalize
        x = (points_2d[:, 0] - self.intrinsics.cx) / self.intrinsics.fx
        y = (points_2d[:, 1] - self.intrinsics.cy) / self.intrinsics.fy

        # Undistort: solve quadratic
        r2_dist = x**2 + y**2

        if self.division_k == 0:
            x_undist, y_undist = x, y
        else:
            # r_undist = r_dist / (1 + k * r_dist²)
            # Solving: r² * (1 + k*r²)² = r_dist²
            # This gives: r_undist² = r_dist² / (1 + k*r_dist²)²
            scale = 1 / (1 + self.division_k * r2_dist)
            x_undist = x * scale
            y_undist = y * scale

        # To 3D vectors
        vectors = np.column_stack([
            x_undist,
            y_undist,
            np.ones(len(points_2d))
        ])

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        return vectors[0] if single_point else vectors

    def undistort_point(self, point: np.ndarray) -> np.ndarray:
        """Remove division distortion."""
        vec = self.unproject(point)
        if vec[2] > 0:
            u = vec[0] / vec[2] * self.intrinsics.fx + self.intrinsics.cx
            v = vec[1] / vec[2] * self.intrinsics.fy + self.intrinsics.cy
            return np.array([u, v])
        return np.array([np.nan, np.nan])

    def distort_point(self, point: np.ndarray) -> np.ndarray:
        """Apply division distortion."""
        x = (point[0] - self.intrinsics.cx) / self.intrinsics.fx
        y = (point[1] - self.intrinsics.cy) / self.intrinsics.fy

        r2 = x**2 + y**2
        scale = 1 + self.division_k * r2

        u = x * scale * self.intrinsics.fx + self.intrinsics.cx
        v = y * scale * self.intrinsics.fy + self.intrinsics.cy

        return np.array([u, v])


def create_gopro_camera() -> PinholeCamera:
    """Create camera model for GoPro Hero 7 Black (wide mode)."""
    intrinsics = CameraIntrinsics(
        fx=1520.0,
        fy=1520.0,
        cx=960.0,
        cy=540.0,
        width=1920,
        height=1080,
        distortion=np.array([-0.25, 0.08, 0.0, 0.0, -0.01]),
        readout_time=0.030  # ~30ms rolling shutter
    )
    return PinholeCamera(intrinsics)


def create_asi585mc_camera() -> FisheyeCamera:
    """Create camera model for ASI585MC with Entaniya 220° lens."""
    intrinsics = CameraIntrinsics(
        fx=500.0,  # Approx for 220° fisheye
        fy=500.0,
        cx=1936.0,  # 3840 / 2
        cy=1096.0,  # 2192 / 2
        width=3840,
        height=2192,
        distortion=np.array([0.0, 0.0, 0.0, 0.0]),
        readout_time=0.0  # Global shutter
    )
    return FisheyeCamera(intrinsics)


def demonstrate_camera_models():
    """Demonstrate camera model functionality."""
    print("=" * 60)
    print("Camera Model Demonstration")
    print("=" * 60)

    # Create GoPro camera
    gopro = create_gopro_camera()
    print(f"\nGoPro Camera:")
    print(f"  FOV: {np.rad2deg(gopro.intrinsics.fov_horizontal):.1f}° x "
          f"{np.rad2deg(gopro.intrinsics.fov_vertical):.1f}°")

    # Test projection
    test_points = np.array([
        [0, 0, 1],      # Center
        [0.5, 0.5, 1],  # Off-center
        [1, 0, 1],      # Far right
        [0, 0, -1],     # Behind camera
    ])

    print("\n  Projection test:")
    for pt in test_points:
        proj = gopro.project(pt)
        print(f"    {pt} -> {proj}")

    # Test unproject
    print("\n  Unproject test:")
    test_pixels = np.array([
        [960, 540],     # Center
        [100, 100],     # Corner
        [1800, 1000],   # Other corner
    ])

    for px in test_pixels:
        vec = gopro.unproject(px)
        print(f"    {px} -> {vec}")

    # Test roundtrip
    print("\n  Roundtrip test:")
    for pt in test_points[:3]:
        proj = gopro.project(pt)
        unproj = gopro.unproject(proj)
        unproj_normalized = pt / np.linalg.norm(pt)
        error = np.linalg.norm(unproj - unproj_normalized)
        print(f"    {pt} -> error: {error:.6f}")

    # Create fisheye camera
    print(f"\n\nASI585MC Fisheye Camera:")
    fisheye = create_asi585mc_camera()
    print(f"  FOV: {np.rad2deg(fisheye.intrinsics.fov_diagonal):.1f}° diagonal")

    # Test extreme angles
    print("\n  Extreme angle projection:")
    angles = [0, 30, 60, 90, 110]
    for angle in angles:
        pt = np.array([
            np.sin(np.deg2rad(angle)),
            0,
            np.cos(np.deg2rad(angle))
        ])
        proj = fisheye.project(pt)
        print(f"    {angle}° from axis -> {proj}")

    return gopro, fisheye


if __name__ == "__main__":
    demonstrate_camera_models()

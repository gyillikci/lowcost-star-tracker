"""
Sub-pixel frame alignment using star positions.

This module provides algorithms for precise frame registration
using detected star positions as reference points.
"""

from typing import Literal, Optional, Tuple

import numpy as np
import cv2

from .star_detector import StarField, StarMatcher


class FrameAligner:
    """
    Align frames using star position matching and geometric transformation.
    """
    
    def __init__(
        self,
        reference_frame: Literal["best", "first", "median"] = "best",
        min_match_stars: int = 5,
        max_alignment_error_pixels: float = 1.0,
        transform_type: Literal["translation", "rigid", "affine", "homography"] = "affine",
        ransac_threshold: float = 3.0,
    ):
        self.reference_frame = reference_frame
        self.min_match_stars = min_match_stars
        self.max_alignment_error = max_alignment_error_pixels
        self.transform_type = transform_type
        self.ransac_threshold = ransac_threshold
        
        self.star_matcher = StarMatcher(min_matches=min_match_stars)
    
    def compute_transform(
        self,
        source_stars: StarField,
        target_stars: StarField,
    ) -> Optional[np.ndarray]:
        """
        Compute transformation matrix from source to target frame.
        
        Args:
            source_stars: Stars detected in source frame
            target_stars: Stars detected in target (reference) frame
            
        Returns:
            Transformation matrix, or None if alignment failed
        """
        # Match stars between frames
        matches = self.star_matcher.match(source_stars, target_stars)
        
        if len(matches) < self.min_match_stars:
            return None
        
        # Extract matched positions
        src_pts = np.array([source_stars.stars[i].position for i, _ in matches])
        dst_pts = np.array([target_stars.stars[j].position for _, j in matches])
        
        # Compute transformation
        if self.transform_type == "translation":
            transform = self._compute_translation(src_pts, dst_pts)
        elif self.transform_type == "rigid":
            transform = self._compute_rigid(src_pts, dst_pts)
        elif self.transform_type == "affine":
            transform = self._compute_affine(src_pts, dst_pts)
        elif self.transform_type == "homography":
            transform = self._compute_homography(src_pts, dst_pts)
        else:
            raise ValueError(f"Unknown transform type: {self.transform_type}")
        
        # Validate alignment quality
        if transform is not None:
            error = self._compute_alignment_error(src_pts, dst_pts, transform)
            if error > self.max_alignment_error:
                return None
        
        return transform
    
    def apply_transform(
        self,
        frame: np.ndarray,
        transform: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Apply transformation to align a frame.
        
        Args:
            frame: Input image
            transform: Transformation matrix
            output_size: Output size (width, height), defaults to input size
            
        Returns:
            Aligned frame
        """
        if output_size is None:
            output_size = (frame.shape[1], frame.shape[0])
        
        if transform.shape == (2, 3):
            # Affine transformation
            aligned = cv2.warpAffine(
                frame, transform, output_size,
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            # Homography (3x3)
            aligned = cv2.warpPerspective(
                frame, transform, output_size,
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        
        return aligned
    
    def _compute_translation(
        self, 
        src_pts: np.ndarray, 
        dst_pts: np.ndarray
    ) -> np.ndarray:
        """Compute translation-only transformation."""
        # Simple mean offset
        offset = np.mean(dst_pts - src_pts, axis=0)
        
        transform = np.array([
            [1, 0, offset[0]],
            [0, 1, offset[1]]
        ], dtype=np.float32)
        
        return transform
    
    def _compute_rigid(
        self, 
        src_pts: np.ndarray, 
        dst_pts: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute rigid (rotation + translation) transformation."""
        # Estimate using RANSAC
        if len(src_pts) < 2:
            return None
        
        transform, inliers = cv2.estimateAffinePartial2D(
            src_pts.reshape(-1, 1, 2).astype(np.float32),
            dst_pts.reshape(-1, 1, 2).astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold
        )
        
        return transform
    
    def _compute_affine(
        self, 
        src_pts: np.ndarray, 
        dst_pts: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute full affine transformation."""
        if len(src_pts) < 3:
            return None
        
        transform, inliers = cv2.estimateAffine2D(
            src_pts.reshape(-1, 1, 2).astype(np.float32),
            dst_pts.reshape(-1, 1, 2).astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold
        )
        
        return transform
    
    def _compute_homography(
        self, 
        src_pts: np.ndarray, 
        dst_pts: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute homography transformation."""
        if len(src_pts) < 4:
            return None
        
        transform, mask = cv2.findHomography(
            src_pts.reshape(-1, 1, 2).astype(np.float32),
            dst_pts.reshape(-1, 1, 2).astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold
        )
        
        return transform
    
    def _compute_alignment_error(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        transform: np.ndarray,
    ) -> float:
        """Compute RMS alignment error after transformation."""
        # Transform source points
        if transform.shape == (2, 3):
            # Affine
            ones = np.ones((len(src_pts), 1))
            src_homogeneous = np.hstack([src_pts, ones])
            transformed = src_homogeneous @ transform.T
        else:
            # Homography
            ones = np.ones((len(src_pts), 1))
            src_homogeneous = np.hstack([src_pts, ones])
            transformed = src_homogeneous @ transform.T
            transformed = transformed[:, :2] / transformed[:, 2:3]
        
        # Compute RMS error
        errors = np.linalg.norm(transformed - dst_pts, axis=1)
        return float(np.sqrt(np.mean(errors**2)))
    
    def select_reference_frame(
        self,
        star_fields: list[StarField],
    ) -> int:
        """
        Select the best reference frame based on quality metrics.
        
        Args:
            star_fields: List of StarField objects for each frame
            
        Returns:
            Index of the selected reference frame
        """
        if self.reference_frame == "first":
            return 0
        elif self.reference_frame == "median":
            return len(star_fields) // 2
        else:  # "best"
            # Score based on star count and FWHM
            scores = []
            for sf in star_fields:
                if sf.num_stars == 0:
                    scores.append(0)
                else:
                    star_score = min(sf.num_stars / 100, 1.0)  # Normalize to [0, 1]
                    fwhm_score = max(0, 1 - sf.median_fwhm / 10)  # Lower FWHM is better
                    scores.append(star_score * 0.5 + fwhm_score * 0.5)
            
            return int(np.argmax(scores))


class DrizzleStacker:
    """
    Drizzle algorithm for super-resolution stacking.
    
    Not yet implemented - placeholder for future enhancement.
    """
    
    def __init__(
        self,
        scale_factor: float = 2.0,
        drop_size: float = 0.7,
    ):
        self.scale_factor = scale_factor
        self.drop_size = drop_size
    
    def drizzle(
        self,
        frames: list[np.ndarray],
        transforms: list[np.ndarray],
    ) -> np.ndarray:
        """
        Drizzle frames onto higher resolution output grid.
        """
        raise NotImplementedError("Drizzle stacking not yet implemented")

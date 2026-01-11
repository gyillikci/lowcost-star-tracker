"""
Star detection and measurement for astrophotography frames.

This module provides algorithms for detecting stars, measuring their
properties, and filtering detections for quality assessment.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


@dataclass
class Star:
    """Detected star with measured properties."""
    
    x: float  # Centroid X position (sub-pixel)
    y: float  # Centroid Y position (sub-pixel)
    flux: float  # Total integrated flux
    peak: float  # Peak pixel value
    fwhm: float  # Full width at half maximum
    ellipticity: float  # Ellipticity (0 = circular)
    snr: float  # Signal-to-noise ratio
    area: int  # Area in pixels
    
    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class StarField:
    """Collection of detected stars in a frame."""
    
    stars: list[Star]
    background_mean: float
    background_std: float
    frame_shape: tuple[int, int]
    
    @property
    def num_stars(self) -> int:
        return len(self.stars)
    
    @property
    def median_fwhm(self) -> float:
        if not self.stars:
            return 0.0
        return float(np.median([s.fwhm for s in self.stars]))
    
    def get_positions(self) -> np.ndarray:
        """Get star positions as Nx2 array."""
        if not self.stars:
            return np.zeros((0, 2))
        return np.array([[s.x, s.y] for s in self.stars])


class StarDetector:
    """
    Detect and measure stars in astronomical images.
    
    Uses a combination of background estimation, thresholding,
    connected component analysis, and centroid refinement.
    """
    
    def __init__(
        self,
        detection_threshold_sigma: float = 3.0,
        min_area_pixels: int = 3,
        max_area_pixels: int = 500,
        max_ellipticity: float = 0.5,
        background_box_size: int = 64,
        filter_sigma: float = 1.5,
    ):
        self.threshold_sigma = detection_threshold_sigma
        self.min_area = min_area_pixels
        self.max_area = max_area_pixels
        self.max_ellipticity = max_ellipticity
        self.background_box_size = background_box_size
        self.filter_sigma = filter_sigma
    
    def detect(self, image: np.ndarray) -> StarField:
        """
        Detect stars in an image.
        
        Args:
            image: Grayscale or color image (will be converted to grayscale)
            
        Returns:
            StarField containing detected stars and background statistics
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Convert to float
        gray = gray.astype(np.float32)
        
        # Estimate and subtract background
        background = self._estimate_background(gray)
        subtracted = gray - background
        
        # Compute background statistics from the subtracted image
        bg_mean = 0.0  # After subtraction
        bg_std = self._robust_std(subtracted)
        
        # Apply Gaussian filter for noise reduction
        filtered = gaussian_filter(subtracted, sigma=self.filter_sigma)
        
        # Threshold
        threshold = self.threshold_sigma * bg_std
        binary = (filtered > threshold).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # Process each detected object
        stars = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding box
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y0 = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Extract region
            mask = (labels[y0:y0+h, x0:x0+w] == i)
            region = subtracted[y0:y0+h, x0:x0+w]
            
            # Compute properties
            star = self._measure_star(
                region, mask, x0, y0, bg_std
            )
            
            if star is not None:
                # Filter by ellipticity
                if star.ellipticity <= self.max_ellipticity:
                    stars.append(star)
        
        return StarField(
            stars=stars,
            background_mean=float(np.mean(background)),
            background_std=bg_std,
            frame_shape=gray.shape,
        )
    
    def _estimate_background(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate smooth background using sigma-clipped statistics in boxes.
        """
        h, w = image.shape
        box = self.background_box_size
        
        # Compute number of boxes
        ny = max(1, h // box)
        nx = max(1, w // box)
        
        # Compute median in each box
        bg_grid = np.zeros((ny, nx))
        for iy in range(ny):
            for ix in range(nx):
                y0 = iy * h // ny
                y1 = (iy + 1) * h // ny
                x0 = ix * w // nx
                x1 = (ix + 1) * w // nx
                
                region = image[y0:y1, x0:x1]
                bg_grid[iy, ix] = self._sigma_clipped_mean(region)
        
        # Interpolate to full resolution
        background = cv2.resize(
            bg_grid.astype(np.float32), 
            (w, h), 
            interpolation=cv2.INTER_LINEAR
        )
        
        return background
    
    def _sigma_clipped_mean(
        self, 
        data: np.ndarray, 
        sigma: float = 3.0, 
        max_iter: int = 5
    ) -> float:
        """Compute sigma-clipped mean."""
        data = data.flatten()
        
        for _ in range(max_iter):
            mean = np.median(data)
            std = self._robust_std(data)
            
            if std == 0:
                break
            
            mask = np.abs(data - mean) < sigma * std
            if mask.sum() == len(data):
                break
            
            data = data[mask]
        
        return float(np.median(data))
    
    def _robust_std(self, data: np.ndarray) -> float:
        """Compute robust standard deviation using MAD."""
        data = data.flatten()
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return float(mad * 1.4826)  # Scale factor for normal distribution
    
    def _measure_star(
        self,
        region: np.ndarray,
        mask: np.ndarray,
        x_offset: int,
        y_offset: int,
        background_std: float,
    ) -> Optional[Star]:
        """
        Measure star properties from extracted region.
        """
        # Compute flux and peak
        values = region[mask]
        flux = float(values.sum())
        peak = float(values.max())
        
        if flux <= 0:
            return None
        
        # Compute centroid
        y_indices, x_indices = np.where(mask)
        weights = region[mask]
        
        if weights.sum() <= 0:
            return None
        
        cx = float(np.average(x_indices, weights=weights)) + x_offset
        cy = float(np.average(y_indices, weights=weights)) + y_offset
        
        # Compute second moments for FWHM and ellipticity
        dx = x_indices - (cx - x_offset)
        dy = y_indices - (cy - y_offset)
        
        mxx = np.average(dx**2, weights=weights)
        myy = np.average(dy**2, weights=weights)
        mxy = np.average(dx * dy, weights=weights)
        
        # Compute ellipse parameters
        trace = mxx + myy
        det = mxx * myy - mxy**2
        
        if trace <= 0:
            return None
        
        # Semi-major and semi-minor axes
        discriminant = max(0, trace**2 / 4 - det)
        a = np.sqrt(trace / 2 + np.sqrt(discriminant))
        b = np.sqrt(max(0.01, trace / 2 - np.sqrt(discriminant)))
        
        # FWHM from geometric mean of axes
        fwhm = 2.355 * np.sqrt(a * b)  # Convert sigma to FWHM
        
        # Ellipticity
        ellipticity = 1 - b / a if a > 0 else 0
        
        # SNR
        snr = peak / background_std if background_std > 0 else 0
        
        return Star(
            x=cx,
            y=cy,
            flux=flux,
            peak=peak,
            fwhm=float(fwhm),
            ellipticity=float(ellipticity),
            snr=float(snr),
            area=int(mask.sum()),
        )


class StarMatcher:
    """
    Match stars between frames using triangle matching.
    """
    
    def __init__(
        self,
        max_match_distance: float = 10.0,
        min_matches: int = 5,
    ):
        self.max_match_distance = max_match_distance
        self.min_matches = min_matches
    
    def match(
        self,
        stars1: StarField,
        stars2: StarField,
    ) -> list[tuple[int, int]]:
        """
        Find matching stars between two frames.
        
        Args:
            stars1: Stars in first frame
            stars2: Stars in second frame
            
        Returns:
            List of (idx1, idx2) pairs of matching star indices
        """
        if stars1.num_stars < self.min_matches or stars2.num_stars < self.min_matches:
            return []
        
        pos1 = stars1.get_positions()
        pos2 = stars2.get_positions()
        
        # Build triangle descriptors
        triangles1 = self._build_triangles(pos1)
        triangles2 = self._build_triangles(pos2)
        
        # Match triangles
        triangle_matches = self._match_triangles(triangles1, triangles2)
        
        # Vote for star matches
        match_votes = {}
        for t1_idx, t2_idx in triangle_matches:
            for i in range(3):
                star1_idx = triangles1[t1_idx]["indices"][i]
                star2_idx = triangles2[t2_idx]["indices"][i]
                key = (star1_idx, star2_idx)
                match_votes[key] = match_votes.get(key, 0) + 1
        
        # Filter by votes and distance
        matches = []
        used1, used2 = set(), set()
        
        for (idx1, idx2), votes in sorted(match_votes.items(), key=lambda x: -x[1]):
            if idx1 in used1 or idx2 in used2:
                continue
            
            dist = np.linalg.norm(pos1[idx1] - pos2[idx2])
            if dist < self.max_match_distance or votes >= 3:
                matches.append((idx1, idx2))
                used1.add(idx1)
                used2.add(idx2)
        
        return matches
    
    def _build_triangles(
        self, 
        positions: np.ndarray, 
        max_triangles: int = 200
    ) -> list[dict]:
        """Build triangle descriptors from star positions."""
        n = len(positions)
        if n < 3:
            return []
        
        triangles = []
        
        # Use brightest/largest stars for triangle building
        indices = list(range(min(n, 15)))  # Limit to 15 stars to keep triangles manageable
        
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                for k in range(j + 1, len(indices)):
                    if len(triangles) >= max_triangles:
                        return triangles
                    
                    idx = [indices[i], indices[j], indices[k]]
                    pts = positions[idx]
                    
                    # Compute side lengths
                    sides = [
                        np.linalg.norm(pts[1] - pts[0]),
                        np.linalg.norm(pts[2] - pts[1]),
                        np.linalg.norm(pts[0] - pts[2]),
                    ]
                    
                    # Normalize by longest side
                    max_side = max(sides)
                    if max_side == 0:
                        continue
                    
                    sides = sorted([s / max_side for s in sides])
                    
                    triangles.append({
                        "indices": idx,
                        "descriptor": tuple(sides),
                    })
        
        return triangles
    
    def _match_triangles(
        self,
        triangles1: list[dict],
        triangles2: list[dict],
        tolerance: float = 0.05,
    ) -> list[tuple[int, int]]:
        """Match triangles by descriptor similarity using binning for speed."""
        if not triangles1 or not triangles2:
            return []
        
        matches = []
        
        # Build a hash table for triangles2 using quantized descriptors
        # This gives O(1) lookup instead of O(n) for each triangle
        bin_size = tolerance
        bins = {}
        
        for j, t2 in enumerate(triangles2):
            # Quantize descriptor to create hash key
            d = t2["descriptor"]
            key = (round(d[0] / bin_size), round(d[1] / bin_size), round(d[2] / bin_size))
            if key not in bins:
                bins[key] = []
            bins[key].append(j)
        
        # For each triangle in triangles1, check nearby bins
        for i, t1 in enumerate(triangles1):
            d1 = np.array(t1["descriptor"])
            key = (round(d1[0] / bin_size), round(d1[1] / bin_size), round(d1[2] / bin_size))
            
            # Check the key and neighboring bins
            for dk0 in (-1, 0, 1):
                for dk1 in (-1, 0, 1):
                    for dk2 in (-1, 0, 1):
                        neighbor_key = (key[0] + dk0, key[1] + dk1, key[2] + dk2)
                        if neighbor_key in bins:
                            for j in bins[neighbor_key]:
                                d2 = np.array(triangles2[j]["descriptor"])
                                if np.max(np.abs(d1 - d2)) < tolerance:
                                    matches.append((i, j))
            
            # Early termination if we have enough matches
            if len(matches) >= 50:
                break
        
        return matches

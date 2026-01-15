"""
Star Detection for Plate Solving

Detects point sources (stars) in astronomical images using adaptive
thresholding and centroid refinement.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class DetectedStar:
    """A detected star in an image."""
    x: float            # X centroid position (pixels)
    y: float            # Y centroid position (pixels)
    flux: float         # Total flux (ADU)
    fwhm: float         # Full width half maximum (pixels)
    snr: float          # Signal to noise ratio
    peak: float         # Peak pixel value

    @property
    def position(self) -> Tuple[float, float]:
        """Get (x, y) position tuple."""
        return (self.x, self.y)


class StarDetector:
    """
    Detects stars in astronomical images.

    Uses a combination of:
    - Background estimation and subtraction
    - Adaptive thresholding
    - Connected component analysis
    - Centroid refinement
    - Quality filtering (SNR, FWHM, roundness)
    """

    def __init__(self,
                 sigma_threshold: float = 3.0,
                 min_area: int = 4,
                 max_area: int = 500,
                 min_snr: float = 5.0,
                 max_fwhm: float = 20.0,
                 max_elongation: float = 2.0):
        """
        Initialize the star detector.

        Args:
            sigma_threshold: Detection threshold in sigma above background
            min_area: Minimum star area in pixels
            max_area: Maximum star area in pixels
            min_snr: Minimum signal-to-noise ratio
            max_fwhm: Maximum FWHM (filter hot pixels)
            max_elongation: Maximum elongation (filter cosmic rays/satellites)
        """
        self.sigma_threshold = sigma_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.min_snr = min_snr
        self.max_fwhm = max_fwhm
        self.max_elongation = max_elongation

    def detect(self, image: np.ndarray,
               mask: Optional[np.ndarray] = None) -> List[DetectedStar]:
        """
        Detect stars in an image.

        Args:
            image: Input image (grayscale or color)
            mask: Optional mask (0 = ignore, 255 = process)

        Returns:
            List of DetectedStar objects, sorted by flux
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required for star detection")

        # Convert to grayscale float
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = gray.astype(np.float32)

        # Estimate background
        background, noise = self._estimate_background(gray)

        # Subtract background
        subtracted = gray - background

        # Create detection mask
        threshold = self.sigma_threshold * noise
        detection_mask = (subtracted > threshold).astype(np.uint8) * 255

        # Apply user mask if provided
        if mask is not None:
            detection_mask = cv2.bitwise_and(detection_mask, mask)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            detection_mask, connectivity=8
        )

        # Process each component
        stars = []
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]

            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue

            # Get bounding box
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y0 = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Skip edge stars
            if x0 < 2 or y0 < 2 or x0 + w > gray.shape[1] - 2 or y0 + h > gray.shape[0] - 2:
                continue

            # Extract star region
            star_mask = (labels[y0:y0+h, x0:x0+w] == i)
            star_data = subtracted[y0:y0+h, x0:x0+w]

            # Refine centroid
            cx, cy = self._centroid(star_data, star_mask)
            cx += x0
            cy += y0

            # Calculate properties
            flux = np.sum(star_data[star_mask])
            peak = np.max(star_data[star_mask])
            snr = flux / (noise * np.sqrt(area))

            # Calculate FWHM
            fwhm = self._estimate_fwhm(star_data, star_mask)

            # Calculate elongation
            elongation = max(w, h) / (min(w, h) + 1e-6)

            # Filter by quality
            if snr < self.min_snr:
                continue
            if fwhm > self.max_fwhm or fwhm < 1.0:
                continue
            if elongation > self.max_elongation:
                continue

            stars.append(DetectedStar(
                x=cx,
                y=cy,
                flux=flux,
                fwhm=fwhm,
                snr=snr,
                peak=peak
            ))

        # Sort by flux (brightest first)
        stars.sort(key=lambda s: s.flux, reverse=True)

        return stars

    def _estimate_background(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Estimate background level and noise using sigma-clipped statistics.

        Args:
            image: Input grayscale image

        Returns:
            (background_image, noise_level)
        """
        # Use median filter for background estimation
        if min(image.shape) > 100:
            # Downsample, compute median, then upsample
            scale = 0.1
            small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            median_small = cv2.medianBlur(small.astype(np.float32), 21).astype(np.float32)
            background = cv2.resize(median_small, (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
        else:
            background = np.median(image) * np.ones_like(image)

        # Estimate noise using MAD (median absolute deviation)
        residuals = image - background
        mad = np.median(np.abs(residuals - np.median(residuals)))
        noise = 1.4826 * mad  # Convert MAD to sigma

        return background, max(noise, 1.0)

    def _centroid(self, data: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
        """
        Calculate intensity-weighted centroid.

        Args:
            data: Star data patch
            mask: Boolean mask of star pixels

        Returns:
            (cx, cy) centroid position relative to patch origin
        """
        h, w = data.shape
        y_grid, x_grid = np.mgrid[0:h, 0:w]

        masked_data = np.where(mask, data, 0)
        total = np.sum(masked_data)

        if total > 0:
            cx = np.sum(x_grid * masked_data) / total
            cy = np.sum(y_grid * masked_data) / total
        else:
            cx, cy = w / 2, h / 2

        return cx, cy

    def _estimate_fwhm(self, data: np.ndarray, mask: np.ndarray) -> float:
        """
        Estimate FWHM from star data.

        Uses the second moment method.

        Args:
            data: Star data patch
            mask: Boolean mask

        Returns:
            Estimated FWHM in pixels
        """
        h, w = data.shape
        cy, cx = h / 2, w / 2
        y_grid, x_grid = np.mgrid[0:h, 0:w]

        masked_data = np.where(mask, np.maximum(data, 0), 0)
        total = np.sum(masked_data)

        if total > 0:
            # Calculate centroid
            cx = np.sum(x_grid * masked_data) / total
            cy = np.sum(y_grid * masked_data) / total

            # Calculate second moments
            var_x = np.sum((x_grid - cx)**2 * masked_data) / total
            var_y = np.sum((y_grid - cy)**2 * masked_data) / total

            # FWHM from variance (assuming Gaussian)
            sigma = np.sqrt((var_x + var_y) / 2)
            fwhm = 2.355 * sigma  # FWHM = 2.355 * sigma for Gaussian
        else:
            fwhm = 2.0

        return max(fwhm, 1.0)

    def detect_with_visualization(self, image: np.ndarray,
                                 mask: Optional[np.ndarray] = None) -> Tuple[List[DetectedStar], np.ndarray]:
        """
        Detect stars and return visualization image.

        Args:
            image: Input image
            mask: Optional mask

        Returns:
            (stars, visualization_image)
        """
        stars = self.detect(image, mask)

        # Create visualization
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        # Normalize for display
        vis = ((vis - vis.min()) / (vis.max() - vis.min() + 1) * 255).astype(np.uint8)

        # Draw detected stars
        for i, star in enumerate(stars[:100]):  # Limit to top 100
            x, y = int(star.x), int(star.y)
            r = max(3, int(star.fwhm))

            # Color by brightness rank
            if i < 10:
                color = (0, 255, 0)  # Green for top 10
            elif i < 30:
                color = (0, 255, 255)  # Yellow for next 20
            else:
                color = (0, 128, 255)  # Orange for rest

            cv2.circle(vis, (x, y), r, color, 1)
            cv2.circle(vis, (x, y), 1, color, -1)

        return stars, vis


def extract_star_triangles(stars: List[DetectedStar],
                          max_stars: int = 30,
                          min_side: float = 20.0,
                          max_side: float = 500.0) -> List[Tuple]:
    """
    Extract triangle patterns from detected stars for matching.

    Args:
        stars: List of detected stars
        max_stars: Maximum number of stars to use
        min_side: Minimum triangle side length in pixels
        max_side: Maximum triangle side length in pixels

    Returns:
        List of (star_indices, side_lengths, hash_key) tuples
    """
    use_stars = stars[:max_stars]
    triangles = []

    for i in range(len(use_stars)):
        for j in range(i + 1, len(use_stars)):
            for k in range(j + 1, len(use_stars)):
                s1, s2, s3 = use_stars[i], use_stars[j], use_stars[k]

                # Calculate side lengths in pixels
                d12 = math.hypot(s1.x - s2.x, s1.y - s2.y)
                d23 = math.hypot(s2.x - s3.x, s2.y - s3.y)
                d13 = math.hypot(s1.x - s3.x, s1.y - s3.y)

                sides = sorted([d12, d23, d13])

                # Filter by size
                if sides[0] < min_side or sides[2] > max_side:
                    continue

                # Compute hash
                r1 = sides[0] / sides[2]
                r2 = sides[1] / sides[2]
                hash_key = (int(r1 * 100), int(r2 * 100))

                triangles.append(((i, j, k), sides, hash_key))

    return triangles

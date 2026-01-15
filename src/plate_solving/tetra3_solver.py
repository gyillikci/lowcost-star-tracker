"""
Plate Solving using Tetra3 (ESA)

Uses the tetra3 library from the European Space Agency for fast, reliable
lost-in-space plate solving. Tetra3 is specifically designed for star trackers.

Installation:
    pip install tetra3

References:
    - GitHub: https://github.com/esa/tetra3
    - Docs: https://tetra3.readthedocs.io/
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Try to import tetra3
try:
    import tetra3
    TETRA3_AVAILABLE = True
except ImportError:
    TETRA3_AVAILABLE = False
    logger.warning("tetra3 not installed. Install with: pip install tetra3")

# Try to import OpenCV for image processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Try to import astropy for WCS
try:
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


@dataclass
class PlateSolution:
    """Result from plate solving."""
    # Pointing direction
    ra: float               # Right ascension in degrees
    dec: float              # Declination in degrees
    roll: float             # Roll angle in degrees

    # Field of view
    fov: float              # Field of view in degrees

    # Quality metrics
    n_matches: int          # Number of matched stars
    prob: float             # Match probability (0-1)
    t_solve: float          # Solve time in seconds

    # Pattern match info
    pattern_centroids: Optional[np.ndarray] = None
    matched_centroids: Optional[np.ndarray] = None
    matched_catID: Optional[np.ndarray] = None

    # Raw tetra3 result
    raw_result: Optional[Dict] = None

    @property
    def ra_hms(self) -> str:
        """RA in hours:minutes:seconds format."""
        hours = self.ra / 15.0
        h = int(hours)
        m = int((hours - h) * 60)
        s = ((hours - h) * 60 - m) * 60
        return f"{h:02d}h {m:02d}m {s:05.2f}s"

    @property
    def dec_dms(self) -> str:
        """Dec in degrees:arcmin:arcsec format."""
        sign = '+' if self.dec >= 0 else '-'
        d = abs(self.dec)
        deg = int(d)
        m = int((d - deg) * 60)
        s = ((d - deg) * 60 - m) * 60
        return f"{sign}{deg:02d}° {m:02d}' {s:04.1f}\""


class Tetra3Solver:
    """
    Plate solver using tetra3 from ESA.

    Tetra3 is a fast lost-in-space plate solver specifically designed
    for star trackers. It can solve images in ~10ms with 10 arcsec accuracy.

    Usage:
        solver = Tetra3Solver()
        solution = solver.solve(image, fov_estimate=10.0)
    """

    def __init__(self, database_path: Optional[str] = None,
                 max_fov: float = 30.0,
                 min_fov: float = 5.0):
        """
        Initialize the tetra3 solver.

        Args:
            database_path: Path to tetra3 database (None = use default)
            max_fov: Maximum field of view in degrees
            min_fov: Minimum field of view in degrees
        """
        if not TETRA3_AVAILABLE:
            raise ImportError(
                "tetra3 is required for plate solving.\n"
                "Install with: pip install tetra3"
            )

        self.max_fov = max_fov
        self.min_fov = min_fov
        self._solver = None
        self._database_path = database_path

        # Initialize solver
        self._init_solver()

    def _init_solver(self):
        """Initialize the tetra3 Tetra3 solver."""
        try:
            if self._database_path:
                self._solver = tetra3.Tetra3(self._database_path)
            else:
                # Use default database
                self._solver = tetra3.Tetra3()

            logger.info("Tetra3 solver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tetra3: {e}")
            raise

    def solve(self, image: np.ndarray,
              fov_estimate: Optional[float] = None,
              fov_max_error: float = 0.5,
              match_radius: float = 0.01,
              match_threshold: float = 1e-9,
              num_stars: int = 20,
              distortion: Optional[float] = None,
              return_matches: bool = True,
              timeout: float = None) -> Optional[PlateSolution]:
        """
        Solve an image to determine pointing direction.

        Args:
            image: Input image (grayscale or BGR)
            fov_estimate: Estimated field of view in degrees (None = search range)
            fov_max_error: Maximum FOV error fraction (default 0.5 = ±50%)
            match_radius: Star matching radius (fraction of FOV)
            match_threshold: Probability threshold for valid match
            num_stars: Number of brightest stars to use
            distortion: Radial distortion coefficient (None = auto)
            return_matches: Return matched star information
            timeout: Maximum solve time in seconds

        Returns:
            PlateSolution if successful, None otherwise
        """
        if self._solver is None:
            raise RuntimeError("Solver not initialized")

        start_time = time.time()

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Extract star centroids from image
        centroids = self._extract_centroids(gray, num_stars=num_stars)

        if centroids is None or len(centroids) < 4:
            logger.warning(f"Not enough stars detected ({len(centroids) if centroids is not None else 0})")
            return None

        logger.info(f"Detected {len(centroids)} stars")

        # Prepare solve parameters
        solve_kwargs = {
            'match_radius': match_radius,
            'match_threshold': match_threshold,
            'return_matches': return_matches,
        }

        if fov_estimate:
            solve_kwargs['fov_estimate'] = fov_estimate
            solve_kwargs['fov_max_error'] = fov_max_error
        else:
            # Search full range
            solve_kwargs['fov_estimate'] = (self.min_fov + self.max_fov) / 2
            solve_kwargs['fov_max_error'] = (self.max_fov - self.min_fov) / (self.min_fov + self.max_fov)

        if distortion is not None:
            solve_kwargs['distortion'] = distortion

        # Solve
        try:
            result = self._solver.solve_from_centroids(
                centroids,
                (gray.shape[0], gray.shape[1]),
                **solve_kwargs
            )
        except Exception as e:
            logger.error(f"Solve failed: {e}")
            return None

        solve_time = time.time() - start_time

        # Parse result
        if result is None or 'RA' not in result:
            logger.warning("No solution found")
            return None

        solution = PlateSolution(
            ra=float(result['RA']),
            dec=float(result['Dec']),
            roll=float(result['Roll']),
            fov=float(result['FOV']),
            n_matches=int(result.get('Matches', 0)),
            prob=float(result.get('Prob', 0)),
            t_solve=solve_time,
            pattern_centroids=result.get('pattern_centroids'),
            matched_centroids=result.get('matched_centroids'),
            matched_catID=result.get('matched_catID'),
            raw_result=result
        )

        logger.info(f"Solved: RA={solution.ra:.4f}°, Dec={solution.dec:.4f}°, "
                   f"FOV={solution.fov:.2f}°, matches={solution.n_matches}, "
                   f"time={solve_time*1000:.1f}ms")

        return solution

    def solve_from_centroids(self, centroids: np.ndarray,
                            image_size: Tuple[int, int],
                            **kwargs) -> Optional[PlateSolution]:
        """
        Solve from pre-extracted star centroids.

        Args:
            centroids: Star positions as Nx2 array (x, y)
            image_size: Image dimensions (height, width)
            **kwargs: Additional arguments passed to solve()

        Returns:
            PlateSolution if successful
        """
        if self._solver is None:
            raise RuntimeError("Solver not initialized")

        start_time = time.time()

        # Default parameters
        fov_estimate = kwargs.pop('fov_estimate', (self.min_fov + self.max_fov) / 2)
        fov_max_error = kwargs.pop('fov_max_error', 0.5)
        match_radius = kwargs.pop('match_radius', 0.01)
        match_threshold = kwargs.pop('match_threshold', 1e-9)
        return_matches = kwargs.pop('return_matches', True)

        try:
            result = self._solver.solve_from_centroids(
                centroids,
                image_size,
                fov_estimate=fov_estimate,
                fov_max_error=fov_max_error,
                match_radius=match_radius,
                match_threshold=match_threshold,
                return_matches=return_matches,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Solve failed: {e}")
            return None

        solve_time = time.time() - start_time

        if result is None or 'RA' not in result:
            return None

        return PlateSolution(
            ra=float(result['RA']),
            dec=float(result['Dec']),
            roll=float(result['Roll']),
            fov=float(result['FOV']),
            n_matches=int(result.get('Matches', 0)),
            prob=float(result.get('Prob', 0)),
            t_solve=solve_time,
            pattern_centroids=result.get('pattern_centroids'),
            matched_centroids=result.get('matched_centroids'),
            matched_catID=result.get('matched_catID'),
            raw_result=result
        )

    def _extract_centroids(self, image: np.ndarray,
                          num_stars: int = 20,
                          sigma: float = 3.0) -> Optional[np.ndarray]:
        """
        Extract star centroids from image using tetra3's built-in method.

        Args:
            image: Grayscale image
            num_stars: Maximum number of stars to return
            sigma: Detection threshold in sigma above background

        Returns:
            Nx2 array of (x, y) centroids, sorted by brightness
        """
        try:
            # Use tetra3's centroid extraction
            centroids = tetra3.get_centroids_from_image(
                image,
                sigma=sigma,
                filtsize=15,
                max_area=100,
                min_area=3,
                binary_open=True
            )

            if centroids is None or len(centroids) == 0:
                return None

            # Sort by brightness (tetra3 returns [y, x, brightness])
            if centroids.shape[1] >= 3:
                idx = np.argsort(centroids[:, 2])[::-1]  # Descending brightness
                centroids = centroids[idx]

            # Return top N as (x, y)
            result = centroids[:num_stars, :2]
            if centroids.shape[1] >= 2:
                result = result[:, [1, 0]]  # Swap to (x, y) if needed

            return result

        except Exception as e:
            logger.error(f"Centroid extraction failed: {e}")
            return None

    def get_wcs(self, solution: PlateSolution,
               image_shape: Tuple[int, int]) -> Optional['WCS']:
        """
        Get an astropy WCS object from the solution.

        Args:
            solution: Plate solution
            image_shape: Image dimensions (height, width)

        Returns:
            astropy.wcs.WCS object or None
        """
        if not ASTROPY_AVAILABLE:
            logger.warning("astropy not available for WCS")
            return None

        height, width = image_shape

        # Create WCS
        w = WCS(naxis=2)
        w.wcs.crpix = [width / 2, height / 2]
        w.wcs.crval = [solution.ra, solution.dec]

        # Pixel scale (degrees per pixel)
        scale = solution.fov / max(width, height)
        w.wcs.cdelt = [-scale, scale]

        # Rotation
        roll_rad = np.radians(solution.roll)
        w.wcs.pc = [
            [np.cos(roll_rad), -np.sin(roll_rad)],
            [np.sin(roll_rad), np.cos(roll_rad)]
        ]

        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']

        return w


class AllSkySolver:
    """
    Specialized solver for 180° AllSky fisheye images.

    AllSky images require different handling due to:
    - 180° field of view with fisheye distortion
    - Stars near horizon have significant distortion
    - Full hemisphere visible in single image

    This solver subdivides the image and solves regions separately.
    """

    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize AllSky solver.

        Args:
            database_path: Path to tetra3 database
        """
        self.solver = Tetra3Solver(
            database_path=database_path,
            max_fov=60.0,  # Solve sub-regions
            min_fov=10.0
        )

    def solve(self, image: np.ndarray,
             n_regions: int = 5,
             min_altitude: float = 20.0) -> List[PlateSolution]:
        """
        Solve AllSky image by dividing into regions.

        Args:
            image: AllSky image
            n_regions: Number of altitude regions to try
            min_altitude: Minimum altitude to consider (degrees from horizon)

        Returns:
            List of solutions from different regions
        """
        solutions = []
        height, width = image.shape[:2]
        cx, cy = width // 2, height // 2
        radius = min(width, height) // 2

        # Try concentric rings at different altitudes
        for i in range(n_regions):
            # Altitude from zenith (center) to horizon (edge)
            alt_frac = (i + 1) / (n_regions + 1)
            ring_radius = int(radius * alt_frac)

            # Extract ring region
            for angle in range(0, 360, 90):
                # Sample point on ring
                x = int(cx + ring_radius * np.cos(np.radians(angle)))
                y = int(cy + ring_radius * np.sin(np.radians(angle)))

                # Extract sub-image around this point
                sub_size = radius // 3
                x1 = max(0, x - sub_size)
                y1 = max(0, y - sub_size)
                x2 = min(width, x + sub_size)
                y2 = min(height, y + sub_size)

                sub_image = image[y1:y2, x1:x2]

                if sub_image.size == 0:
                    continue

                # Try to solve this region
                try:
                    solution = self.solver.solve(
                        sub_image,
                        fov_estimate=30.0,
                        fov_max_error=0.5
                    )
                    if solution and solution.n_matches >= 4:
                        solutions.append(solution)
                        logger.info(f"Solved region at alt={90*(1-alt_frac):.0f}°, "
                                   f"az={angle}°: RA={solution.ra:.2f}°")
                except Exception as e:
                    continue

        return solutions


def check_dependencies():
    """Check if required dependencies are available."""
    issues = []

    if not TETRA3_AVAILABLE:
        issues.append("tetra3 not installed: pip install tetra3")

    if not CV2_AVAILABLE:
        issues.append("OpenCV not installed: pip install opencv-python")

    if not ASTROPY_AVAILABLE:
        issues.append("astropy not installed (optional): pip install astropy")

    return issues


# Convenience function for quick solving
def solve_image(image_path: str,
                fov_estimate: Optional[float] = None) -> Optional[PlateSolution]:
    """
    Quick plate solve for an image file.

    Args:
        image_path: Path to image file
        fov_estimate: Estimated FOV in degrees

    Returns:
        PlateSolution if successful
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV required: pip install opencv-python")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    solver = Tetra3Solver()
    return solver.solve(image, fov_estimate=fov_estimate)

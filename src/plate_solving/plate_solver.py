"""
Plate Solving Algorithm

Implements blind astrometric plate solving using geometric pattern matching.
Based on the approach used by astrometry.net but simplified for star tracker use.

The algorithm:
1. Detect stars in the image
2. Build triangle patterns from detected stars
3. Match triangles against catalog using geometric hashing
4. Verify matches using star positions
5. Compute WCS solution
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import time

from .star_catalog import StarCatalog, CatalogStar
from .star_detector import StarDetector, DetectedStar, extract_star_triangles
from .fisheye_model import AllSkyCamera, FisheyeProjection, ProjectionType


@dataclass
class WCSSolution:
    """World Coordinate System solution from plate solving."""
    # Solved center coordinates
    ra_center: float          # Right ascension of image center (degrees)
    dec_center: float         # Declination of image center (degrees)

    # Plate scale
    pixel_scale: float        # Arcseconds per pixel

    # Rotation
    rotation: float           # Image rotation (degrees, E of N)

    # Field of view
    fov_x: float             # FOV in X direction (degrees)
    fov_y: float             # FOV in Y direction (degrees)

    # Solution quality
    n_matches: int           # Number of matched stars
    rms_error: float         # RMS position error (arcseconds)
    confidence: float        # Solution confidence (0-1)

    # Matched stars
    matched_stars: List[Tuple[DetectedStar, CatalogStar]] = field(default_factory=list)

    # Transformation matrix (3x3)
    transform: Optional[np.ndarray] = None

    # Timestamp
    solve_time: float = 0.0  # Time taken to solve (seconds)

    def pixel_to_radec(self, x: float, y: float,
                       width: int, height: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to RA/Dec using the solution.

        Uses gnomonic (tangent plane) projection for small fields,
        or the full transform for larger fields.
        """
        if self.transform is not None:
            # Use affine transform
            px = np.array([x, y, 1.0])
            result = self.transform @ px
            return result[0], result[1]

        # Simple tangent plane approximation
        cx, cy = width / 2, height / 2
        dx = (x - cx) * self.pixel_scale / 3600.0  # degrees
        dy = (y - cy) * self.pixel_scale / 3600.0

        # Apply rotation
        rot_rad = math.radians(self.rotation)
        xi = dx * math.cos(rot_rad) - dy * math.sin(rot_rad)
        eta = dx * math.sin(rot_rad) + dy * math.cos(rot_rad)

        # Gnomonic projection
        dec_rad = math.radians(self.dec_center)
        ra_rad = math.radians(self.ra_center)

        denom = math.cos(dec_rad) - eta * math.sin(dec_rad)
        ra = self.ra_center + math.degrees(math.atan2(xi, denom))
        dec = math.degrees(math.atan2(
            (eta * math.cos(dec_rad) + math.sin(dec_rad)) * math.cos(math.radians(ra - self.ra_center)),
            denom
        ))

        return ra % 360, dec


class PlateSolver:
    """
    Blind plate solver using geometric pattern matching.

    Algorithm:
    1. Detect stars in image using adaptive thresholding
    2. Extract triangle patterns from brightest stars
    3. Hash triangles by their side ratios (scale-invariant)
    4. Match against pre-computed catalog triangle index
    5. Verify candidate matches with RANSAC
    6. Compute final WCS transformation
    """

    def __init__(self,
                 catalog: Optional[StarCatalog] = None,
                 detector: Optional[StarDetector] = None,
                 max_catalog_stars: int = 100,
                 max_detected_stars: int = 50):
        """
        Initialize the plate solver.

        Args:
            catalog: Star catalog (default creates new with mag < 4.5)
            detector: Star detector (default creates new)
            max_catalog_stars: Max stars to use from catalog
            max_detected_stars: Max detected stars to use
        """
        self.catalog = catalog or StarCatalog(max_magnitude=4.5)
        self.detector = detector or StarDetector(sigma_threshold=3.0, min_snr=5.0)
        self.max_catalog_stars = max_catalog_stars
        self.max_detected_stars = max_detected_stars

        # Build triangle index from catalog
        self.triangle_index = self.catalog.build_triangle_index(max_catalog_stars)

    def solve(self, image: np.ndarray,
              camera: Optional[AllSkyCamera] = None,
              lst_hours: Optional[float] = None,
              timeout: float = 30.0) -> Optional[WCSSolution]:
        """
        Solve plate for an image.

        Args:
            image: Input image (grayscale or color)
            camera: Optional camera model (for fisheye/allsky)
            lst_hours: Local sidereal time in hours (for RA/Dec conversion)
            timeout: Maximum solve time in seconds

        Returns:
            WCSSolution if successful, None otherwise
        """
        start_time = time.time()

        # 1. Detect stars
        detected = self.detector.detect(image)
        if len(detected) < 4:
            print(f"Only {len(detected)} stars detected, need at least 4")
            return None

        print(f"Detected {len(detected)} stars")

        # Limit to brightest stars
        detected = detected[:self.max_detected_stars]

        # 2. Extract triangles from detected stars
        image_triangles = extract_star_triangles(
            detected,
            max_stars=min(30, len(detected)),
            min_side=20.0,
            max_side=max(image.shape) / 2
        )

        if len(image_triangles) < 10:
            print(f"Only {len(image_triangles)} triangles extracted")
            return None

        print(f"Extracted {len(image_triangles)} triangles from image")

        # 3. Find matching triangles
        candidate_matches = self._find_triangle_matches(
            image_triangles, detected, timeout, start_time
        )

        if len(candidate_matches) < 3:
            print(f"Only {len(candidate_matches)} candidate matches found")
            return None

        print(f"Found {len(candidate_matches)} candidate star matches")

        # 4. Verify and refine matches using RANSAC
        verified_matches = self._verify_matches(
            candidate_matches, detected, image.shape
        )

        if len(verified_matches) < 3:
            print(f"Only {len(verified_matches)} verified matches")
            return None

        print(f"Verified {len(verified_matches)} star matches")

        # 5. Compute WCS solution
        solution = self._compute_wcs(
            verified_matches, detected, image.shape, camera, lst_hours
        )

        if solution:
            solution.solve_time = time.time() - start_time

        return solution

    def _find_triangle_matches(self, image_triangles: List[Tuple],
                               detected: List[DetectedStar],
                               timeout: float, start_time: float) -> List[Tuple]:
        """
        Find candidate star matches by matching triangles.

        Returns list of (detected_star, catalog_star, vote_count) tuples.
        """
        # Vote accumulator for star matches
        votes: Dict[Tuple[int, int], int] = {}  # (detected_idx, catalog_hip) -> votes

        for img_tri in image_triangles:
            if time.time() - start_time > timeout:
                break

            det_indices, det_sides, det_hash = img_tri

            # Look for matching catalog triangles
            for tolerance in [0, 1, 2]:  # Hash tolerance
                for dh1 in range(-tolerance, tolerance + 1):
                    for dh2 in range(-tolerance, tolerance + 1):
                        lookup_hash = (det_hash[0] + dh1, det_hash[1] + dh2)

                        if lookup_hash in self.triangle_index:
                            for cat_tri in self.triangle_index[lookup_hash]:
                                cat_stars, cat_sides = cat_tri[:2]

                                # Check scale consistency
                                scale1 = det_sides[0] / cat_sides[0]
                                scale2 = det_sides[1] / cat_sides[1]
                                scale3 = det_sides[2] / cat_sides[2]

                                if abs(scale1 - scale2) / scale1 < 0.1 and \
                                   abs(scale2 - scale3) / scale2 < 0.1:
                                    # Vote for this correspondence
                                    for det_idx, cat_star in zip(det_indices, cat_stars):
                                        key = (det_idx, cat_star.hip_id)
                                        votes[key] = votes.get(key, 0) + 1

        # Convert votes to candidate matches
        candidates = []
        for (det_idx, hip_id), count in votes.items():
            if count >= 2:  # Minimum votes
                det_star = detected[det_idx]
                cat_star = next((s for s in self.catalog.stars if s.hip_id == hip_id), None)
                if cat_star:
                    candidates.append((det_star, cat_star, count))

        # Sort by vote count
        candidates.sort(key=lambda x: x[2], reverse=True)

        return candidates

    def _verify_matches(self, candidates: List[Tuple],
                       detected: List[DetectedStar],
                       image_shape: Tuple) -> List[Tuple[DetectedStar, CatalogStar]]:
        """
        Verify candidate matches using geometric consistency.

        Uses RANSAC-like approach to find consistent subset.
        """
        if len(candidates) < 3:
            return []

        best_inliers = []

        # Try different combinations
        for i in range(min(len(candidates), 10)):
            for j in range(i + 1, min(len(candidates), 15)):
                for k in range(j + 1, min(len(candidates), 20)):
                    # Use these 3 as reference
                    ref_matches = [candidates[i], candidates[j], candidates[k]]

                    # Check if angular separations match
                    det_d12 = math.hypot(
                        ref_matches[0][0].x - ref_matches[1][0].x,
                        ref_matches[0][0].y - ref_matches[1][0].y
                    )
                    det_d13 = math.hypot(
                        ref_matches[0][0].x - ref_matches[2][0].x,
                        ref_matches[0][0].y - ref_matches[2][0].y
                    )
                    det_d23 = math.hypot(
                        ref_matches[1][0].x - ref_matches[2][0].x,
                        ref_matches[1][0].y - ref_matches[2][0].y
                    )

                    cat_d12 = ref_matches[0][1].angular_distance(ref_matches[1][1])
                    cat_d13 = ref_matches[0][1].angular_distance(ref_matches[2][1])
                    cat_d23 = ref_matches[1][1].angular_distance(ref_matches[2][1])

                    # Estimate pixel scale from this triplet
                    scales = []
                    if cat_d12 > 0.1:
                        scales.append(det_d12 / cat_d12)
                    if cat_d13 > 0.1:
                        scales.append(det_d13 / cat_d13)
                    if cat_d23 > 0.1:
                        scales.append(det_d23 / cat_d23)

                    if len(scales) < 2:
                        continue

                    avg_scale = np.mean(scales)
                    scale_std = np.std(scales)

                    if scale_std / avg_scale > 0.2:  # Too inconsistent
                        continue

                    # Count inliers
                    inliers = []
                    for det_star, cat_star, _ in candidates:
                        # Check distance to reference stars
                        is_inlier = True
                        for ref_det, ref_cat, _ in ref_matches:
                            det_dist = math.hypot(det_star.x - ref_det.x, det_star.y - ref_det.y)
                            cat_dist = cat_star.angular_distance(ref_cat)

                            predicted_dist = cat_dist * avg_scale
                            error = abs(det_dist - predicted_dist)

                            if error > 0.2 * predicted_dist + 10:  # Allow some tolerance
                                is_inlier = False
                                break

                        if is_inlier:
                            inliers.append((det_star, cat_star))

                    if len(inliers) > len(best_inliers):
                        best_inliers = inliers

        return best_inliers

    def _compute_wcs(self, matches: List[Tuple[DetectedStar, CatalogStar]],
                    detected: List[DetectedStar],
                    image_shape: Tuple,
                    camera: Optional[AllSkyCamera],
                    lst_hours: Optional[float]) -> Optional[WCSSolution]:
        """
        Compute WCS solution from verified matches.
        """
        if len(matches) < 3:
            return None

        height, width = image_shape[:2]
        cx, cy = width / 2, height / 2

        # Calculate center of matched stars
        det_positions = np.array([[m[0].x, m[0].y] for m in matches])
        cat_positions = np.array([[m[1].ra, m[1].dec] for m in matches])

        # Estimate center RA/Dec
        det_center = np.mean(det_positions, axis=0)
        ra_center = np.mean(cat_positions[:, 0])
        dec_center = np.mean(cat_positions[:, 1])

        # Estimate pixel scale from pairwise distances
        scales = []
        for i in range(len(matches)):
            for j in range(i + 1, len(matches)):
                det_dist = math.hypot(
                    matches[i][0].x - matches[j][0].x,
                    matches[i][0].y - matches[j][0].y
                )
                cat_dist = matches[i][1].angular_distance(matches[j][1])

                if cat_dist > 0.5 and det_dist > 10:
                    # pixels per degree
                    scales.append(det_dist / cat_dist)

        if not scales:
            return None

        pixel_scale_deg = 1.0 / np.median(scales)  # degrees per pixel
        pixel_scale_arcsec = pixel_scale_deg * 3600

        # Estimate rotation
        rotations = []
        for i in range(len(matches)):
            for j in range(i + 1, len(matches)):
                det_dx = matches[j][0].x - matches[i][0].x
                det_dy = matches[j][0].y - matches[i][0].y

                cat_dra = (matches[j][1].ra - matches[i][1].ra) * math.cos(math.radians(dec_center))
                cat_ddec = matches[j][1].dec - matches[i][1].dec

                if abs(det_dx) > 10 and abs(cat_dra) > 0.1:
                    det_angle = math.atan2(det_dy, det_dx)
                    cat_angle = math.atan2(cat_ddec, cat_dra)
                    rot = math.degrees(det_angle - cat_angle)
                    rotations.append(rot % 360)

        rotation = np.median(rotations) if rotations else 0.0

        # Calculate RMS error
        errors = []
        for det_star, cat_star in matches:
            # Predicted position
            dra = (cat_star.ra - ra_center) * math.cos(math.radians(dec_center))
            ddec = cat_star.dec - dec_center

            rot_rad = math.radians(rotation)
            pred_dx = (dra * math.cos(rot_rad) - ddec * math.sin(rot_rad)) / pixel_scale_deg
            pred_dy = (dra * math.sin(rot_rad) + ddec * math.cos(rot_rad)) / pixel_scale_deg

            pred_x = cx + pred_dx
            pred_y = cy - pred_dy  # Flip Y for image coordinates

            error_pix = math.hypot(det_star.x - pred_x, det_star.y - pred_y)
            error_arcsec = error_pix * pixel_scale_arcsec
            errors.append(error_arcsec)

        rms_error = np.sqrt(np.mean(np.array(errors)**2))

        # Calculate confidence
        n_expected = len([s for s in self.catalog.stars if s.mag < 3.5])
        confidence = min(1.0, len(matches) / (n_expected * 0.3))
        if rms_error > 60:  # More than 1 arcmin error
            confidence *= 0.5

        # Calculate FOV
        fov_x = width * pixel_scale_deg
        fov_y = height * pixel_scale_deg

        return WCSSolution(
            ra_center=ra_center,
            dec_center=dec_center,
            pixel_scale=pixel_scale_arcsec,
            rotation=rotation,
            fov_x=fov_x,
            fov_y=fov_y,
            n_matches=len(matches),
            rms_error=rms_error,
            confidence=confidence,
            matched_stars=matches
        )

    def solve_allsky(self, image: np.ndarray,
                     lat: float = 20.02,
                     lon: float = -155.67,
                     lst_hours: Optional[float] = None,
                     timestamp: Optional[datetime] = None) -> Optional[WCSSolution]:
        """
        Solve an AllSky image.

        Special handling for 180° fisheye images.

        Args:
            image: AllSky image
            lat: Observer latitude
            lon: Observer longitude
            lst_hours: Local sidereal time (calculated from timestamp if not provided)
            timestamp: UTC timestamp of image

        Returns:
            WCSSolution if successful
        """
        # Calculate LST if not provided
        if lst_hours is None and timestamp:
            lst_hours = self._calculate_lst(lon, timestamp)

        # Create camera model
        height, width = image.shape[:2]
        camera = AllSkyCamera.create_typical_allsky(
            width=width, height=height,
            fov=180.0, lat=lat, lon=lon
        )

        # Detect stars with AllSky-appropriate settings
        detector = StarDetector(
            sigma_threshold=2.5,  # Lower threshold for wide field
            min_snr=3.0,
            max_fwhm=30.0,        # Larger FWHM for fisheye distortion
            max_elongation=3.0
        )

        detected = detector.detect(image)

        if len(detected) < 10:
            print(f"Only {len(detected)} stars detected in AllSky image")
            return None

        print(f"Detected {len(detected)} stars in AllSky image")

        # For AllSky, we need to handle the fisheye projection
        # Convert detected pixel positions to Alt/Az
        if lst_hours is not None:
            # Get expected visible stars
            visible = self.catalog.get_visible_stars(lat, lon, lst_hours, min_altitude=10.0)
            print(f"{len(visible)} catalog stars should be visible")

            # Match by position in Alt/Az space
            matches = self._match_allsky_stars(detected, visible, camera, lst_hours)

            if len(matches) >= 3:
                return self._compute_allsky_wcs(matches, camera, image.shape)

        # Fall back to standard solving
        return self.solve(image, camera, lst_hours)

    def _match_allsky_stars(self, detected: List[DetectedStar],
                           visible: List[Tuple[CatalogStar, float, float]],
                           camera: AllSkyCamera,
                           lst_hours: float) -> List[Tuple[DetectedStar, CatalogStar]]:
        """
        Match detected stars to catalog for AllSky images.

        Uses Alt/Az positions for matching.
        """
        matches = []

        for det_star in detected[:50]:
            # Convert pixel to Alt/Az
            det_alt, det_az = camera.pixel_to_altaz(det_star.x, det_star.y)

            # Find closest catalog star
            best_dist = float('inf')
            best_cat = None

            for cat_star, cat_alt, cat_az in visible:
                # Angular distance in Alt/Az
                dist = math.sqrt((det_alt - cat_alt)**2 +
                               ((det_az - cat_az) * math.cos(math.radians(det_alt)))**2)

                if dist < best_dist and dist < 5.0:  # 5 degree tolerance
                    best_dist = dist
                    best_cat = cat_star

            if best_cat:
                matches.append((det_star, best_cat))

        return matches

    def _compute_allsky_wcs(self, matches: List[Tuple[DetectedStar, CatalogStar]],
                           camera: AllSkyCamera,
                           image_shape: Tuple) -> WCSSolution:
        """Compute WCS for AllSky image."""
        height, width = image_shape[:2]

        # Calculate zenith RA/Dec (this is our "center")
        # For zenith-pointed camera, the center is at (90° alt, 0° az)
        ra_center, dec_center = camera.altaz_to_radec(90, 0, 0)  # Approximate

        # Pixel scale varies across fisheye, use average
        pixel_scale = camera.projection.fov_degrees * 3600 / min(width, height)

        # Calculate errors
        errors = []
        for det_star, cat_star in matches:
            det_alt, det_az = camera.pixel_to_altaz(det_star.x, det_star.y)
            cat_alt, cat_az = camera.radec_to_altaz(cat_star.ra, cat_star.dec, 0)

            if cat_alt is not None:
                error = math.sqrt((det_alt - cat_alt)**2 +
                                ((det_az - cat_az) * math.cos(math.radians(det_alt)))**2)
                errors.append(error * 3600)  # Convert to arcsec

        rms_error = np.sqrt(np.mean(np.array(errors)**2)) if errors else 999.0

        return WCSSolution(
            ra_center=ra_center,
            dec_center=dec_center,
            pixel_scale=pixel_scale,
            rotation=camera.rotation,
            fov_x=camera.projection.fov_degrees,
            fov_y=camera.projection.fov_degrees,
            n_matches=len(matches),
            rms_error=rms_error,
            confidence=min(1.0, len(matches) / 20),
            matched_stars=matches
        )

    def _calculate_lst(self, longitude: float, timestamp: datetime) -> float:
        """Calculate Local Sidereal Time from UTC timestamp."""
        # Julian Date
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day
        hour = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600

        if month <= 2:
            year -= 1
            month += 12

        A = int(year / 100)
        B = 2 - A + int(A / 4)
        JD = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + hour / 24 + B - 1524.5

        # Julian centuries from J2000
        T = (JD - 2451545.0) / 36525.0

        # Greenwich Mean Sidereal Time
        GMST = 280.46061837 + 360.98564736629 * (JD - 2451545.0) + T * T * (0.000387933 - T / 38710000)
        GMST = GMST % 360

        # Local Sidereal Time
        LST = GMST + longitude
        LST = LST % 360

        return LST / 15.0  # Convert to hours

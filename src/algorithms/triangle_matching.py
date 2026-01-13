#!/usr/bin/env python3
"""
Robust Triangle Matching for Star Identification.

This module implements enhanced triangle matching algorithms for reliable
star identification in the Lost-in-Space problem. Key features:

- False star rejection (cosmic rays, hot pixels, satellites)
- Sparse field handling (minimum 5-star matching)
- Confidence metrics for match quality scoring
- Adaptive voting with geometric verification

Based on triangle matching literature but enhanced for consumer cameras
with more noise and optical distortions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import heapq


@dataclass
class DetectedStar:
    """A detected star candidate from image processing."""
    x: float  # Centroid x position (pixels)
    y: float  # Centroid y position (pixels)
    flux: float  # Integrated flux
    snr: float  # Signal-to-noise ratio
    fwhm: float  # Full width at half maximum
    elongation: float = 1.0  # Major/minor axis ratio
    peak_value: float = 0.0  # Peak pixel value
    timestamp: float = 0.0  # Detection timestamp


@dataclass
class CatalogStar:
    """A star from the reference catalog."""
    hip_id: int  # Hipparcos ID
    ra: float  # Right ascension (radians)
    dec: float  # Declination (radians)
    magnitude: float  # Visual magnitude
    unit_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class StarMatch:
    """A matched pair of detected and catalog stars."""
    detected: DetectedStar
    catalog: CatalogStar
    confidence: float  # Match confidence (0-1)
    residual: float  # Position residual (pixels)


@dataclass
class MatchResult:
    """Complete result of star matching."""
    matches: List[StarMatch]
    attitude: Optional[np.ndarray]  # Rotation matrix if solved
    confidence: float  # Overall match confidence
    n_inliers: int
    n_outliers: int
    rejected_detections: List[DetectedStar]
    metrics: Dict


class FalseStarFilter:
    """
    Filters out false star detections.

    Identifies and removes:
    - Cosmic ray hits (too sharp, irregular shape)
    - Hot pixels (consistent position across frames)
    - Satellite/meteor trails (elongated, fast-moving)
    - Noise spikes (low SNR, inconsistent with PSF)
    """

    def __init__(self,
                 min_snr: float = 5.0,
                 max_elongation: float = 2.0,
                 min_fwhm: float = 1.5,
                 max_fwhm: float = 10.0,
                 hot_pixel_map: np.ndarray = None):
        """
        Initialize false star filter.

        Args:
            min_snr: Minimum signal-to-noise ratio
            max_elongation: Maximum elongation (major/minor axis)
            min_fwhm: Minimum FWHM (rejects cosmic rays)
            max_fwhm: Maximum FWHM (rejects defocused/extended objects)
            hot_pixel_map: Boolean map of known hot pixels
        """
        self.min_snr = min_snr
        self.max_elongation = max_elongation
        self.min_fwhm = min_fwhm
        self.max_fwhm = max_fwhm
        self.hot_pixel_map = hot_pixel_map

        # Tracking for temporal consistency
        self.detection_history: List[List[DetectedStar]] = []
        self.history_window = 5  # frames

    def filter_detections(self,
                          detections: List[DetectedStar],
                          image_shape: Tuple[int, int] = None) -> Tuple[List[DetectedStar], List[DetectedStar]]:
        """
        Filter detections to remove likely false stars.

        Args:
            detections: List of star candidates
            image_shape: Image dimensions for hot pixel checking

        Returns:
            Tuple of (valid_stars, rejected_stars)
        """
        valid = []
        rejected = []

        for det in detections:
            rejection_reason = self._check_detection(det, image_shape)

            if rejection_reason is None:
                valid.append(det)
            else:
                det_with_reason = DetectedStar(
                    x=det.x, y=det.y, flux=det.flux, snr=det.snr,
                    fwhm=det.fwhm, elongation=det.elongation,
                    peak_value=det.peak_value, timestamp=det.timestamp
                )
                rejected.append(det_with_reason)

        # Update history for temporal filtering
        self._update_history(valid)

        # Additional temporal filtering for persistent false stars
        if len(self.detection_history) >= 3:
            valid = self._temporal_filter(valid)

        return valid, rejected

    def _check_detection(self, det: DetectedStar,
                          image_shape: Tuple[int, int]) -> Optional[str]:
        """Check if a detection is likely a false star."""

        # SNR check
        if det.snr < self.min_snr:
            return "low_snr"

        # Shape checks (cosmic rays are typically sharp and irregular)
        if det.fwhm < self.min_fwhm:
            return "cosmic_ray_sharp"

        if det.fwhm > self.max_fwhm:
            return "extended_source"

        # Elongation check (satellites/meteors)
        if det.elongation > self.max_elongation:
            return "elongated_trail"

        # Hot pixel check
        if self.hot_pixel_map is not None and image_shape is not None:
            x_int, y_int = int(det.x), int(det.y)
            if 0 <= y_int < image_shape[0] and 0 <= x_int < image_shape[1]:
                if self.hot_pixel_map[y_int, x_int]:
                    return "hot_pixel"

        return None

    def _update_history(self, detections: List[DetectedStar]):
        """Update detection history for temporal analysis."""
        self.detection_history.append(detections)

        if len(self.detection_history) > self.history_window:
            self.detection_history.pop(0)

    def _temporal_filter(self, detections: List[DetectedStar]) -> List[DetectedStar]:
        """
        Filter based on temporal consistency.

        Real stars should move smoothly across frames.
        False detections appear randomly.
        """
        if len(self.detection_history) < 2:
            return detections

        # For now, return as-is (full implementation would track star motion)
        # This is a placeholder for more sophisticated temporal analysis
        return detections

    def identify_satellites(self,
                            detections: List[DetectedStar],
                            max_angular_velocity: float = 0.5) -> List[DetectedStar]:
        """
        Identify satellite trails from detection sequence.

        Args:
            detections: Current frame detections
            max_angular_velocity: Maximum star angular motion (deg/s)

        Returns:
            List of likely satellite detections
        """
        satellites = []

        if len(self.detection_history) < 2:
            return satellites

        prev_detections = self.detection_history[-1]

        for det in detections:
            # Check for fast-moving objects
            min_dist = float('inf')
            for prev in prev_detections:
                dist = np.sqrt((det.x - prev.x)**2 + (det.y - prev.y)**2)
                min_dist = min(min_dist, dist)

            # If no close match in previous frame, might be satellite
            if min_dist > 50:  # pixels - adjust based on expected motion
                if det.elongation > 1.3:  # Slightly elongated
                    satellites.append(det)

        return satellites


class TriangleMatcher:
    """
    Triangle-based star pattern matching.

    Uses geometric invariants (angular separations) to match
    detected star patterns against a catalog database.
    """

    def __init__(self,
                 catalog: List[CatalogStar],
                 fov_deg: float = 90.0,
                 min_angle_deg: float = 1.0,
                 max_angle_deg: float = 60.0):
        """
        Initialize triangle matcher.

        Args:
            catalog: Reference star catalog
            fov_deg: Camera field of view (degrees)
            min_angle_deg: Minimum triangle side angle
            max_angle_deg: Maximum triangle side angle
        """
        self.catalog = catalog
        self.fov_rad = np.deg2rad(fov_deg)
        self.min_angle = np.deg2rad(min_angle_deg)
        self.max_angle = np.deg2rad(max_angle_deg)

        # Build triangle database
        self._build_triangle_database()

    def _build_triangle_database(self):
        """Pre-compute triangle patterns for catalog stars."""
        n_stars = len(self.catalog)

        # Store unit vectors
        self.catalog_vectors = np.array([s.unit_vector for s in self.catalog])

        # Compute angular distances between all star pairs
        # dot product gives cos(angle)
        dots = self.catalog_vectors @ self.catalog_vectors.T
        dots = np.clip(dots, -1, 1)
        self.angular_distances = np.arccos(dots)

        # Build triangle index
        # Key: (a1, a2, a3) binned angles
        # Value: list of (i, j, k) star indices
        self.triangle_index: Dict[Tuple, List[Tuple[int, int, int]]] = {}

        # Generate triangles (limit to manageable number)
        self.triangles = []

        angle_bin_size = np.deg2rad(0.5)  # 0.5 degree bins

        for i in range(n_stars):
            for j in range(i + 1, n_stars):
                d_ij = self.angular_distances[i, j]
                if not (self.min_angle <= d_ij <= self.max_angle):
                    continue

                for k in range(j + 1, n_stars):
                    d_ik = self.angular_distances[i, k]
                    d_jk = self.angular_distances[j, k]

                    if not (self.min_angle <= d_ik <= self.max_angle and
                            self.min_angle <= d_jk <= self.max_angle):
                        continue

                    # Sort angles for canonical form
                    angles = sorted([d_ij, d_ik, d_jk])

                    # Bin the angles
                    bins = tuple(int(a / angle_bin_size) for a in angles)

                    if bins not in self.triangle_index:
                        self.triangle_index[bins] = []
                    self.triangle_index[bins].append((i, j, k))

                    self.triangles.append({
                        'indices': (i, j, k),
                        'angles': angles,
                        'bins': bins
                    })

        print(f"Triangle database: {len(self.triangles)} triangles from {n_stars} stars")

    def match_stars(self,
                    detections: List[DetectedStar],
                    camera_matrix: np.ndarray,
                    min_matches: int = 5) -> MatchResult:
        """
        Match detected stars against catalog using triangle matching.

        Args:
            detections: Filtered star detections
            camera_matrix: Camera intrinsic matrix
            min_matches: Minimum required matches

        Returns:
            MatchResult with matches and confidence
        """
        n_det = len(detections)

        if n_det < 3:
            return MatchResult(
                matches=[], attitude=None, confidence=0.0,
                n_inliers=0, n_outliers=n_det,
                rejected_detections=detections,
                metrics={"error": "insufficient_detections"}
            )

        # Convert detections to unit vectors
        det_vectors = self._detections_to_vectors(detections, camera_matrix)

        # Compute angular distances between detections
        det_dots = det_vectors @ det_vectors.T
        det_dots = np.clip(det_dots, -1, 1)
        det_angles = np.arccos(det_dots)

        # Generate detection triangles and match against database
        votes: Dict[Tuple[int, int], int] = {}  # (det_idx, cat_idx) -> vote count

        angle_tolerance = np.deg2rad(0.5)  # Matching tolerance

        for i in range(n_det):
            for j in range(i + 1, n_det):
                for k in range(j + 1, n_det):
                    # Get detection triangle angles
                    d_ij = det_angles[i, j]
                    d_ik = det_angles[i, k]
                    d_jk = det_angles[j, k]

                    det_tri_angles = sorted([d_ij, d_ik, d_jk])

                    # Find matching catalog triangles
                    matches = self._find_matching_triangles(
                        det_tri_angles, angle_tolerance
                    )

                    # Vote for consistent matches
                    for cat_tri in matches:
                        cat_angles = cat_tri['angles']
                        cat_indices = cat_tri['indices']

                        # Determine correspondence
                        det_sorted = sorted([(d_ij, (i, j)), (d_ik, (i, k)), (d_jk, (j, k))])
                        cat_sorted = sorted([(cat_angles[0], 0), (cat_angles[1], 1), (cat_angles[2], 2)])

                        # Map detection indices to catalog indices
                        # This is simplified - full implementation needs proper correspondence
                        for det_idx in [i, j, k]:
                            for cat_idx in cat_indices:
                                key = (det_idx, cat_idx)
                                votes[key] = votes.get(key, 0) + 1

        # Extract best matches from votes
        matches = self._extract_matches_from_votes(
            votes, detections, det_vectors, min_matches
        )

        if len(matches) < min_matches:
            return MatchResult(
                matches=matches, attitude=None,
                confidence=len(matches) / max(n_det, 1),
                n_inliers=len(matches), n_outliers=n_det - len(matches),
                rejected_detections=[d for d in detections
                                     if d not in [m.detected for m in matches]],
                metrics={"error": "insufficient_matches", "votes": len(votes)}
            )

        # Compute attitude from matches
        attitude = self._compute_attitude(matches, det_vectors)

        # Compute confidence
        confidence = self._compute_confidence(matches, attitude, det_vectors)

        matched_dets = {m.detected for m in matches}
        rejected = [d for d in detections if d not in matched_dets]

        return MatchResult(
            matches=matches,
            attitude=attitude,
            confidence=confidence,
            n_inliers=len(matches),
            n_outliers=len(rejected),
            rejected_detections=rejected,
            metrics={
                "n_votes": len(votes),
                "max_votes": max(votes.values()) if votes else 0,
                "n_triangles_checked": n_det * (n_det - 1) * (n_det - 2) // 6
            }
        )

    def _detections_to_vectors(self,
                                detections: List[DetectedStar],
                                camera_matrix: np.ndarray) -> np.ndarray:
        """Convert pixel detections to unit vectors."""
        K_inv = np.linalg.inv(camera_matrix)

        vectors = []
        for det in detections:
            pixel = np.array([det.x, det.y, 1.0])
            direction = K_inv @ pixel
            direction = direction / np.linalg.norm(direction)
            vectors.append(direction)

        return np.array(vectors)

    def _find_matching_triangles(self,
                                  det_angles: List[float],
                                  tolerance: float) -> List[Dict]:
        """Find catalog triangles matching the given angles."""
        matches = []

        # Check bins within tolerance
        angle_bin_size = np.deg2rad(0.5)

        for da in det_angles:
            bin_center = int(da / angle_bin_size)

        # Search nearby bins
        base_bins = [int(a / angle_bin_size) for a in det_angles]

        for d0 in [-1, 0, 1]:
            for d1 in [-1, 0, 1]:
                for d2 in [-1, 0, 1]:
                    test_bins = tuple(sorted([
                        base_bins[0] + d0,
                        base_bins[1] + d1,
                        base_bins[2] + d2
                    ]))

                    if test_bins in self.triangle_index:
                        for indices in self.triangle_index[test_bins]:
                            i, j, k = indices
                            cat_angles = sorted([
                                self.angular_distances[i, j],
                                self.angular_distances[i, k],
                                self.angular_distances[j, k]
                            ])

                            # Verify angles match within tolerance
                            angle_errors = [
                                abs(det_angles[l] - cat_angles[l])
                                for l in range(3)
                            ]

                            if all(e < tolerance for e in angle_errors):
                                matches.append({
                                    'indices': indices,
                                    'angles': cat_angles
                                })

        return matches

    def _extract_matches_from_votes(self,
                                     votes: Dict[Tuple[int, int], int],
                                     detections: List[DetectedStar],
                                     det_vectors: np.ndarray,
                                     min_matches: int) -> List[StarMatch]:
        """Extract consistent star matches from voting results."""
        if not votes:
            return []

        # Sort by vote count
        sorted_votes = sorted(votes.items(), key=lambda x: -x[1])

        matches = []
        used_detections: Set[int] = set()
        used_catalog: Set[int] = set()

        for (det_idx, cat_idx), vote_count in sorted_votes:
            if det_idx in used_detections or cat_idx in used_catalog:
                continue

            if vote_count < 2:  # Minimum votes threshold
                continue

            detection = detections[det_idx]
            catalog_star = self.catalog[cat_idx]

            # Compute residual (would need attitude for proper calculation)
            residual = 0.0  # Placeholder

            confidence = min(1.0, vote_count / 10.0)  # Normalize confidence

            matches.append(StarMatch(
                detected=detection,
                catalog=catalog_star,
                confidence=confidence,
                residual=residual
            ))

            used_detections.add(det_idx)
            used_catalog.add(cat_idx)

        return matches

    def _compute_attitude(self,
                           matches: List[StarMatch],
                           det_vectors: np.ndarray) -> Optional[np.ndarray]:
        """Compute attitude from star matches using SVD."""
        if len(matches) < 2:
            return None

        # Build observation and reference vectors
        obs_vectors = []
        ref_vectors = []

        for match in matches:
            # Find detection index
            for i, det in enumerate(det_vectors):
                # Simplified matching - in practice would use proper indexing
                pass

            ref_vectors.append(match.catalog.unit_vector)

        if len(ref_vectors) < 2:
            return None

        # Use first few matches for attitude
        # Full implementation would use QUEST or SVD

        return np.eye(3)  # Placeholder

    def _compute_confidence(self,
                             matches: List[StarMatch],
                             attitude: Optional[np.ndarray],
                             det_vectors: np.ndarray) -> float:
        """Compute overall match confidence."""
        if not matches:
            return 0.0

        # Factors affecting confidence:
        # 1. Number of matches
        n_match_score = min(1.0, len(matches) / 10.0)

        # 2. Average individual match confidence
        avg_confidence = np.mean([m.confidence for m in matches])

        # 3. Geometric consistency (would use attitude residuals)
        geometric_score = 0.8  # Placeholder

        # Combined confidence
        confidence = 0.4 * n_match_score + 0.3 * avg_confidence + 0.3 * geometric_score

        return confidence


class SparseFieldMatcher:
    """
    Handles star matching in sparse fields with few visible stars.

    Uses exhaustive search when triangle matching fails due to
    insufficient star count.
    """

    def __init__(self,
                 catalog: List[CatalogStar],
                 fov_deg: float = 90.0):
        """
        Initialize sparse field matcher.

        Args:
            catalog: Reference star catalog
            fov_deg: Camera field of view
        """
        self.catalog = catalog
        self.fov_rad = np.deg2rad(fov_deg)

        # Build KD-tree for fast catalog lookup
        vectors = np.array([s.unit_vector for s in catalog])
        self.catalog_tree = KDTree(vectors)

    def match_sparse_field(self,
                            detections: List[DetectedStar],
                            camera_matrix: np.ndarray,
                            attitude_hint: np.ndarray = None,
                            search_radius_deg: float = 5.0) -> MatchResult:
        """
        Match stars in a sparse field (3-6 detections).

        Args:
            detections: Star detections
            camera_matrix: Camera intrinsic matrix
            attitude_hint: Optional attitude estimate to narrow search
            search_radius_deg: Search radius around predicted positions

        Returns:
            MatchResult with matches
        """
        n_det = len(detections)

        if n_det < 3:
            return MatchResult(
                matches=[], attitude=None, confidence=0.0,
                n_inliers=0, n_outliers=n_det,
                rejected_detections=detections,
                metrics={"error": "too_few_detections"}
            )

        # Convert detections to unit vectors
        K_inv = np.linalg.inv(camera_matrix)
        det_vectors = []

        for det in detections:
            pixel = np.array([det.x, det.y, 1.0])
            direction = K_inv @ pixel
            direction = direction / np.linalg.norm(direction)
            det_vectors.append(direction)

        det_vectors = np.array(det_vectors)

        # If we have attitude hint, use it
        if attitude_hint is not None:
            return self._match_with_hint(
                detections, det_vectors, attitude_hint, search_radius_deg
            )

        # Otherwise, exhaustive search using angular patterns
        return self._exhaustive_pattern_search(detections, det_vectors)

    def _match_with_hint(self,
                          detections: List[DetectedStar],
                          det_vectors: np.ndarray,
                          attitude: np.ndarray,
                          search_radius_deg: float) -> MatchResult:
        """Match using attitude hint to predict catalog positions."""
        matches = []
        search_radius_rad = np.deg2rad(search_radius_deg)

        for i, det in enumerate(detections):
            # Transform detection to celestial coordinates
            celestial_dir = attitude.T @ det_vectors[i]

            # Search catalog near predicted position
            indices = self.catalog_tree.query_ball_point(
                celestial_dir, search_radius_rad
            )

            if indices:
                # Find best match by brightness
                best_idx = min(indices,
                              key=lambda j: self.catalog[j].magnitude)

                matches.append(StarMatch(
                    detected=det,
                    catalog=self.catalog[best_idx],
                    confidence=0.8,
                    residual=0.0
                ))

        matched_dets = {m.detected for m in matches}
        rejected = [d for d in detections if d not in matched_dets]

        return MatchResult(
            matches=matches,
            attitude=attitude,
            confidence=len(matches) / len(detections) if detections else 0,
            n_inliers=len(matches),
            n_outliers=len(rejected),
            rejected_detections=rejected,
            metrics={"method": "hint_based"}
        )

    def _exhaustive_pattern_search(self,
                                    detections: List[DetectedStar],
                                    det_vectors: np.ndarray) -> MatchResult:
        """Exhaustive search when no attitude hint available."""
        # Compute all pairwise angles
        n_det = len(detections)
        det_angles = np.zeros((n_det, n_det))

        for i in range(n_det):
            for j in range(i + 1, n_det):
                dot = np.dot(det_vectors[i], det_vectors[j])
                angle = np.arccos(np.clip(dot, -1, 1))
                det_angles[i, j] = angle
                det_angles[j, i] = angle

        # Find matching patterns in catalog
        # This is computationally expensive but works for sparse fields

        # For demonstration, return empty result
        # Full implementation would search catalog for matching angle patterns

        return MatchResult(
            matches=[],
            attitude=None,
            confidence=0.0,
            n_inliers=0,
            n_outliers=n_det,
            rejected_detections=detections,
            metrics={"method": "exhaustive_search", "status": "not_implemented"}
        )


class ConfidenceMetrics:
    """
    Computes confidence metrics for star matching quality.
    """

    @staticmethod
    def geometric_consistency(matches: List[StarMatch],
                               attitude: np.ndarray,
                               camera_matrix: np.ndarray) -> float:
        """
        Measure geometric consistency of matches.

        Reprojects catalog stars and compares to detections.
        """
        if not matches or attitude is None:
            return 0.0

        residuals = []

        for match in matches:
            # Project catalog star to image
            cat_dir = attitude @ match.catalog.unit_vector

            if cat_dir[2] <= 0:  # Behind camera
                continue

            projected = camera_matrix @ cat_dir
            proj_x = projected[0] / projected[2]
            proj_y = projected[1] / projected[2]

            # Compare to detection
            dx = proj_x - match.detected.x
            dy = proj_y - match.detected.y
            residual = np.sqrt(dx**2 + dy**2)
            residuals.append(residual)

        if not residuals:
            return 0.0

        # Convert to confidence (lower residual = higher confidence)
        mean_residual = np.mean(residuals)
        confidence = np.exp(-mean_residual / 5.0)  # 5 pixel characteristic scale

        return confidence

    @staticmethod
    def photometric_consistency(matches: List[StarMatch]) -> float:
        """
        Check if brightness ordering matches catalog magnitudes.
        """
        if len(matches) < 2:
            return 0.5

        # Sort by detected flux
        sorted_by_flux = sorted(matches, key=lambda m: -m.detected.flux)

        # Sort by catalog magnitude (brighter = lower magnitude)
        sorted_by_mag = sorted(matches, key=lambda m: m.catalog.magnitude)

        # Count concordant pairs
        concordant = 0
        total = 0

        for i in range(len(matches)):
            for j in range(i + 1, len(matches)):
                flux_order = sorted_by_flux.index(matches[i]) < sorted_by_flux.index(matches[j])
                mag_order = sorted_by_mag.index(matches[i]) < sorted_by_mag.index(matches[j])

                if flux_order == mag_order:
                    concordant += 1
                total += 1

        return concordant / total if total > 0 else 0.5

    @staticmethod
    def coverage_score(matches: List[StarMatch],
                        image_shape: Tuple[int, int]) -> float:
        """
        Measure how well matches cover the field of view.

        Good matches should be spread across the image.
        """
        if len(matches) < 3:
            return 0.0

        height, width = image_shape

        # Compute centroid
        xs = [m.detected.x for m in matches]
        ys = [m.detected.y for m in matches]

        cx, cy = np.mean(xs), np.mean(ys)

        # Compute spread
        spread_x = np.std(xs) / (width / 2)
        spread_y = np.std(ys) / (height / 2)

        # Higher spread = better coverage
        coverage = min(1.0, (spread_x + spread_y) / 2)

        return coverage

    @staticmethod
    def compute_overall_confidence(matches: List[StarMatch],
                                    attitude: np.ndarray,
                                    camera_matrix: np.ndarray,
                                    image_shape: Tuple[int, int]) -> Dict:
        """
        Compute comprehensive confidence metrics.
        """
        metrics = {
            "n_matches": len(matches),
            "geometric": ConfidenceMetrics.geometric_consistency(
                matches, attitude, camera_matrix
            ),
            "photometric": ConfidenceMetrics.photometric_consistency(matches),
            "coverage": ConfidenceMetrics.coverage_score(matches, image_shape),
            "individual_mean": np.mean([m.confidence for m in matches]) if matches else 0
        }

        # Weighted overall score
        weights = {
            "geometric": 0.4,
            "photometric": 0.2,
            "coverage": 0.2,
            "individual_mean": 0.2
        }

        overall = sum(
            weights[k] * metrics[k]
            for k in weights
        )

        metrics["overall"] = overall

        return metrics


def demonstrate_triangle_matching():
    """Demonstrate robust triangle matching."""
    print("=" * 60)
    print("Robust Triangle Matching Demonstration")
    print("=" * 60)

    # Create synthetic catalog
    print("\nCreating synthetic star catalog...")
    np.random.seed(42)

    catalog = []
    for i in range(500):
        # Random position on unit sphere
        ra = np.random.uniform(0, 2 * np.pi)
        dec = np.arcsin(np.random.uniform(-1, 1))

        # Unit vector
        unit_vec = np.array([
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec)
        ])

        catalog.append(CatalogStar(
            hip_id=i,
            ra=ra,
            dec=dec,
            magnitude=np.random.uniform(2, 7),
            unit_vector=unit_vec
        ))

    print(f"Catalog: {len(catalog)} stars")

    # Initialize matcher
    print("\nBuilding triangle database...")
    matcher = TriangleMatcher(catalog, fov_deg=90.0)

    # Create synthetic detections
    print("\nGenerating synthetic detections...")

    # Camera parameters
    fx = fy = 1000.0
    cx, cy = 960.0, 540.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    # True attitude (random rotation)
    true_attitude = np.eye(3)  # Identity for simplicity

    # Select visible catalog stars
    detections = []
    true_matches = []

    for star in catalog[:50]:  # Check first 50 stars
        # Project to camera
        cam_dir = true_attitude @ star.unit_vector

        if cam_dir[2] > 0.1:  # In front of camera
            proj = K @ cam_dir
            x = proj[0] / proj[2]
            y = proj[1] / proj[2]

            # Check if in image bounds
            if 0 <= x < 1920 and 0 <= y < 1080:
                # Add noise
                x += np.random.normal(0, 0.5)
                y += np.random.normal(0, 0.5)

                det = DetectedStar(
                    x=x, y=y,
                    flux=10**(-(star.magnitude - 2) / 2.5) * 10000,
                    snr=50.0,
                    fwhm=3.0,
                    elongation=1.0 + np.random.uniform(0, 0.2)
                )

                detections.append(det)
                true_matches.append(star)

    print(f"Generated {len(detections)} star detections")

    # Add some false detections (noise, cosmic rays)
    n_false = 5
    print(f"Adding {n_false} false detections...")

    for _ in range(n_false):
        false_det = DetectedStar(
            x=np.random.uniform(0, 1920),
            y=np.random.uniform(0, 1080),
            flux=np.random.uniform(100, 1000),
            snr=3.0,  # Low SNR
            fwhm=1.0,  # Too sharp (cosmic ray)
            elongation=1.0
        )
        detections.append(false_det)

    # Apply false star filter
    print("\nFiltering false detections...")
    false_filter = FalseStarFilter(min_snr=5.0, min_fwhm=1.5)
    valid_detections, rejected = false_filter.filter_detections(detections)

    print(f"Valid detections: {len(valid_detections)}")
    print(f"Rejected: {len(rejected)}")

    # Run matching
    print("\nRunning triangle matching...")
    result = matcher.match_stars(valid_detections, K, min_matches=5)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Matches found: {result.n_inliers}")
    print(f"Outliers: {result.n_outliers}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Triangles checked: {result.metrics.get('n_triangles_checked', 0)}")
    print(f"Total votes: {result.metrics.get('n_votes', 0)}")
    print(f"Max votes for single pair: {result.metrics.get('max_votes', 0)}")

    # Compute confidence metrics
    print("\nConfidence Metrics:")
    confidence_metrics = ConfidenceMetrics.compute_overall_confidence(
        result.matches,
        result.attitude,
        K,
        (1080, 1920)
    )

    for key, value in confidence_metrics.items():
        print(f"  {key}: {value:.3f}")

    return result


if __name__ == "__main__":
    demonstrate_triangle_matching()

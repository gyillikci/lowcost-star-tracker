"""
Frame quality assessment for astrophotography.

This module evaluates frame quality based on various metrics
to filter out unsuitable frames before stacking.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .star_detector import StarField


@dataclass
class QualityScore:
    """Quality assessment result for a single frame."""
    
    frame_index: int
    overall_score: float  # Combined score [0, 1]
    star_count: int
    median_fwhm: float
    background_mean: float
    background_std: float
    background_uniformity: float
    is_acceptable: bool
    rejection_reason: Optional[str] = None


class QualityAssessor:
    """
    Assess frame quality for stacking suitability.
    """
    
    def __init__(
        self,
        min_stars: int = 10,
        max_fwhm_pixels: float = 8.0,
        max_background_std: float = 50.0,
        reject_fraction: float = 0.2,
        score_weights: Optional[dict] = None,
    ):
        self.min_stars = min_stars
        self.max_fwhm = max_fwhm_pixels
        self.max_background_std = max_background_std
        self.reject_fraction = reject_fraction
        
        self.score_weights = score_weights or {
            "star_count": 0.3,
            "fwhm": 0.4,
            "background_uniformity": 0.3,
        }
    
    def assess_frame(
        self,
        star_field: StarField,
        frame_index: int,
    ) -> QualityScore:
        """
        Assess quality of a single frame.
        
        Args:
            star_field: Detected stars and background statistics
            frame_index: Index of the frame
            
        Returns:
            QualityScore with metrics and pass/fail status
        """
        # Check hard limits
        rejection_reason = None
        
        if star_field.num_stars < self.min_stars:
            rejection_reason = f"Too few stars ({star_field.num_stars} < {self.min_stars})"
        elif star_field.median_fwhm > self.max_fwhm:
            rejection_reason = f"FWHM too large ({star_field.median_fwhm:.1f} > {self.max_fwhm})"
        elif star_field.background_std > self.max_background_std:
            rejection_reason = f"Background too noisy ({star_field.background_std:.1f})"
        
        # Compute component scores
        star_score = self._score_star_count(star_field.num_stars)
        fwhm_score = self._score_fwhm(star_field.median_fwhm)
        bg_score = self._score_background(star_field.background_std)
        
        # Compute overall score
        overall = (
            self.score_weights["star_count"] * star_score +
            self.score_weights["fwhm"] * fwhm_score +
            self.score_weights["background_uniformity"] * bg_score
        )
        
        return QualityScore(
            frame_index=frame_index,
            overall_score=overall,
            star_count=star_field.num_stars,
            median_fwhm=star_field.median_fwhm,
            background_mean=star_field.background_mean,
            background_std=star_field.background_std,
            background_uniformity=bg_score,
            is_acceptable=rejection_reason is None,
            rejection_reason=rejection_reason,
        )
    
    def filter_frames(
        self,
        quality_scores: list[QualityScore],
    ) -> list[int]:
        """
        Filter frames based on quality scores.
        
        Args:
            quality_scores: List of quality scores for all frames
            
        Returns:
            List of frame indices that passed quality filtering
        """
        # First, filter by hard limits
        acceptable = [q for q in quality_scores if q.is_acceptable]
        
        if not acceptable:
            return []
        
        # Then, filter by relative quality (reject worst N%)
        n_to_keep = max(1, int(len(acceptable) * (1 - self.reject_fraction)))
        
        # Sort by overall score (descending)
        sorted_scores = sorted(acceptable, key=lambda q: q.overall_score, reverse=True)
        
        # Return indices of best frames
        return [q.frame_index for q in sorted_scores[:n_to_keep]]
    
    def _score_star_count(self, count: int) -> float:
        """Score based on number of detected stars."""
        # Linear ramp from min_stars to 5x min_stars
        if count < self.min_stars:
            return 0.0
        normalized = (count - self.min_stars) / (4 * self.min_stars)
        return min(1.0, normalized)
    
    def _score_fwhm(self, fwhm: float) -> float:
        """Score based on median FWHM (lower is better)."""
        if fwhm <= 0:
            return 0.0
        if fwhm >= self.max_fwhm:
            return 0.0
        
        # Ideal FWHM around 2-3 pixels
        ideal_fwhm = 2.5
        
        if fwhm <= ideal_fwhm:
            return 1.0
        else:
            # Linear decay from ideal to max
            return 1.0 - (fwhm - ideal_fwhm) / (self.max_fwhm - ideal_fwhm)
    
    def _score_background(self, std: float) -> float:
        """Score based on background uniformity (lower std is better)."""
        if std <= 0:
            return 1.0
        if std >= self.max_background_std:
            return 0.0
        
        # Exponential decay
        return np.exp(-std / (self.max_background_std / 3))
    
    def generate_report(
        self,
        quality_scores: list[QualityScore],
    ) -> str:
        """
        Generate a text report of quality assessment results.
        """
        lines = ["=" * 60]
        lines.append("FRAME QUALITY ASSESSMENT REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Summary statistics
        total = len(quality_scores)
        acceptable = sum(1 for q in quality_scores if q.is_acceptable)
        rejected = total - acceptable
        
        lines.append(f"Total frames analyzed: {total}")
        lines.append(f"Acceptable frames: {acceptable} ({100*acceptable/total:.1f}%)")
        lines.append(f"Rejected frames: {rejected} ({100*rejected/total:.1f}%)")
        lines.append("")
        
        if acceptable > 0:
            scores = [q.overall_score for q in quality_scores if q.is_acceptable]
            lines.append(f"Quality score range: {min(scores):.3f} - {max(scores):.3f}")
            lines.append(f"Mean quality score: {np.mean(scores):.3f}")
            lines.append("")
            
            fwhms = [q.median_fwhm for q in quality_scores if q.is_acceptable]
            lines.append(f"FWHM range: {min(fwhms):.2f} - {max(fwhms):.2f} pixels")
            lines.append(f"Median FWHM: {np.median(fwhms):.2f} pixels")
            lines.append("")
            
            stars = [q.star_count for q in quality_scores if q.is_acceptable]
            lines.append(f"Star count range: {min(stars)} - {max(stars)}")
            lines.append(f"Median star count: {int(np.median(stars))}")
        
        lines.append("")
        
        # Rejection reasons
        if rejected > 0:
            lines.append("Rejection reasons:")
            reasons = {}
            for q in quality_scores:
                if not q.is_acceptable and q.rejection_reason:
                    reason_type = q.rejection_reason.split("(")[0].strip()
                    reasons[reason_type] = reasons.get(reason_type, 0) + 1
            
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                lines.append(f"  {reason}: {count} frames")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)

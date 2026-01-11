"""
Main processing pipeline for the star tracker.

This module orchestrates the complete workflow from video input
to stacked star field output.
"""

from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass
import logging
import time

import numpy as np
import cv2
from tqdm import tqdm

from .config import Config
from .gyro_extractor import GyroExtractor, GyroData
from .motion_compensator import MotionCompensator, CameraIntrinsics
from .frame_extractor import FrameExtractor
from .star_detector import StarDetector, StarField
from .quality_assessor import QualityAssessor, QualityScore
from .frame_aligner import FrameAligner
from .stacker import Stacker


logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of the complete processing pipeline."""
    
    stacked_image: np.ndarray
    stacked_image_path: Optional[Path]
    num_input_frames: int
    num_stacked_frames: int
    processing_time_seconds: float
    quality_report: str
    
    @property
    def success(self) -> bool:
        return self.num_stacked_frames > 0


class Pipeline:
    """
    Complete processing pipeline for low-cost star tracker.
    
    Orchestrates all processing stages from video input to stacked output.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._setup_components()
    
    def _setup_components(self) -> None:
        """Initialize all processing components."""
        self.gyro_extractor = GyroExtractor(
            sample_rate_hz=self.config.gyro.sample_rate_hz,
            bias_estimation=self.config.gyro.bias_estimation,
            bias_window_seconds=self.config.gyro.bias_window_seconds,
            low_pass_cutoff_hz=self.config.gyro.low_pass_cutoff_hz,
            integration_method=self.config.gyro.integration_method,
        )
        
        self.frame_extractor = FrameExtractor()
        
        self.star_detector = StarDetector(
            detection_threshold_sigma=self.config.star_detection.detection_threshold_sigma,
            min_area_pixels=self.config.star_detection.min_area_pixels,
            max_area_pixels=self.config.star_detection.max_area_pixels,
            max_ellipticity=self.config.star_detection.max_ellipticity,
        )
        
        self.quality_assessor = QualityAssessor(
            min_stars=self.config.quality.min_stars,
            max_fwhm_pixels=self.config.quality.max_fwhm_pixels,
            max_background_std=self.config.quality.max_background_std,
            reject_fraction=self.config.quality.reject_fraction,
            score_weights=self.config.quality.score_weights,
        )
        
        self.frame_aligner = FrameAligner(
            reference_frame=self.config.alignment.reference_frame,
            min_match_stars=self.config.alignment.min_match_stars,
            max_alignment_error_pixels=self.config.alignment.max_alignment_error_pixels,
            transform_type=self.config.alignment.transform_type,
            ransac_threshold=self.config.alignment.ransac_threshold,
        )
        
        self.stacker = Stacker(
            method=self.config.stacking.method,
            sigma_low=self.config.stacking.sigma_low,
            sigma_high=self.config.stacking.sigma_high,
            max_iterations=self.config.stacking.max_iterations,
            output_bit_depth=self.config.stacking.output_bit_depth,
            normalize=self.config.stacking.normalize,
        )
    
    def process(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> PipelineResult:
        """
        Process a video file through the complete pipeline.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path for output image
            progress_callback: Optional callback for progress updates
            
        Returns:
            PipelineResult with stacked image and metadata
        """
        start_time = time.time()
        
        if output_path is None:
            output_path = self.config.output.output_dir / f"{video_path.stem}_stacked.tiff"
        
        self.config.output.output_dir.mkdir(parents=True, exist_ok=True)
        
        def report(stage: str, progress: float) -> None:
            if progress_callback:
                progress_callback(stage, progress)
            if self.config.verbose:
                logger.info(f"{stage}: {progress*100:.1f}%")
        
        # Stage 1: Extract gyroscope data
        report("Extracting gyroscope data", 0.0)
        gyro_data = self.gyro_extractor.extract(video_path)
        report("Extracting gyroscope data", 1.0)
        
        # Stage 2: Get video info
        video_info = self.frame_extractor.get_video_info(video_path)
        num_frames = video_info["total_frames"]
        
        # Stage 3: Setup motion compensator
        camera_intrinsics = CameraIntrinsics.from_gopro_hero7(
            (video_info["width"], video_info["height"])
        )
        
        motion_compensator = MotionCompensator(
            camera_intrinsics=camera_intrinsics,
            target_orientation=self.config.motion_compensation.target_orientation,
            interpolation=self.config.motion_compensation.interpolation,
            crop_black_borders=self.config.motion_compensation.crop_black_borders,
            crop_margin_percent=self.config.motion_compensation.crop_margin_percent,
        )
        
        target_orientation = motion_compensator.compute_target_orientation(gyro_data)
        
        # Stage 4: Process frames
        report("Processing frames", 0.0)
        
        frames = []
        star_fields = []
        quality_scores = []
        
        for idx, timestamp, frame in tqdm(
            self.frame_extractor.iterate_frames(video_path),
            total=num_frames,
            desc="Processing",
            disable=not self.config.verbose,
        ):
            # Get frame orientation
            frame_orientation = self._get_orientation_at_time(gyro_data, timestamp)
            
            # Apply motion compensation
            stabilized = motion_compensator.compensate_frame(
                frame, frame_orientation, target_orientation
            )
            
            # Detect stars
            star_field = self.star_detector.detect(stabilized)
            
            # Assess quality
            quality = self.quality_assessor.assess_frame(star_field, idx)
            
            frames.append(stabilized)
            star_fields.append(star_field)
            quality_scores.append(quality)
            
            progress = (idx + 1) / num_frames
            report("Processing frames", progress)
        
        # Stage 5: Filter frames by quality
        report("Filtering frames", 0.0)
        accepted_indices = self.quality_assessor.filter_frames(quality_scores)
        
        if not accepted_indices:
            logger.warning("No frames passed quality filtering!")
            return PipelineResult(
                stacked_image=np.zeros((video_info["height"], video_info["width"], 3), dtype=np.uint8),
                stacked_image_path=None,
                num_input_frames=num_frames,
                num_stacked_frames=0,
                processing_time_seconds=time.time() - start_time,
                quality_report=self.quality_assessor.generate_report(quality_scores),
            )
        
        logger.info(f"Accepted {len(accepted_indices)}/{num_frames} frames")
        report("Filtering frames", 1.0)
        
        # Stage 6: Align frames
        report("Aligning frames", 0.0)
        
        # Select reference frame
        filtered_star_fields = [star_fields[i] for i in accepted_indices]
        ref_idx_local = self.frame_aligner.select_reference_frame(filtered_star_fields)
        ref_idx = accepted_indices[ref_idx_local]
        ref_star_field = star_fields[ref_idx]
        
        aligned_frames = []
        for i, idx in enumerate(tqdm(
            accepted_indices,
            desc="Aligning",
            disable=not self.config.verbose,
        )):
            if idx == ref_idx:
                aligned_frames.append(frames[idx])
            else:
                transform = self.frame_aligner.compute_transform(
                    star_fields[idx], ref_star_field
                )
                
                if transform is not None:
                    aligned = self.frame_aligner.apply_transform(frames[idx], transform)
                    aligned_frames.append(aligned)
                else:
                    # Fallback: use unaligned frame
                    aligned_frames.append(frames[idx])
            
            progress = (i + 1) / len(accepted_indices)
            report("Aligning frames", progress)
        
        # Stage 7: Stack frames
        report("Stacking frames", 0.0)
        
        # Get quality weights
        weights = [quality_scores[idx].overall_score for idx in accepted_indices]
        
        stacked = self.stacker.stack(aligned_frames, weights)
        report("Stacking frames", 1.0)
        
        # Stage 8: Save result
        report("Saving result", 0.0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() in (".tif", ".tiff"):
            cv2.imwrite(str(output_path), stacked)
        elif output_path.suffix.lower() == ".png":
            cv2.imwrite(str(output_path), stacked)
        elif output_path.suffix.lower() in (".jpg", ".jpeg"):
            # Convert to 8-bit for JPEG
            if stacked.dtype != np.uint8:
                stacked_8bit = (stacked / stacked.max() * 255).astype(np.uint8)
            else:
                stacked_8bit = stacked
            cv2.imwrite(str(output_path), stacked_8bit, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(str(output_path), stacked)
        
        report("Saving result", 1.0)
        
        # Generate quality report
        quality_report = self.quality_assessor.generate_report(quality_scores)
        
        processing_time = time.time() - start_time
        logger.info(f"Processing complete in {processing_time:.1f}s")
        
        return PipelineResult(
            stacked_image=stacked,
            stacked_image_path=output_path,
            num_input_frames=num_frames,
            num_stacked_frames=len(aligned_frames),
            processing_time_seconds=processing_time,
            quality_report=quality_report,
        )
    
    def _get_orientation_at_time(
        self, 
        gyro_data: GyroData, 
        timestamp: float
    ) -> np.ndarray:
        """Get interpolated orientation at a specific timestamp."""
        idx = np.searchsorted(gyro_data.timestamps, timestamp)
        
        if idx == 0:
            return gyro_data.orientations[0]
        if idx >= len(gyro_data.timestamps):
            return gyro_data.orientations[-1]
        
        # Linear interpolation
        t0 = gyro_data.timestamps[idx - 1]
        t1 = gyro_data.timestamps[idx]
        alpha = (timestamp - t0) / (t1 - t0)
        
        q0 = gyro_data.orientations[idx - 1]
        q1 = gyro_data.orientations[idx]
        
        q = (1 - alpha) * q0 + alpha * q1
        return q / np.linalg.norm(q)


class CameraCalibration:
    """
    Camera calibration and lens profile management.
    
    Placeholder for future lens calibration features.
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def load_profile(profile_path: Path) -> CameraIntrinsics:
        """Load camera profile from JSON file."""
        import json
        
        with open(profile_path) as f:
            data = json.load(f)
        
        return CameraIntrinsics(
            focal_length_x=data["focal_length_x"],
            focal_length_y=data["focal_length_y"],
            principal_point_x=data["principal_point_x"],
            principal_point_y=data["principal_point_y"],
            distortion_coeffs=np.array(data.get("distortion_coeffs", [])) if data.get("distortion_coeffs") else None,
        )

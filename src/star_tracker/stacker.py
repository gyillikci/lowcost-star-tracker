"""
Image stacking algorithms for astrophotography.

This module implements various stacking methods including mean, median,
sigma-clipping, and winsorized mean for combining aligned frames.
"""

from typing import Literal, Optional, Iterator
from pathlib import Path

import numpy as np
import cv2


class Stacker:
    """
    Stack aligned frames using various statistical methods.
    """
    
    def __init__(
        self,
        method: Literal["mean", "median", "sigma_clip", "winsorized"] = "sigma_clip",
        sigma_low: float = 3.0,
        sigma_high: float = 3.0,
        max_iterations: int = 5,
        output_bit_depth: Literal[8, 16, 32] = 16,
        normalize: bool = True,
    ):
        self.method = method
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.max_iterations = max_iterations
        self.output_bit_depth = output_bit_depth
        self.normalize = normalize
    
    def stack(
        self,
        frames: list[np.ndarray],
        weights: Optional[list[float]] = None,
    ) -> np.ndarray:
        """
        Stack multiple frames into a single image.
        
        Args:
            frames: List of aligned frames (same shape)
            weights: Optional weights for each frame
            
        Returns:
            Stacked image
        """
        if not frames:
            raise ValueError("No frames to stack")
        
        # Stack into 3D array
        stack = np.stack(frames, axis=0).astype(np.float32)
        
        # Apply stacking method
        if self.method == "mean":
            result = self._mean_stack(stack, weights)
        elif self.method == "median":
            result = self._median_stack(stack)
        elif self.method == "sigma_clip":
            result = self._sigma_clip_stack(stack, weights)
        elif self.method == "winsorized":
            result = self._winsorized_stack(stack)
        else:
            raise ValueError(f"Unknown stacking method: {self.method}")
        
        # Normalize and convert to output bit depth
        result = self._convert_output(result)
        
        return result
    
    def stack_from_files(
        self,
        frame_paths: list[Path],
        weights: Optional[list[float]] = None,
        batch_size: int = 100,
    ) -> np.ndarray:
        """
        Stack frames from files, processing in batches to manage memory.
        
        Args:
            frame_paths: Paths to frame images
            weights: Optional weights for each frame
            batch_size: Number of frames to load at once
            
        Returns:
            Stacked image
        """
        if not frame_paths:
            raise ValueError("No frames to stack")
        
        # For small number of frames, load all at once
        if len(frame_paths) <= batch_size:
            frames = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in frame_paths]
            return self.stack(frames, weights)
        
        # For large number of frames, use running statistics
        return self._streaming_stack(frame_paths, weights, batch_size)
    
    def _mean_stack(
        self, 
        stack: np.ndarray, 
        weights: Optional[list[float]] = None
    ) -> np.ndarray:
        """Simple mean stacking."""
        if weights is None:
            return np.mean(stack, axis=0)
        else:
            weights = np.array(weights).reshape(-1, 1, 1, 1) if stack.ndim == 4 else np.array(weights).reshape(-1, 1, 1)
            weights = weights / weights.sum()
            return np.sum(stack * weights, axis=0)
    
    def _median_stack(self, stack: np.ndarray) -> np.ndarray:
        """Median stacking for outlier rejection."""
        return np.median(stack, axis=0)
    
    def _sigma_clip_stack(
        self, 
        stack: np.ndarray,
        weights: Optional[list[float]] = None,
    ) -> np.ndarray:
        """
        Sigma-clipping stack with iterative outlier rejection.
        """
        n_frames = stack.shape[0]
        
        # Create mask for valid pixels (all True initially)
        mask = np.ones(stack.shape, dtype=bool)
        
        for iteration in range(self.max_iterations):
            # Compute masked statistics
            masked_stack = np.ma.array(stack, mask=~mask)
            mean = np.ma.mean(masked_stack, axis=0).data
            std = np.ma.std(masked_stack, axis=0).data
            
            # Compute new mask
            deviation = stack - mean
            new_mask = (
                (deviation >= -self.sigma_low * std) &
                (deviation <= self.sigma_high * std)
            )
            
            # Check for convergence
            if np.array_equal(mask, new_mask):
                break
            
            mask = new_mask
        
        # Compute final mean with mask
        masked_stack = np.ma.array(stack, mask=~mask)
        
        if weights is not None:
            # Weight by frame quality
            weight_array = np.array(weights)
            weight_array = weight_array.reshape(-1, *([1] * (stack.ndim - 1)))
            weighted = masked_stack * weight_array
            result = np.ma.sum(weighted, axis=0) / np.ma.sum(weight_array * mask, axis=0)
        else:
            result = np.ma.mean(masked_stack, axis=0)
        
        return result.data
    
    def _winsorized_stack(self, stack: np.ndarray) -> np.ndarray:
        """
        Winsorized mean - clips extreme values to percentiles.
        """
        low_percentile = self.sigma_low * 10  # Approximate mapping
        high_percentile = 100 - self.sigma_high * 10
        
        low = np.percentile(stack, low_percentile, axis=0)
        high = np.percentile(stack, high_percentile, axis=0)
        
        clipped = np.clip(stack, low, high)
        return np.mean(clipped, axis=0)
    
    def _streaming_stack(
        self,
        frame_paths: list[Path],
        weights: Optional[list[float]] = None,
        batch_size: int = 100,
    ) -> np.ndarray:
        """
        Memory-efficient streaming stack using Welford's algorithm.
        
        For very large numbers of frames, uses running mean and variance.
        """
        # Read first frame to get shape
        first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_UNCHANGED)
        shape = first_frame.shape
        dtype = np.float64
        
        # Initialize running statistics
        count = np.zeros(shape, dtype=dtype)
        mean = np.zeros(shape, dtype=dtype)
        M2 = np.zeros(shape, dtype=dtype)  # Sum of squared differences
        
        for i, path in enumerate(frame_paths):
            frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(dtype)
            
            weight = weights[i] if weights else 1.0
            
            # Welford's online algorithm
            count += weight
            delta = frame - mean
            mean += delta * weight / count
            delta2 = frame - mean
            M2 += weight * delta * delta2
        
        # For streaming, we just use the mean (no outlier rejection)
        # Full sigma clipping would require multiple passes
        return self._convert_output(mean.astype(np.float32))
    
    def _convert_output(self, image: np.ndarray) -> np.ndarray:
        """Convert to output bit depth with optional normalization."""
        if self.normalize:
            # Normalize to [0, 1] range
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
            else:
                image = np.zeros_like(image)
        
        if self.output_bit_depth == 8:
            return (image * 255).clip(0, 255).astype(np.uint8)
        elif self.output_bit_depth == 16:
            return (image * 65535).clip(0, 65535).astype(np.uint16)
        else:  # 32-bit float
            return image.astype(np.float32)


class CalibrationStacker:
    """
    Create master calibration frames (darks, flats, bias).
    """
    
    def __init__(self, method: Literal["mean", "median"] = "median"):
        self.method = method
    
    def create_master_dark(
        self,
        dark_frames: list[np.ndarray],
    ) -> np.ndarray:
        """Create master dark frame from multiple dark exposures."""
        stack = np.stack(dark_frames, axis=0).astype(np.float32)
        
        if self.method == "median":
            return np.median(stack, axis=0)
        else:
            return np.mean(stack, axis=0)
    
    def create_master_flat(
        self,
        flat_frames: list[np.ndarray],
        master_dark: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Create normalized master flat field."""
        stack = np.stack(flat_frames, axis=0).astype(np.float32)
        
        # Subtract dark if provided
        if master_dark is not None:
            stack = stack - master_dark
        
        # Stack
        if self.method == "median":
            master = np.median(stack, axis=0)
        else:
            master = np.mean(stack, axis=0)
        
        # Normalize by mean
        mean_value = np.mean(master)
        if mean_value > 0:
            master = master / mean_value
        
        return master
    
    def apply_calibration(
        self,
        frame: np.ndarray,
        master_dark: Optional[np.ndarray] = None,
        master_flat: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply calibration frames to a light frame."""
        result = frame.astype(np.float32)
        
        # Subtract dark
        if master_dark is not None:
            result = result - master_dark
        
        # Divide by flat
        if master_flat is not None:
            # Avoid division by zero
            safe_flat = np.where(master_flat > 0.1, master_flat, 1.0)
            result = result / safe_flat
        
        return result

"""
Step 1: OpenCV-based Video Stabilization (V3 - Fixed Reference Frame)

All frames are aligned to the first frame for maximum stabilization.
This completely removes all camera motion relative to frame 0.
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm
import imageio


def enhance_stars(gray: np.ndarray) -> np.ndarray:
    """Enhance star visibility with high-pass filter."""
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    enhanced = cv2.subtract(gray, blurred * 0.8)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced


def stabilize_to_reference(video_path: str, output_path: str):
    """
    Stabilize all frames to the first frame (fixed reference).
    Maximum stabilization - removes ALL camera motion.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Processing scale
    proc_scale = 0.25
    proc_width = int(width * proc_scale)
    proc_height = int(height * proc_scale)
    
    # Read reference frame (frame 0)
    ret, ref_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read video")
    
    ref_small = cv2.resize(ref_frame, (proc_width, proc_height))
    ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref_enhanced = enhance_stars(ref_gray)
    
    # Hanning window for phase correlation
    hann = cv2.createHanningWindow((proc_width, proc_height), cv2.CV_32F)
    
    # Store all frames and transforms
    frames = [ref_frame]
    cumulative_transforms = [(0.0, 0.0, 0.0)]  # First frame has no transform
    
    # Cumulative motion tracking
    cum_dx, cum_dy, cum_da = 0.0, 0.0, 0.0
    
    prev_gray = ref_enhanced
    
    print("Extracting cumulative motion (all frames → frame 0)...")
    for i in tqdm(range(frame_count - 1), desc="Analyzing"):
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        
        # Get current frame
        curr_small = cv2.resize(frame, (proc_width, proc_height))
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        curr_enhanced = enhance_stars(curr_gray)
        
        # Phase correlation: current vs previous
        shift, response = cv2.phaseCorrelate(prev_gray, curr_enhanced, hann)
        dx, dy = shift[0] / proc_scale, shift[1] / proc_scale
        
        # Estimate rotation using ECC
        da = 0.0
        if response > 0.05:
            try:
                h, w = prev_gray.shape
                M_trans = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
                curr_aligned = cv2.warpAffine(curr_enhanced, M_trans, (w, h))
                
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
                
                margin = int(min(h, w) * 0.2)
                prev_crop = prev_gray[margin:-margin, margin:-margin]
                curr_crop = curr_aligned[margin:-margin, margin:-margin]
                
                prev_norm = cv2.normalize(prev_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                curr_norm = cv2.normalize(curr_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                _, warp_matrix = cv2.findTransformECC(
                    prev_norm, curr_norm, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
                )
                da = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])
            except cv2.error:
                da = 0.0
        
        # Accumulate motion
        cum_dx += dx
        cum_dy += dy
        cum_da += da
        
        # Store cumulative transform (how much this frame has moved from frame 0)
        cumulative_transforms.append((cum_dx, cum_dy, cum_da))
        
        prev_gray = curr_enhanced
    
    cap.release()
    
    # Print statistics
    print(f"\n=== TOTAL MOTION FROM FRAME 0 ===")
    print(f"Final translation: ({cum_dx:.1f}, {cum_dy:.1f}) px")
    print(f"Final rotation: {np.degrees(cum_da):.2f} deg")
    print(f"Max translation: ({max(abs(t[0]) for t in cumulative_transforms):.1f}, {max(abs(t[1]) for t in cumulative_transforms):.1f}) px")
    print(f"Max rotation: {np.degrees(max(abs(t[2]) for t in cumulative_transforms)):.2f} deg")
    
    # Apply stabilization - reverse the cumulative motion
    print("\nApplying fixed-reference stabilization...")
    
    # Output scale
    out_scale = 0.5
    out_width = int(width * out_scale)
    out_height = int(height * out_scale)
    
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=7,
                                 macro_block_size=1)
    
    for i, (frame, (cdx, cdy, cda)) in enumerate(tqdm(zip(frames, cumulative_transforms), 
                                                        total=len(frames), desc="Stabilizing")):
        # Correction = negative of cumulative motion (bring back to frame 0 position)
        corr_dx = -cdx
        corr_dy = -cdy
        corr_da = -cda
        
        # Build transform matrix (rotate around center)
        cx, cy = width / 2, height / 2
        cos_a = np.cos(corr_da)
        sin_a = np.sin(corr_da)
        
        M = np.array([
            [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy + corr_dx],
            [sin_a,  cos_a, -sin_a * cx + (1 - cos_a) * cy + corr_dy]
        ], dtype=np.float32)
        
        stabilized = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        
        # Resize for output
        orig_small = cv2.resize(frame, (out_width, out_height))
        stab_small = cv2.resize(stabilized, (out_width, out_height))
        
        # Add labels
        cv2.putText(orig_small, "ORIGINAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(stab_small, "FIXED REF", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show correction being applied
        info = f"Corr: dx={corr_dx:.0f} dy={corr_dy:.0f} da={np.degrees(corr_da):.2f}deg"
        cv2.putText(stab_small, info, (20, out_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine
        combined = np.hstack([orig_small, stab_small])
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        writer.append_data(combined_rgb)
    
    writer.close()
    print(f"\nSaved: {output_path}")
    
    return cumulative_transforms


def main():
    if len(sys.argv) < 2:
        print("Usage: python step1_opencv_stabilization_v3.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    output_dir = Path("experiments/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_name = Path(video_path).stem
    output_path = str(output_dir / f"{video_name}_fixed_ref.mp4")
    
    print(f"Processing: {video_path}")
    print("Mode: FIXED REFERENCE (all frames → frame 0)")
    print("=" * 50)
    
    transforms = stabilize_to_reference(video_path, output_path)
    
    # Save data
    np.save(str(output_dir / f"{video_name}_cumulative_transforms.npy"), transforms)
    
    os.startfile(output_path)


if __name__ == "__main__":
    main()

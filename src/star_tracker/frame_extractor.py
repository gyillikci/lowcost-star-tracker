"""
Video frame extraction utilities.

This module handles extraction of individual frames from video files
using FFmpeg for efficient decoding.
"""

from pathlib import Path
from typing import Iterator, Optional, Tuple
import subprocess
import tempfile
import shutil

import numpy as np
import cv2


class FrameExtractor:
    """
    Extract frames from video files.
    """
    
    def __init__(
        self,
        output_format: str = "png",
        quality: int = 95,
    ):
        self.output_format = output_format
        self.quality = quality
    
    def get_video_info(self, video_path: Path) -> dict:
        """
        Get video metadata using FFprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video properties
        """
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            import json
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break
            
            if video_stream is None:
                raise ValueError("No video stream found")
            
            # Parse frame rate
            fps_str = video_stream.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = map(int, fps_str.split("/"))
                fps = num / den
            else:
                fps = float(fps_str)
            
            return {
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "fps": fps,
                "duration": float(data.get("format", {}).get("duration", 0)),
                "codec": video_stream.get("codec_name", "unknown"),
                "total_frames": int(video_stream.get("nb_frames", 0)) or int(fps * float(data.get("format", {}).get("duration", 0))),
            }
        except Exception as e:
            # Fallback to OpenCV
            cap = cv2.VideoCapture(str(video_path))
            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                "codec": "unknown",
            }
            cap.release()
            return info
    
    def extract_all_frames(
        self,
        video_path: Path,
        output_dir: Path,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        frame_interval: int = 1,
    ) -> list[Path]:
        """
        Extract all frames from video to disk.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            frame_interval: Extract every Nth frame
            
        Returns:
            List of paths to extracted frames
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build FFmpeg command
        cmd = ["ffmpeg", "-y", "-i", str(video_path)]
        
        if start_time is not None:
            cmd.extend(["-ss", str(start_time)])
        if end_time is not None:
            cmd.extend(["-t", str(end_time - (start_time or 0))])
        
        # Frame selection filter
        if frame_interval > 1:
            cmd.extend(["-vf", f"select='not(mod(n\\,{frame_interval}))'", "-vsync", "vfr"])
        
        # Output format
        output_pattern = str(output_dir / f"frame_%06d.{self.output_format}")
        
        if self.output_format == "png":
            cmd.extend(["-compression_level", "3"])
        elif self.output_format in ("jpg", "jpeg"):
            cmd.extend(["-qscale:v", str(int((100 - self.quality) / 5))])
        
        cmd.append(output_pattern)
        
        # Run FFmpeg
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Return list of extracted frames
        frames = sorted(output_dir.glob(f"frame_*.{self.output_format}"))
        return frames
    
    def iterate_frames(
        self,
        video_path: Path,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Iterator[Tuple[int, float, np.ndarray]]:
        """
        Iterate over video frames without saving to disk.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Yields:
            Tuples of (frame_index, timestamp, frame_array)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range
        start_frame = int((start_time or 0) * fps)
        end_frame = int((end_time * fps) if end_time else total_frames)
        
        # Seek to start
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_idx = start_frame
        
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            yield frame_idx, timestamp, frame
            
            frame_idx += 1
        
        cap.release()
    
    def extract_single_frame(
        self,
        video_path: Path,
        timestamp: float,
    ) -> np.ndarray:
        """
        Extract a single frame at a specific timestamp.
        
        Args:
            video_path: Path to video file
            timestamp: Time in seconds
            
        Returns:
            Frame as numpy array
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError(f"Could not read frame at {timestamp}s")
        
        return frame


class VideoWriter:
    """
    Write processed frames back to video format.
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        codec: str = "mp4v",
        quality: int = 95,
    ):
        self.fps = fps
        self.codec = codec
        self.quality = quality
    
    def write_video(
        self,
        frame_paths: list[Path],
        output_path: Path,
    ) -> None:
        """
        Combine frames into a video file.
        
        Args:
            frame_paths: List of paths to frame images
            output_path: Output video path
        """
        if not frame_paths:
            raise ValueError("No frames to write")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_paths[0]))
        height, width = first_frame.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, (width, height)
        )
        
        try:
            for frame_path in frame_paths:
                frame = cv2.imread(str(frame_path))
                writer.write(frame)
        finally:
            writer.release()
    
    def write_from_array(
        self,
        frames: list[np.ndarray],
        output_path: Path,
    ) -> None:
        """
        Write frames from numpy arrays.
        """
        if not frames:
            raise ValueError("No frames to write")
        
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, (width, height)
        )
        
        try:
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                writer.write(frame)
        finally:
            writer.release()

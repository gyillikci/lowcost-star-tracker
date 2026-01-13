#!/usr/bin/env python3
"""
Video Stream Module.

Handles video frame loading and feature extraction for
camera-IMU calibration.

Features:
- Video file loading (OpenCV)
- Feature detection (corners, stars)
- Frame-to-frame tracking
- Optical flow computation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Generator
from pathlib import Path
import cv2


@dataclass
class FrameData:
    """Container for a single video frame."""
    timestamp: float  # seconds
    image: np.ndarray  # grayscale or color image
    features: Optional[np.ndarray] = None  # detected features (N, 2)
    feature_ids: Optional[np.ndarray] = None  # feature track IDs


@dataclass
class FeatureTrack:
    """Track of a feature across multiple frames."""
    track_id: int
    positions: List[np.ndarray]  # list of (x, y) positions
    timestamps: List[float]  # corresponding timestamps
    first_frame: int
    last_frame: int


class VideoStream:
    """
    Video stream manager for calibration.

    Handles video loading, feature detection, and tracking
    for camera-IMU temporal and spatial calibration.
    """

    def __init__(self, camera_model=None):
        """
        Initialize video stream.

        Args:
            camera_model: CameraModel for undistortion (optional)
        """
        self.camera_model = camera_model

        # Video properties
        self.filepath: Optional[str] = None
        self.frame_rate: float = 0.0
        self.frame_count: int = 0
        self.width: int = 0
        self.height: int = 0
        self.duration: float = 0.0

        # Frame data
        self.frames: List[FrameData] = []
        self.timestamps: np.ndarray = np.array([])

        # Feature tracking
        self.feature_tracks: Dict[int, FeatureTrack] = {}
        self._next_track_id: int = 0

        # OpenCV capture
        self._capture: Optional[cv2.VideoCapture] = None

    def load_video(self, filepath: str,
                   start_time: float = 0.0,
                   end_time: float = None,
                   max_frames: int = None):
        """
        Load video file.

        Args:
            filepath: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds (None = end of video)
            max_frames: Maximum frames to load (None = all)
        """
        self.filepath = filepath
        self._capture = cv2.VideoCapture(filepath)

        if not self._capture.isOpened():
            raise ValueError(f"Cannot open video: {filepath}")

        # Get video properties
        self.frame_rate = self._capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.frame_rate

        # Set start position
        if start_time > 0:
            self._capture.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        # Calculate end frame
        if end_time is not None:
            end_frame = int(end_time * self.frame_rate)
        else:
            end_frame = self.frame_count

        # Load frames
        self.frames = []
        self.timestamps = []

        frame_idx = 0
        while True:
            ret, frame = self._capture.read()
            if not ret:
                break

            current_time = self._capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if current_time >= start_time:
                # Convert to grayscale
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame

                self.frames.append(FrameData(
                    timestamp=current_time,
                    image=gray
                ))
                self.timestamps.append(current_time)

                frame_idx += 1

                if max_frames and frame_idx >= max_frames:
                    break

            if current_time >= end_time if end_time else False:
                break

        self.timestamps = np.array(self.timestamps)
        self._capture.release()

        print(f"Loaded {len(self.frames)} frames from {filepath}")
        print(f"  Duration: {self.timestamps[-1] - self.timestamps[0]:.2f}s")
        print(f"  Frame rate: {self.frame_rate:.2f} fps")

    def load_frames_from_array(self,
                               images: List[np.ndarray],
                               timestamps: np.ndarray):
        """
        Load frames from numpy arrays.

        Args:
            images: List of grayscale images
            timestamps: Array of timestamps
        """
        self.frames = []
        self.timestamps = np.asarray(timestamps)

        for img, t in zip(images, timestamps):
            self.frames.append(FrameData(timestamp=t, image=img))

        if len(self.frames) > 0:
            self.height, self.width = self.frames[0].image.shape[:2]
            self.duration = self.timestamps[-1] - self.timestamps[0]
            self.frame_rate = len(self.frames) / self.duration if self.duration > 0 else 0

    def detect_features(self,
                        detector_type: str = "shi-tomasi",
                        max_features: int = 500,
                        quality_level: float = 0.01,
                        min_distance: float = 10.0):
        """
        Detect features in all frames.

        Args:
            detector_type: "shi-tomasi", "harris", "orb", or "star"
            max_features: Maximum features per frame
            quality_level: Quality threshold (0-1)
            min_distance: Minimum distance between features
        """
        for frame in self.frames:
            if detector_type == "shi-tomasi":
                corners = cv2.goodFeaturesToTrack(
                    frame.image,
                    maxCorners=max_features,
                    qualityLevel=quality_level,
                    minDistance=min_distance
                )
                if corners is not None:
                    frame.features = corners.reshape(-1, 2)
                else:
                    frame.features = np.array([]).reshape(0, 2)

            elif detector_type == "harris":
                harris = cv2.cornerHarris(
                    frame.image.astype(np.float32), 2, 3, 0.04
                )
                harris = cv2.dilate(harris, None)
                threshold = quality_level * harris.max()
                coords = np.where(harris > threshold)
                frame.features = np.column_stack([coords[1], coords[0]])[:max_features]

            elif detector_type == "orb":
                orb = cv2.ORB_create(nfeatures=max_features)
                keypoints = orb.detect(frame.image, None)
                frame.features = np.array([kp.pt for kp in keypoints])

            elif detector_type == "star":
                # Simple star detection (bright local maxima)
                frame.features = self._detect_stars(
                    frame.image, max_features, quality_level
                )

    def _detect_stars(self, image: np.ndarray,
                      max_stars: int, threshold: float) -> np.ndarray:
        """Simple star detection for astronomical images."""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Find local maxima
        kernel = np.ones((7, 7), np.float32)
        kernel[3, 3] = 0
        local_max = cv2.dilate(blurred, kernel)

        # Stars are pixels where value equals local maximum
        stars_mask = (blurred >= local_max) & (blurred > threshold * blurred.max())

        # Get coordinates
        coords = np.column_stack(np.where(stars_mask))[:, ::-1]  # xy format

        # Sort by brightness and take top N
        if len(coords) > 0:
            brightness = blurred[coords[:, 1].astype(int), coords[:, 0].astype(int)]
            sorted_idx = np.argsort(-brightness)[:max_stars]
            return coords[sorted_idx].astype(np.float32)

        return np.array([]).reshape(0, 2)

    def track_features(self, window_size: int = 21):
        """
        Track features across frames using optical flow.

        Args:
            window_size: Lucas-Kanade window size
        """
        if len(self.frames) < 2:
            return

        # Initialize tracks from first frame
        self.feature_tracks = {}
        self._next_track_id = 0

        first_frame = self.frames[0]
        if first_frame.features is None or len(first_frame.features) == 0:
            self.detect_features()

        # Initialize track IDs
        first_frame.feature_ids = np.arange(len(first_frame.features))
        for i, pt in enumerate(first_frame.features):
            self.feature_tracks[i] = FeatureTrack(
                track_id=i,
                positions=[pt.copy()],
                timestamps=[first_frame.timestamp],
                first_frame=0,
                last_frame=0
            )
        self._next_track_id = len(first_frame.features)

        # LK parameters
        lk_params = dict(
            winSize=(window_size, window_size),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Track through frames
        for i in range(1, len(self.frames)):
            prev_frame = self.frames[i - 1]
            curr_frame = self.frames[i]

            if prev_frame.features is None or len(prev_frame.features) == 0:
                curr_frame.features = np.array([]).reshape(0, 2)
                curr_frame.feature_ids = np.array([])
                continue

            # Compute optical flow
            prev_pts = prev_frame.features.reshape(-1, 1, 2).astype(np.float32)
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_frame.image, curr_frame.image,
                prev_pts, None, **lk_params
            )

            # Filter good tracks
            status = status.flatten()
            good_prev = prev_pts[status == 1].reshape(-1, 2)
            good_curr = curr_pts[status == 1].reshape(-1, 2)
            good_ids = prev_frame.feature_ids[status == 1]

            # Update tracks
            for pt, track_id in zip(good_curr, good_ids):
                if track_id in self.feature_tracks:
                    self.feature_tracks[track_id].positions.append(pt.copy())
                    self.feature_tracks[track_id].timestamps.append(curr_frame.timestamp)
                    self.feature_tracks[track_id].last_frame = i

            curr_frame.features = good_curr
            curr_frame.feature_ids = good_ids

            # Detect new features in areas without tracks
            if len(good_curr) < 100:
                mask = np.ones_like(curr_frame.image)
                for pt in good_curr:
                    cv2.circle(mask, tuple(pt.astype(int)), 10, 0, -1)

                new_pts = cv2.goodFeaturesToTrack(
                    curr_frame.image, maxCorners=200,
                    qualityLevel=0.01, minDistance=10, mask=mask
                )

                if new_pts is not None:
                    new_pts = new_pts.reshape(-1, 2)
                    new_ids = np.arange(
                        self._next_track_id,
                        self._next_track_id + len(new_pts)
                    )
                    self._next_track_id += len(new_pts)

                    # Initialize new tracks
                    for pt, track_id in zip(new_pts, new_ids):
                        self.feature_tracks[track_id] = FeatureTrack(
                            track_id=track_id,
                            positions=[pt.copy()],
                            timestamps=[curr_frame.timestamp],
                            first_frame=i,
                            last_frame=i
                        )

                    curr_frame.features = np.vstack([good_curr, new_pts])
                    curr_frame.feature_ids = np.concatenate([good_ids, new_ids])

        # Count long tracks
        long_tracks = sum(1 for t in self.feature_tracks.values()
                         if len(t.positions) >= 10)
        print(f"Tracked {len(self.feature_tracks)} features, "
              f"{long_tracks} long tracks (>10 frames)")

    def compute_frame_rotation(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Compute rotation between consecutive frames from feature tracks.

        Args:
            frame_idx: Index of the current frame

        Returns:
            Rotation vector (Rodrigues) or None if insufficient features
        """
        if frame_idx < 1 or frame_idx >= len(self.frames):
            return None

        prev_frame = self.frames[frame_idx - 1]
        curr_frame = self.frames[frame_idx]

        if (prev_frame.features is None or curr_frame.features is None or
            len(prev_frame.features) < 5 or len(curr_frame.features) < 5):
            return None

        # Find common features
        prev_ids = set(prev_frame.feature_ids)
        curr_ids = set(curr_frame.feature_ids)
        common_ids = prev_ids & curr_ids

        if len(common_ids) < 5:
            return None

        # Get corresponding points
        prev_pts = []
        curr_pts = []

        for track_id in common_ids:
            prev_idx = np.where(prev_frame.feature_ids == track_id)[0][0]
            curr_idx = np.where(curr_frame.feature_ids == track_id)[0][0]
            prev_pts.append(prev_frame.features[prev_idx])
            curr_pts.append(curr_frame.features[curr_idx])

        prev_pts = np.array(prev_pts)
        curr_pts = np.array(curr_pts)

        # Compute essential matrix
        if self.camera_model is not None:
            K = self.camera_model.intrinsics.K
        else:
            # Assume simple camera
            K = np.array([
                [self.width, 0, self.width / 2],
                [0, self.width, self.height / 2],
                [0, 0, 1]
            ])

        E, mask = cv2.findEssentialMat(
            prev_pts, curr_pts, K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is None:
            return None

        # Recover rotation
        _, R, _, _ = cv2.recoverPose(E, prev_pts, curr_pts, K)

        # Convert to rotation vector
        rvec, _ = cv2.Rodrigues(R)

        return rvec.flatten()

    def get_frame_at(self, timestamp: float) -> Optional[FrameData]:
        """Get frame closest to timestamp."""
        if len(self.timestamps) == 0:
            return None

        idx = np.argmin(np.abs(self.timestamps - timestamp))
        return self.frames[idx]

    def get_angular_velocity_from_video(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate angular velocity from video feature tracking.

        Returns:
            Tuple of (timestamps, angular_velocities)
        """
        timestamps = []
        angular_velocities = []

        for i in range(1, len(self.frames)):
            rvec = self.compute_frame_rotation(i)
            if rvec is not None:
                dt = self.frames[i].timestamp - self.frames[i-1].timestamp
                if dt > 0:
                    omega = rvec / dt
                    t = (self.frames[i].timestamp + self.frames[i-1].timestamp) / 2
                    timestamps.append(t)
                    angular_velocities.append(omega)

        return np.array(timestamps), np.array(angular_velocities)

    def iterate_frames(self) -> Generator[FrameData, None, None]:
        """Iterate over frames."""
        for frame in self.frames:
            yield frame

    def __len__(self) -> int:
        """Return number of frames."""
        return len(self.frames)

    def __repr__(self) -> str:
        return (f"VideoStream(frames={len(self)}, "
                f"rate={self.frame_rate:.1f}fps, "
                f"size={self.width}x{self.height})")


def demonstrate_video_stream():
    """Demonstrate VideoStream functionality with synthetic data."""
    print("=" * 60)
    print("VideoStream Demonstration")
    print("=" * 60)

    # Generate synthetic frames with moving features
    np.random.seed(42)
    width, height = 640, 480
    n_frames = 50
    frame_rate = 30.0

    # Generate feature positions (simulating camera rotation)
    n_features = 100
    base_positions = np.random.uniform(
        [50, 50], [width - 50, height - 50], (n_features, 2)
    )

    images = []
    timestamps = []

    for i in range(n_frames):
        t = i / frame_rate
        timestamps.append(t)

        # Simulate rotation (features move)
        angle = 0.01 * t
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        center = np.array([width / 2, height / 2])
        positions = (base_positions - center) @ rotation.T + center

        # Add random noise
        positions += np.random.normal(0, 0.5, positions.shape)

        # Create image with features
        img = np.zeros((height, width), dtype=np.uint8)
        img += 20  # Background

        for pos in positions:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(img, (x, y), 3, 255, -1)

        # Add noise
        img = img.astype(np.float32)
        img += np.random.normal(0, 5, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)

        images.append(img)

    # Create video stream
    stream = VideoStream()
    stream.load_frames_from_array(images, np.array(timestamps))

    print(f"\n{stream}")

    # Detect features
    print("\nDetecting features...")
    stream.detect_features(max_features=200)

    n_detected = sum(len(f.features) for f in stream.frames if f.features is not None)
    print(f"  Total features detected: {n_detected}")

    # Track features
    print("\nTracking features...")
    stream.track_features()

    # Compute angular velocity from video
    print("\nComputing angular velocity from video...")
    t_video, omega_video = stream.get_angular_velocity_from_video()

    if len(omega_video) > 0:
        print(f"  Computed {len(omega_video)} angular velocity samples")
        print(f"  Mean omega: {np.mean(omega_video, axis=0)} rad/s")

    return stream


if __name__ == "__main__":
    demonstrate_video_stream()

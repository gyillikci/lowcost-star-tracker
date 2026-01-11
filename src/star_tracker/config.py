"""
Configuration management for the star tracker pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import yaml


@dataclass
class CameraConfig:
    """Camera-specific configuration."""
    
    model: str = "gopro_hero7_black"
    sensor_width: int = 4000
    sensor_height: int = 3000
    pixel_size_um: float = 1.55
    lens_profile: Optional[str] = None
    
    # Intrinsic parameters (will be loaded from lens profile if not specified)
    focal_length_px: Optional[float] = None
    principal_point: Optional[tuple[float, float]] = None
    distortion_coeffs: Optional[list[float]] = None


@dataclass
class GyroConfig:
    """Gyroscope processing configuration."""
    
    sample_rate_hz: float = 200.0
    bias_estimation: bool = True
    bias_window_seconds: float = 2.0
    low_pass_cutoff_hz: float = 50.0
    integration_method: Literal["euler", "rk4", "vqf"] = "vqf"
    
    # VQF sensor fusion options
    use_accelerometer: bool = True
    vqf_tau_acc: float = 3.0  # Accelerometer time constant
    vqf_rest_detection: bool = True
    vqf_rest_threshold_gyr: float = 2.0  # deg/s
    vqf_rest_threshold_acc: float = 0.5  # m/s²
    
    # Velocity-adaptive smoothing
    velocity_adaptive_smoothing: bool = True
    smoothing_time_constant: float = 0.5  # seconds


@dataclass
class MotionCompensationConfig:
    """Motion compensation configuration."""
    
    target_orientation: Literal["mean", "median", "first", "custom"] = "mean"
    interpolation: Literal["nearest", "linear", "cubic"] = "cubic"
    crop_black_borders: bool = True
    crop_margin_percent: float = 5.0
    
    # Rolling shutter correction
    rolling_shutter_correction: bool = True
    frame_readout_time_ms: float = 8.3  # 4K60 GoPro default
    
    # Lens profile
    lens_profile_path: Optional[str] = None  # Path to lens profile JSON
    use_smoothed_orientations: bool = True


@dataclass
class StarDetectionConfig:
    """Star detection configuration."""
    
    detection_threshold_sigma: float = 3.0
    min_area_pixels: int = 3
    max_area_pixels: int = 500
    max_ellipticity: float = 0.5
    deblend: bool = True
    deblend_threshold: float = 0.005


@dataclass
class QualityConfig:
    """Frame quality assessment configuration."""
    
    min_stars: int = 10
    max_fwhm_pixels: float = 8.0
    max_background_std: float = 50.0
    reject_fraction: float = 0.2
    score_weights: dict = field(default_factory=lambda: {
        "star_count": 0.3,
        "fwhm": 0.4,
        "background_uniformity": 0.3,
    })


@dataclass
class AlignmentConfig:
    """Frame alignment configuration."""
    
    reference_frame: Literal["best", "first", "median"] = "best"
    min_match_stars: int = 5
    max_alignment_error_pixels: float = 1.0
    transform_type: Literal["translation", "rigid", "affine", "homography"] = "affine"
    ransac_threshold: float = 3.0


@dataclass 
class StackingConfig:
    """Image stacking configuration."""
    
    method: Literal["mean", "median", "sigma_clip", "winsorized"] = "sigma_clip"
    sigma_low: float = 3.0
    sigma_high: float = 3.0
    max_iterations: int = 5
    output_bit_depth: Literal[8, 16, 32] = 16
    normalize: bool = True


@dataclass
class OutputConfig:
    """Output configuration."""
    
    output_dir: Path = field(default_factory=lambda: Path("output"))
    save_intermediate: bool = False
    intermediate_format: Literal["png", "tiff", "fits"] = "tiff"
    final_format: Literal["png", "tiff", "fits", "jpg"] = "tiff"
    create_preview: bool = True
    preview_size: tuple[int, int] = (1920, 1080)


@dataclass
class Config:
    """Main configuration container."""
    
    camera: CameraConfig = field(default_factory=CameraConfig)
    gyro: GyroConfig = field(default_factory=GyroConfig)
    motion_compensation: MotionCompensationConfig = field(default_factory=MotionCompensationConfig)
    star_detection: StarDetectionConfig = field(default_factory=StarDetectionConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    stacking: StackingConfig = field(default_factory=StackingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Processing options
    num_workers: int = 4
    use_gpu: bool = False
    verbose: bool = True
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Create Config from dictionary."""
        config = cls()
        
        if "camera" in data:
            config.camera = CameraConfig(**data["camera"])
        if "gyro" in data:
            config.gyro = GyroConfig(**data["gyro"])
        if "motion_compensation" in data:
            config.motion_compensation = MotionCompensationConfig(**data["motion_compensation"])
        if "star_detection" in data:
            config.star_detection = StarDetectionConfig(**data["star_detection"])
        if "quality" in data:
            config.quality = QualityConfig(**data["quality"])
        if "alignment" in data:
            config.alignment = AlignmentConfig(**data["alignment"])
        if "stacking" in data:
            config.stacking = StackingConfig(**data["stacking"])
        if "output" in data:
            out_data = data["output"]
            if "output_dir" in out_data:
                out_data["output_dir"] = Path(out_data["output_dir"])
            config.output = OutputConfig(**out_data)
        
        config.num_workers = data.get("num_workers", 4)
        config.use_gpu = data.get("use_gpu", False)
        config.verbose = data.get("verbose", True)
        
        return config
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import dataclasses
        
        def convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, Path):
                return str(obj)
            return obj
        
        data = convert(self)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

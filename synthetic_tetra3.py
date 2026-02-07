#!/usr/bin/env python3
"""
Synthetic Star Field Generator for tetra3 Testing.

Generates realistic synthetic star field images with known ground truth
for validating the tetra3 plate solver.

Features:
- Generate stars from Hipparcos/Yale catalogs or random positions
- Configurable FOV, image size, and noise levels
- Known RA/Dec for ground truth validation
- PSF models (Gaussian, Moffat, Airy)
- Add realistic noise (read noise, shot noise, background)
- Support for distortion simulation

Usage:
    python synthetic_tetra3.py --ra 180 --dec 45 --fov 15 --output test_field.png
"""

import sys
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import argparse

# Add tetra3 to path
_script_dir = Path(__file__).parent.resolve()
_tetra3_path = _script_dir / 'external' / 'tetra3'
if _tetra3_path.exists():
    sys.path.insert(0, str(_tetra3_path))

try:
    import tetra3
    TETRA3_AVAILABLE = True
except ImportError:
    TETRA3_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class Star:
    """A star with catalog information."""
    ra: float           # Right Ascension in degrees
    dec: float          # Declination in degrees
    magnitude: float    # Visual magnitude
    catalog_id: int = 0 # Catalog ID (e.g., Hipparcos number)


@dataclass
class SyntheticFieldConfig:
    """Configuration for synthetic star field generation."""
    # Image parameters
    width: int = 1024
    height: int = 1024

    # Pointing direction
    ra: float = 180.0       # Right Ascension in degrees
    dec: float = 45.0       # Declination in degrees
    roll: float = 0.0       # Roll angle in degrees

    # Field of view
    fov: float = 15.0       # Field of view in degrees

    # Star parameters
    max_magnitude: float = 7.0      # Limiting magnitude
    min_magnitude: float = -1.5     # Brightest star
    base_brightness: float = 10000  # ADU for magnitude 0 star

    # PSF parameters
    psf_sigma: float = 1.5          # Gaussian sigma in pixels

    # Noise parameters
    background: float = 100.0       # Background level in ADU
    read_noise: float = 10.0        # Read noise in electrons
    dark_current: float = 0.1       # Dark current (electrons/pixel/sec)
    exposure_time: float = 1.0      # Exposure time in seconds
    gain: float = 1.0               # Camera gain (e-/ADU)

    # Distortion
    distortion: float = 0.0         # Radial distortion coefficient

    # Output
    bit_depth: int = 16             # Output bit depth


class StarCatalog:
    """
    Simple star catalog for synthetic field generation.

    Uses embedded bright stars or can load from tetra3's catalog.
    """

    @staticmethod
    def from_tetra3() -> 'StarCatalog':
        """
        Load star catalog from tetra3's database.

        tetra3 star_table structure (6 columns):
        - Column 0: RA in radians
        - Column 1-3: Unit vector components
        - Column 4: z = sin(Dec)
        - Column 5: Apparent magnitude
        """
        if not TETRA3_AVAILABLE:
            raise ImportError("tetra3 not available")

        catalog = StarCatalog()
        catalog.stars = []

        # Load tetra3
        t3 = tetra3.Tetra3()
        st = t3.star_table

        # Also get catalog IDs if available
        catalog_ids = t3.star_catalog_IDs if hasattr(t3, 'star_catalog_IDs') and t3.star_catalog_IDs is not None else None

        # Convert tetra3 catalog to our format
        for i in range(len(st)):
            # Column 0 is RA in radians
            ra_rad = float(st[i][0])
            ra_deg = np.degrees(ra_rad)

            # Normalize RA to 0-360
            if ra_deg < 0:
                ra_deg += 360
            elif ra_deg >= 360:
                ra_deg -= 360

            # Column 4 is z = sin(Dec), so Dec = arcsin(z)
            z = float(st[i][4])
            z = max(-1.0, min(1.0, z))  # Clamp for numerical stability
            dec_deg = np.degrees(np.arcsin(z))

            # Column 5 is apparent magnitude
            mag = float(st[i][5])

            # Get catalog ID if available
            cat_id = int(catalog_ids[i]) if catalog_ids is not None else i

            catalog.stars.append(Star(
                ra=ra_deg,
                dec=dec_deg,
                magnitude=mag,
                catalog_id=cat_id
            ))

        return catalog

    # Bright stars (Hipparcos) - subset for quick testing
    BRIGHT_STARS = [
        # (RA, Dec, Magnitude, Name)
        (101.287, -16.716, -1.46, "Sirius"),
        (114.825, 5.225, 0.34, "Procyon"),
        (219.896, -60.835, -0.27, "Alpha Centauri"),
        (310.358, 45.280, 1.25, "Deneb"),
        (279.235, 38.784, 0.03, "Vega"),
        (88.793, 7.407, 0.50, "Betelgeuse"),
        (78.634, -8.202, 0.12, "Rigel"),
        (68.980, 16.509, 0.85, "Aldebaran"),
        (37.955, 89.264, 1.98, "Polaris"),
        (95.988, -52.696, -0.72, "Canopus"),
        (201.298, -11.161, 0.98, "Spica"),
        (152.093, 11.967, 1.35, "Regulus"),
        (213.915, 19.182, -0.05, "Arcturus"),
        (247.352, -26.432, 0.96, "Antares"),
        (344.413, -29.622, 1.16, "Fomalhaut"),
        (186.650, -63.099, 0.77, "Beta Centauri"),
        (263.402, -37.104, 1.62, "Shaula"),
        (116.329, 28.026, 1.58, "Pollux"),
        (113.650, 31.888, 1.93, "Castor"),
        (141.897, -8.659, 1.98, "Alphard"),
    ]

    def __init__(self):
        """Initialize catalog with bright stars."""
        self.stars: List[Star] = []
        self._load_bright_stars()

    def _load_bright_stars(self):
        """Load the embedded bright star list."""
        for i, (ra, dec, mag, name) in enumerate(self.BRIGHT_STARS):
            self.stars.append(Star(
                ra=ra,
                dec=dec,
                magnitude=mag,
                catalog_id=i
            ))

    def generate_random_stars(self, n_stars: int,
                               min_mag: float = 3.0,
                               max_mag: float = 7.0,
                               seed: int = None,
                               ra_center: float = None,
                               dec_center: float = None,
                               fov: float = None) -> List[Star]:
        """
        Generate random stars for testing.

        Args:
            n_stars: Number of stars to generate
            min_mag: Minimum (brightest) magnitude
            max_mag: Maximum (dimmest) magnitude
            seed: Random seed for reproducibility
            ra_center: If provided, generate stars around this RA
            dec_center: If provided, generate stars around this Dec
            fov: If provided, generate stars within this FOV

        Returns:
            List of Star objects
        """
        if seed is not None:
            np.random.seed(seed)

        stars = []
        for i in range(n_stars):
            if ra_center is not None and dec_center is not None and fov is not None:
                # Generate stars within FOV of center
                # Random offset from center
                offset = np.random.uniform(0, fov / 2)
                angle = np.random.uniform(0, 2 * np.pi)

                # Apply offset (simplified, works well for small FOV)
                ra = ra_center + offset * np.cos(angle) / np.cos(np.radians(dec_center))
                dec = dec_center + offset * np.sin(angle)

                # Clamp dec
                dec = max(-89, min(89, dec))
                # Normalize RA
                ra = ra % 360
            else:
                # Random RA (0-360) and Dec (-90 to +90)
                ra = np.random.uniform(0, 360)
                # Uniform distribution on sphere
                dec = np.degrees(np.arcsin(np.random.uniform(-1, 1)))

            # Magnitude distribution (more faint stars)
            mag = np.random.uniform(min_mag, max_mag)

            stars.append(Star(ra=ra, dec=dec, magnitude=mag, catalog_id=1000+i))

        return stars

    def get_stars_in_fov(self, ra_center: float, dec_center: float,
                          fov: float, max_mag: float = 7.0) -> List[Star]:
        """
        Get stars within a field of view.

        Args:
            ra_center: Center RA in degrees
            dec_center: Center Dec in degrees
            fov: Field of view in degrees
            max_mag: Limiting magnitude

        Returns:
            List of stars within FOV
        """
        result = []
        fov_rad = math.radians(fov / 2)

        for star in self.stars:
            if star.magnitude > max_mag:
                continue

            # Angular separation
            sep = self._angular_separation(
                ra_center, dec_center,
                star.ra, star.dec
            )

            if sep <= fov / 2:
                result.append(star)

        return result

    def _angular_separation(self, ra1: float, dec1: float,
                            ra2: float, dec2: float) -> float:
        """Calculate angular separation in degrees."""
        ra1_rad = math.radians(ra1)
        dec1_rad = math.radians(dec1)
        ra2_rad = math.radians(ra2)
        dec2_rad = math.radians(dec2)

        cos_sep = (math.sin(dec1_rad) * math.sin(dec2_rad) +
                   math.cos(dec1_rad) * math.cos(dec2_rad) *
                   math.cos(ra1_rad - ra2_rad))

        cos_sep = max(-1, min(1, cos_sep))
        return math.degrees(math.acos(cos_sep))


class SyntheticStarField:
    """
    Generates synthetic star field images for tetra3 testing.
    """

    def __init__(self, config: SyntheticFieldConfig, use_tetra3_catalog: bool = True):
        """
        Initialize the generator.

        Args:
            config: Configuration for field generation
            use_tetra3_catalog: Try to load tetra3's star catalog (8000+ stars)
        """
        self.config = config

        # Try to load tetra3's catalog for realistic star patterns
        if use_tetra3_catalog and TETRA3_AVAILABLE:
            try:
                print("Loading star catalog from tetra3...")
                self.catalog = StarCatalog.from_tetra3()
                print(f"  Loaded {len(self.catalog.stars)} stars from tetra3 database")
            except Exception as e:
                print(f"  Warning: Could not load tetra3 catalog: {e}")
                print("  Falling back to built-in bright star catalog")
                self.catalog = StarCatalog()
        else:
            self.catalog = StarCatalog()

    def generate(self, use_catalog: bool = True,
                 n_random_stars: int = 50,
                 seed: int = None) -> Tuple[np.ndarray, Dict]:
        """
        Generate a synthetic star field image.

        Args:
            use_catalog: Use real star catalog positions
            n_random_stars: Number of random stars if not using catalog
            seed: Random seed for reproducibility

        Returns:
            Tuple of (image array, metadata dict)
        """
        cfg = self.config

        # Create base image
        image = np.ones((cfg.height, cfg.width), dtype=np.float64) * cfg.background

        # Get stars
        if use_catalog:
            # Get stars from catalog within FOV
            stars = self.catalog.get_stars_in_fov(
                cfg.ra, cfg.dec, cfg.fov * 1.5,  # Slightly larger to catch edge stars
                max_mag=cfg.max_magnitude
            )
            # Add some random faint stars within FOV
            stars.extend(self.catalog.generate_random_stars(
                n_random_stars // 2,
                min_mag=5.0,
                max_mag=cfg.max_magnitude,
                seed=seed,
                ra_center=cfg.ra,
                dec_center=cfg.dec,
                fov=cfg.fov
            ))
        else:
            # Generate random stars within FOV
            stars = self.catalog.generate_random_stars(
                n_random_stars,
                min_mag=cfg.min_magnitude,
                max_mag=cfg.max_magnitude,
                seed=seed,
                ra_center=cfg.ra,
                dec_center=cfg.dec,
                fov=cfg.fov
            )

        # Project stars onto image
        star_positions = []
        for star in stars:
            px, py = self._project_star(star.ra, star.dec)

            # Check if within image bounds
            if 0 <= px < cfg.width and 0 <= py < cfg.height:
                # Calculate brightness from magnitude
                brightness = self._magnitude_to_adu(star.magnitude)

                # Add star PSF to image
                self._add_star_psf(image, px, py, brightness)

                star_positions.append({
                    'x': px,
                    'y': py,
                    'ra': star.ra,
                    'dec': star.dec,
                    'magnitude': star.magnitude,
                    'brightness': brightness
                })

        # Add noise
        image = self._add_noise(image, seed)

        # Clip and convert to integer
        max_val = 2**cfg.bit_depth - 1
        image = np.clip(image, 0, max_val)

        if cfg.bit_depth <= 8:
            image = image.astype(np.uint8)
        else:
            image = image.astype(np.uint16)

        # Build metadata
        metadata = {
            'ra': cfg.ra,
            'dec': cfg.dec,
            'roll': cfg.roll,
            'fov': cfg.fov,
            'width': cfg.width,
            'height': cfg.height,
            'n_stars': len(star_positions),
            'stars': star_positions,
            'config': cfg.__dict__.copy()
        }

        return image, metadata

    def _project_star(self, ra: float, dec: float) -> Tuple[float, float]:
        """
        Project star RA/Dec to image pixel coordinates.

        Uses gnomonic (tangent plane) projection.
        """
        cfg = self.config

        # Convert to radians
        ra_rad = math.radians(ra)
        dec_rad = math.radians(dec)
        ra0_rad = math.radians(cfg.ra)
        dec0_rad = math.radians(cfg.dec)
        roll_rad = math.radians(cfg.roll)

        # Gnomonic projection
        cos_c = (math.sin(dec0_rad) * math.sin(dec_rad) +
                 math.cos(dec0_rad) * math.cos(dec_rad) *
                 math.cos(ra_rad - ra0_rad))

        if cos_c <= 0:
            return -1, -1  # Behind camera

        x = (math.cos(dec_rad) * math.sin(ra_rad - ra0_rad)) / cos_c
        y = (math.cos(dec0_rad) * math.sin(dec_rad) -
             math.sin(dec0_rad) * math.cos(dec_rad) *
             math.cos(ra_rad - ra0_rad)) / cos_c

        # Apply roll rotation
        x_rot = x * math.cos(roll_rad) - y * math.sin(roll_rad)
        y_rot = x * math.sin(roll_rad) + y * math.cos(roll_rad)

        # Apply distortion
        if cfg.distortion != 0:
            r2 = x_rot**2 + y_rot**2
            factor = 1 + cfg.distortion * r2
            x_rot *= factor
            y_rot *= factor

        # Scale to pixels
        scale = cfg.width / (2 * math.tan(math.radians(cfg.fov / 2)))

        px = cfg.width / 2 + x_rot * scale
        py = cfg.height / 2 - y_rot * scale  # Flip Y

        return px, py

    def _magnitude_to_adu(self, magnitude: float) -> float:
        """Convert stellar magnitude to ADU counts."""
        cfg = self.config
        # Pogson's law: m1 - m2 = -2.5 * log10(F1/F2)
        # F = F0 * 10^(-0.4 * m)
        return cfg.base_brightness * 10**(-0.4 * magnitude)

    def _add_star_psf(self, image: np.ndarray, x: float, y: float,
                       brightness: float):
        """Add a star PSF to the image."""
        cfg = self.config
        sigma = cfg.psf_sigma

        # PSF size (3-sigma radius)
        size = int(3 * sigma) + 1

        # Generate PSF
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                px = int(x + dx)
                py = int(y + dy)

                if 0 <= px < cfg.width and 0 <= py < cfg.height:
                    # Gaussian PSF
                    r2 = (dx - (x - int(x)))**2 + (dy - (y - int(y)))**2
                    psf = math.exp(-r2 / (2 * sigma**2))
                    image[py, px] += brightness * psf

    def _add_noise(self, image: np.ndarray, seed: int = None) -> np.ndarray:
        """Add realistic noise to the image."""
        cfg = self.config

        if seed is not None:
            np.random.seed(seed + 1)

        # Shot noise (Poisson)
        # Convert to electrons, apply Poisson, convert back
        electrons = image * cfg.gain
        electrons = np.maximum(electrons, 0)
        noisy_electrons = np.random.poisson(electrons.astype(np.float64))

        # Add read noise (Gaussian)
        read_noise = np.random.normal(0, cfg.read_noise, image.shape)
        noisy_electrons = noisy_electrons + read_noise

        # Add dark current
        dark = cfg.dark_current * cfg.exposure_time
        dark_noise = np.random.poisson(dark, image.shape)
        noisy_electrons = noisy_electrons + dark_noise

        # Convert back to ADU
        return noisy_electrons / cfg.gain


def validate_with_tetra3(image: np.ndarray, metadata: Dict) -> Dict:
    """
    Validate synthetic image with tetra3 solver.

    Args:
        image: Synthetic star field image
        metadata: Ground truth metadata

    Returns:
        Validation results dict
    """
    if not TETRA3_AVAILABLE:
        return {'error': 'tetra3 not available'}

    # Initialize tetra3
    t3 = tetra3.Tetra3()

    # Convert to PIL Image if needed
    if PIL_AVAILABLE:
        pil_image = Image.fromarray(image)
        solution = t3.solve_from_image(
            pil_image,
            fov_estimate=metadata['fov'],
            fov_max_error=0.5
        )
    else:
        # Use centroids directly
        centroids = tetra3.get_centroids_from_image(image)
        if centroids is None or len(centroids) < 4:
            return {'error': 'Not enough stars detected'}

        solution = t3.solve_from_centroids(
            centroids,
            (image.shape[0], image.shape[1]),
            fov_estimate=metadata['fov'],
            fov_max_error=0.5
        )

    if solution is None or solution.get('RA') is None:
        return {'solved': False, 'error': 'No solution found'}

    # Calculate errors
    ra_error = abs(solution['RA'] - metadata['ra'])
    if ra_error > 180:
        ra_error = 360 - ra_error

    dec_error = abs(solution['Dec'] - metadata['dec'])
    fov_error = abs(solution['FOV'] - metadata['fov'])

    return {
        'solved': True,
        'solution': solution,
        'ground_truth': {
            'ra': metadata['ra'],
            'dec': metadata['dec'],
            'fov': metadata['fov']
        },
        'errors': {
            'ra_deg': ra_error,
            'dec_deg': dec_error,
            'fov_deg': fov_error,
            'ra_arcsec': ra_error * 3600,
            'dec_arcsec': dec_error * 3600
        }
    }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic star fields for tetra3 testing"
    )

    parser.add_argument("--ra", type=float, default=180.0,
                       help="Right Ascension in degrees")
    parser.add_argument("--dec", type=float, default=45.0,
                       help="Declination in degrees")
    parser.add_argument("--roll", type=float, default=0.0,
                       help="Roll angle in degrees")
    parser.add_argument("--fov", type=float, default=15.0,
                       help="Field of view in degrees")
    parser.add_argument("--width", type=int, default=1024,
                       help="Image width in pixels")
    parser.add_argument("--height", type=int, default=1024,
                       help="Image height in pixels")
    parser.add_argument("--magnitude", type=float, default=7.0,
                       help="Limiting magnitude")
    parser.add_argument("--noise", type=float, default=10.0,
                       help="Read noise in electrons")
    parser.add_argument("--background", type=float, default=100.0,
                       help="Background level in ADU")
    parser.add_argument("--output", type=str, default="synthetic_field.png",
                       help="Output image path")
    parser.add_argument("--validate", action="store_true",
                       help="Validate with tetra3 after generation")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--use-tetra3-catalog", action="store_true", default=True,
                       help="Use tetra3's star catalog (default: True)")
    parser.add_argument("--no-tetra3-catalog", dest="use_tetra3_catalog", action="store_false",
                       help="Use built-in bright stars only")

    args = parser.parse_args()

    # Create configuration
    config = SyntheticFieldConfig(
        width=args.width,
        height=args.height,
        ra=args.ra,
        dec=args.dec,
        roll=args.roll,
        fov=args.fov,
        max_magnitude=args.magnitude,
        read_noise=args.noise,
        background=args.background
    )

    print("=" * 60)
    print("Synthetic Star Field Generator for tetra3")
    print("=" * 60)
    print(f"Pointing: RA={args.ra:.2f}°, Dec={args.dec:.2f}°, Roll={args.roll:.2f}°")
    print(f"FOV: {args.fov:.1f}°")
    print(f"Image: {args.width}x{args.height}")
    print(f"Limiting magnitude: {args.magnitude}")
    print()

    # Generate field
    print("Generating synthetic star field...")
    generator = SyntheticStarField(config, use_tetra3_catalog=args.use_tetra3_catalog)
    image, metadata = generator.generate(use_catalog=True, seed=args.seed)

    print(f"Generated {metadata['n_stars']} stars")

    # Save image
    if CV2_AVAILABLE:
        cv2.imwrite(args.output, image)
        print(f"Saved to: {args.output}")
    elif PIL_AVAILABLE:
        pil_image = Image.fromarray(image)
        pil_image.save(args.output)
        print(f"Saved to: {args.output}")
    else:
        print("Warning: Neither OpenCV nor PIL available for saving")

    # Validate with tetra3
    if args.validate:
        print()
        print("Validating with tetra3...")
        result = validate_with_tetra3(image, metadata)

        if result.get('solved'):
            print(f"  ✓ SOLVED!")
            print(f"  Solution: RA={result['solution']['RA']:.4f}°, "
                  f"Dec={result['solution']['Dec']:.4f}°, "
                  f"FOV={result['solution']['FOV']:.2f}°")
            print(f"  Errors: RA={result['errors']['ra_arcsec']:.1f} arcsec, "
                  f"Dec={result['errors']['dec_arcsec']:.1f} arcsec")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")

    print()
    print("=" * 60)

    return metadata


if __name__ == "__main__":
    main()

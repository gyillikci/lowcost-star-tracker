#!/usr/bin/env python3
"""
Demo: Plate Solving for AllSky Images

Demonstrates blind astrometric plate solving using geometric pattern matching.
Can process video frames or static images to determine celestial coordinates.

Usage:
    python demo_plate_solver.py                    # Demo with synthetic stars
    python demo_plate_solver.py --image allsky.jpg # Solve single image
    python demo_plate_solver.py --video allsky.mp4 # Process video frames

Camera assumptions (typical AllSky setup):
    - Raspberry Pi HQ Camera (Sony IMX477, 12.3MP)
    - 180° fisheye lens (1.55mm focal length)
    - Location: Waimea, Hawaii (20.02°N, 155.67°W)
"""

import sys
import argparse
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available")

from src.plate_solving import (
    PlateSolver, StarCatalog, StarDetector,
    AllSkyCamera, WCSSolution
)


def generate_synthetic_allsky(width: int = 1920, height: int = 1080,
                              lat: float = 20.02, lon: float = -155.67,
                              timestamp: datetime = None) -> np.ndarray:
    """
    Generate a synthetic AllSky image with stars.

    Args:
        width, height: Image dimensions
        lat, lon: Observer location
        timestamp: UTC timestamp (default: now)

    Returns:
        Synthetic star field image
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    # Create camera model
    camera = AllSkyCamera.create_typical_allsky(
        width=width, height=height,
        fov=180.0, lat=lat, lon=lon
    )

    # Calculate LST
    solver = PlateSolver()
    lst_hours = solver._calculate_lst(lon, timestamp)

    # Get visible stars
    catalog = StarCatalog(max_magnitude=4.5)
    visible = catalog.get_visible_stars(lat, lon, lst_hours, min_altitude=5.0)

    print(f"Generating synthetic AllSky image")
    print(f"  Location: {lat:.2f}°N, {lon:.2f}°W")
    print(f"  Time: {timestamp.isoformat()}")
    print(f"  LST: {lst_hours:.2f} hours")
    print(f"  Visible stars: {len(visible)}")

    # Create dark background with noise
    image = np.random.normal(10, 5, (height, width)).astype(np.float32)
    image = np.clip(image, 0, 255)

    # Add horizon glow
    for y in range(height):
        for x in range(width):
            alt, _ = camera.pixel_to_altaz(x, y)
            if alt < 20:
                glow = max(0, 20 - alt) * 2
                image[y, x] += glow

    # Add stars
    stars_added = 0
    for star, alt, az in visible:
        pixel = camera.altaz_to_pixel(alt, az)
        if pixel is None:
            continue

        x, y = int(pixel[0]), int(pixel[1])
        if 0 <= x < width and 0 <= y < height:
            # Star brightness based on magnitude
            # Brighter stars (lower mag) -> higher intensity
            intensity = 255 * 10 ** (-(star.mag + 1.5) / 2.5)
            intensity = min(255, max(50, intensity))

            # Draw star with Gaussian profile
            size = max(2, int(5 - star.mag / 2))
            for dy in range(-size*2, size*2+1):
                for dx in range(-size*2, size*2+1):
                    px, py = x + dx, y + dy
                    if 0 <= px < width and 0 <= py < height:
                        r = np.sqrt(dx*dx + dy*dy)
                        gauss = np.exp(-r*r / (2 * (size/2.355)**2))
                        image[py, px] += intensity * gauss

            stars_added += 1

    print(f"  Stars rendered: {stars_added}")

    # Clip and convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def solve_image(image_path: str, lat: float, lon: float,
                timestamp: datetime = None, visualize: bool = True):
    """
    Solve plate for a single image.

    Args:
        image_path: Path to image file
        lat, lon: Observer location
        timestamp: Image timestamp
        visualize: Show visualization
    """
    print(f"\nLoading image: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image")
        return None

    height, width = image.shape[:2]
    print(f"Image size: {width}x{height}")

    # Create solver
    solver = PlateSolver(max_catalog_stars=100, max_detected_stars=50)

    # Solve
    print("\nRunning plate solver...")
    if timestamp:
        solution = solver.solve_allsky(image, lat, lon, timestamp=timestamp)
    else:
        solution = solver.solve(image)

    if solution:
        print_solution(solution)

        if visualize and CV2_AVAILABLE:
            show_solution(image, solution, solver.detector)
    else:
        print("Plate solving failed")

    return solution


def solve_video_frame(video_path: str, frame_number: int,
                     lat: float, lon: float, timestamp: datetime = None):
    """
    Solve plate for a specific video frame.

    Args:
        video_path: Path to video file
        frame_number: Frame to solve
        lat, lon: Observer location
        timestamp: Video start timestamp
    """
    print(f"\nLoading video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames, {fps:.2f} FPS")

    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        return None

    # Adjust timestamp for frame time
    if timestamp:
        frame_time = frame_number / fps
        from datetime import timedelta
        timestamp = timestamp + timedelta(seconds=frame_time)

    return solve_image_array(frame, lat, lon, timestamp)


def solve_image_array(image: np.ndarray, lat: float, lon: float,
                     timestamp: datetime = None) -> WCSSolution:
    """
    Solve plate for image array.
    """
    solver = PlateSolver(max_catalog_stars=100, max_detected_stars=50)

    print("\nRunning plate solver...")
    if timestamp:
        solution = solver.solve_allsky(image, lat, lon, timestamp=timestamp)
    else:
        solution = solver.solve(image)

    if solution:
        print_solution(solution)
    else:
        print("Plate solving failed")

    return solution


def print_solution(solution: WCSSolution):
    """Print plate solving solution details."""
    print("\n" + "=" * 50)
    print("PLATE SOLVING SOLUTION")
    print("=" * 50)

    # Convert RA to hours
    ra_hours = solution.ra_center / 15.0
    ra_h = int(ra_hours)
    ra_m = int((ra_hours - ra_h) * 60)
    ra_s = ((ra_hours - ra_h) * 60 - ra_m) * 60

    # Convert Dec to DMS
    dec_sign = '+' if solution.dec_center >= 0 else '-'
    dec_abs = abs(solution.dec_center)
    dec_d = int(dec_abs)
    dec_m = int((dec_abs - dec_d) * 60)
    dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60

    print(f"\nImage Center:")
    print(f"  RA:  {ra_h:02d}h {ra_m:02d}m {ra_s:05.2f}s ({solution.ra_center:.4f}°)")
    print(f"  Dec: {dec_sign}{dec_d:02d}° {dec_m:02d}' {dec_s:04.1f}\" ({solution.dec_center:.4f}°)")

    print(f"\nField of View:")
    print(f"  {solution.fov_x:.2f}° x {solution.fov_y:.2f}°")

    print(f"\nPixel Scale: {solution.pixel_scale:.2f} arcsec/pixel")
    print(f"Rotation: {solution.rotation:.2f}° (E of N)")

    print(f"\nQuality:")
    print(f"  Matched stars: {solution.n_matches}")
    print(f"  RMS error: {solution.rms_error:.1f} arcsec")
    print(f"  Confidence: {solution.confidence:.1%}")
    print(f"  Solve time: {solution.solve_time:.2f}s")

    if solution.matched_stars:
        print(f"\nMatched Stars:")
        for i, (det, cat) in enumerate(solution.matched_stars[:10]):
            print(f"  {i+1}. {cat.name or f'HIP {cat.hip_id}'} "
                  f"(mag {cat.mag:.1f}) at ({det.x:.1f}, {det.y:.1f})")


def show_solution(image: np.ndarray, solution: WCSSolution,
                 detector: StarDetector):
    """Show visualization of plate solving solution."""
    # Detect stars
    detected, vis = detector.detect_with_visualization(image)

    # Add matched star labels
    for det_star, cat_star in solution.matched_stars:
        x, y = int(det_star.x), int(det_star.y)
        label = cat_star.name or f"HIP{cat_star.hip_id}"

        cv2.circle(vis, (x, y), 10, (0, 255, 0), 2)
        cv2.putText(vis, label, (x + 12, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Add solution info
    info = [
        f"RA: {solution.ra_center:.2f}°",
        f"Dec: {solution.dec_center:.2f}°",
        f"FOV: {solution.fov_x:.1f}°x{solution.fov_y:.1f}°",
        f"Matches: {solution.n_matches}",
        f"RMS: {solution.rms_error:.1f}\"",
    ]

    y_pos = 30
    for line in info:
        cv2.putText(vis, line, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_pos += 25

    # Show
    cv2.imshow("Plate Solving Result", vis)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_synthetic():
    """Run demo with synthetic star field."""
    print("=" * 60)
    print("PLATE SOLVING DEMO - Synthetic AllSky Image")
    print("=" * 60)

    # Waimea, Hawaii location
    lat = 20.02
    lon = -155.67

    # Current time
    timestamp = datetime.now(timezone.utc)

    # Generate synthetic image
    image = generate_synthetic_allsky(
        width=1920, height=1080,
        lat=lat, lon=lon,
        timestamp=timestamp
    )

    # Solve
    solution = solve_image_array(image, lat, lon, timestamp)

    if solution and CV2_AVAILABLE:
        detector = StarDetector(sigma_threshold=2.5, min_snr=3.0)
        show_solution(image, solution, detector)

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Plate Solving Demo for AllSky Images"
    )
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image file to solve')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file')
    parser.add_argument('--frame', type=int, default=0,
                       help='Video frame number to solve (default: 0)')
    parser.add_argument('--lat', type=float, default=20.02,
                       help='Observer latitude (default: 20.02 for Waimea, HI)')
    parser.add_argument('--lon', type=float, default=-155.67,
                       help='Observer longitude (default: -155.67 for Waimea, HI)')
    parser.add_argument('--date', type=str, default=None,
                       help='Image date/time in ISO format (e.g., 2026-01-13T22:00:00)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')

    args = parser.parse_args()

    # Parse timestamp
    timestamp = None
    if args.date:
        try:
            timestamp = datetime.fromisoformat(args.date.replace('Z', '+00:00'))
        except:
            print(f"Warning: Could not parse date '{args.date}'")

    if args.image:
        solve_image(args.image, args.lat, args.lon, timestamp,
                   visualize=not args.no_viz)
    elif args.video:
        solve_video_frame(args.video, args.frame, args.lat, args.lon, timestamp)
    else:
        demo_synthetic()


if __name__ == '__main__':
    main()

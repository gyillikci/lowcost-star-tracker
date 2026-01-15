#!/usr/bin/env python3
"""
Demo: Plate Solving using Tetra3 (ESA)

Uses the tetra3 library from the European Space Agency for fast, reliable
astrometric plate solving. Tetra3 is specifically designed for star trackers.

Installation:
    pip install tetra3 opencv-python astropy

Usage:
    python demo_plate_solver.py                    # Check dependencies
    python demo_plate_solver.py --image star.jpg  # Solve single image
    python demo_plate_solver.py --video allsky.mp4 --frame 100  # Video frame

References:
    - tetra3: https://github.com/esa/tetra3
    - Docs: https://tetra3.readthedocs.io/
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available: pip install opencv-python")

from src.plate_solving import (
    Tetra3Solver, AllSkySolver, PlateSolution,
    check_dependencies, TETRA3_AVAILABLE
)


def print_solution(solution: PlateSolution):
    """Print plate solving solution."""
    print("\n" + "=" * 60)
    print("PLATE SOLVING RESULT (tetra3)")
    print("=" * 60)

    print(f"\nPointing Direction:")
    print(f"  RA:   {solution.ra_hms} ({solution.ra:.4f}°)")
    print(f"  Dec:  {solution.dec_dms} ({solution.dec:.4f}°)")
    print(f"  Roll: {solution.roll:.2f}°")

    print(f"\nField of View: {solution.fov:.2f}°")

    print(f"\nQuality Metrics:")
    print(f"  Matched stars: {solution.n_matches}")
    print(f"  Probability:   {solution.prob:.2e}")
    print(f"  Solve time:    {solution.t_solve*1000:.1f} ms")

    if solution.matched_catID is not None:
        print(f"\nMatched Star IDs: {solution.matched_catID[:10]}...")

    print("=" * 60)


def solve_single_image(image_path: str, fov_estimate: float = None,
                       visualize: bool = True):
    """Solve a single image file."""
    print(f"\nLoading: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image")
        return None

    height, width = image.shape[:2]
    print(f"Image size: {width}x{height}")

    # Check if tetra3 available
    if not TETRA3_AVAILABLE:
        print("\nError: tetra3 not installed")
        print("Install with: pip install tetra3")
        return None

    # Create solver
    print("\nInitializing tetra3 solver...")
    try:
        solver = Tetra3Solver()
    except Exception as e:
        print(f"Error initializing solver: {e}")
        return None

    # Solve
    print("Solving...")
    solution = solver.solve(image, fov_estimate=fov_estimate)

    if solution:
        print_solution(solution)

        if visualize and CV2_AVAILABLE:
            visualize_solution(image, solution)
    else:
        print("\nNo solution found.")
        print("Tips:")
        print("  - Ensure image contains visible stars")
        print("  - Try providing --fov estimate")
        print("  - Check image is not too noisy")

    return solution


def solve_video_frame(video_path: str, frame_num: int,
                     fov_estimate: float = None):
    """Extract and solve a video frame."""
    print(f"\nLoading video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames, {fps:.2f} FPS")

    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_num}")
        return None

    print(f"Extracted frame {frame_num}")

    # Save frame for inspection
    frame_path = f"frame_{frame_num}.jpg"
    cv2.imwrite(frame_path, frame)
    print(f"Saved to: {frame_path}")

    return solve_single_image(frame_path, fov_estimate)


def visualize_solution(image: np.ndarray, solution: PlateSolution):
    """Show visualization of the solution."""
    vis = image.copy()

    # Normalize for display
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    height, width = vis.shape[:2]

    # Draw matched stars if available
    if solution.matched_centroids is not None:
        for i, (x, y) in enumerate(solution.matched_centroids):
            x, y = int(x), int(y)
            cv2.circle(vis, (x, y), 10, (0, 255, 0), 2)
            if solution.matched_catID is not None and i < len(solution.matched_catID):
                label = f"ID:{solution.matched_catID[i]}"
                cv2.putText(vis, label, (x+12, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw pattern centroids
    if solution.pattern_centroids is not None:
        for x, y in solution.pattern_centroids:
            x, y = int(x), int(y)
            cv2.circle(vis, (x, y), 5, (0, 255, 255), -1)

    # Add text overlay
    info_lines = [
        f"RA: {solution.ra_hms}",
        f"Dec: {solution.dec_dms}",
        f"FOV: {solution.fov:.2f} deg",
        f"Matches: {solution.n_matches}",
        f"Time: {solution.t_solve*1000:.1f}ms",
    ]

    y_pos = 30
    for line in info_lines:
        cv2.putText(vis, line, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30

    # Show
    cv2.imshow("Plate Solution (tetra3)", vis)
    print("\nPress any key to close visualization...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def check_install():
    """Check installation status."""
    print("\n" + "=" * 60)
    print("PLATE SOLVING - Dependency Check")
    print("=" * 60)

    issues = check_dependencies()

    print("\nRequired packages:")
    print(f"  tetra3:  {'OK' if TETRA3_AVAILABLE else 'MISSING'}")
    print(f"  OpenCV:  {'OK' if CV2_AVAILABLE else 'MISSING'}")

    try:
        from astropy.wcs import WCS
        print(f"  astropy: OK (optional, for WCS)")
    except ImportError:
        print(f"  astropy: MISSING (optional, for WCS)")

    if issues:
        print("\nTo install missing packages:")
        for issue in issues:
            print(f"  {issue}")
        print("\nOr install all at once:")
        print("  pip install tetra3 opencv-python astropy")
    else:
        print("\nAll dependencies satisfied!")

    # Try to initialize solver
    if TETRA3_AVAILABLE:
        print("\nTesting tetra3 initialization...")
        try:
            solver = Tetra3Solver()
            print("  Solver initialized successfully!")
        except Exception as e:
            print(f"  Initialization failed: {e}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Plate Solving Demo using tetra3 (ESA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Check dependencies
  %(prog)s --image stars.jpg         # Solve image
  %(prog)s --image stars.jpg --fov 15  # With FOV estimate
  %(prog)s --video allsky.mp4 --frame 100  # Video frame

For best results:
  - Provide FOV estimate with --fov (in degrees)
  - Ensure image has clear star field
  - tetra3 works best with 10-30 degree FOV
"""
    )

    parser.add_argument('--image', type=str,
                       help='Image file to solve')
    parser.add_argument('--video', type=str,
                       help='Video file')
    parser.add_argument('--frame', type=int, default=0,
                       help='Video frame number (default: 0)')
    parser.add_argument('--fov', type=float, default=None,
                       help='Field of view estimate in degrees')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--check', action='store_true',
                       help='Only check dependencies')

    args = parser.parse_args()

    # Always show dependency check first if no specific task
    if args.check or (not args.image and not args.video):
        check_install()
        if not args.image and not args.video:
            return

    if args.image:
        solve_single_image(args.image, args.fov, not args.no_viz)
    elif args.video:
        solve_video_frame(args.video, args.frame, args.fov)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Display checkerboard pattern on screen for camera calibration.
Press ESC or Q to quit.
"""

import cv2
import numpy as np
import sys
import argparse


def create_checkerboard(inner_cols=9, inner_rows=6, square_size=100):
    """Create checkerboard pattern image.
    
    Args:
        inner_cols: Number of inner corners in horizontal direction (default 9)
        inner_rows: Number of inner corners in vertical direction (default 6)
        square_size: Size of each square in pixels
        
    Note: To get inner_cols x inner_rows inner corners, we need 
          (inner_cols + 1) x (inner_rows + 1) squares
    """
    # We need one more square than inner corners in each dimension
    num_squares_x = inner_cols + 1  # 10 squares for 9 inner corners
    num_squares_y = inner_rows + 1  # 7 squares for 6 inner corners
    
    # Add white border around the pattern for better detection
    border = square_size
    width = num_squares_x * square_size + 2 * border
    height = num_squares_y * square_size + 2 * border
    
    # Create white background
    pattern = np.ones((height, width), dtype=np.uint8) * 255
    
    # Draw black squares (starting with white in top-left)
    for row in range(num_squares_y):
        for col in range(num_squares_x):
            if (row + col) % 2 == 1:
                y1 = border + row * square_size
                y2 = border + (row + 1) * square_size
                x1 = border + col * square_size
                x2 = border + (col + 1) * square_size
                pattern[y1:y2, x1:x2] = 0
    
    return pattern


def main():
    parser = argparse.ArgumentParser(description='Display checkerboard pattern')
    parser.add_argument('--size', type=str, default='9x6',
                       help='Checkerboard size (default: 9x6)')
    parser.add_argument('--square', type=int, default=150,
                       help='Square size in pixels (default: 150)')
    parser.add_argument('--screen', type=int, default=1,
                       help='Screen number (0=primary, 1=secondary)')
    parser.add_argument('--camera', type=int, default=1,
                       help='Camera index to display (default: 1)')
    parser.add_argument('--no-camera', action='store_true',
                       help='Do not show camera feed')
    args = parser.parse_args()
    
    # Parse size (inner corners)
    try:
        inner_cols, inner_rows = map(int, args.size.split('x'))
    except:
        print(f"Invalid size: {args.size}")
        return 1
    
    # Create pattern with correct number of inner corners
    pattern = create_checkerboard(inner_cols, inner_rows, args.square)
    
    # Open camera if requested
    cap = None
    if not args.no_camera:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Warning: Could not open camera {args.camera}")
            cap = None
    
    # Create window
    window_name = 'Checkerboard Calibration Pattern'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Create camera window if available
    if cap:
        camera_window = 'Camera Feed'
        cv2.namedWindow(camera_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(camera_window, 640, 480)
    
    # Try to move to second screen (Windows)
    if args.screen == 1:
        try:
            # Move window to second monitor (approximate position)
            cv2.moveWindow(window_name, 1920, 0)  # Assuming 1920px primary monitor
        except:
            pass
    
    # Set fullscreen
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print("\n" + "="*60)
    print("CHECKERBOARD PATTERN DISPLAYED")
    print("="*60)
    print(f"Inner corners: {inner_cols}x{inner_rows}")
    print(f"Squares: {inner_cols + 1}x{inner_rows + 1}")
    print(f"Square size: {args.square}px")
    if cap:
        print(f"Camera {args.camera}: Active")
    print(f"\nInstructions:")
    print("  - Point your camera at this screen")
    if cap:
        print("  - Watch the 'Camera Feed' window for visual confirmation")
    print("  - Capture images from different angles")
    print("  - Press ESC or Q to quit")
    print("="*60 + "\n")
    
    # Checkerboard size for detection (inner corners)
    cb_size = (inner_cols, inner_rows)
    
    while True:
        cv2.imshow(window_name, pattern)
        
        # Display camera feed if available
        if cap:
            ret, frame = cap.read()
            if ret:
                # Detect checkerboard in camera feed for visual feedback
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret_corners, corners = cv2.findChessboardCorners(
                    gray, cb_size, 
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
                )
                
                # Draw corners if found
                if ret_corners:
                    cv2.drawChessboardCorners(frame, cb_size, corners, ret_corners)
                    cv2.putText(frame, "PATTERN DETECTED", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No pattern detected", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow(camera_window, frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q
            break
    
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    exit(main())

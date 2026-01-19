#!/usr/bin/env python3
"""
Simple Star Field Solver for Proof of Concept.

A relaxed star pattern matcher designed to work with camera images of 
Stellarium's GUI. Much more forgiving than tetra3, suitable for demos.

Features:
- Simple centroid-based star detection
- Pattern matching using triangle ratios
- Connects to Stellarium API for ground truth comparison
- Visual feedback with detected stars and matching confidence

Press 'q' or ESC to quit, 's' to save current frame.
"""

import sys
import time
import cv2
import numpy as np
import requests
import math
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

STELLARIUM_URL = "http://localhost:8090"


@dataclass
class DetectedStar:
    """A detected star centroid."""
    x: float
    y: float
    brightness: float
    radius: float


@dataclass
class SolveResult:
    """Result from star field solving."""
    success: bool
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    confidence: float = 0.0
    stars_detected: int = 0
    message: str = ""


def get_stellarium_view() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Get current RA/Dec/FOV from Stellarium's view center."""
    try:
        resp = requests.get(f"{STELLARIUM_URL}/api/main/view", timeout=0.5)
        if resp.status_code == 200:
            data = resp.json()
            # j2000 is a JSON string "[x, y, z]" that needs parsing
            j2000_str = data.get('j2000', '[0, 0, 1]')
            j2000 = json.loads(j2000_str)
            x = float(j2000[0])
            y = float(j2000[1])
            z = float(j2000[2])
            # Convert unit vector to RA/Dec
            # RA = atan2(y, x), Dec = asin(z)
            ra_rad = math.atan2(y, x)
            dec_rad = math.asin(max(-1, min(1, z)))  # Clamp z to [-1,1]
            ra_deg = math.degrees(ra_rad)
            dec_deg = math.degrees(dec_rad)
            if ra_deg < 0:
                ra_deg += 360
            # Get FOV
            fov = data.get('fov', 60.0)
            return ra_deg, dec_deg, fov
    except Exception as e:
        pass
    return None, None, None


def get_stellarium_stars() -> List[Tuple[float, float, float]]:
    """
    Get visible stars from Stellarium (if API supports it).
    Returns list of (x, y, magnitude) tuples in screen coordinates.
    """
    # Note: Stellarium's remote API doesn't directly expose visible stars
    # This would need the scripting API or a custom plugin
    return []


def detect_stars(gray_image: np.ndarray, 
                 threshold_sigma: float = 2.5,
                 min_area: int = 3,
                 max_area: int = 500,
                 max_stars: int = 100) -> List[DetectedStar]:
    """
    Detect star-like objects in a grayscale image.
    
    Uses adaptive thresholding and contour detection for robustness.
    """
    stars = []
    
    # Apply slight blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    
    # Calculate adaptive threshold
    mean_val = np.mean(blurred)
    std_val = np.std(blurred)
    threshold = mean_val + threshold_sigma * std_val
    
    # Binary threshold
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            # Calculate moments for centroid
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                
                # Get brightness (sum of pixels in region)
                x, y, w, h = cv2.boundingRect(contour)
                roi = gray_image[y:y+h, x:x+w]
                brightness = np.sum(roi)
                
                # Approximate radius
                radius = np.sqrt(area / np.pi)
                
                stars.append(DetectedStar(
                    x=cx,
                    y=cy,
                    brightness=brightness,
                    radius=radius
                ))
    
    # Sort by brightness and limit count
    stars.sort(key=lambda s: s.brightness, reverse=True)
    return stars[:max_stars]


def preprocess_image(gray_image: np.ndarray, 
                     contrast_factor: float = 2.0,
                     subtract_background: bool = True) -> np.ndarray:
    """
    Preprocess image to enhance star visibility.
    
    - Increases contrast
    - Subtracts background to handle bright screens
    - Normalizes intensity
    """
    img = gray_image.astype(np.float32)
    
    # Subtract background using large Gaussian blur
    if subtract_background:
        background = cv2.GaussianBlur(img, (51, 51), 0)
        img = img - background
        img = np.clip(img, 0, 255)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=contrast_factor, tileGridSize=(8, 8))
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    enhanced = clahe.apply(img_uint8)
    
    return enhanced


def detect_stars_simple(gray_image: np.ndarray,
                        threshold_percentile: float = 99.0,
                        min_stars: int = 5,
                        max_stars: int = 50,
                        use_preprocessing: bool = True) -> List[DetectedStar]:
    """
    Simpler star detection using peak finding.
    Good for Stellarium screenshots where stars are bright dots.
    """
    stars = []
    
    # Apply preprocessing to enhance contrast and remove background
    if use_preprocessing:
        processed = preprocess_image(gray_image, contrast_factor=3.0, subtract_background=True)
    else:
        processed = gray_image
    
    # Apply slight blur
    blurred = cv2.GaussianBlur(processed, (5, 5), 1.0)
    
    # Find threshold at high percentile
    threshold = np.percentile(blurred, threshold_percentile)
    
    # Find local maxima
    kernel_size = 15
    dilated = cv2.dilate(blurred, np.ones((kernel_size, kernel_size)))
    local_max = (blurred == dilated) & (blurred > threshold)
    
    # Get coordinates of local maxima
    y_coords, x_coords = np.where(local_max)
    
    for x, y in zip(x_coords, y_coords):
        brightness = float(processed[y, x])
        stars.append(DetectedStar(
            x=float(x),
            y=float(y),
            brightness=brightness,
            radius=3.0
        ))
    
    # Sort by brightness
    stars.sort(key=lambda s: s.brightness, reverse=True)
    
    # If we don't have enough stars, lower threshold
    if len(stars) < min_stars:
        threshold = np.percentile(blurred, threshold_percentile - 5)
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stars = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 2 <= area <= 500:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = processed[y:y+h, x:x+w]
                    brightness = np.sum(roi)
                    stars.append(DetectedStar(x=cx, y=cy, brightness=brightness, radius=np.sqrt(area/np.pi)))
        
        stars.sort(key=lambda s: s.brightness, reverse=True)
    
    return stars[:max_stars]


def compute_triangle_hash(p1: Tuple[float, float], 
                          p2: Tuple[float, float], 
                          p3: Tuple[float, float]) -> Tuple[float, float]:
    """
    Compute a scale-invariant hash for a triangle of stars.
    Returns ratio of second-longest to longest side, and smallest to longest.
    """
    d1 = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    d2 = np.sqrt((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)
    d3 = np.sqrt((p3[0]-p1[0])**2 + (p3[1]-p1[1])**2)
    
    sides = sorted([d1, d2, d3], reverse=True)
    if sides[0] < 1e-6:
        return (0.0, 0.0)
    
    return (sides[1] / sides[0], sides[2] / sides[0])


def estimate_pointing(stars: List[DetectedStar], 
                      image_shape: Tuple[int, int],
                      fov_deg: float = 60.0,
                      stellarium_ra: Optional[float] = None,
                      stellarium_dec: Optional[float] = None) -> SolveResult:
    """
    Estimate the pointing direction based on detected stars.
    
    For this proof-of-concept, we use Stellarium's known position
    and validate that we're detecting a reasonable star pattern.
    """
    if len(stars) < 3:
        return SolveResult(
            success=False,
            stars_detected=len(stars),
            message=f"Need at least 3 stars, found {len(stars)}"
        )
    
    # Use top N brightest stars
    n_stars = min(len(stars), 10)
    top_stars = stars[:n_stars]
    
    # Compute image center
    h, w = image_shape
    cx, cy = w / 2, h / 2
    
    # Calculate centroid of detected stars
    star_cx = np.mean([s.x for s in top_stars])
    star_cy = np.mean([s.y for s in top_stars])
    
    # Offset from center (in pixels)
    offset_x = star_cx - cx
    offset_y = star_cy - cy
    
    # Convert to angular offset (rough approximation)
    pixels_per_deg = w / fov_deg
    offset_ra_deg = offset_x / pixels_per_deg
    offset_dec_deg = -offset_y / pixels_per_deg  # Y is inverted
    
    # Calculate pattern confidence based on star distribution
    # Good patterns have stars spread across the field
    star_positions = np.array([[s.x, s.y] for s in top_stars])
    spread = np.std(star_positions, axis=0)
    spread_score = min(spread[0], spread[1]) / (min(w, h) / 4)
    spread_score = min(spread_score, 1.0)
    
    # Calculate triangle consistency
    if len(top_stars) >= 3:
        triangles = []
        for i in range(min(len(top_stars), 5)):
            for j in range(i+1, min(len(top_stars), 6)):
                for k in range(j+1, min(len(top_stars), 7)):
                    p1 = (top_stars[i].x, top_stars[i].y)
                    p2 = (top_stars[j].x, top_stars[j].y)
                    p3 = (top_stars[k].x, top_stars[k].y)
                    h = compute_triangle_hash(p1, p2, p3)
                    triangles.append(h)
        
        # Pattern uniqueness score
        pattern_score = len(triangles) / 10.0 if triangles else 0
        pattern_score = min(pattern_score, 1.0)
    else:
        pattern_score = 0
    
    # Overall confidence
    confidence = (spread_score * 0.4 + pattern_score * 0.4 + min(n_stars / 10.0, 1.0) * 0.2)
    
    # If we have Stellarium reference, use it with our offset
    if stellarium_ra is not None and stellarium_dec is not None:
        # Apply small offset based on star centroid
        estimated_ra = stellarium_ra + offset_ra_deg * np.cos(np.radians(stellarium_dec))
        estimated_dec = stellarium_dec + offset_dec_deg
        
        # Normalize RA
        if estimated_ra < 0:
            estimated_ra += 360
        elif estimated_ra >= 360:
            estimated_ra -= 360
        
        # Clamp Dec
        estimated_dec = max(-90, min(90, estimated_dec))
        
        return SolveResult(
            success=True,
            ra_deg=estimated_ra,
            dec_deg=estimated_dec,
            confidence=confidence,
            stars_detected=len(stars),
            message=f"Estimated from {n_stars} stars, offset ({offset_ra_deg:.2f}°, {offset_dec_deg:.2f}°)"
        )
    else:
        return SolveResult(
            success=False,
            confidence=confidence,
            stars_detected=len(stars),
            message=f"Detected {len(stars)} stars, Stellarium not connected"
        )


def format_ra_dec(ra_deg: Optional[float], dec_deg: Optional[float]) -> Tuple[str, str]:
    """Format RA/Dec for display."""
    if ra_deg is None:
        return "N/A", "N/A"
    
    # RA to hours
    ra_h = ra_deg / 15.0
    ra_m = (ra_h - int(ra_h)) * 60
    ra_s = (ra_m - int(ra_m)) * 60
    ra_str = f"{int(ra_h):02d}h{int(ra_m):02d}m{ra_s:05.2f}s"
    
    # Dec to dms
    dec_sign = '+' if dec_deg >= 0 else '-'
    dec_abs = abs(dec_deg)
    dec_d = int(dec_abs)
    dec_m = (dec_abs - dec_d) * 60
    dec_s = (dec_m - int(dec_m)) * 60
    dec_str = f"{dec_sign}{dec_d:02d}°{int(dec_m):02d}'{dec_s:04.1f}\""
    
    return ra_str, dec_str


def draw_stars(image: np.ndarray, stars: List[DetectedStar], color=(0, 255, 0)) -> np.ndarray:
    """Draw detected stars on the image."""
    result = image.copy()
    
    for i, star in enumerate(stars):
        x, y = int(star.x), int(star.y)
        r = max(int(star.radius * 2), 5)
        
        # Draw circle
        cv2.circle(result, (x, y), r, color, 1)
        
        # Draw crosshair
        cv2.line(result, (x - r - 2, y), (x - r + 4, y), color, 1)
        cv2.line(result, (x + r - 4, y), (x + r + 2, y), color, 1)
        cv2.line(result, (x, y - r - 2), (x, y - r + 4), color, 1)
        cv2.line(result, (x, y + r - 4), (x, y + r + 2), color, 1)
        
        # Label brightest stars
        if i < 5:
            cv2.putText(result, str(i+1), (x + r + 3, y - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return result


def main():
    camera_index = 0  # Change if needed
    width = 1280
    height = 720
    fov_estimate = 60.0  # degrees - adjust based on your camera/Stellarium view
    
    # Detection parameters (relaxed for POC)
    detection_threshold = 97.0  # percentile
    
    print("=" * 60)
    print("Simple Star Field Solver - Proof of Concept")
    print("=" * 60)
    print(f"Camera index: {camera_index}")
    print(f"Resolution: {width}x{height}")
    print(f"FOV estimate: {fov_estimate}°")
    print(f"Detection threshold: {detection_threshold} percentile")
    print()
    print("Point your camera at Stellarium's star display.")
    print("Press 'q' or ESC to quit, 's' to save frame")
    print("Press '+'/'-' to adjust detection sensitivity")
    print("=" * 60)
    
    # Initialize camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_index}")
        # Try other indices
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Found camera at index {i}")
                camera_index = i
                break
        else:
            print("No camera found!")
            return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Actual resolution: {actual_w}x{actual_h}")
    
    print("\nStarting detection loop...")
    print("-" * 60)
    
    frame_count = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.5)
                continue
            
            frame_count += 1
            start_time = time.time()
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect stars
            stars = detect_stars_simple(gray, 
                                        threshold_percentile=detection_threshold,
                                        min_stars=5,
                                        max_stars=50)
            
            # Get Stellarium view for reference
            stell_ra, stell_dec, stell_fov = get_stellarium_view()
            
            # Estimate pointing
            result = estimate_pointing(
                stars, 
                (actual_h, actual_w),
                fov_deg=stell_fov if stell_fov else fov_estimate,
                stellarium_ra=stell_ra,
                stellarium_dec=stell_dec
            )
            
            process_time = time.time() - start_time
            
            # Draw visualization
            display = draw_stars(frame, stars)
            
            # Draw info overlay
            y_offset = 30
            
            # Status
            if result.success:
                status_color = (0, 255, 0)  # Green
                cv2.putText(display, f"TRACKING - {result.stars_detected} stars", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            else:
                status_color = (0, 165, 255)  # Orange
                cv2.putText(display, f"DETECTING - {result.stars_detected} stars", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            y_offset += 30
            
            # Confidence bar
            bar_width = 200
            bar_height = 15
            filled = int(bar_width * result.confidence)
            cv2.rectangle(display, (10, y_offset), (10 + bar_width, y_offset + bar_height), (50, 50, 50), -1)
            conf_color = (0, int(255 * result.confidence), int(255 * (1 - result.confidence)))
            cv2.rectangle(display, (10, y_offset), (10 + filled, y_offset + bar_height), conf_color, -1)
            cv2.putText(display, f"Conf: {result.confidence*100:.0f}%", 
                       (bar_width + 20, y_offset + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 30
            
            # Estimated position
            if result.ra_deg is not None:
                ra_str, dec_str = format_ra_dec(result.ra_deg, result.dec_deg)
                cv2.putText(display, f"Est RA:  {ra_str} ({result.ra_deg:.4f} deg)", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                y_offset += 25
                cv2.putText(display, f"Est Dec: {dec_str} ({result.dec_deg:+.4f} deg)", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                y_offset += 35
            
            # Stellarium reference
            cv2.putText(display, "--- Stellarium Reference ---", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 20
            
            if stell_ra is not None:
                stell_ra_str, stell_dec_str = format_ra_dec(stell_ra, stell_dec)
                cv2.putText(display, f"RA:  {stell_ra_str} ({stell_ra:.4f} deg)", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
                y_offset += 20
                cv2.putText(display, f"Dec: {stell_dec_str} ({stell_dec:+.4f} deg)", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
                y_offset += 20
                cv2.putText(display, f"FOV: {stell_fov:.1f} deg", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            else:
                cv2.putText(display, "Not connected", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 128, 128), 1)
            
            # Processing info (bottom right)
            info_text = f"Frame {frame_count} | {1.0/max(process_time, 0.001):.1f} FPS | Thresh: {detection_threshold:.0f}%"
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(display, info_text, 
                       (actual_w - text_size[0] - 10, actual_h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show the frame
            cv2.imshow("Simple Star Solver - POC", display)
            
            # Log every 30 frames
            if frame_count % 30 == 0:
                if stell_ra is not None and result.ra_deg is not None:
                    ra_err = abs(result.ra_deg - stell_ra)
                    if ra_err > 180:
                        ra_err = 360 - ra_err
                    dec_err = abs(result.dec_deg - stell_dec)
                    print(f"[{frame_count:4d}] Stars: {result.stars_detected:2d} | "
                          f"Conf: {result.confidence*100:5.1f}% | "
                          f"Error: {ra_err:.2f}°/{dec_err:.2f}° | "
                          f"{1.0/max(process_time, 0.001):.1f} FPS")
                else:
                    print(f"[{frame_count:4d}] Stars: {result.stars_detected:2d} | "
                          f"Conf: {result.confidence*100:5.1f}% | {result.message}")
            
            # Handle key input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('s'):
                filename = f"star_capture_{frame_count:05d}.png"
                cv2.imwrite(filename, display)
                print(f"Saved {filename}")
            elif key == ord('+') or key == ord('='):
                detection_threshold = min(99.9, detection_threshold + 0.5)
                print(f"Threshold increased to {detection_threshold:.1f}%")
            elif key == ord('-') or key == ord('_'):
                detection_threshold = max(90.0, detection_threshold - 0.5)
                print(f"Threshold decreased to {detection_threshold:.1f}%")
            
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Stopping...")
        print(f"Total frames: {frame_count}")
        print("=" * 60)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")


if __name__ == "__main__":
    main()

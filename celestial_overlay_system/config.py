"""
config.py - System Configuration

Edit this file to match your hardware setup.
"""

from ephemeris import ObserverLocation


# =============================================================================
# OBSERVER LOCATION
# =============================================================================
# IMPORTANT: Set this to your actual location!
# Without GPS, this must be manually configured for accurate celestial positions.

OBSERVER_LOCATION = ObserverLocation(
    latitude=41.0082,      # Degrees, positive = North, negative = South
    longitude=28.9784,     # Degrees, positive = East, negative = West  
    elevation=50.0         # Meters above sea level
)


# =============================================================================
# ORANGE CUBE (MAVLink)
# =============================================================================
# Find the correct COM port in Windows Device Manager under "Ports (COM & LPT)"

MAVLINK_PORT = 'COM3'      # e.g., 'COM3', 'COM4', etc.
MAVLINK_BAUD = 115200      # Usually 115200 for USB connection


# =============================================================================
# HARRIER 10x CAMERA
# =============================================================================
# Camera index is usually 0 for the first camera

CAMERA_INDEX = 0           # OpenCV camera index
CAMERA_WIDTH = 1920        # Requested width
CAMERA_HEIGHT = 1080       # Requested height


# =============================================================================
# VISCA ZOOM CONTROL (Optional)
# =============================================================================
# If you have VISCA control connected to the Harrier, set the COM port here.
# Set to None if not using VISCA control.

VISCA_PORT = None          # e.g., 'COM4' or None


# =============================================================================
# CAMERA-IMU MOUNTING
# =============================================================================
# These offsets correct for any misalignment between the Orange Cube
# and the Harrier camera. Values in degrees.

MOUNTING_ROLL_OFFSET = 0.0
MOUNTING_PITCH_OFFSET = 0.0
MOUNTING_YAW_OFFSET = 0.0


# =============================================================================
# DISPLAY OPTIONS
# =============================================================================

SHOW_HORIZON = True
SHOW_CARDINALS = True
SHOW_ALTITUDE_GRID = False
SHOW_LABELS = True
SHOW_CROSSHAIR = True
SHOW_INFO_PANEL = True

STAR_MAGNITUDE_LIMIT = 3.0  # Only show stars brighter than this


# =============================================================================
# SIMULATION MODE
# =============================================================================
# Set to True to run without hardware (for testing)

USE_SIMULATION = False
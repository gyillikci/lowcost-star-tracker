"""
IMU Reader Modules for Star Tracker.

Provides interfaces to various IMU sensors:
- WitMotion WT9011DCL (Bluetooth/USB)
- Orange Cube (MAVLink)

IMU Implementations:
    1. WitMotionReader - Custom implementation with full protocol support
    2. PyWitMotionAdapter - Uses pywitmotion library (github.com/askuric/pywitmotion)

Usage:
    # Use default implementation
    from imu import create_witmotion_reader
    reader = create_witmotion_reader(port='/dev/rfcomm0')

    # Or specify implementation
    reader = create_witmotion_reader(port='/dev/rfcomm0', use_pywitmotion=True)
"""

from .witmotion_reader import WitMotionReader, AttitudeData, IMUData, find_witmotion_ports

# Try to import pywitmotion adapter
try:
    from .pywitmotion_adapter import PyWitMotionAdapter, PYWITMOTION_AVAILABLE
except ImportError:
    PyWitMotionAdapter = None
    PYWITMOTION_AVAILABLE = False


def create_witmotion_reader(port: str = '/dev/rfcomm0',
                            baudrate: int = 115200,
                            use_pywitmotion: bool = False):
    """
    Create a WitMotion IMU reader.

    Factory function that allows switching between implementations.

    Args:
        port: Serial port path
        baudrate: Baud rate (default 115200)
        use_pywitmotion: If True, use pywitmotion library parser

    Returns:
        WitMotionReader or PyWitMotionAdapter instance
    """
    if use_pywitmotion:
        if PyWitMotionAdapter is not None and PYWITMOTION_AVAILABLE:
            return PyWitMotionAdapter(port=port, baudrate=baudrate)
        else:
            print("Warning: pywitmotion not available, falling back to WitMotionReader")

    return WitMotionReader(port=port, baudrate=baudrate)


__all__ = [
    # Main reader
    'WitMotionReader',
    # PyWitMotion adapter
    'PyWitMotionAdapter',
    'PYWITMOTION_AVAILABLE',
    # Data structures
    'AttitudeData',
    'IMUData',
    # Utilities
    'find_witmotion_ports',
    'create_witmotion_reader',
]

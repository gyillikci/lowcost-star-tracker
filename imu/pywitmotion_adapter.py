#!/usr/bin/env python3
"""
PyWitMotion Adapter - Integration with pywitmotion library.

This module provides an adapter that uses the pywitmotion library
(https://github.com/askuric/pywitmotion) for parsing WitMotion IMU data.

The pywitmotion library supports BWT901CL and similar WitMotion IMUs.
It has been tested to work with WT9011DCL as well (same protocol).

Usage:
    # Using pywitmotion parser with serial port
    adapter = PyWitMotionAdapter(port='/dev/rfcomm0')

    if adapter.connect():
        while True:
            if adapter.read_and_process():
                print(f"Angle: {adapter.get_angle()}")
                print(f"Quaternion: {adapter.get_quaternion()}")

References:
    - https://github.com/askuric/pywitmotion
    - BWT901CL Datasheet (compatible protocol)
"""

import sys
import time
import math
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import numpy as np

# Add local pywitmotion to path
_pywitmotion_path = Path(__file__).parent.parent / 'external' / 'pywitmotion'
if _pywitmotion_path.exists():
    sys.path.insert(0, str(_pywitmotion_path))

# Try to import pywitmotion
try:
    import pywitmotion as wit
    PYWITMOTION_AVAILABLE = True
except ImportError:
    PYWITMOTION_AVAILABLE = False
    wit = None

# Import serial
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    serial = None


@dataclass
class AttitudeData:
    """Container for IMU attitude data (compatible with OrangeCubeReader)."""
    roll: float = 0.0       # Roll angle in radians
    pitch: float = 0.0      # Pitch angle in radians
    yaw: float = 0.0        # Yaw angle in radians
    rollspeed: float = 0.0  # Roll rate in rad/s
    pitchspeed: float = 0.0 # Pitch rate in rad/s
    yawspeed: float = 0.0   # Yaw rate in rad/s
    timestamp: float = 0.0  # Unix timestamp


class PyWitMotionAdapter:
    """
    WitMotion IMU reader using pywitmotion library for parsing.

    Compatible with celestial_sphere_3d.py viewer interface.
    Supports BWT901CL, WT9011DCL, and similar WitMotion IMUs.
    """

    # Message header
    HEADER = b'U'  # 0x55

    def __init__(self, port: str = '/dev/rfcomm0', baudrate: int = 115200):
        """
        Initialize PyWitMotion adapter.

        Args:
            port: Serial port (e.g., '/dev/rfcomm0' for Bluetooth,
                  '/dev/ttyUSB0' for USB, 'COM3' for Windows)
            baudrate: Serial baudrate (default 115200 for WT9011DCL)
        """
        if not PYWITMOTION_AVAILABLE:
            raise ImportError(
                "pywitmotion not available.\n"
                "Clone from: https://github.com/askuric/pywitmotion"
            )

        if not SERIAL_AVAILABLE:
            raise ImportError("pyserial required: pip install pyserial")

        self.port = port
        self.baudrate = baudrate

        self._serial: Optional[serial.Serial] = None
        self.connected = False

        # Data containers
        self.attitude_data = AttitudeData()

        # Current sensor readings
        self._acceleration: Optional[np.ndarray] = None  # [ax, ay, az] in g
        self._gyro: Optional[np.ndarray] = None          # [wx, wy, wz] in deg/s
        self._angle: Optional[np.ndarray] = None         # [roll, pitch, yaw] in deg
        self._quaternion: Optional[np.ndarray] = None    # [x, y, z, w]
        self._magnetic: Optional[np.ndarray] = None      # [mx, my, mz]

        # Buffer for message parsing
        self._buffer = bytearray()

        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._reader_thread: Optional[threading.Thread] = None

        # Callback
        self._attitude_callback: Optional[Callable] = None

        # Statistics
        self.packets_received = 0
        self.errors = 0
        self.last_update_time = 0.0

    def connect(self) -> bool:
        """
        Connect to the IMU sensor.

        Returns:
            True if connection successful
        """
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0
            )

            # Clear any pending data
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()

            self.connected = True
            print(f"PyWitMotion: Connected to IMU on {self.port}")

            return True

        except serial.SerialException as e:
            print(f"PyWitMotion: Failed to connect to {self.port}: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from the IMU sensor."""
        self._running = False

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)

        if self._serial and self._serial.is_open:
            self._serial.close()

        self.connected = False
        print("PyWitMotion: Disconnected from IMU")

    def stop(self):
        """Alias for disconnect() - compatibility with OrangeCubeReader."""
        self.disconnect()

    def request_data_streams(self, rate_hz: int = 50):
        """
        Request data streams at specified rate.

        Note: Actual rate depends on sensor configuration.

        Args:
            rate_hz: Desired output rate in Hz (informational only)
        """
        print(f"PyWitMotion: Data stream requested at {rate_hz} Hz")

    def read_and_process(self) -> bool:
        """
        Read and process messages from the IMU.

        Uses pywitmotion library for parsing.

        Returns:
            True if new attitude data was received
        """
        if not self.connected or not self._serial:
            return False

        try:
            # Read until we get header byte
            if self._serial.in_waiting > 0:
                # Read available data
                data = self._serial.read(self._serial.in_waiting)
                self._buffer.extend(data)

            # Process complete messages
            attitude_updated = False

            # Split on header byte 'U' (0x55)
            while self.HEADER in self._buffer:
                idx = self._buffer.index(ord(self.HEADER))

                # Discard data before header
                if idx > 0:
                    self._buffer = self._buffer[idx:]

                # Check if we have enough data for a message
                # Format: U + cmd(1) + data(6-8) + checksum (total ~10-11 bytes)
                if len(self._buffer) < 11:
                    break

                # Extract potential message (skip header for pywitmotion)
                msg = bytes(self._buffer[1:11])  # After 'U', up to 10 bytes

                # Try parsing with pywitmotion
                parsed = False

                # Try quaternion (needs 9 bytes after cmd)
                q = wit.get_quaternion(msg)
                if q is not None:
                    with self._lock:
                        self._quaternion = q
                    parsed = True

                # Try angle (needs 7 bytes after cmd)
                angle = wit.get_angle(msg)
                if angle is not None:
                    with self._lock:
                        self._angle = angle
                        # Update attitude in radians
                        self.attitude_data.roll = math.radians(angle[0])
                        self.attitude_data.pitch = math.radians(angle[1])
                        self.attitude_data.yaw = math.radians(angle[2])
                        self.attitude_data.timestamp = time.time()
                    attitude_updated = True
                    self.last_update_time = time.time()

                    # Invoke callback
                    if self._attitude_callback:
                        self._attitude_callback(self.attitude_data)
                    parsed = True

                # Try gyro
                gyro = wit.get_gyro(msg)
                if gyro is not None:
                    with self._lock:
                        self._gyro = gyro
                        # Convert deg/s to rad/s
                        self.attitude_data.rollspeed = math.radians(gyro[0])
                        self.attitude_data.pitchspeed = math.radians(gyro[1])
                        self.attitude_data.yawspeed = math.radians(gyro[2])
                    parsed = True

                # Try acceleration
                accel = wit.get_acceleration(msg)
                if accel is not None:
                    with self._lock:
                        self._acceleration = accel
                    parsed = True

                # Try magnetic
                mag = wit.get_magnetic(msg)
                if mag is not None:
                    with self._lock:
                        self._magnetic = mag
                    parsed = True

                if parsed:
                    self.packets_received += 1
                    # Move past this message
                    self._buffer = self._buffer[11:]
                else:
                    # Skip this header and look for next
                    self._buffer = self._buffer[1:]
                    self.errors += 1

            return attitude_updated

        except Exception as e:
            print(f"PyWitMotion: Read error: {e}")
            self.errors += 1
            return False

    def start_continuous_read(self):
        """Start continuous reading in a background thread."""
        if self._running:
            return

        self._running = True
        self._reader_thread = threading.Thread(
            target=self._continuous_read_loop,
            daemon=True
        )
        self._reader_thread.start()
        print("PyWitMotion: Started continuous reading")

    def _continuous_read_loop(self):
        """Background thread for continuous reading."""
        while self._running and self.connected:
            try:
                self.read_and_process()
                time.sleep(0.001)  # 1ms poll
            except Exception as e:
                print(f"PyWitMotion: Read loop error: {e}")
                time.sleep(0.1)

    def set_attitude_callback(self, callback: Callable):
        """Set callback function for attitude updates."""
        self._attitude_callback = callback

    # Data accessor methods

    def get_euler_degrees(self) -> Tuple[float, float, float]:
        """Get Euler angles in degrees (roll, pitch, yaw)."""
        with self._lock:
            if self._angle is not None:
                return (float(self._angle[0]),
                        float(self._angle[1]),
                        float(self._angle[2]))
            return (0.0, 0.0, 0.0)

    def get_euler_radians(self) -> Tuple[float, float, float]:
        """Get Euler angles in radians."""
        with self._lock:
            return (
                self.attitude_data.roll,
                self.attitude_data.pitch,
                self.attitude_data.yaw
            )

    def get_angle(self) -> Optional[np.ndarray]:
        """Get angle array from pywitmotion [roll, pitch, yaw] in degrees."""
        with self._lock:
            return self._angle.copy() if self._angle is not None else None

    def get_gyro_radians(self) -> Tuple[float, float, float]:
        """Get gyroscope readings in rad/s."""
        with self._lock:
            return (
                self.attitude_data.rollspeed,
                self.attitude_data.pitchspeed,
                self.attitude_data.yawspeed
            )

    def get_gyro(self) -> Optional[np.ndarray]:
        """Get gyro array from pywitmotion [wx, wy, wz] in deg/s."""
        with self._lock:
            return self._gyro.copy() if self._gyro is not None else None

    def get_acceleration(self) -> Optional[np.ndarray]:
        """Get acceleration array from pywitmotion [ax, ay, az] in g."""
        with self._lock:
            return self._acceleration.copy() if self._acceleration is not None else None

    def get_magnetometer(self) -> Optional[np.ndarray]:
        """Get magnetometer array from pywitmotion."""
        with self._lock:
            return self._magnetic.copy() if self._magnetic is not None else None

    def get_quaternion(self) -> Optional[np.ndarray]:
        """Get quaternion array from pywitmotion [x, y, z, w]."""
        with self._lock:
            return self._quaternion.copy() if self._quaternion is not None else None

    def get_stats(self) -> dict:
        """Get reader statistics."""
        return {
            'packets_received': self.packets_received,
            'errors': self.errors,
            'last_update': self.last_update_time,
            'connected': self.connected,
            'port': self.port,
            'parser': 'pywitmotion'
        }


def check_pywitmotion():
    """Check if pywitmotion is available."""
    if PYWITMOTION_AVAILABLE:
        print("pywitmotion: Available")
        print(f"  Path: {_pywitmotion_path}")
        print("  Functions: get_quaternion, get_angle, get_gyro, get_acceleration, get_magnetic")
        return True
    else:
        print("pywitmotion: NOT AVAILABLE")
        print("  Clone from: https://github.com/askuric/pywitmotion")
        return False


def demo():
    """Demo function to test PyWitMotion adapter."""
    print("=" * 60)
    print("PyWitMotion Adapter Demo")
    print("=" * 60)

    # Check availability
    if not check_pywitmotion():
        return

    if not SERIAL_AVAILABLE:
        print("\npyserial not available: pip install pyserial")
        return

    # Find port
    from imu.witmotion_reader import find_witmotion_ports
    ports = find_witmotion_ports()

    if ports:
        print(f"\nFound potential ports: {ports}")
    else:
        print("\nNo WitMotion ports found automatically")

    port = ports[0] if ports else '/dev/rfcomm0'
    print(f"\nTrying to connect to {port}...")

    try:
        adapter = PyWitMotionAdapter(port=port)
    except ImportError as e:
        print(f"Import error: {e}")
        return

    if not adapter.connect():
        print("Connection failed.")
        return

    print("\nReading IMU data using pywitmotion parser...")
    print("Press Ctrl+C to stop")
    print("-" * 60)

    try:
        start_time = time.time()
        while time.time() - start_time < 30:
            if adapter.read_and_process():
                angle = adapter.get_angle()
                if angle is not None:
                    print(f"\rAngle: [{angle[0]:7.2f}, {angle[1]:7.2f}, {angle[2]:7.2f}]  "
                          f"Packets: {adapter.packets_received}", end='')

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        adapter.disconnect()

    print(f"\nStats: {adapter.get_stats()}")


if __name__ == "__main__":
    demo()

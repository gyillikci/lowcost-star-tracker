#!/usr/bin/env python3
"""
WitMotion WT9011DCL IMU Reader.

Provides an interface to the WitMotion WT9011DCL 9-axis IMU sensor
via Bluetooth (serial port) or USB connection.

The WT9011DCL provides:
- 3-axis accelerometer
- 3-axis gyroscope
- 3-axis magnetometer
- Euler angles (roll, pitch, yaw)
- Quaternion output

This module is compatible with the celestial_sphere_3d.py viewer.

References:
- https://github.com/enthusiasticgeek/witmotion_python_wt9011dcl
- https://github.com/askuric/pywitmotion

Usage:
    # Via Bluetooth (paired as rfcomm)
    reader = WitMotionReader(port='/dev/rfcomm0', baudrate=115200)

    # Via USB
    reader = WitMotionReader(port='/dev/ttyUSB0', baudrate=115200)

    if reader.connect():
        while True:
            if reader.read_and_process():
                print(reader.attitude_data)
"""

import time
import struct
import threading
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import serial


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


@dataclass
class IMUData:
    """Full IMU data container."""
    # Accelerometer [m/s²]
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0

    # Gyroscope [rad/s]
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0

    # Magnetometer [μT]
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0

    # Euler angles [degrees]
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    # Quaternion
    q0: float = 1.0
    q1: float = 0.0
    q2: float = 0.0
    q3: float = 0.0

    # Temperature [°C]
    temperature: float = 0.0

    timestamp: float = 0.0


# WitMotion protocol constants
WIT_HEADER = 0x55
WIT_TIME = 0x50
WIT_ACCEL = 0x51
WIT_GYRO = 0x52
WIT_ANGLE = 0x53
WIT_MAG = 0x54
WIT_DPORT = 0x55
WIT_PRESSURE = 0x56
WIT_GPS_LON = 0x57
WIT_GPS_LAT = 0x58
WIT_GPS_HEIGHT = 0x59
WIT_GPS_YAW = 0x5A
WIT_GPS_V = 0x5B
WIT_QUATERNION = 0x59


class WitMotionReader:
    """
    Reader for WitMotion WT9011DCL IMU sensor.

    Compatible with celestial_sphere_3d.py viewer interface.
    """

    def __init__(self, port: str = '/dev/rfcomm0', baudrate: int = 115200):
        """
        Initialize WitMotion IMU reader.

        Args:
            port: Serial port (e.g., '/dev/rfcomm0' for Bluetooth,
                  '/dev/ttyUSB0' for USB, 'COM3' for Windows)
            baudrate: Serial baudrate (default 115200 for WT9011DCL)
        """
        self.port = port
        self.baudrate = baudrate

        self.serial: Optional[serial.Serial] = None
        self.connected = False

        # Data containers
        self.attitude_data = AttitudeData()
        self.imu_data = IMUData()

        # Data buffer
        self._buffer = bytearray()

        # Callbacks
        self._attitude_callback: Optional[Callable] = None

        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._reader_thread: Optional[threading.Thread] = None

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
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0
            )

            # Clear any pending data
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()

            self.connected = True
            print(f"Connected to WitMotion IMU on {self.port}")

            return True

        except serial.SerialException as e:
            print(f"Failed to connect to {self.port}: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from the IMU sensor."""
        self._running = False

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)

        if self.serial and self.serial.is_open:
            self.serial.close()

        self.connected = False
        print("Disconnected from WitMotion IMU")

    def stop(self):
        """Alias for disconnect() - compatibility with OrangeCubeReader."""
        self.disconnect()

    def request_data_streams(self, rate_hz: int = 50):
        """
        Request data streams at specified rate.

        Note: WT9011DCL output rate is typically configured via
        the WitMotion app. This method is for API compatibility.

        Args:
            rate_hz: Desired output rate in Hz (informational only)
        """
        print(f"WitMotion IMU data stream requested at {rate_hz} Hz")
        print("Note: Actual rate depends on sensor configuration")

    def read_message(self, timeout: float = 0.1) -> Optional[bytes]:
        """
        Read a complete message from the IMU.

        Args:
            timeout: Read timeout in seconds

        Returns:
            Message bytes or None if no complete message
        """
        if not self.connected or not self.serial:
            return None

        try:
            # Read available data
            if self.serial.in_waiting > 0:
                data = self.serial.read(self.serial.in_waiting)
                self._buffer.extend(data)

            # Look for complete packet (11 bytes: 0x55 + type + 8 data + checksum)
            while len(self._buffer) >= 11:
                # Find header
                try:
                    header_idx = self._buffer.index(WIT_HEADER)

                    if header_idx > 0:
                        # Discard bytes before header
                        self._buffer = self._buffer[header_idx:]

                    if len(self._buffer) >= 11:
                        # Extract packet
                        packet = bytes(self._buffer[:11])

                        # Verify checksum
                        if self._verify_checksum(packet):
                            self._buffer = self._buffer[11:]
                            return packet
                        else:
                            # Invalid checksum, skip this header
                            self._buffer = self._buffer[1:]
                            self.errors += 1
                    else:
                        break

                except ValueError:
                    # No header found, clear buffer
                    self._buffer.clear()
                    break

        except serial.SerialException as e:
            print(f"Serial read error: {e}")
            self.errors += 1

        return None

    def process_message(self, msg: bytes) -> bool:
        """
        Process a received message and update attitude data.

        Args:
            msg: Raw message bytes (11 bytes)

        Returns:
            True if message was processed successfully
        """
        if len(msg) < 11:
            return False

        msg_type = msg[1]
        data = msg[2:10]

        try:
            if msg_type == WIT_ACCEL:
                self._parse_acceleration(data)

            elif msg_type == WIT_GYRO:
                self._parse_gyro(data)

            elif msg_type == WIT_ANGLE:
                self._parse_angle(data)
                self.packets_received += 1
                self.last_update_time = time.time()
                return True  # Angle is the main attitude update

            elif msg_type == WIT_MAG:
                self._parse_magnetometer(data)

            elif msg_type == WIT_QUATERNION:
                self._parse_quaternion(data)

            self.packets_received += 1
            return True

        except Exception as e:
            print(f"Parse error: {e}")
            self.errors += 1
            return False

    def read_and_process(self) -> bool:
        """
        Read and process a single message.

        Returns:
            True if attitude was updated
        """
        msg = self.read_message()
        if msg:
            return self.process_message(msg)
        return False

    def _verify_checksum(self, packet: bytes) -> bool:
        """Verify packet checksum."""
        if len(packet) != 11:
            return False
        checksum = sum(packet[0:10]) & 0xFF
        return checksum == packet[10]

    def _parse_acceleration(self, data: bytes):
        """Parse acceleration data (0x51)."""
        # Data format: AxL AxH AyL AyH AzL AzH TL TH
        ax = struct.unpack('<h', data[0:2])[0] / 32768.0 * 16 * 9.8
        ay = struct.unpack('<h', data[2:4])[0] / 32768.0 * 16 * 9.8
        az = struct.unpack('<h', data[4:6])[0] / 32768.0 * 16 * 9.8
        temp = struct.unpack('<h', data[6:8])[0] / 100.0

        with self._lock:
            self.imu_data.accel_x = ax
            self.imu_data.accel_y = ay
            self.imu_data.accel_z = az
            self.imu_data.temperature = temp

    def _parse_gyro(self, data: bytes):
        """Parse gyroscope data (0x52)."""
        # Data format: wxL wxH wyL wyH wzL wzH TL TH
        # Scale: 2000 deg/s range
        wx = struct.unpack('<h', data[0:2])[0] / 32768.0 * 2000
        wy = struct.unpack('<h', data[2:4])[0] / 32768.0 * 2000
        wz = struct.unpack('<h', data[4:6])[0] / 32768.0 * 2000

        # Convert to rad/s
        wx_rad = math.radians(wx)
        wy_rad = math.radians(wy)
        wz_rad = math.radians(wz)

        with self._lock:
            self.imu_data.gyro_x = wx_rad
            self.imu_data.gyro_y = wy_rad
            self.imu_data.gyro_z = wz_rad

            self.attitude_data.rollspeed = wx_rad
            self.attitude_data.pitchspeed = wy_rad
            self.attitude_data.yawspeed = wz_rad

    def _parse_angle(self, data: bytes):
        """Parse angle data (0x53)."""
        # Data format: RollL RollH PitchL PitchH YawL YawH VL VH
        roll = struct.unpack('<h', data[0:2])[0] / 32768.0 * 180
        pitch = struct.unpack('<h', data[2:4])[0] / 32768.0 * 180
        yaw = struct.unpack('<h', data[4:6])[0] / 32768.0 * 180
        version = struct.unpack('<h', data[6:8])[0]

        with self._lock:
            self.imu_data.roll_deg = roll
            self.imu_data.pitch_deg = pitch
            self.imu_data.yaw_deg = yaw

            # Update attitude_data in radians (for celestial_sphere_3d.py)
            self.attitude_data.roll = math.radians(roll)
            self.attitude_data.pitch = math.radians(pitch)
            self.attitude_data.yaw = math.radians(yaw)
            self.attitude_data.timestamp = time.time()

        # Invoke callback if set
        if self._attitude_callback:
            self._attitude_callback(self.attitude_data)

    def _parse_magnetometer(self, data: bytes):
        """Parse magnetometer data (0x54)."""
        # Data format: HxL HxH HyL HyH HzL HzH TL TH
        hx = struct.unpack('<h', data[0:2])[0]
        hy = struct.unpack('<h', data[2:4])[0]
        hz = struct.unpack('<h', data[4:6])[0]

        with self._lock:
            self.imu_data.mag_x = hx
            self.imu_data.mag_y = hy
            self.imu_data.mag_z = hz

    def _parse_quaternion(self, data: bytes):
        """Parse quaternion data (0x59)."""
        # Data format: q0L q0H q1L q1H q2L q2H q3L q3H
        q0 = struct.unpack('<h', data[0:2])[0] / 32768.0
        q1 = struct.unpack('<h', data[2:4])[0] / 32768.0
        q2 = struct.unpack('<h', data[4:6])[0] / 32768.0
        q3 = struct.unpack('<h', data[6:8])[0] / 32768.0

        with self._lock:
            self.imu_data.q0 = q0
            self.imu_data.q1 = q1
            self.imu_data.q2 = q2
            self.imu_data.q3 = q3

    def set_attitude_callback(self, callback: Callable):
        """Set callback function for attitude updates."""
        self._attitude_callback = callback

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
        print("Started continuous IMU reading")

    def _continuous_read_loop(self):
        """Background thread for continuous reading."""
        while self._running and self.connected:
            try:
                msg = self.read_message(timeout=0.05)
                if msg:
                    self.process_message(msg)
            except Exception as e:
                print(f"Read loop error: {e}")
                time.sleep(0.1)

    def get_euler_degrees(self) -> Tuple[float, float, float]:
        """Get Euler angles in degrees."""
        with self._lock:
            return (
                self.imu_data.roll_deg,
                self.imu_data.pitch_deg,
                self.imu_data.yaw_deg
            )

    def get_euler_radians(self) -> Tuple[float, float, float]:
        """Get Euler angles in radians."""
        with self._lock:
            return (
                self.attitude_data.roll,
                self.attitude_data.pitch,
                self.attitude_data.yaw
            )

    def get_gyro_radians(self) -> Tuple[float, float, float]:
        """Get gyroscope readings in rad/s."""
        with self._lock:
            return (
                self.imu_data.gyro_x,
                self.imu_data.gyro_y,
                self.imu_data.gyro_z
            )

    def get_acceleration(self) -> Tuple[float, float, float]:
        """Get acceleration in m/s²."""
        with self._lock:
            return (
                self.imu_data.accel_x,
                self.imu_data.accel_y,
                self.imu_data.accel_z
            )

    def get_magnetometer(self) -> Tuple[float, float, float]:
        """Get magnetometer readings."""
        with self._lock:
            return (
                self.imu_data.mag_x,
                self.imu_data.mag_y,
                self.imu_data.mag_z
            )

    def get_quaternion(self) -> Tuple[float, float, float, float]:
        """Get quaternion (w, x, y, z)."""
        with self._lock:
            return (
                self.imu_data.q0,
                self.imu_data.q1,
                self.imu_data.q2,
                self.imu_data.q3
            )

    def get_stats(self) -> dict:
        """Get reader statistics."""
        return {
            'packets_received': self.packets_received,
            'errors': self.errors,
            'last_update': self.last_update_time,
            'connected': self.connected,
            'port': self.port
        }

    def calibrate_accelerometer(self):
        """
        Send accelerometer calibration command.

        Note: Place sensor on flat surface before calling.
        """
        if not self.connected:
            print("Not connected")
            return

        # WitMotion unlock command
        unlock_cmd = bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5])
        # Accelerometer calibration command
        accel_cal_cmd = bytes([0xFF, 0xAA, 0x01, 0x01, 0x00])

        try:
            self.serial.write(unlock_cmd)
            time.sleep(0.1)
            self.serial.write(accel_cal_cmd)
            print("Accelerometer calibration command sent")
            print("Keep sensor level for 5 seconds...")
        except Exception as e:
            print(f"Calibration error: {e}")

    def calibrate_magnetometer(self):
        """
        Send magnetometer calibration start command.

        Rotate sensor 360° in all axes after calling.
        """
        if not self.connected:
            print("Not connected")
            return

        # WitMotion unlock command
        unlock_cmd = bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5])
        # Magnetometer calibration start
        mag_cal_cmd = bytes([0xFF, 0xAA, 0x01, 0x07, 0x00])

        try:
            self.serial.write(unlock_cmd)
            time.sleep(0.1)
            self.serial.write(mag_cal_cmd)
            print("Magnetometer calibration started")
            print("Rotate sensor 360° in all axes...")
            print("Call end_magnetometer_calibration() when done")
        except Exception as e:
            print(f"Calibration error: {e}")

    def end_magnetometer_calibration(self):
        """End magnetometer calibration."""
        if not self.connected:
            print("Not connected")
            return

        # Magnetometer calibration end
        mag_cal_end_cmd = bytes([0xFF, 0xAA, 0x01, 0x00, 0x00])

        try:
            self.serial.write(mag_cal_end_cmd)
            print("Magnetometer calibration ended")
        except Exception as e:
            print(f"Calibration error: {e}")

    def factory_reset(self):
        """Send factory reset command."""
        if not self.connected:
            print("Not connected")
            return

        # WitMotion unlock + factory reset
        unlock_cmd = bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5])
        reset_cmd = bytes([0xFF, 0xAA, 0x00, 0x01, 0x00])

        try:
            self.serial.write(unlock_cmd)
            time.sleep(0.1)
            self.serial.write(reset_cmd)
            print("Factory reset command sent")
        except Exception as e:
            print(f"Reset error: {e}")


def find_witmotion_ports() -> list:
    """
    Find potential WitMotion IMU serial ports.

    Returns:
        List of port names that might be WitMotion IMUs
    """
    import serial.tools.list_ports

    ports = []

    for port in serial.tools.list_ports.comports():
        # Common WitMotion identifiers
        if any(x in port.description.lower() for x in
               ['witmotion', 'ch340', 'cp210', 'ftdi', 'usb serial', 'bluetooth']):
            ports.append(port.device)
        # Also include rfcomm devices (Bluetooth)
        if 'rfcomm' in port.device.lower():
            ports.append(port.device)

    return ports


def demo():
    """Demo function to test WitMotion IMU."""
    print("=" * 60)
    print("WitMotion WT9011DCL IMU Demo")
    print("=" * 60)

    # Find available ports
    print("\nSearching for WitMotion IMU...")
    ports = find_witmotion_ports()

    if ports:
        print(f"Found potential ports: {ports}")
    else:
        print("No WitMotion ports found automatically")
        print("Common ports:")
        print("  Linux Bluetooth: /dev/rfcomm0")
        print("  Linux USB: /dev/ttyUSB0")
        print("  Windows: COM3, COM4, etc.")

    # Try to connect
    port = ports[0] if ports else '/dev/rfcomm0'
    print(f"\nTrying to connect to {port}...")

    reader = WitMotionReader(port=port)

    if not reader.connect():
        print("Connection failed. Make sure:")
        print("  1. IMU is powered on")
        print("  2. Bluetooth is paired (for BLE): sudo rfcomm bind 0 <MAC_ADDRESS>")
        print("  3. User has permission: sudo usermod -a -G dialout $USER")
        return

    print("\nReading IMU data (press Ctrl+C to stop)...")
    print("-" * 60)

    try:
        start_time = time.time()
        while time.time() - start_time < 30:  # Run for 30 seconds
            if reader.read_and_process():
                roll, pitch, yaw = reader.get_euler_degrees()
                gx, gy, gz = reader.get_gyro_radians()

                print(f"\rRoll: {roll:7.2f}°  Pitch: {pitch:7.2f}°  Yaw: {yaw:7.2f}°  "
                      f"Gyro: [{gx:6.3f}, {gy:6.3f}, {gz:6.3f}] rad/s  "
                      f"Packets: {reader.packets_received}", end='')

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        reader.disconnect()

    print(f"\nStats: {reader.get_stats()}")


if __name__ == "__main__":
    demo()

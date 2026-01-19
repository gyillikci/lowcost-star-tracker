#!/usr/bin/env python3
"""
Orange Cube Flight Controller - MAVLink Data Reader

Reads IMU, gyroscope, accelerometer, and attitude data from
an Orange Cube (or any ArduPilot-based) flight controller via MAVLink.

Requirements:
    pip install pymavlink pyserial matplotlib

Usage:
    python orange_cube_reader.py              # Auto-detect COM port
    python orange_cube_reader.py COM3         # Specify COM port
    python orange_cube_reader.py --plot       # Real-time attitude plot
"""

import sys
import time
import serial.tools.list_ports
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import deque
import threading

# MAVLink imports
from pymavlink import mavutil


@dataclass
class IMUData:
    """Container for IMU sensor data."""
    timestamp_ms: int = 0
    
    # Gyroscope (rad/s)
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    
    # Accelerometer (m/s²)
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    
    # Magnetometer (milligauss)
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0
    
    @property
    def gyro_deg_s(self) -> Tuple[float, float, float]:
        """Gyroscope in degrees/second."""
        return (
            np.degrees(self.gyro_x),
            np.degrees(self.gyro_y),
            np.degrees(self.gyro_z)
        )


@dataclass
class AttitudeData:
    """Container for attitude (orientation) data."""
    timestamp_ms: int = 0
    
    # Euler angles (radians)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    # Angular rates (rad/s)
    rollspeed: float = 0.0
    pitchspeed: float = 0.0
    yawspeed: float = 0.0
    
    # Quaternion (w, x, y, z)
    q_w: float = 1.0
    q_x: float = 0.0
    q_y: float = 0.0
    q_z: float = 0.0
    
    @property
    def euler_deg(self) -> Tuple[float, float, float]:
        """Euler angles in degrees."""
        return (
            np.degrees(self.roll),
            np.degrees(self.pitch),
            np.degrees(self.yaw)
        )
    
    @property
    def quaternion(self) -> Tuple[float, float, float, float]:
        """Quaternion as (w, x, y, z)."""
        return (self.q_w, self.q_x, self.q_y, self.q_z)
    
    @property
    def quaternion_normalized(self) -> Tuple[float, float, float, float]:
        """Normalized quaternion."""
        norm = np.sqrt(self.q_w**2 + self.q_x**2 + self.q_y**2 + self.q_z**2)
        if norm > 0:
            return (self.q_w/norm, self.q_x/norm, self.q_y/norm, self.q_z/norm)
        return (1.0, 0.0, 0.0, 0.0)


class OrangeCubeReader:
    """
    MAVLink reader for Orange Cube flight controller.
    
    Supports reading:
    - RAW_IMU: Raw accelerometer/gyro/magnetometer data
    - SCALED_IMU: Scaled sensor data
    - ATTITUDE: Current attitude (roll, pitch, yaw)
    - ATTITUDE_QUATERNION: Attitude as quaternion
    - HIGHRES_IMU: High resolution IMU data
    """
    
    def __init__(self, 
                 port: Optional[str] = None, 
                 baudrate: int = 115200,
                 source_system: int = 255,
                 source_component: int = 0):
        """
        Initialize connection to flight controller.
        
        Args:
            port: Serial port (e.g., 'COM3' or '/dev/ttyUSB0'). 
                  If None, will attempt auto-detection.
            baudrate: Serial baud rate (default 115200 for USB)
            source_system: MAVLink source system ID
            source_component: MAVLink source component ID
        """
        self.port = port
        self.baudrate = baudrate
        self.source_system = source_system
        self.source_component = source_component
        
        self.connection: Optional[mavutil.mavlink_connection] = None
        self.is_connected = False
        
        # Latest data
        self.imu_data = IMUData()
        self.attitude_data = AttitudeData()
        
        # Data buffers for logging
        self.imu_history: List[IMUData] = []
        self.attitude_history: List[AttitudeData] = []
        self.max_history = 10000  # Max samples to keep
        
    def find_serial_ports(self) -> List[Dict]:
        """
        Find available serial ports that might be the Orange Cube.
        
        Returns:
            List of port info dictionaries
        """
        ports = []
        for port in serial.tools.list_ports.comports():
            port_info = {
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid,
                'manufacturer': port.manufacturer,
                'product': port.product,
                'vid': port.vid,
                'pid': port.pid,
            }
            ports.append(port_info)
            
            # Check for typical ArduPilot/PX4 identifiers
            desc_lower = (port.description or '').lower()
            if any(x in desc_lower for x in ['cube', 'pixhawk', 'ardupilot', 'px4', 'stm32', 'fmu']):
                port_info['likely_fc'] = True
            else:
                port_info['likely_fc'] = False
                
        return ports
    
    def auto_detect_port(self) -> Optional[str]:
        """
        Attempt to auto-detect the flight controller port.
        
        Returns:
            Port device string or None if not found
        """
        ports = self.find_serial_ports()
        
        # First, try ports that look like flight controllers
        for port in ports:
            if port.get('likely_fc'):
                return port['device']
        
        # If no obvious FC, return first available port
        if ports:
            return ports[0]['device']
            
        return None
    
    def connect(self) -> bool:
        """
        Establish MAVLink connection to the flight controller.
        
        Returns:
            True if connection successful
        """
        # Auto-detect port if not specified
        if self.port is None:
            self.port = self.auto_detect_port()
            if self.port is None:
                print("ERROR: No serial ports found!")
                return False
            print(f"Auto-detected port: {self.port}")
        
        try:
            # Create MAVLink connection
            connection_string = f"{self.port}"
            print(f"Connecting to {connection_string} at {self.baudrate} baud...")
            
            self.connection = mavutil.mavlink_connection(
                connection_string,
                baud=self.baudrate,
                source_system=self.source_system,
                source_component=self.source_component
            )
            
            # Wait for heartbeat to confirm connection
            print("Waiting for heartbeat...")
            heartbeat = self.connection.wait_heartbeat(timeout=10)
            
            if heartbeat:
                print(f"✓ Connected to system {self.connection.target_system}, "
                      f"component {self.connection.target_component}")
                print(f"  Vehicle type: {mavutil.mavlink.enums['MAV_TYPE'][heartbeat.type].name}")
                print(f"  Autopilot: {mavutil.mavlink.enums['MAV_AUTOPILOT'][heartbeat.autopilot].name}")
                self.is_connected = True
                return True
            else:
                print("ERROR: No heartbeat received!")
                return False
                
        except Exception as e:
            print(f"ERROR: Connection failed: {e}")
            return False
    
    def request_data_streams(self, rate_hz: int = 50):
        """
        Request specific data streams from the flight controller.
        
        Args:
            rate_hz: Desired update rate in Hz
        """
        if not self.connection:
            return
            
        # Request all data streams at specified rate
        # MAV_DATA_STREAM_ALL = 0
        # MAV_DATA_STREAM_RAW_SENSORS = 1
        # MAV_DATA_STREAM_EXTENDED_STATUS = 2
        # MAV_DATA_STREAM_RC_CHANNELS = 3
        # MAV_DATA_STREAM_RAW_CONTROLLER = 4
        # MAV_DATA_STREAM_POSITION = 6
        # MAV_DATA_STREAM_EXTRA1 = 10 (attitude)
        # MAV_DATA_STREAM_EXTRA2 = 11
        # MAV_DATA_STREAM_EXTRA3 = 12
        
        streams = [
            (1, rate_hz),   # RAW_SENSORS - IMU data
            (10, rate_hz),  # EXTRA1 - Attitude
        ]
        
        for stream_id, rate in streams:
            self.connection.mav.request_data_stream_send(
                self.connection.target_system,
                self.connection.target_component,
                stream_id,
                rate,
                1  # Start sending
            )
        
        print(f"Requested data streams at {rate_hz} Hz")
    
    def read_message(self, timeout: float = 1.0) -> Optional[object]:
        """
        Read a single MAVLink message.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            MAVLink message or None
        """
        if not self.connection:
            return None
            
        msg = self.connection.recv_match(blocking=True, timeout=timeout)
        return msg
    
    def process_message(self, msg) -> Optional[str]:
        """
        Process a MAVLink message and update internal state.
        
        Args:
            msg: MAVLink message
            
        Returns:
            Message type string or None
        """
        if msg is None:
            return None
            
        msg_type = msg.get_type()
        
        if msg_type == 'RAW_IMU':
            self.imu_data = IMUData(
                timestamp_ms=msg.time_usec // 1000,
                # Raw values need scaling - typically divide by 1000 for m/s² and rad/s
                accel_x=msg.xacc / 1000.0,
                accel_y=msg.yacc / 1000.0,
                accel_z=msg.zacc / 1000.0,
                gyro_x=msg.xgyro / 1000.0,
                gyro_y=msg.ygyro / 1000.0,
                gyro_z=msg.zgyro / 1000.0,
                mag_x=msg.xmag,
                mag_y=msg.ymag,
                mag_z=msg.zmag,
            )
            if len(self.imu_history) < self.max_history:
                self.imu_history.append(self.imu_data)
            return 'RAW_IMU'
            
        elif msg_type == 'SCALED_IMU':
            self.imu_data = IMUData(
                timestamp_ms=msg.time_boot_ms,
                accel_x=msg.xacc / 1000.0,  # mg to m/s²
                accel_y=msg.yacc / 1000.0,
                accel_z=msg.zacc / 1000.0,
                gyro_x=msg.xgyro / 1000.0,  # mrad/s to rad/s
                gyro_y=msg.ygyro / 1000.0,
                gyro_z=msg.zgyro / 1000.0,
                mag_x=msg.xmag,
                mag_y=msg.ymag,
                mag_z=msg.zmag,
            )
            if len(self.imu_history) < self.max_history:
                self.imu_history.append(self.imu_data)
            return 'SCALED_IMU'
            
        elif msg_type == 'HIGHRES_IMU':
            self.imu_data = IMUData(
                timestamp_ms=int(msg.time_usec / 1000),
                accel_x=msg.xacc,
                accel_y=msg.yacc,
                accel_z=msg.zacc,
                gyro_x=msg.xgyro,
                gyro_y=msg.ygyro,
                gyro_z=msg.zgyro,
                mag_x=msg.xmag,
                mag_y=msg.ymag,
                mag_z=msg.zmag,
            )
            if len(self.imu_history) < self.max_history:
                self.imu_history.append(self.imu_data)
            return 'HIGHRES_IMU'
            
        elif msg_type == 'ATTITUDE':
            self.attitude_data = AttitudeData(
                timestamp_ms=msg.time_boot_ms,
                roll=msg.roll,
                pitch=msg.pitch,
                yaw=msg.yaw,
                rollspeed=msg.rollspeed,
                pitchspeed=msg.pitchspeed,
                yawspeed=msg.yawspeed,
            )
            if len(self.attitude_history) < self.max_history:
                self.attitude_history.append(self.attitude_data)
            return 'ATTITUDE'
            
        elif msg_type == 'ATTITUDE_QUATERNION':
            # Full attitude with quaternion
            self.attitude_data = AttitudeData(
                timestamp_ms=msg.time_boot_ms,
                roll=msg.roll,
                pitch=msg.pitch,
                yaw=msg.yaw,
                rollspeed=msg.rollspeed,
                pitchspeed=msg.pitchspeed,
                yawspeed=msg.yawspeed,
                q_w=msg.q[0],
                q_x=msg.q[1],
                q_y=msg.q[2],
                q_z=msg.q[3],
            )
            if len(self.attitude_history) < self.max_history:
                self.attitude_history.append(self.attitude_data)
            return 'ATTITUDE_QUATERNION'
            
        return msg_type
    
    def read_imu(self, timeout: float = 1.0) -> Optional[IMUData]:
        """
        Read until we get IMU data.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            IMUData or None
        """
        start = time.time()
        while time.time() - start < timeout:
            msg = self.read_message(timeout=0.1)
            msg_type = self.process_message(msg)
            if msg_type in ['RAW_IMU', 'SCALED_IMU', 'HIGHRES_IMU']:
                return self.imu_data
        return None
    
    def read_attitude(self, timeout: float = 1.0) -> Optional[AttitudeData]:
        """
        Read until we get attitude data.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            AttitudeData or None
        """
        start = time.time()
        while time.time() - start < timeout:
            msg = self.read_message(timeout=0.1)
            msg_type = self.process_message(msg)
            if msg_type == 'ATTITUDE':
                return self.attitude_data
        return None
    
    def stream_data(self, duration: float = 10.0, rate_hz: int = 50):
        """
        Stream and display data for a specified duration.
        
        Args:
            duration: Duration in seconds
            rate_hz: Requested data rate
        """
        self.request_data_streams(rate_hz)
        
        print(f"\n{'='*70}")
        print("Streaming IMU and Attitude data (Press Ctrl+C to stop)")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        msg_count = 0
        imu_count = 0
        att_count = 0
        
        try:
            while time.time() - start_time < duration:
                msg = self.read_message(timeout=0.1)
                if msg is None:
                    continue
                    
                msg_type = self.process_message(msg)
                msg_count += 1
                
                if msg_type in ['RAW_IMU', 'SCALED_IMU', 'HIGHRES_IMU']:
                    imu_count += 1
                    gyro = self.imu_data.gyro_deg_s
                    print(f"[IMU] Gyro: ({gyro[0]:+10.5f}, {gyro[1]:+10.5f}, {gyro[2]:+10.5f}) deg/s | "
                          f"Accel: ({self.imu_data.accel_x:+9.4f}, {self.imu_data.accel_y:+9.4f}, {self.imu_data.accel_z:+9.4f}) m/s²")
                          
                elif msg_type == 'ATTITUDE':
                    att_count += 1
                    euler = self.attitude_data.euler_deg
                    print(f"[ATT] Roll: {euler[0]:+10.5f}° | Pitch: {euler[1]:+10.5f}° | Yaw: {euler[2]:+10.5f}°")
                
                elif msg_type == 'ATTITUDE_QUATERNION':
                    att_count += 1
                    euler = self.attitude_data.euler_deg
                    q = self.attitude_data.quaternion
                    print(f"[ATT] Roll: {euler[0]:+10.5f}° | Pitch: {euler[1]:+10.5f}° | Yaw: {euler[2]:+10.5f}° | "
                          f"Quat: ({q[0]:+.6f}, {q[1]:+.6f}, {q[2]:+.6f}, {q[3]:+.6f})")
                    
        except KeyboardInterrupt:
            print("\n\nStreaming stopped by user.")
        
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Summary: {msg_count} messages in {elapsed:.1f}s ({msg_count/elapsed:.1f} msg/s)")
        print(f"  IMU messages: {imu_count} ({imu_count/elapsed:.1f} Hz)")
        print(f"  Attitude messages: {att_count} ({att_count/elapsed:.1f} Hz)")
        print(f"{'='*70}")
    
    def close(self):
        """Close the MAVLink connection."""
        if self.connection:
            self.connection.close()
            self.is_connected = False
            print("Connection closed.")


def list_ports():
    """List all available serial ports."""
    print("\n" + "="*60)
    print("Available Serial Ports")
    print("="*60)
    
    reader = OrangeCubeReader()
    ports = reader.find_serial_ports()
    
    if not ports:
        print("No serial ports found!")
        return
    
    for port in ports:
        fc_marker = " ★ [Likely Flight Controller]" if port.get('likely_fc') else ""
        print(f"\n  Device: {port['device']}{fc_marker}")
        print(f"    Description: {port['description']}")
        if port['manufacturer']:
            print(f"    Manufacturer: {port['manufacturer']}")
        if port['product']:
            print(f"    Product: {port['product']}")
        if port['vid'] and port['pid']:
            print(f"    VID:PID: {port['vid']:04X}:{port['pid']:04X}")
    
    print("\n" + "="*60)


class RealTimePlotter:
    """
    Real-time matplotlib plotter for attitude data.
    Uses background thread for data acquisition to eliminate delay.
    """
    
    def __init__(self, max_points: int = 500):
        self.max_points = max_points
        self.times = deque(maxlen=max_points)
        self.rolls = deque(maxlen=max_points)
        self.pitches = deque(maxlen=max_points)
        self.yaws = deque(maxlen=max_points)
        self.gyro_x = deque(maxlen=max_points)
        self.gyro_y = deque(maxlen=max_points)
        self.gyro_z = deque(maxlen=max_points)
        
        # Quaternion data
        self.q_w = deque(maxlen=max_points)
        self.q_x = deque(maxlen=max_points)
        self.q_y = deque(maxlen=max_points)
        self.q_z = deque(maxlen=max_points)
        
        self.start_time = time.time()
        self.running = False
        self.fig = None
        self.axes = None
        self.lock = threading.Lock()
        self.reader = None
    
    def update_data(self, attitude: AttitudeData, imu: IMUData):
        """Add new data point (thread-safe)."""
        t = time.time() - self.start_time
        
        with self.lock:
            self.times.append(t)
            
            euler = attitude.euler_deg
            self.rolls.append(euler[0])
            self.pitches.append(euler[1])
            self.yaws.append(euler[2])
            
            gyro = imu.gyro_deg_s
            self.gyro_x.append(gyro[0])
            self.gyro_y.append(gyro[1])
            self.gyro_z.append(gyro[2])
            
            self.q_w.append(attitude.q_w)
            self.q_x.append(attitude.q_x)
            self.q_y.append(attitude.q_y)
            self.q_z.append(attitude.q_z)
    
    def data_reader_thread(self):
        """Background thread to continuously read MAVLink data."""
        while self.running:
            try:
                msg = self.reader.read_message(timeout=0.001)
                if msg:
                    msg_type = self.reader.process_message(msg)
                    if msg_type in ['ATTITUDE', 'ATTITUDE_QUATERNION']:
                        self.update_data(self.reader.attitude_data, self.reader.imu_data)
            except Exception:
                pass
    
    def run(self, reader: OrangeCubeReader, duration: float = 60.0):
        """Run the real-time plot."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        self.running = True
        self.start_time = time.time()
        self.reader = reader
        
        # Request data streams at high rate
        reader.request_data_streams(100)
        
        # Start background data reader thread
        data_thread = threading.Thread(target=self.data_reader_thread, daemon=True)
        data_thread.start()
        
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 9))
        self.fig.suptitle('Orange Cube IMU - Real-Time Attitude', fontsize=14, fontweight='bold')
        
        # Euler angles plot
        ax1 = self.axes[0]
        ax1.set_ylabel('Euler Angles (°)')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(-180, 180)
        ax1.grid(True, alpha=0.3)
        line_roll, = ax1.plot([], [], 'r-', label='Roll', linewidth=1.5)
        line_pitch, = ax1.plot([], [], 'g-', label='Pitch', linewidth=1.5)
        line_yaw, = ax1.plot([], [], 'b-', label='Yaw', linewidth=1.5)
        ax1.legend(loc='upper right')
        
        # Gyroscope plot
        ax2 = self.axes[1]
        ax2.set_ylabel('Angular Rate (°/s)')
        ax2.set_xlim(0, 10)
        ax2.set_ylim(-50, 50)
        ax2.grid(True, alpha=0.3)
        line_gx, = ax2.plot([], [], 'r-', label='Gyro X', linewidth=1)
        line_gy, = ax2.plot([], [], 'g-', label='Gyro Y', linewidth=1)
        line_gz, = ax2.plot([], [], 'b-', label='Gyro Z', linewidth=1)
        ax2.legend(loc='upper right')
        
        # Quaternion plot
        ax3 = self.axes[2]
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Quaternion')
        ax3.set_xlim(0, 10)
        ax3.set_ylim(-1.1, 1.1)
        ax3.grid(True, alpha=0.3)
        line_qw, = ax3.plot([], [], 'k-', label='q_w', linewidth=1.5)
        line_qx, = ax3.plot([], [], 'r-', label='q_x', linewidth=1)
        line_qy, = ax3.plot([], [], 'g-', label='q_y', linewidth=1)
        line_qz, = ax3.plot([], [], 'b-', label='q_z', linewidth=1)
        ax3.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Status text
        status_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                               fontsize=10, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        def init():
            return line_roll, line_pitch, line_yaw, line_gx, line_gy, line_gz, line_qw, line_qx, line_qy, line_qz, status_text
        
        def update(frame):
            # Get data snapshot (thread-safe)
            with self.lock:
                if len(self.times) < 2:
                    return line_roll, line_pitch, line_yaw, line_gx, line_gy, line_gz, line_qw, line_qx, line_qy, line_qz, status_text
                
                times = list(self.times)
                rolls = list(self.rolls)
                pitches = list(self.pitches)
                yaws = list(self.yaws)
                gyro_x = list(self.gyro_x)
                gyro_y = list(self.gyro_y)
                gyro_z = list(self.gyro_z)
                qw = list(self.q_w)
                qx = list(self.q_x)
                qy = list(self.q_y)
                qz = list(self.q_z)
            
            # Update x-axis limits to scroll
            if times[-1] > 10:
                for ax in self.axes:
                    ax.set_xlim(times[-1] - 10, times[-1])
            
            # Update Euler lines
            line_roll.set_data(times, rolls)
            line_pitch.set_data(times, pitches)
            line_yaw.set_data(times, yaws)
            
            # Update Gyro lines
            line_gx.set_data(times, gyro_x)
            line_gy.set_data(times, gyro_y)
            line_gz.set_data(times, gyro_z)
            
            # Update Quaternion lines
            line_qw.set_data(times, qw)
            line_qx.set_data(times, qx)
            line_qy.set_data(times, qy)
            line_qz.set_data(times, qz)
            
            # Update status text
            status_text.set_text(f'Roll: {rolls[-1]:+.4f}°  Pitch: {pitches[-1]:+.4f}°  Yaw: {yaws[-1]:+.4f}°')
            
            return line_roll, line_pitch, line_yaw, line_gx, line_gy, line_gz, line_qw, line_qx, line_qy, line_qz, status_text
        
        # Animation
        ani = FuncAnimation(self.fig, update, init_func=init, 
                           interval=20, blit=True, cache_frame_data=False)
        
        print("\nReal-time plot started. Close the window to stop.")
        plt.show()
        
        self.running = False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Read IMU/Gyro data from Orange Cube flight controller via MAVLink"
    )
    parser.add_argument('port', nargs='?', default=None,
                        help='Serial port (e.g., COM3, /dev/ttyUSB0). Auto-detects if not specified.')
    parser.add_argument('--baud', '-b', type=int, default=115200,
                        help='Baud rate (default: 115200)')
    parser.add_argument('--rate', '-r', type=int, default=50,
                        help='Data request rate in Hz (default: 50)')
    parser.add_argument('--duration', '-d', type=float, default=30.0,
                        help='Streaming duration in seconds (default: 30)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available serial ports and exit')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Show real-time attitude plot')
    
    args = parser.parse_args()
    
    if args.list:
        list_ports()
        return
    
    print("\n" + "="*60)
    print("Orange Cube MAVLink Reader")
    print("="*60)
    
    # List ports first
    list_ports()
    
    # Create reader and connect
    reader = OrangeCubeReader(port=args.port, baudrate=args.baud)
    
    if not reader.connect():
        print("\nFailed to connect. Please check:")
        print("  1. Orange Cube is connected via USB")
        print("  2. Correct COM port is selected")
        print("  3. No other application (Mission Planner, QGC) is using the port")
        return
    
    try:
        if args.plot:
            # Real-time plot mode
            plotter = RealTimePlotter()
            plotter.run(reader, duration=args.duration)
        else:
            # Text stream mode
            reader.stream_data(duration=args.duration, rate_hz=args.rate)
        
    finally:
        reader.close()


if __name__ == '__main__':
    main()

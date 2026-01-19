#!/usr/bin/env python3
"""Test WitMotion on COM8 with pywitmotion protocol."""

import sys
sys.path.insert(0, 'external/pywitmotion')

import serial
import time
from pywitmotion import pywitmotion

port = 'COM8'
baudrate = 115200

print(f"Opening {port} at {baudrate} baud...")
ser = serial.Serial(port, baudrate, timeout=1)
time.sleep(0.5)
ser.reset_input_buffer()

print("Reading data for 5 seconds...")
print("Expecting packets with 'Q' (accel), 'R' (gyro), 'S' (angle), 'T' (mag), 'Y' (quat)")

start = time.time()
packet_count = 0

while time.time() - start < 5:
    if ser.in_waiting > 0:
        # Read potential packet (11 bytes)
        data = ser.read(11)
        
        if len(data) >= 11:
            # Try to parse with pywitmotion
            angle = pywitmotion.get_angle(data)
            if angle is not None:
                packet_count += 1
                print(f"Angle: Roll={angle[0]:6.1f}°  Pitch={angle[1]:6.1f}°  Yaw={angle[2]:6.1f}°")
            
            accel = pywitmotion.get_acceleration(data)
            if accel is not None:
                print(f"Accel: X={accel[0]:6.2f}  Y={accel[1]:6.2f}  Z={accel[2]:6.2f} g")
            
            gyro = pywitmotion.get_gyro(data)
            if gyro is not None:
                print(f"Gyro: X={gyro[0]:7.1f}  Y={gyro[1]:7.1f}  Z={gyro[2]:7.1f} deg/s")
    
    time.sleep(0.01)

print(f"\nReceived {packet_count} packets")
ser.close()
print("Done")

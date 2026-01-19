#!/usr/bin/env python3
"""
Find WitMotion IMU COM ports on Windows.

Usage:
    python find_witmotion_windows.py
"""

import sys

def find_com_ports():
    """Find available COM ports on Windows."""
    try:
        import serial.tools.list_ports
    except ImportError:
        print("PySerial not installed. Run: pip install pyserial")
        return []

    print("=" * 60)
    print("Searching for COM ports...")
    print("=" * 60)

    ports = list(serial.tools.list_ports.comports())

    if not ports:
        print("\nNo COM ports found!")
        print("\nTroubleshooting:")
        print("  1. Make sure the WitMotion IMU is powered on")
        print("  2. Pair it via Windows Settings > Bluetooth")
        print("  3. Check Device Manager > Ports (COM & LPT)")
        return []

    print(f"\nFound {len(ports)} COM port(s):\n")

    witmotion_candidates = []

    for port in ports:
        print(f"  {port.device}")
        print(f"    Description: {port.description}")
        print(f"    HWID: {port.hwid}")
        print()

        # Check for Bluetooth or common USB-serial chips
        desc_lower = port.description.lower()
        if any(x in desc_lower for x in ['bluetooth', 'serial', 'ch340', 'cp210', 'ftdi', 'witmotion']):
            witmotion_candidates.append(port.device)

    if witmotion_candidates:
        print("=" * 60)
        print("Likely WitMotion ports:")
        for p in witmotion_candidates:
            print(f"  {p}")
        print("=" * 60)

    return witmotion_candidates


def test_connection(port: str, baudrate: int = 115200):
    """Test connection to a COM port."""
    import serial
    import time

    print(f"\nTesting connection to {port} at {baudrate} baud...")

    try:
        ser = serial.Serial(port, baudrate, timeout=2)
        print(f"  Opened {port} successfully")

        # Try to read some data
        print("  Waiting for data (5 seconds)...")
        start = time.time()
        data_received = False

        while time.time() - start < 5:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                print(f"  Received {len(data)} bytes: {data[:20].hex()}...")
                data_received = True
                break
            time.sleep(0.1)

        ser.close()

        if data_received:
            print(f"\n  SUCCESS! {port} is receiving data from IMU")
            return True
        else:
            print(f"\n  No data received. IMU may not be sending or wrong port.")
            return False

    except serial.SerialException as e:
        print(f"  ERROR: {e}")
        return False


def main():
    print("\nWitMotion IMU Port Finder for Windows")
    print("=" * 60)

    ports = find_com_ports()

    if not ports:
        print("\nNo candidate ports found.")
        print("\nManual steps:")
        print("  1. Open Device Manager")
        print("  2. Look under 'Ports (COM & LPT)'")
        print("  3. Find 'Standard Serial over Bluetooth link'")
        print("  4. Note the COM port number (e.g., COM5)")
        return

    # Ask user if they want to test
    print("\nWould you like to test these ports? (y/n): ", end="")
    try:
        response = input().strip().lower()
        if response == 'y':
            for port in ports:
                if test_connection(port):
                    print(f"\n\nUse this command to run the celestial viewer:")
                    print(f"  python camera/celestial_sphere_3d.py --witmotion {port}")
                    break
    except:
        pass


if __name__ == "__main__":
    main()

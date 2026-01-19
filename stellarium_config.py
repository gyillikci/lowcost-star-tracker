#!/usr/bin/env python3
"""
Connect to Stellarium Remote Control API and configure view.

Stellarium must be running with Remote Control plugin enabled on port 8090.
Enable in Stellarium: Configuration (F2) > Plugins > Remote Control > Load at startup
"""

import requests
import json
import time

STELLARIUM_URL = "http://localhost:8090/api"

def stellarium_command(endpoint, method="GET", data=None):
    """Send command to Stellarium."""
    url = f"{STELLARIUM_URL}/{endpoint}"
    try:
        if method == "POST":
            response = requests.post(url, json=data)
        else:
            response = requests.get(url, params=data)
        
        if response.status_code == 200:
            return response.json() if response.text else {}
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to Stellarium on port 8090")
        print("Make sure Stellarium is running and Remote Control plugin is enabled")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def hide_star_names():
    """Hide star names."""
    print("Hiding star names...")
    # Use actionapi to execute action
    result = stellarium_command("stelaction/do", "POST", {
        "id": "actionShow_Stars_Labels"
    })
    if result is not None:
        print("  ✓ Star name visibility toggled")
    return result

def hide_constellation_labels():
    """Hide constellation labels."""
    print("Hiding constellation labels...")
    result = stellarium_command("stelproperty/set", "POST", {
        "id": "ConstellationMgr.labelsAmount",
        "value": 0
    })
    if result is not None:
        print("  ✓ Constellation labels hidden")
    return result

def hide_ground():
    """Hide ground/landscape."""
    print("Hiding ground...")
    result = stellarium_command("stelproperty/set", "POST", {
        "id": "LandscapeMgr.flagLandscape",
        "value": False
    })
    if result is not None:
        print("  ✓ Ground hidden")
    return result

def hide_atmosphere():
    """Hide atmosphere."""
    print("Hiding atmosphere...")
    result = stellarium_command("stelproperty/set", "POST", {
        "id": "LandscapeMgr.flagAtmosphere",
        "value": False
    })
    if result is not None:
        print("  ✓ Atmosphere hidden")
    return result

def get_view_info():
    """Get current view information."""
    print("\nGetting view info...")
    result = stellarium_command("main/view")
    if result:
        print(f"  Altitude: {result.get('altitude', 0):.1f}°")
        print(f"  Azimuth: {result.get('azimuth', 0):.1f}°")
        print(f"  FOV: {result.get('fov', 0):.1f}°")
    return result

def set_fov(fov_degrees):
    """Set field of view."""
    print(f"Setting FOV to {fov_degrees}°...")
    result = stellarium_command("main/fov", "POST", {
        "fov": fov_degrees
    })
    if result is not None:
        print(f"  ✓ FOV set to {fov_degrees}°")
    return result

def main():
    print("=" * 60)
    print("Stellarium Remote Control Configuration")
    print("=" * 60)
    
    # Test connection
    print("\nTesting connection to Stellarium...")
    status = stellarium_command("main/status")
    if status is None:
        print("\nFailed to connect. Please:")
        print("  1. Start Stellarium")
        print("  2. Press F2 (Configuration)")
        print("  3. Go to Plugins tab")
        print("  4. Enable 'Remote Control' plugin")
        print("  5. Check 'Load at startup'")
        print("  6. Restart Stellarium")
        return
    
    print("  ✓ Connected to Stellarium")
    print(f"  Version: {status.get('version', 'unknown')}")
    
    # Configure view
    print("\n" + "=" * 60)
    print("Configuring clean view for star tracking...")
    print("=" * 60)
    
    hide_star_names()
    hide_constellation_labels()
    hide_ground()
    hide_atmosphere()
    
    # Show current view
    get_view_info()
    
    print("\n" + "=" * 60)
    print("Configuration complete!")
    print("=" * 60)
    print("\nYou can now use Stellarium as a reference sky view.")
    print("Use the celestial sphere viewer to overlay camera/IMU data.")

if __name__ == "__main__":
    main()

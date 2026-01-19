#!/usr/bin/env python3
"""Query and toggle Stellarium labels."""

import requests

STELLARIUM_URL = "http://localhost:8090/api"

def get_property(prop_id):
    """Get a property value."""
    try:
        response = requests.get(f"{STELLARIUM_URL}/stelproperty/list")
        if response.status_code == 200:
            props = response.json()
            for prop in props:
                if prop_id in prop.get('id', ''):
                    print(f"{prop['id']}: {prop.get('value', 'unknown')}")
        
        # Try direct get
        response = requests.get(f"{STELLARIUM_URL}/stelproperty/get", params={"id": prop_id})
        if response.status_code == 200:
            print(f"\nDirect get: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

def set_property(prop_id, value):
    """Set a property value."""
    try:
        # Try using query parameters
        response = requests.post(f"{STELLARIUM_URL}/stelproperty/set?id={prop_id}&value={value}")
        print(f"Response: {response.status_code} - {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

print("Checking star label properties...")
get_property("StarMgr.labelsAmount")
get_property("StarMgr.flagLabels")

print("\n" + "="*60)
print("Setting star labels to 0...")
if set_property("StarMgr.labelsAmount", 0):
    print("✓ Star labels amount set to 0")
else:
    print("✗ Failed to set star labels")

print("\nTrying to disable star labels flag...")
if set_property("StarMgr.flagLabels", "false"):
    print("✓ Star labels flag disabled")
else:
    print("✗ Failed to disable flag")

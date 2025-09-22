#!/usr/bin/env python3
"""
Debug coordinate system and heading calculations
"""

import numpy as np

def test_coordinate_system():
    """Test coordinate system and heading relationships"""
    
    print("=== Coordinate System Debug ===")
    print("Expected coordinate system:")
    print("- X axis: left/right (0=left, 120=right)")  
    print("- Y axis: forward/backward (0=start, 380=goal)")
    print("- Heading: 0=forward, -pi/2=right, pi/2=left, pi=backward")
    print()
    
    # Test different headings and expected movements
    test_cases = [
        (0, "Forward", "Should increase Y only"),
        (-np.pi/2, "Right", "Should increase X only"), 
        (np.pi/2, "Left", "Should decrease X only"),
        (np.pi, "Backward", "Should decrease Y only"),
        (-np.pi/4, "45° Right", "Should increase both X and Y"),
    ]
    
    distance = 50  # 50cm movement
    
    for heading, direction, expected in test_cases:
        # CORRECTED position calculation 
        dx = -distance * np.sin(heading)  # Fixed with minus sign
        dy = distance * np.cos(heading)
        
        print(f"{direction} (heading={np.degrees(heading):.0f}°):")
        print(f"  dx = -{distance} * sin({np.degrees(heading):.0f}°) = {dx:.1f}")
        print(f"  dy = {distance} * cos({np.degrees(heading):.0f}°) = {dy:.1f}")
        print(f"  Expected: {expected}")
        
        # Check if it matches expectation
        if direction == "Forward" and abs(dx) < 1 and dy > 0:
            print("  [OK] Correct")
        elif direction == "Right" and dx > 0 and abs(dy) < 1:
            print("  [OK] Correct") 
        elif direction == "Left" and dx < 0 and abs(dy) < 1:
            print("  [OK] Correct")
        elif direction == "Backward" and abs(dx) < 1 and dy < 0:
            print("  [OK] Correct")
        elif direction == "45° Right" and dx > 0 and dy > 0:
            print("  [OK] Correct")
        else:
            print("  [ERROR] WRONG - coordinate system issue!")
        
        print()

if __name__ == "__main__":
    test_coordinate_system()
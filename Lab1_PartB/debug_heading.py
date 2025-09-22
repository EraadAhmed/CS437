#!/usr/bin/env python3
"""
Debug the coordinate system and heading calculation
"""

import numpy as np

def debug_coordinate_system():
    """Debug the coordinate system and heading issues"""
    
    print("=== Coordinate System Debug ===")
    print()
    
    # Test the straight-line movement case
    current_pos = (60, 0, 0)    # Start position
    target_pos = (60, 50, 0)    # Target position (straight ahead)
    
    current_x, current_y, current_theta = current_pos
    target_x, target_y, target_theta = target_pos
    
    dx = target_x - current_x  # Should be 0
    dy = target_y - current_y  # Should be 50
    
    print(f"Movement: ({current_x}, {current_y}) -> ({target_x}, {target_y})")
    print(f"dx = {dx}, dy = {dy}")
    print()
    
    # This is the calculation from integrated_main.py
    target_heading = np.arctan2(dy, dx)
    print(f"Calculated target_heading = arctan2({dy}, {dx}) = {target_heading:.3f} rad")
    print(f"In degrees: {np.degrees(target_heading):.1f}°")
    print()
    
    # Analyze what this means
    if dx == 0 and dy > 0:
        print("Movement type: Straight forward (positive Y)")
        print("Expected heading for 'straight forward': depends on coordinate system")
        print()
        print("If Y-axis represents 'forward':")
        print("  - Heading should be 0° (or pi/2 rad)")
        print("  - Current calculation gives pi/2 rad (90°)")
        print()
        print("If X-axis represents 'forward':")
        print("  - Heading should be pi/2 rad (90°)")
        print("  - Current calculation gives pi/2 rad (90°)")
        print()
        
    # Check what arctan2 gives for different directions
    print("=== Coordinate System Reference ===")
    directions = [
        (1, 0, "East (+X)"),
        (0, 1, "North (+Y)"), 
        (-1, 0, "West (-X)"),
        (0, -1, "South (-Y)")
    ]
    
    for dx, dy, name in directions:
        heading = np.arctan2(dy, dx)
        print(f"{name:12s}: arctan2({dy:2d}, {dx:2d}) = {heading:6.3f} rad = {np.degrees(heading):6.1f}°")
    
    print()
    print("=== Problem Analysis ===")
    print("If the car should go 'straight forward' with heading=0,")
    print("then the coordinate system might be:")
    print("  - X = forward/backward")
    print("  - Y = left/right")
    print()
    print("But current movement (60,0) -> (60,50) suggests:")
    print("  - X = left/right (stays at 60)")
    print("  - Y = forward/backward (increases)")
    print()
    print("This mismatch might be causing the turning behavior!")

if __name__ == "__main__":
    debug_coordinate_system()
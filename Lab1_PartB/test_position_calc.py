#!/usr/bin/env python3
"""
Test the corrected position calculation
"""

import numpy as np

def test_position_calculation():
    """Test the corrected position calculation"""
    
    print("=== Position Calculation Test ===")
    print()
    
    # Test the straight forward movement case
    current_pos = (60, 0, 0)
    target_pos = (60, 50, 0)
    
    current_x, current_y, current_theta = current_pos
    target_x, target_y, target_theta = target_pos
    
    # Distance and heading calculation
    dx = target_x - current_x
    dy = target_y - current_y
    distance = np.sqrt(dx**2 + dy**2)
    
    # New heading calculation
    if abs(dx) < 1:  # Moving straight forward/backward
        if dy > 0:
            target_heading = 0  # Forward
        else:
            target_heading = np.pi  # Backward
    else:
        target_heading = np.arctan2(-dx, dy)
    
    actual_distance_moved = distance  # Assume perfect movement for this test
    
    print(f"Movement: ({current_x}, {current_y}) -> target ({target_x}, {target_y})")
    print(f"Distance: {distance}")
    print(f"Target heading: {target_heading:.3f} rad = {np.degrees(target_heading):.1f}°")
    print()
    
    # OLD position calculation (WRONG)
    old_new_x = current_x + actual_distance_moved * np.cos(target_heading)
    old_new_y = current_y + actual_distance_moved * np.sin(target_heading)
    print(f"OLD calculation:")
    print(f"  new_x = {current_x} + {actual_distance_moved} * cos({np.degrees(target_heading):.1f}°) = {old_new_x:.1f}")
    print(f"  new_y = {current_y} + {actual_distance_moved} * sin({np.degrees(target_heading):.1f}°) = {old_new_y:.1f}")
    print(f"  Result: ({old_new_x:.0f}, {old_new_y:.0f}) [WRONG]")
    print()
    
    # NEW position calculation (CORRECT)
    new_x = current_x + actual_distance_moved * np.sin(target_heading)
    new_y = current_y + actual_distance_moved * np.cos(target_heading)
    print(f"NEW calculation:")
    print(f"  new_x = {current_x} + {actual_distance_moved} * sin({np.degrees(target_heading):.1f}°) = {new_x:.1f}")
    print(f"  new_y = {current_y} + {actual_distance_moved} * cos({np.degrees(target_heading):.1f}°) = {new_y:.1f}")
    print(f"  Result: ({new_x:.0f}, {new_y:.0f}) [CORRECT]")
    print()
    
    print(f"Expected: ({target_x}, {target_y})")
    if abs(new_x - target_x) < 1 and abs(new_y - target_y) < 1:
        print("[SUCCESS] New calculation matches target!")
    else:
        print("[ERROR] Still not matching target")

if __name__ == "__main__":
    test_position_calculation()
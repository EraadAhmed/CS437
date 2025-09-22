#!/usr/bin/env python3
"""
Test the updated heading calculation
"""

import numpy as np

def test_new_heading_calculation():
    """Test the new heading calculation logic"""
    
    print("=== Updated Heading Calculation Test ===")
    print()
    
    # Test cases: (current_pos, target_pos, expected_heading, description)
    test_cases = [
        ((60, 0, 0), (60, 50, 0), 0, "Straight forward"),
        ((60, 50, 0), (60, 0, 0), np.pi, "Straight backward"),
        ((60, 50, 0), (80, 50, 0), -np.pi/2, "Straight right"),
        ((60, 50, 0), (40, 50, 0), np.pi/2, "Straight left"),
    ]
    
    for current_pos, target_pos, expected_heading, description in test_cases:
        current_x, current_y, current_theta = current_pos
        target_x, target_y, target_theta = target_pos
        
        dx = target_x - current_x
        dy = target_y - current_y
        
        print(f"Test: {description}")
        print(f"  Movement: ({current_x}, {current_y}) -> ({target_x}, {target_y})")
        print(f"  dx = {dx}, dy = {dy}")
        
        # Apply the new logic
        if abs(dx) < 1:  # Moving straight forward/backward
            if dy > 0:
                target_heading = 0  # Forward
            else:
                target_heading = np.pi  # Backward
        elif abs(dy) < 1:  # Moving straight left/right
            if dx > 0:
                target_heading = -np.pi/2  # Right
            else:
                target_heading = np.pi/2   # Left
        else:
            # Diagonal movement
            target_heading = np.arctan2(-dx, dy)
        
        print(f"  Calculated heading: {target_heading:.3f} rad = {np.degrees(target_heading):.1f}°")
        print(f"  Expected heading: {expected_heading:.3f} rad = {np.degrees(expected_heading):.1f}°")
        
        if abs(target_heading - expected_heading) < 0.01:
            print("  [CORRECT]")
        else:
            print("  [WRONG]")
        print()

if __name__ == "__main__":
    test_new_heading_calculation()
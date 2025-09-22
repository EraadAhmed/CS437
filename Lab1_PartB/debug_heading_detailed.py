#!/usr/bin/env python3
"""
Debug heading and movement calculations
"""

import numpy as np

def test_heading_calculations():
    """Test if heading calculations are correct for straight movement"""
    
    print("=== Heading Calculation Debug ===")
    print()
    
    # Test straight forward movement cases
    test_cases = [
        ((60, 0, 0), (60, 50, 0)),    # First movement
        ((60, 50, 0), (60, 100, 0)),  # Second movement  
        ((60, 100, 0), (60, 150, 0)), # Third movement
    ]
    
    for i, (current, target) in enumerate(test_cases):
        current_x, current_y, current_theta = current
        target_x, target_y, target_theta = target
        
        print(f"Test {i+1}: ({current_x}, {current_y}) -> ({target_x}, {target_y})")
        
        # Calculate required motion
        dx = target_x - current_x
        dy = target_y - current_y
        
        print(f"  dx = {dx}, dy = {dy}")
        
        # Calculate required heading
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
        
        print(f"  target_heading = {target_heading:.3f} rad = {np.degrees(target_heading):.1f}°")
        
        # Calculate heading error
        heading_error = target_heading - current_theta
        
        # Normalize heading error
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        print(f"  current_theta = {current_theta:.3f} rad = {np.degrees(current_theta):.1f}°")
        print(f"  heading_error = {heading_error:.3f} rad = {np.degrees(heading_error):.1f}°")
        
        # Check if turn would be executed
        if abs(heading_error) > 0.1:  # 0.1 radians ~ 6 degrees
            turn_angle = np.degrees(heading_error)
            print(f"  TURN REQUIRED: {turn_angle:.1f}° [This could cause drift!]")
        else:
            print(f"  No turn needed (error < 6°)")
        
        print()

if __name__ == "__main__":
    test_heading_calculations()
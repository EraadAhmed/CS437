#!/usr/bin/env python3
"""
Debug script to test actual movement vs logged movement
"""

import numpy as np

def debug_movement_calculation():
    """Debug the movement calculation logic"""
    
    print("=== Movement Calculation Debug ===")
    
    # Test the exact scenario from the log
    test_cases = [
        ((60, 0, 0), (60, 50, 0)),    # First movement
        ((60, 50, 0), (60, 100, 0)),  # Second movement  
        ((60, 100, 0), (60, 150, 0)), # Third movement
    ]
    
    SAMPLING = 1
    SPEED = 20  # Updated to match integrated_main.py
    
    for i, (current_pos, target_state) in enumerate(test_cases):
        print(f"\nTest {i+1}: {current_pos} -> {target_state}")
        
        current_x, current_y, current_theta = current_pos
        target_x, target_y, target_theta = target_state
        
        # Calculate required motion (same as integrated_main.py)
        dx = target_x - current_x
        dy = target_y - current_y
        
        distance = np.sqrt(dx**2 + dy**2) * SAMPLING  # Convert to cm
        print(f"  Distance to move: {distance:.2f} cm")
        
        if distance > 1:
            # Calculate required heading
            target_heading = np.arctan2(dy, dx)
            print(f"  Target heading: {target_heading:.3f} rad = {np.degrees(target_heading):.1f}°")
            
            heading_error = target_heading - current_theta
            print(f"  Heading error: {heading_error:.3f} rad = {np.degrees(heading_error):.1f}°")
            
            # Normalize heading error
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi
            
            print(f"  Normalized heading error: {heading_error:.3f} rad = {np.degrees(heading_error):.1f}°")
            
            # Execute motion calculation
            move_time = distance / SPEED
            print(f"  Move time: {move_time:.3f} seconds")
            
            actual_distance_moved = SPEED * move_time / SAMPLING
            print(f"  Actual distance moved: {actual_distance_moved:.2f} grid units")
            
            # Calculate actual new position based on heading
            new_x = current_x + actual_distance_moved * np.cos(target_heading) 
            new_y = current_y + actual_distance_moved * np.sin(target_heading)
            
            final_pos = (int(round(new_x)), int(round(new_y)), target_heading)
            print(f"  Final position: {final_pos}")
            print(f"  Expected: {target_state}")
            
            if abs(final_pos[0] - target_state[0]) > 1 or abs(final_pos[1] - target_state[1]) > 1:
                print(f"  ⚠️  MISMATCH: Significant difference between expected and calculated position!")

if __name__ == "__main__":
    debug_movement_calculation()
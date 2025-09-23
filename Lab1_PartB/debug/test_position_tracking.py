#!/usr/bin/env python3
"""
Test position tracking during turns and movements
"""

import numpy as np
import asyncio
from picarx import Picarx

async def test_position_tracking():
    """Test position tracking during various movements"""
    
    print("=== Position Tracking Test ===")
    print()
    
    # Simulate the coordinate system and position tracking logic
    SAMPLING = 1  # 1cm per grid unit
    SPEED = 30.5  # cm/s 
    SERVO_OFFSET = 0  # Hardware fixed: no offset needed
    
    # Starting position
    current_pos = (60, 200, 0)  # x, y, theta
    print(f"Starting position: {current_pos}")
    print()
    
    # Test 1: Recovery maneuver simulation
    print("=== Test 1: Recovery Maneuver Simulation ===")
    current_x, current_y, current_theta = current_pos
    
    # Backup 20cm
    backup_distance = 20
    backup_distance_grid = backup_distance / SAMPLING
    
    # CORRECTED backup calculation
    new_x = current_x + backup_distance_grid * np.sin(current_theta)  # Backup reverses direction
    new_y = current_y - backup_distance_grid * np.cos(current_theta)  # Backup reverses direction
    
    print(f"After backing up 20cm:")
    print(f"  Position: ({current_x}, {current_y}) -> ({new_x:.1f}, {new_y:.1f})")
    
    # Turn 45 degrees right
    turn_angle = 45
    # COORDINATE SYSTEM: 0=forward, -pi/2=right, pi/2=left
    # Right turn = subtract angle (clockwise)
    new_theta = (current_theta - np.radians(turn_angle)) % (2 * np.pi)
    
    recovery_pos = (int(round(new_x)), int(round(new_y)), new_theta)
    print(f"After 45° right turn:")
    print(f"  Heading: {np.degrees(current_theta):.1f}° -> {np.degrees(new_theta):.1f}°")
    print(f"  Final recovery position: {recovery_pos}")
    print()
    
    # Test 2: Forward movement after turn
    print("=== Test 2: Forward Movement After Turn ===")
    current_x, current_y, current_theta = recovery_pos
    
    # Move forward 50cm at the new heading
    move_distance = 50
    actual_distance_moved = move_distance / SAMPLING
    
    # CORRECTED forward movement calculation
    forward_x = current_x - actual_distance_moved * np.sin(current_theta)  # Fixed formula
    forward_y = current_y + actual_distance_moved * np.cos(current_theta)  # Forward direction
    
    final_pos = (int(round(forward_x)), int(round(forward_y)), current_theta)
    
    print(f"Moving forward 50cm at heading {np.degrees(current_theta):.1f}°:")
    print(f"  Position: ({current_x}, {current_y}) -> {final_pos[:2]}")
    print(f"  Expected: Car should move diagonally right-forward")
    print()
    
    # Test 3: Compare with user's observed position
    print("=== Test 3: Compare with Observed Results ===")
    observed_pos = (120, 185)
    print(f"User observed car at: {observed_pos}")
    print(f"Our calculation gives: {final_pos[:2]}")
    print(f"Difference: X={observed_pos[0] - final_pos[0]}, Y={observed_pos[1] - final_pos[1]}")

if __name__ == "__main__":
    asyncio.run(test_position_tracking())
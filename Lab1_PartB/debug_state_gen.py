#!/usr/bin/env python3
"""
Debug script to check if the state generation is reasonable
"""

import numpy as np
from car_control import next_state_gen, cost

def test_state_generation():
    """Test what states are being generated"""
    
    print("=== State Generation Debug ===")
    
    # Parameters from the hanging system (IMPROVED VALUES)
    current_state = (60, 0, 0)  # start
    SPEED = 50        # Increased for larger steps
    DELTA_T = 0.5     # Increased for larger steps
    CAR_LENGTH = 23   # Correct value from integrated_main.py
    
    print(f"Starting state: {current_state}")
    print(f"Parameters: SPEED={SPEED}, DELTA_T={DELTA_T}, CAR_LENGTH={CAR_LENGTH}")
    print()
    
    # Test a few control inputs to see what happens
    controls = [-45, -30, -15, 0, 15, 30, 45]  # Updated control set
    
    for control in controls:
        try:
            next_state = next_state_gen(current_state, SPEED, DELTA_T, control, CAR_LENGTH)
            distance = cost(current_state, next_state)
            print(f"Control {control:3d}° -> State {next_state} (distance: {distance:.2f})")
        except Exception as e:
            print(f"Control {control:3d}° -> ERROR: {e}")
    
    print()
    print("=== Goal Distance Check ===")
    goal_state = (60, 380, 0)
    goal_distance = cost(current_state, goal_state)
    print(f"Distance to goal: {goal_distance:.2f}")
    
    # Test if we're making meaningful progress
    print()
    print("=== Progress Analysis ===")
    next_state = next_state_gen(current_state, SPEED, DELTA_T, 0, CAR_LENGTH)  # straight ahead
    progress = cost(next_state, goal_state)
    print(f"After one step forward: distance to goal = {progress:.2f}")
    print(f"Progress made: {goal_distance - progress:.2f}")
    
    if abs(goal_distance - progress) < 0.1:
        print("WARNING: Very little progress per step! This could cause infinite loops.")
    
if __name__ == "__main__":
    test_state_generation()
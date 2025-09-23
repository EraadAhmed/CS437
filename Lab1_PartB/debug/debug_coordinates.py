#!/usr/bin/env python3
"""
Debug script to check coordinate system and motion logic
"""

import numpy as np

# Constants from integrated_main.py (corrected)
WIDTH = 120  # 120cm hallway width
LENGTH = 380  # 380cm to endpoint
SAMPLING = 1

# Positions
start_pos = (int(60/SAMPLING), int(0/SAMPLING), 0)    # (x, y, theta) - center of 120cm hallway, at start
goal_pos = (int(60/SAMPLING), int(380/SAMPLING), 0)   # 380cm forward

print("=== Coordinate System Debug ===")
print(f"Map dimensions: {int(LENGTH/SAMPLING)} x {int(WIDTH/SAMPLING)} (Length x Width)")
print(f"Start position: {start_pos}")
print(f"Goal position: {goal_pos}")

# Calculate distance
current_pos_2d = start_pos[:2]
goal_pos_2d = goal_pos[:2]
distance_to_goal = np.sqrt(sum((a - b)**2 for a, b in zip(current_pos_2d, goal_pos_2d)))
print(f"Initial distance to goal: {distance_to_goal}")

print("\n=== Motion Logic Test ===")
current_x, current_y, current_theta = start_pos
target_x, target_y, target_theta = goal_pos

# Calculate required motion (same as in integrated_main.py)
dx = target_x - current_x
dy = target_y - current_y

print(f"dx = {target_x} - {current_x} = {dx}")
print(f"dy = {target_y} - {current_y} = {dy}")

distance = np.sqrt(dx**2 + dy**2) * SAMPLING
print(f"Distance to move: {distance} cm")

# Calculate required heading
target_heading = np.arctan2(dx, dy)
print(f"Target heading: {target_heading} radians = {np.degrees(target_heading)} degrees")

print("\n=== Problem Analysis ===")
if dx == 0 and dy > 0:
    print("Should move straight forward (positive Y direction)")
elif dx == 0 and dy < 0:
    print("WARNING: Would move backward (negative Y direction)")
else:
    print(f"Would move diagonally: dx={dx}, dy={dy}")

print(f"\nIf goal reached immediately, current position might be: {start_pos[:2]}")
print(f"Goal position: {goal_pos[:2]}")
print(f"Distance threshold: 2 grid cells")

if distance_to_goal < 2:
    print("ERROR: Goal is already considered 'reached' at start!")
else:
    print("GOOD: Goal is not immediately reached")
    
print("\n=== Coordinate System Check ===")
print("In typical computer graphics:")
print("- X increases to the right")
print("- Y increases downward")
print("In robotics:")
print("- X often increases forward")  
print("- Y increases to the left")
print("\nCurrent setup:")
print(f"- Start at Y=0, Goal at Y={goal_pos[1]}")
print(f"- This suggests Y increases in the 'forward' direction")
print(f"- Goal is {goal_pos[1]} units 'forward' from start")
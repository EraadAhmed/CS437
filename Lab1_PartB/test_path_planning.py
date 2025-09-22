#!/usr/bin/env python3
"""
Test script for path planning functionality
"""

import asyncio
import numpy as np
from car_control import hybrid_a_star

async def test_path_planning():
    """Test the hybrid A* path planning with correct parameters."""
    
    print("Testing Hybrid A* Path Planning...")
    
    # Test parameters (matching integrated_main.py)
    WIDTH = 120  # 120cm hallway width
    CAR_WIDTH = 14
    CAR_LENGTH = 23
    SPEED = 10
    DELTA_T = 0.25
    SAMPLING = 1
    
    # Create a simple test map
    LENGTH = 380  # 380cm total length
    map_grid = np.zeros((int(LENGTH/SAMPLING), int(WIDTH/SAMPLING)))
    
    # Add some obstacles for testing
    map_grid[50:70, 50:70] = 1  # Small obstacle
    
    # Start and goal positions (real coordinates)
    # Test path planning from start to 380cm goal
    start = Coordinate((60, 0, 0))      # Middle of hallway width, at 0cm length
    goal = Coordinate((60, 380, 0))     # 380cm forward, staying in middle
    
    print(f"Start position: {start_pos}")
    print(f"Goal position: {goal_pos}")
    print(f"Map shape: {map_grid.shape}")
    print(f"Parameters: WIDTH={WIDTH}, CAR_WIDTH={CAR_WIDTH}, CAR_LENGTH={CAR_LENGTH}")
    print(f"Motion: SPEED={SPEED}, DELTA_T={DELTA_T}")
    
    try:
        # Test the path planning
        path = await hybrid_a_star(
            start_pos, 
            goal_pos, 
            map_grid,
            WIDTH, 
            CAR_WIDTH, 
            SPEED, 
            DELTA_T,
            CAR_LENGTH
        )
        
        print(f"Path planning successful!")
        print(f"Path length: {len(path)} waypoints")
        print("First few waypoints:")
        for i, waypoint in enumerate(path[:5]):
            print(f"  {i}: {waypoint}")
        if len(path) > 5:
            print("  ...")
            print(f"  {len(path)-1}: {path[-1]}")
            
        return True
        
    except Exception as e:
        print(f"Path planning failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_path_planning())
    if success:
        print("\nPath planning test completed successfully!")
    else:
        print("\nPath planning test failed!")
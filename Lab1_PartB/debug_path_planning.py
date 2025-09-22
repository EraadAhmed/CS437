#!/usr/bin/env python3
"""
Debug script to isolate path planning hanging issue
"""

import asyncio
import numpy as np
import time
from car_control import hybrid_a_star, Coordinate

async def test_path_planning():
    """Test the hybrid A* function with the current parameters"""
    
    print("=== Path Planning Debug ===")
    print("Testing hybrid_a_star function with current parameters")
    
    # Use the same parameters as integrated_main.py
    start_pos = (60, 0, 0)
    goal_pos = (60, 380, 0)
    
    # Create a simple map (no obstacles for initial test)
    WIDTH = 120
    LENGTH = 380
    SAMPLING = 1
    
    map_grid = np.zeros((int(LENGTH/SAMPLING), int(WIDTH/SAMPLING)))
    print(f"Map dimensions: {map_grid.shape}")
    print(f"Start: {start_pos}")
    print(f"Goal: {goal_pos}")
    
    # Parameters from integrated_main.py
    CAR_WIDTH = 10
    SPEED = 30
    DELTA_T = 0.1
    CAR_LENGTH = 18
    
    print("\nStarting path planning...")
    start_time = time.time()
    
    try:
        # Add timeout to prevent infinite hanging
        path = await asyncio.wait_for(
            hybrid_a_star(
                start_pos,
                goal_pos,
                map_grid,
                WIDTH,
                CAR_WIDTH,
                SPEED,
                DELTA_T,
                CAR_LENGTH
            ),
            timeout=30.0  # 30 second timeout
        )
        
        end_time = time.time()
        print(f"Path planning completed in {end_time - start_time:.2f} seconds")
        
        if path:
            print(f"Path found with {len(path)} waypoints")
            print(f"First few waypoints: {path[:5]}")
            print(f"Last few waypoints: {path[-5:]}")
        else:
            print("No path found")
            
    except asyncio.TimeoutError:
        print("ERROR: Path planning timed out after 30 seconds!")
        print("This indicates the algorithm is hanging or stuck in an infinite loop")
        
    except Exception as e:
        print(f"ERROR: Path planning failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_path_planning())
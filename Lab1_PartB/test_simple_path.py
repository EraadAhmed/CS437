#!/usr/bin/env python3
"""
Test a simple straight-line path to see if the algorithm can handle it
"""

import asyncio
import numpy as np
import time
from car_control import hybrid_a_star

async def test_simple_path():
    """Test a much simpler path planning scenario"""
    
    print("=== Simple Path Planning Test ===")
    
    # Test a MUCH shorter distance first
    start_pos = (60, 0, 0)
    goal_pos = (60, 50, 0)  # Only 50 units away instead of 380
    
    WIDTH = 120
    CAR_WIDTH = 14
    SPEED = 50
    DELTA_T = 0.5
    CAR_LENGTH = 23
    
    # Create empty map
    map_grid = np.zeros((100, 120))  # 100 length for 50-unit goal
    
    print(f"Start: {start_pos}")
    print(f"Goal: {goal_pos}")
    print(f"Distance: {np.sqrt((goal_pos[0]-start_pos[0])**2 + (goal_pos[1]-start_pos[1])**2)}")
    
    print("\nStarting path planning...")
    start_time = time.time()
    
    try:
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
            timeout=30.0
        )
        
        end_time = time.time()
        print(f"SUCCESS: Path found in {end_time - start_time:.2f} seconds")
        print(f"Path length: {len(path)} waypoints")
        if path:
            print(f"First few: {path[:3]}")
            print(f"Last few: {path[-3:]}")
            
    except asyncio.TimeoutError:
        print("FAILED: Still timed out even with shorter distance")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_simple_path())
#!/usr/bin/env python3
"""
Simple test to isolate the system hanging issue.
"""

import asyncio
import sys
import logging

# Set up simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def simple_test():
    """Simple test to see if the basic system works."""
    logger.info("Starting simple test...")
    
    try:
        # Test basic navigation logic without hardware
        from integrated import Position, SystemConfig
        
        # Create basic objects
        config = SystemConfig()
        start_pos = Position(10, 10, 0)
        goal_pos = Position(60, 200, 0)
        
        logger.info(f"Start: {start_pos}")
        logger.info(f"Goal: {goal_pos}")
        logger.info(f"Distance: {start_pos.distance_to(goal_pos):.2f}")
        
        # Test grid coordinates
        grid_start = start_pos.to_grid(1.0)
        grid_goal = goal_pos.to_grid(1.0)
        
        logger.info(f"Grid start: {grid_start}")
        logger.info(f"Grid goal: {grid_goal}")
        
        logger.info("Basic test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(simple_test())
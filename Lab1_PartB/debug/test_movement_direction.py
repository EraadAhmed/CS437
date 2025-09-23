#!/usr/bin/env python3
"""
Test script to verify car movement direction
"""

import asyncio
import time
from picarx import Picarx

async def test_movement_direction():
    """Test which direction the car actually moves."""
    
    print("=== Car Movement Direction Test ===")
    print("This test will help identify if the car moves in the expected direction")
    print()
    
    # Initialize car
    picarx = Picarx(servo_pins=["P0", "P1", "P3"])
    
    print("Car initialized. Please observe the car's movement:")
    print()
    
    # Test 1: Forward command
    print("TEST 1: Executing picarx.forward() for 2 seconds...")
    print("Expected: Car should move toward the 380cm goal")
    print("Observe: Does the car move toward or away from the goal?")
    
    input("Press Enter when ready to test forward movement...")
    
    picarx.forward(40)
    await asyncio.sleep(2)
    picarx.stop()
    
    print("Forward test complete.")
    forward_result = input("Did the car move TOWARD the goal? (y/n): ").lower().strip()
    
    print()
    
    # Test 2: Backward command  
    print("TEST 2: Executing picarx.backward() for 2 seconds...")
    print("Expected: Car should move away from the 380cm goal")
    print("Observe: Does the car move toward or away from the goal?")
    
    input("Press Enter when ready to test backward movement...")
    
    picarx.backward(40)
    await asyncio.sleep(2)
    picarx.stop()
    
    print("Backward test complete.")
    backward_result = input("Did the car move AWAY from the goal? (y/n): ").lower().strip()
    
    print()
    print("=== Test Results ===")
    print(f"Forward command moves toward goal: {forward_result == 'y'}")
    print(f"Backward command moves away from goal: {backward_result == 'y'}")
    
    if forward_result == 'y' and backward_result == 'y':
        print("✓ GOOD: Car movement matches expected direction")
        print("  The issue is likely in the coordinate system or path planning")
    elif forward_result == 'n' and backward_result == 'n':
        print("✗ PROBLEM: Car movement is reversed!")
        print("  Solution: Use picarx.backward() instead of picarx.forward()")
        print("  Or fix the motor wiring/orientation")
    else:
        print("? UNCLEAR: Mixed results, please retest")
    
    print()
    print("=== Recommended Action ===")
    if forward_result == 'n':
        print("Since forward moves away from goal, you should:")
        print("1. Replace picarx.forward() with picarx.backward() in the code")
        print("2. Or physically flip the car around")
        print("3. Or check motor wiring")

if __name__ == "__main__":
    asyncio.run(test_movement_direction())
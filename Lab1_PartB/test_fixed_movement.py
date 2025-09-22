#!/usr/bin/env python3
"""
Quick test to verify car moves toward goal
"""

import asyncio
from picarx import Picarx

async def test_forward_movement():
    """Test if car now moves toward the goal."""
    
    print("=== Quick Movement Test ===")
    print("This will test if the car now moves toward the goal at 380cm")
    print("Layout: 120cm wide hallway, 380cm total length")
    print("Route: From start (0cm) to goal (380cm)")
    
    # Initialize car
    picarx = Picarx(servo_pins=["P0", "P1", "P3"])
    
    print("HARDWARE FIXED: Car wiring has been corrected!")
    print("NOW TESTING: Car should move forward toward goal with normal commands")
    print()
    print("The software now uses picarx.forward() (normal command)")
    print("This should work correctly with the fixed wiring")
    print()
    
    input("Press Enter to test the movement (should go toward goal)...")
    
    # Use the normal forward command now that hardware is fixed
    print("Executing: picarx.forward(40) for 2 seconds")
    print("(This is the normal forward command)")
    
    picarx.forward(40)
    await asyncio.sleep(2)
    picarx.stop()
    
    print()
    result = input("Did the car move TOWARD the 380cm goal? (y/n): ").lower().strip()
    
    if result == 'y':
        print("SUCCESS: Car movement direction is now FIXED!")
        print("  The car should now properly navigate toward the goal")
    else:
        print("PROBLEM: Movement still wrong")
        print("  You may need to:")
        print("  1. Check physical car orientation") 
        print("  2. Verify motor wiring")
        print("  3. Try switching back to picarx.forward()")

if __name__ == "__main__":
    asyncio.run(test_forward_movement())
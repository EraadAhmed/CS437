#!/usr/bin/env python3
"""
Debug steering alignment and servo centering
"""

import asyncio
from picarx import Picarx

async def test_steering():
    """Test steering centering and forward movement"""
    
    print("=== Steering Debug Test ===")
    
    picarx = Picarx()
    
    try:
        print("1. Testing servo centering...")
        picarx.set_dir_servo_angle(0)
        await asyncio.sleep(1)
        print("   Servo set to 0 degrees (center)")
        
        print("2. Testing slight left/right moves...")
        
        # Test small left turn
        picarx.set_dir_servo_angle(-10)
        await asyncio.sleep(0.5)
        print("   Servo set to -10 degrees (slight left)")
        
        # Back to center
        picarx.set_dir_servo_angle(0)
        await asyncio.sleep(0.5)
        print("   Servo returned to center")
        
        # Test small right turn
        picarx.set_dir_servo_angle(10)
        await asyncio.sleep(0.5)
        print("   Servo set to +10 degrees (slight right)")
        
        # Back to center
        picarx.set_dir_servo_angle(0)
        await asyncio.sleep(0.5)
        print("   Servo returned to center")
        
        print("3. Testing forward movement with explicit centering...")
        
        # Ensure servo is centered before moving
        picarx.set_dir_servo_angle(0)
        await asyncio.sleep(0.2)  # Give time for servo to center
        
        print("   Moving forward for 2 seconds...")
        picarx.forward(40)  # Same power as main system
        await asyncio.sleep(2)
        picarx.stop()
        
        print("   Movement complete")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        picarx.stop()
        picarx.set_dir_servo_angle(0)  # Ensure centered at end

if __name__ == "__main__":
    asyncio.run(test_steering())
#!/usr/bin/env python3
"""
Servo offset calibration tool - find the angle that makes the car go straight
"""

import asyncio
from picarx import Picarx

async def test_servo_offsets():
    """Test different servo offsets to find the one that goes straight"""
    
    print("=== Servo Offset Calibration Tool ===")
    print()
    print("This will test different servo angles to find the one that makes the car go straight.")
    print("Place the car at the start of the hallway, pointing forward.")
    print()
    
    picarx = Picarx()
    
    # Test different offsets
    offsets_to_test = [0, -2, -4, -6, 2, 4]  # Start with current and try others
    
    try:
        for offset in offsets_to_test:
            print(f"\n--- Testing servo offset: {offset}° ---")
            print("Setting up for test...")
            
            # Set servo to this offset
            picarx.set_dir_servo_angle(offset)
            await asyncio.sleep(0.5)
            
            input(f"Position car at start. Press Enter to test offset {offset}°...")
            
            # Move forward for 3 seconds
            print(f"Moving forward with {offset}° offset for 3 seconds...")
            picarx.forward(40)  # Same power as main system
            await asyncio.sleep(3)
            picarx.stop()
            
            print("Movement complete!")
            
            # Get user feedback
            response = input(f"Did the car go straight with {offset}°? (y/n/left/right): ").strip().lower()
            
            if response == 'y':
                print(f"FOUND IT! Use SERVO_OFFSET = {offset}")
                break
            elif response == 'left':
                print("Car drifted left - try a more negative offset")
            elif response == 'right':  
                print("Car drifted right - try a more positive offset")
            else:
                print("Continuing to next test...")
            
            # Return servo to center for repositioning
            picarx.set_dir_servo_angle(0)
            await asyncio.sleep(0.5)
            
        print(f"\nCalibration complete!")
        
    except KeyboardInterrupt:
        print("\nCalibration interrupted")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        picarx.set_dir_servo_angle(0)
        picarx.stop()

if __name__ == "__main__":
    asyncio.run(test_servo_offsets())
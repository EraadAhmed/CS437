#!/usr/bin/env python3
"""
Test servo calibration and centering
"""

import asyncio
from picarx import Picarx

async def test_servo_calibration():
    """Test if servo 0 degrees is actually straight"""
    
    print("=== Servo Calibration Test ===")
    print()
    print("This test will help determine if the servo is properly calibrated.")
    print("Observe the front wheels during each position:")
    print()
    
    picarx = Picarx()
    
    try:
        positions = [
            (-30, "Full Left"),
            (-15, "Half Left"), 
            (0, "CENTER (should be straight)"),
            (15, "Half Right"),
            (30, "Full Right"),
            (0, "Back to CENTER")
        ]
        
        for angle, description in positions:
            print(f"Setting servo to {angle}° - {description}")
            picarx.set_dir_servo_angle(angle)
            await asyncio.sleep(2)  # Give time to observe
            input("Press Enter when ready for next position...")
        
        print()
        print("Now testing slight adjustments around center...")
        
        # Test small adjustments around center
        for offset in [-5, -3, -1, 0, 1, 3, 5]:
            print(f"Testing center + {offset}° = {offset}°")
            picarx.set_dir_servo_angle(offset)
            await asyncio.sleep(1)
            
            response = input(f"Is {offset}° straight? (y/n/closest): ").strip().lower()
            if response == 'closest':
                print("Mark this angle as the true center!")
                break
            elif response == 'y':
                if offset == 0:
                    print("Servo is properly calibrated!")
                else:
                    print(f"Servo needs {-offset}° offset calibration")
                break
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        picarx.set_dir_servo_angle(0)
        picarx.stop()

if __name__ == "__main__":
    asyncio.run(test_servo_calibration())
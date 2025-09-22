#!/usr/bin/env python3
"""
Simple test to measure actual physical movement distance
"""

import asyncio
from picarx import Picarx

async def test_actual_movement():
    """Test how far the car actually moves with current settings"""
    
    print("=== Physical Movement Test ===")
    print("This test will move the car forward and you can measure the actual distance")
    print()
    
    # Use the same settings as integrated_main.py
    POWER = 40
    SPEED = 20  # cm/sec (for calculation)
    
    picarx = Picarx(servo_pins=["P0", "P1", "P3"])
    
    # Test different durations
    test_cases = [
        (1.0, "1 second (expected: 20cm)"),
        (2.5, "2.5 seconds (expected: 50cm)"), 
        (5.0, "5 seconds (expected: 100cm)")
    ]
    
    for duration, description in test_cases:
        expected_distance = SPEED * duration
        
        print(f"\nTest: {description}")
        print(f"Power: {POWER}%, Duration: {duration}s")
        print(f"Expected distance: {expected_distance}cm")
        print()
        
        input(f"Position the car at a starting point and press Enter to move for {duration}s...")
        
        # Move forward
        picarx.forward(POWER)
        await asyncio.sleep(duration)
        picarx.stop()
        
        print("Movement complete!")
        actual_distance = input("Measure and enter the actual distance moved (cm): ")
        
        try:
            actual_dist = float(actual_distance)
            error = actual_dist - expected_distance
            error_percent = (error / expected_distance) * 100
            
            print(f"Expected: {expected_distance}cm")
            print(f"Actual: {actual_dist}cm") 
            print(f"Error: {error:+.1f}cm ({error_percent:+.1f}%)")
            
            if abs(error_percent) > 20:
                print("⚠️  Large error! Speed setting may need adjustment.")
            else:
                print("✅ Reasonable accuracy")
                
        except ValueError:
            print("Invalid input, skipping calculation")

if __name__ == "__main__":
    asyncio.run(test_actual_movement())
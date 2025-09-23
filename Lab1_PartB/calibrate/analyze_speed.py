#!/usr/bin/env python3
"""
Analyze the speed calibration issue
"""

# Global constants for speed calibration analysis
EXPECTED_DISTANCE = 380  # cm - target navigation distance
ACTUAL_DISTANCE = 500    # cm - measured actual distance
CURRENT_SPEED = 20       # cm/s - assumed speed setting
POWER_SETTING = 40       # % - motor power percentage
STANDARD_STEP_DISTANCE = 50  # cm - standard movement step
FINAL_STEP_DISTANCE = 30     # cm - final movement step
TOTAL_STEPS = 8              # number of movement steps
LONG_STEPS = 7               # number of 50cm steps
SHORT_STEPS = 1              # number of 30cm steps

def analyze_speed_problem():
    """Analyze why car went 500cm instead of 380cm"""
    
    print("=== Speed Calibration Analysis ===")
    print()
    
    # Use global constants instead of magic numbers
    expected_distance = EXPECTED_DISTANCE
    actual_distance = ACTUAL_DISTANCE
    current_speed = CURRENT_SPEED
    power = POWER_SETTING
    
    print(f"Expected distance: {expected_distance} cm")
    print(f"Actual distance: {actual_distance} cm")
    print(f"Distance ratio: {actual_distance/expected_distance:.2f}x")
    print()
    
    # Calculate what the actual speed must be
    print("=== Movement Analysis ===")
    print(f"The car made {TOTAL_STEPS} movements:")
    print(f"- {LONG_STEPS} movements of {STANDARD_STEP_DISTANCE}cm each = {LONG_STEPS * STANDARD_STEP_DISTANCE}cm")
    print(f"- {SHORT_STEPS} movement of {FINAL_STEP_DISTANCE}cm = {FINAL_STEP_DISTANCE}cm")
    print(f"- Total expected = {LONG_STEPS * STANDARD_STEP_DISTANCE + SHORT_STEPS * FINAL_STEP_DISTANCE}cm")
    print()
    
    print(f"But actual distance was ~{actual_distance}cm")
    print(f"Speed error factor: {actual_distance/expected_distance:.2f}")
    print()
    
    # Calculate the real speed
    actual_speed = current_speed * (actual_distance / expected_distance)
    print(f"Actual car speed: {current_speed} * {actual_distance/expected_distance:.2f} = {actual_speed:.1f} cm/s")
    print()
    
    # Calculate corrected movement times
    print("=== Corrected Parameters ===")
    print(f"To travel exactly {expected_distance}cm, we need:")
    
    corrected_speed = actual_speed  # Use the real speed
    corrected_time_50cm = STANDARD_STEP_DISTANCE / corrected_speed
    corrected_time_30cm = FINAL_STEP_DISTANCE / corrected_speed
    
    print(f"SPEED = {corrected_speed:.1f}  # Real speed at {power}% power")
    print(f"Time for {STANDARD_STEP_DISTANCE}cm: {corrected_time_50cm:.2f}s (was 2.5s)")
    print(f"Time for {FINAL_STEP_DISTANCE}cm: {corrected_time_30cm:.2f}s (was 1.5s)")
    print()
    
    # Alternative: reduce movement distance per step
    print("=== Alternative: Reduce Step Distance ===")
    target_step_distance = STANDARD_STEP_DISTANCE * (expected_distance / actual_distance)
    print(f"Or reduce step size from {STANDARD_STEP_DISTANCE}cm to {target_step_distance:.1f}cm")
    print(f"This would give {expected_distance/target_step_distance:.1f} steps instead of {TOTAL_STEPS}")

if __name__ == "__main__":
    analyze_speed_problem()
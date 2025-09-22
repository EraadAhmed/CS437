#!/usr/bin/env python3
"""
Analyze the speed calibration issue
"""

def analyze_speed_problem():
    """Analyze why car went 500cm instead of 380cm"""
    
    print("=== Speed Calibration Analysis ===")
    print()
    
    # Current system parameters
    expected_distance = 380  # cm
    actual_distance = 500    # cm
    current_speed = 20       # cm/s (our assumption)
    power = 40              # %
    
    print(f"Expected distance: {expected_distance} cm")
    print(f"Actual distance: {actual_distance} cm")
    print(f"Distance ratio: {actual_distance/expected_distance:.2f}x")
    print()
    
    # Calculate what the actual speed must be
    print("=== Movement Analysis ===")
    print("The car made 8 movements:")
    print("- 7 movements of 50cm each = 350cm")
    print("- 1 movement of 30cm = 30cm")
    print("- Total expected = 380cm")
    print()
    
    print("But actual distance was ~500cm")
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
    corrected_time_50cm = 50 / corrected_speed
    corrected_time_30cm = 30 / corrected_speed
    
    print(f"SPEED = {corrected_speed:.1f}  # Real speed at {power}% power")
    print(f"Time for 50cm: {corrected_time_50cm:.2f}s (was 2.5s)")
    print(f"Time for 30cm: {corrected_time_30cm:.2f}s (was 1.5s)")
    print()
    
    # Alternative: reduce movement distance per step
    print("=== Alternative: Reduce Step Distance ===")
    target_step_distance = 50 * (expected_distance / actual_distance)
    print(f"Or reduce step size from 50cm to {target_step_distance:.1f}cm")
    print(f"This would give {380/target_step_distance:.1f} steps instead of 8")

if __name__ == "__main__":
    analyze_speed_problem()
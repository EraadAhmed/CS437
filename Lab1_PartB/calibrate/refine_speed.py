#!/usr/bin/env python3
"""
Calculate refined speed calibration
"""

def refine_speed_calibration():
    """Calculate the corrected speed based on latest test"""
    
    print("=== Speed Calibration Refinement ===")
    print()
    
    # Latest test results
    expected_distance = 380  # cm
    actual_distance = 440    # cm (60cm over)
    current_speed = 26.3     # cm/s
    
    print(f"Expected: {expected_distance} cm")
    print(f"Actual: {actual_distance} cm") 
    print(f"Overshoot: {actual_distance - expected_distance} cm")
    print(f"Error ratio: {actual_distance/expected_distance:.3f}")
    print()
    
    # Calculate corrected speed
    corrected_speed = current_speed * (actual_distance / expected_distance)
    
    print(f"Current SPEED setting: {current_speed} cm/s")
    print(f"Corrected SPEED should be: {corrected_speed:.1f} cm/s")
    print()
    
    # Show impact on movement times
    time_50cm_old = 50 / current_speed
    time_30cm_old = 30 / current_speed
    time_50cm_new = 50 / corrected_speed
    time_30cm_new = 30 / corrected_speed
    
    print("Movement time changes:")
    print(f"50cm: {time_50cm_old:.2f}s -> {time_50cm_new:.2f}s")
    print(f"30cm: {time_30cm_old:.2f}s -> {time_30cm_new:.2f}s")

if __name__ == "__main__":
    refine_speed_calibration()
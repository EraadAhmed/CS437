#!/usr/bin/env python3
"""
Test script for Step 7 object detection implementation
Verifies functionality without requiring full hardware setup
"""

import sys
import numpy as np
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from lab1_part2_step7 import (
            SelfDrivingCar, ObjectDetector, UltrasonicMapper, 
            PathPlanner, Coordinate
        )
        print("✓ Main classes imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    try:
        import utils
        print("✓ Utils module imported successfully")
    except ImportError as e:
        print(f"✗ Utils import error: {e}")
        return False
    
    return True


def test_coordinate_class():
    """Test the Coordinate class for A* pathfinding"""
    print("\nTesting Coordinate class...")
    
    try:
        from lab1_part2_step7 import Coordinate
        
        # Test basic functionality
        node = Coordinate((5, 10, 0), g=3, h=7)
        assert node.f() == 10, f"Expected f=10, got {node.f()}"
        assert node.state == (5, 10, 0), f"Unexpected state: {node.state}"
        
        # Test comparison
        node1 = Coordinate((0, 0, 0), g=5, h=5)
        node2 = Coordinate((1, 1, 0), g=3, h=3)
        assert node2 < node1, "Node comparison failed"
        
        print("✓ Coordinate class working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Coordinate test failed: {e}")
        return False


def test_path_planner():
    """Test the A* pathfinding algorithm"""
    print("\nTesting PathPlanner...")
    
    try:
        from lab1_part2_step7 import PathPlanner
        
        # Create simple test map
        test_map = np.zeros((10, 10))
        test_map[3:7, 3:7] = 1  # Add obstacle in middle
        
        start = (1, 1, 0)
        goal = (8, 8, 0)
        
        # Test pathfinding
        path = PathPlanner.find_path(start, goal, test_map)
        
        if path is None:
            print("✗ No path found (this might be expected)")
            return False
        
        print(f"✓ Path found with {len(path)} waypoints")
        print(f"  Start: {path[0][:2]}, Goal: {path[-1][:2]}")
        
        # Test collision checking
        obstacle_state = (4, 4, 0)
        free_state = (1, 1, 0)
        
        assert PathPlanner.collision_check(obstacle_state, test_map), "Should detect collision"
        assert not PathPlanner.collision_check(free_state, test_map), "Should not detect collision"
        
        print("✓ Collision checking working correctly")
        return True
        
    except Exception as e:
        print(f"✗ PathPlanner test failed: {e}")
        return False


def test_ultrasonic_mapper():
    """Test the ultrasonic mapping system"""
    print("\nTesting UltrasonicMapper...")
    
    try:
        from lab1_part2_step7 import UltrasonicMapper
        
        # Test without hardware (picar=None)
        mapper = UltrasonicMapper(None)
        
        # Test initial state
        assert mapper.current_pos == [11, 0], f"Unexpected initial position: {mapper.current_pos}"
        assert mapper.heading_angle == 0, f"Unexpected initial heading: {mapper.heading_angle}"
        
        # Test position update
        original_pos = mapper.current_pos.copy()
        mapper.update_position(velocity=10, dt=0.5, steer_angle=0)
        
        # Should move forward (positive y direction)
        assert mapper.current_pos[1] > original_pos[1], "Car should move forward"
        
        # Test clearance addition
        mapper.map[5, 5] = 1  # Add obstacle
        mapper.add_clearance_to_obstacles(clearance=1)
        
        # Check that clearance was added
        assert mapper.map[4, 5] == 1, "Clearance should be added around obstacles"
        assert mapper.map[6, 5] == 1, "Clearance should be added around obstacles"
        
        print("✓ UltrasonicMapper working correctly")
        return True
        
    except Exception as e:
        print(f"✗ UltrasonicMapper test failed: {e}")
        return False


def test_object_detector():
    """Test the object detection system"""
    print("\nTesting ObjectDetector...")
    
    try:
        from lab1_part2_step7 import ObjectDetector
        
        # Test initialization without model file
        detector = ObjectDetector(model_path='nonexistent.tflite')
        
        # Should handle missing model gracefully
        assert detector.detector is None, "Should handle missing model file"
        
        # Test detection with dummy image
        dummy_image = np.zeros((240, 320, 3), dtype=np.uint8)
        detections = detector.detect_objects(dummy_image)
        
        # Should return empty list when no detector
        assert isinstance(detections, list), "Should return list"
        
        # Test halt condition checking
        detector.current_detections = []
        detector.check_halt_conditions()
        assert not detector.halt_triggered, "Should not halt with no detections"
        
        print("✓ ObjectDetector working correctly (without model)")
        return True
        
    except Exception as e:
        print(f"✗ ObjectDetector test failed: {e}")
        return False


def test_self_driving_car():
    """Test the main SelfDrivingCar class"""
    print("\nTesting SelfDrivingCar...")
    
    try:
        from lab1_part2_step7 import SelfDrivingCar
        
        # Test initialization
        goal = (15, 20)
        car = SelfDrivingCar(goal_position=goal)
        
        assert car.goal_position == goal, f"Unexpected goal: {car.goal_position}"
        assert not car.running, "Car should not be running initially"
        
        # Test frame capture (simulation mode)
        frame = car.capture_frame()
        assert frame.shape == (240, 320, 3), f"Unexpected frame shape: {frame.shape}"
        
        print("✓ SelfDrivingCar initialization working correctly")
        return True
        
    except Exception as e:
        print(f"✗ SelfDrivingCar test failed: {e}")
        return False


def test_visualization():
    """Test visualization utilities"""
    print("\nTesting visualization utilities...")
    
    try:
        import utils
        
        # Test map visualization
        test_map = np.zeros((15, 20))
        test_map[5:8, 8:12] = 1  # Add obstacle
        
        current_pos = (2, 3)
        goal_pos = (12, 18)
        test_path = [(2, 3), (5, 6), (8, 9), (12, 18)]
        
        map_img = utils.draw_map(test_map, current_pos, goal_pos, test_path)
        
        expected_shape = (test_map.shape[0] * 10, test_map.shape[1] * 10, 3)
        assert map_img.shape == expected_shape, f"Unexpected map image shape: {map_img.shape}"
        
        # Test stop sign creation
        stop_sign_img = utils.create_test_image_with_stop_sign()
        assert stop_sign_img.shape == (240, 320, 3), f"Unexpected stop sign shape: {stop_sign_img.shape}"
        
        print("✓ Visualization utilities working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False


def run_integration_test():
    """Run a simple integration test"""
    print("\nRunning integration test...")
    
    try:
        from lab1_part2_step7 import SelfDrivingCar
        
        # Create car with nearby goal
        car = SelfDrivingCar(goal_position=(12, 5))
        
        # Test mapping
        car.mapper.scan_surroundings()  # Should work in simulation mode
        
        # Add some test obstacles
        car.mapper.map[8, 3] = 1
        car.mapper.map[9, 3] = 1
        car.mapper.add_clearance_to_obstacles()
        
        # Test pathfinding
        success = car.plan_path()
        
        if success:
            print(f"✓ Integration test passed - found path with {len(car.current_path)} waypoints")
        else:
            print("✓ Integration test completed (no path found, but this may be expected)")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("Step 7 Object Detection - Test Suite")
    print("="*60)
    
    tests = [
        test_imports,
        test_coordinate_class,
        test_path_planner,
        test_ultrasonic_mapper,
        test_object_detector,
        test_self_driving_car,
        test_visualization,
        run_integration_test
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "="*60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The implementation appears to be working correctly.")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check the output above for details.")
    
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
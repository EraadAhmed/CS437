#!/usr/bin/env python3
"""
Test the integrated system without hardware initialization to isolate hanging issues.
"""

import asyncio
import logging
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class MockCarController:
    """Mock car controller for testing without hardware."""
    def __init__(self):
        self.servo_angle = 0
        
    def set_servo_angle(self, angle):
        self.servo_angle = angle
        
    def read_ultrasonic(self):
        return 50  # Return constant distance
        
    def move_forward(self, power):
        pass
        
    def move_backward(self, power):
        pass
        
    def turn_left(self, power):
        pass
        
    def turn_right(self, power):
        pass
        
    def stop(self):
        pass

class MockObjectDetector:
    """Mock object detector for testing."""
    def __init__(self):
        pass
        
    def start_detection(self):
        pass
        
    def stop_detection(self):
        pass
        
    def get_detections(self):
        return []

async def test_system_logic():
    """Test the system logic without hardware."""
    logger.info("Starting system logic test...")
    
    try:
        # Import the system components
        from integrated import IntegratedSelfDrivingSystem, SystemConfig
        
        # Create config
        config = SystemConfig()
        
        # Create system WITHOUT hardware
        system = IntegratedSelfDrivingSystem(config, enable_hardware=False)
        
        # Replace with mock implementations
        system.car_controller = MockCarController()
        system.object_detector = MockObjectDetector()
        
        logger.info("Mock system created successfully")
        
        # Test just one iteration without the infinite loop
        logger.info("Testing single control loop iteration...")
        
        # Initialize the running flag
        system.running.set()
        
        # Test single grid operation
        logger.info("Testing grid mark operation...")
        try:
            system.grid.mark_car_position(
                system.current_position,
                system.config.CAR_WIDTH,
                system.config.CAR_LENGTH
            )
            logger.info("Grid operation successful!")
        except Exception as e:
            logger.error(f"Grid operation failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        logger.error(f"System logic test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test function."""
    logger.info("Starting no-hardware test...")
    
    try:
        await test_system_logic()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
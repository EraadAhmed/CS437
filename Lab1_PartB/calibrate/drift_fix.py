# fixed_drift_corrected.py
# CS 437 Lab 1: REAL Drift Correction with Sensor Feedback
# Addresses the actual hardware drift issue using ultrasonic wall detection

import asyncio
import time
import numpy as np
import logging
import math
from threading import Event

# Hardware imports
try:
    from picarx import Picarx
    HW_AVAILABLE = True
except ImportError:
    print("WARNING: PiCarx not available")
    HW_AVAILABLE = False

try:
    from calibrate.fixed_object_detection import ObjectDetector
    DETECTION_AVAILABLE = True
except ImportError:
    print("WARNING: Object detection not available")
    DETECTION_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDriftCorrectedSystem:
    """
    REAL drift correction system that uses ultrasonic sensor to measure
    actual position against walls and corrects steering in real-time.
    """
    
    def __init__(self):
        # Field dimensions
        self.FIELD_WIDTH = 120    # cm
        self.FIELD_LENGTH = 380   # cm
        self.TARGET_X = 60        # Center of hallway
        self.TARGET_Y = 375       # Near end
        
        # AGGRESSIVE servo offset - your hardware needs more correction
        self.BASE_SERVO_OFFSET = 0   # Hardware fixed: no offset needed
        self.current_steering_bias = 0  # Additional real-time correction
        
        # Movement parameters
        self.DRIVE_SPEED = 20.0   # cm/s - slower for better control
        self.DRIVE_POWER = 30     # Lower power for precision
        self.STEP_DISTANCE = 15   # Smaller steps for frequent corrections
        
        # Wall detection parameters
        self.WALL_DETECTION_ANGLES = [-90, -45, 0, 45, 90]  # Servo angles to check walls
        self.MAX_WALL_DISTANCE = 80  # cm - max expected distance to wall
        
        # Control parameters
        self.DRIFT_TOLERANCE = 5.0    # cm - how far off center before correction
        self.MAX_STEERING_CORRECTION = 15  # degrees max additional steering
        self.CORRECTION_GAIN = 1.5    # How aggressively to correct
        
        # State variables
        self.estimated_x = 60.0   # Estimated position
        self.estimated_y = 0.0
        self.actual_x = 60.0      # Measured position using sensors
        self.measurement_count = 0
        
        # Hardware
        self.picarx = None
        self.object_detector = None
        self.stop_event = Event()
        
        logger.info("Real drift correction system initialized")

    async def initialize(self):
        """Initialize hardware with aggressive servo offset."""
        logger.info("Initializing with aggressive drift correction...")
        
        if HW_AVAILABLE:
            try:
                self.picarx = Picarx(servo_pins=["P0", "P1", "P3"])
                # Set aggressive initial servo position
                self.picarx.set_dir_servo_angle(self.BASE_SERVO_OFFSET)
                await asyncio.sleep(1.0)  # Let servo settle
                logger.info(f"PiCar initialized with servo offset: {self.BASE_SERVO_OFFSET}")
            except Exception as e:
                logger.error(f"PiCar initialization failed: {e}")
                raise
        
        # Initialize object detection (simplified)
        if DETECTION_AVAILABLE:
            try:
                self.object_detector = ObjectDetector(confidence_threshold=0.5)
                if self.object_detector.test_detection():
                    self.object_detector.start_detection()
                    logger.info("Object detection initialized")
                else:
                    self.object_detector = None
            except Exception as e:
                logger.warning(f"Object detection failed: {e}")
                self.object_detector = None
        
        logger.info("Initialization complete")

    def measure_distance_to_walls(self):
        """
        Use ultrasonic sensor with servo to measure distance to walls
        and calculate actual X position in hallway.
        """
        if not self.picarx:
            return None
        
        try:
            wall_distances = {}
            
            # Measure distances at different angles
            for angle in self.WALL_DETECTION_ANGLES:
                try:
                    # Point sensor at angle
                    self.picarx.set_cam_pan_angle(angle)
                    await asyncio.sleep(0.1)  # Wait for servo to move
                    
                    # Take multiple readings for accuracy
                    readings = []
                    for _ in range(3):
                        distance = self.picarx.ultrasonic.read()
                        if 0 < distance <= self.MAX_WALL_DISTANCE:
                            readings.append(distance)
                        await asyncio.sleep(0.05)
                    
                    if readings:
                        wall_distances[angle] = sum(readings) / len(readings)
                    
                except Exception as e:
                    logger.warning(f"Wall measurement error at {angle}°: {e}")
            
            # Return sensor to center
            self.picarx.set_cam_pan_angle(0)
            await asyncio.sleep(0.1)
            
            return wall_distances
            
        except Exception as e:
            logger.error(f"Wall measurement failed: {e}")
            return None

    def calculate_actual_position(self, wall_distances):
        """
        Calculate actual X position based on wall measurements.
        Assumes driving in a hallway with walls on left and right.
        """
        if not wall_distances:
            return None
        
        try:
            # Get left and right wall distances
            left_distance = wall_distances.get(-90)  # Left side
            right_distance = wall_distances.get(90)  # Right side
            
            if left_distance and right_distance:
                # Calculate position in hallway
                total_width = left_distance + right_distance
                
                # Check if measurements make sense
                if 100 <= total_width <= 140:  # Reasonable hallway width
                    actual_x = left_distance
                    logger.info(f"Wall measurements: Left={left_distance:.1f}cm, Right={right_distance:.1f}cm, Total={total_width:.1f}cm")
                    return actual_x
                else:
                    logger.warning(f"Unrealistic wall measurements: total width {total_width:.1f}cm")
            
            # Try forward measurements to detect drift
            front_distance = wall_distances.get(0)
            if front_distance:
                logger.info(f"Forward obstacle at {front_distance:.1f}cm")
            
            return None
            
        except Exception as e:
            logger.error(f"Position calculation error: {e}")
            return None

    def calculate_steering_correction(self, position_error):
        """
        Calculate steering correction based on position error.
        """
        # position_error = actual_x - target_x
        # Positive error means too far right, need to steer left (negative angle)
        
        correction = -position_error * self.CORRECTION_GAIN
        correction = max(-self.MAX_STEERING_CORRECTION, 
                        min(self.MAX_STEERING_CORRECTION, correction))
        
        logger.info(f"Position error: {position_error:.1f}cm -> Steering correction: {correction:.1f}°")
        return correction

    async def measure_and_correct_position(self):
        """
        Measure actual position and update steering correction.
        """
        try:
            # Measure wall distances
            wall_distances = await self.measure_distance_to_walls()
            
            if wall_distances:
                # Calculate actual position
                actual_x = self.calculate_actual_position(wall_distances)
                
                if actual_x is not None:
                    self.actual_x = actual_x
                    self.measurement_count += 1
                    
                    # Calculate position error
                    position_error = actual_x - self.TARGET_X
                    
                    # Update steering bias
                    if abs(position_error) > self.DRIFT_TOLERANCE:
                        steering_correction = self.calculate_steering_correction(position_error)
                        self.current_steering_bias = steering_correction
                        
                        logger.warning(f"DRIFT DETECTED: Actual X={actual_x:.1f}cm (target={self.TARGET_X}cm)")
                        logger.warning(f"Applying steering bias: {self.current_steering_bias:.1f}°")
                    else:
                        # Gradually reduce steering bias when on track
                        self.current_steering_bias *= 0.8
                        logger.info(f"On track: X={actual_x:.1f}cm, reducing bias to {self.current_steering_bias:.1f}°")
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Position measurement failed: {e}")
            return False

    async def move_forward_with_correction(self, distance):
        """
        Move forward with real-time steering correction based on measured position.
        """
        if not self.picarx:
            return 0
        
        logger.info(f"Moving forward {distance:.1f}cm with real-time correction")
        
        # Calculate total steering angle
        total_steering = self.BASE_SERVO_OFFSET + self.current_steering_bias
        total_steering = max(-40, min(40, total_steering))  # Limit total steering
        
        logger.info(f"Steering: base={self.BASE_SERVO_OFFSET}° + bias={self.current_steering_bias:.1f}° = {total_steering:.1f}°")
        
        try:
            # Set steering with correction
            self.picarx.set_dir_servo_angle(total_steering)
            await asyncio.sleep(0.1)
            
            # Calculate movement time
            movement_time = distance / self.DRIVE_SPEED
            
            # Move forward
            self.picarx.forward(self.DRIVE_POWER)
            
            # Monitor during movement
            elapsed = 0
            increment = 0.2  # Check every 200ms
            
            while elapsed < movement_time and not self.stop_event.is_set():
                # Check for obstacles
                try:
                    obstacle_distance = self.picarx.ultrasonic.read()
                    if 0 < obstacle_distance <= 20:  # Emergency stop
                        logger.warning(f"Emergency stop: obstacle at {obstacle_distance}cm")
                        break
                except:
                    pass
                
                # Check object detection
                if self.object_detector and self.object_detector.is_halt_needed():
                    logger.warning("Object detection stop")
                    break
                
                await asyncio.sleep(increment)
                elapsed += increment
            
            # Stop
            self.picarx.stop()
            
            # Update estimated position
            actual_distance = (elapsed / movement_time) * distance
            self.estimated_y += actual_distance
            
            logger.info(f"Moved {actual_distance:.1f}cm, estimated position: ({self.actual_x:.1f}, {self.estimated_y:.1f})")
            return actual_distance
            
        except Exception as e:
            logger.error(f"Forward movement failed: {e}")
            if self.picarx:
                self.picarx.stop()
            return 0

    async def navigation_step(self):
        """
        Execute one navigation step with position measurement and correction.
        """
        try:
            # Check if goal reached
            distance_to_goal = math.sqrt(
                (self.actual_x - self.TARGET_X)**2 + 
                (self.estimated_y - self.TARGET_Y)**2
            )
            
            if distance_to_goal < 20:
                logger.info("GOAL REACHED!")
                return False
            
            # Measure and correct position every few steps
            if self.measurement_count % 3 == 0:  # Every 3rd step
                logger.info("Measuring actual position...")
                await self.measure_and_correct_position()
            
            # Move forward with correction
            actual_distance = await self.move_forward_with_correction(self.STEP_DISTANCE)
            
            if actual_distance < 2:  # Barely moved, probably obstacle
                logger.warning("Movement blocked, trying to navigate around obstacle")
                return False
            
            # Progress report
            progress = (self.estimated_y / self.TARGET_Y) * 100
            logger.info(f"Progress: {progress:.1f}% - Estimated: ({self.actual_x:.1f}, {self.estimated_y:.1f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Navigation step failed: {e}")
            return False

    async def main_loop(self):
        """Main control loop."""
        logger.info("Starting main navigation loop...")
        
        step_count = 0
        
        try:
            while not self.stop_event.is_set():
                step_count += 1
                logger.info(f"--- Navigation Step {step_count} ---")
                
                # Execute navigation step
                continue_nav = await self.navigation_step()
                
                if not continue_nav:
                    logger.info("Navigation complete or blocked")
                    break
                
                # Small delay between steps
                await asyncio.sleep(0.5)
                
        except KeyboardInterrupt:
            logger.info("Navigation interrupted by user")
        except Exception as e:
            logger.error(f"Main loop error: {e}")
        finally:
            if self.picarx:
                self.picarx.stop()

    async def run(self):
        """Main run method."""
        try:
            await self.initialize()
            await self.main_loop()
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down...")
        self.stop_event.set()
        
        if self.picarx:
            self.picarx.stop()
        
        if self.object_detector:
            self.object_detector.stop_detection()
        
        logger.info("Shutdown complete")


# Calibration function to find optimal servo offset
async def calibrate_aggressive_offset():
    """
    Interactive calibration to find the right servo offset for your hardware.
    """
    if not HW_AVAILABLE:
        logger.error("Hardware not available for calibration")
        return
    
    logger.info("AGGRESSIVE SERVO OFFSET CALIBRATION")
    logger.info("This will test different servo offsets to find the one that makes your car go straight")
    
    picarx = Picarx(servo_pins=["P0", "P1", "P3"])
    
    # Test increasingly aggressive offsets
    test_offsets = [-5, -8, -10, -12, -15, -18, -20]
    
    for offset in test_offsets:
        logger.info(f"\n--- Testing servo offset: {offset} ---")
        logger.info("The car will drive forward for 3 seconds.")
        logger.info("Watch carefully to see if it goes straight or drifts.")
        
        input("Press Enter when ready to test this offset...")
        
        try:
            picarx.set_dir_servo_angle(offset)
            await asyncio.sleep(0.5)
            
            picarx.forward(30)
            await asyncio.sleep(3.0)
            picarx.stop()
            
            response = input(f"With offset {offset}, did the car go straight? (y/n/q to quit): ")
            
            if response.lower() == 'y':
                logger.info(f"SUCCESS! Use SERVO_OFFSET = {offset} in your code")
                break
            elif response.lower() == 'q':
                break
                
        except Exception as e:
            logger.error(f"Test failed: {e}")
    
    picarx.stop()
    logger.info("Calibration complete")


# Quick test function
async def quick_drift_test():
    """
    Quick test to measure actual drift without full navigation.
    """
    system = RealDriftCorrectedSystem()
    await system.initialize()
    
    logger.info("Quick drift test - measuring wall positions...")
    
    try:
        for i in range(5):
            logger.info(f"Measurement {i+1}/5")
            await system.measure_and_correct_position()
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
    
    await system.shutdown()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--calibrate":
            asyncio.run(calibrate_aggressive_offset())
        elif sys.argv[1] == "--test":
            asyncio.run(quick_drift_test())
        else:
            print("Usage: python drift_fix.py [--calibrate|--test]")
    else:
        # Run main system
        system = RealDriftCorrectedSystem()
        asyncio.run(system.run())
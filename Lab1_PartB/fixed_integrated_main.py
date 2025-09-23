# fixed_integrated_main.py
# CS 437 Lab 1: Fixed Self-Driving Car System
# Addresses drift issues, object detection problems, and navigation consistency

import asyncio
import time
import numpy as np
import logging
from threading import Event
import math

# Hardware imports with error handling
try:
    from picarx import Picarx
    HW_AVAILABLE = True
except ImportError:
    print("WARNING: PiCarx not available - running in simulation mode")
    HW_AVAILABLE = False

# Import our fixed object detection
try:
    from fixed_object_detection import ObjectDetector
    DETECTION_AVAILABLE = True
except ImportError:
    print("WARNING: Object detection not available")
    DETECTION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedSelfDrivingSystem:
    """
    Fixed self-driving system with improved:
    - Drift correction and position tracking
    - Reliable object detection integration
    - Consistent coordinate system
    - Better error handling
    """
    
    def __init__(self):
        # Physical constants - well-calibrated for your setup
        self.FIELD_WIDTH = 120    # cm
        self.FIELD_LENGTH = 380   # cm
        self.CAR_WIDTH = 14       # cm
        self.CAR_LENGTH = 23      # cm
        
        # CRITICAL: Servo calibration to fix drift
        # Start with -5 and adjust based on actual drift
        self.SERVO_OFFSET = 0  # Hardware fixed: no offset needed
        self.DRIFT_CORRECTION_FACTOR = 0.95  # Multiply distance by this to account for systematic error
        
        # Movement calibration
        self.DRIVE_SPEED = 25.0   # cm/s - conservative for accuracy
        self.DRIVE_POWER = 35     # PWM power level
        self.TURN_POWER = 35
        
        # Position tracking - CRITICAL for drift detection
        self.current_x = 60.0     # Start at center of hallway
        self.current_y = 0.0      # Start at beginning
        self.current_theta = 0.0  # Facing forward
        self.target_x = 60.0      # Goal at center
        self.target_y = 375.0     # Goal near end (5cm margin)
        
        # Drift detection and correction
        self.position_history = []
        self.expected_x = 60.0    # Expected X position for straight driving
        self.max_drift_tolerance = 8.0  # cm before correction
        
        # Safety distances
        self.EMERGENCY_STOP_DISTANCE = 15  # cm
        self.DETECTION_STOP_DISTANCE = 25  # cm - stop for camera detection
        self.SAFE_DISTANCE = 30            # cm - maintain this distance
        
        # Control timing
        self.CONTROL_FREQUENCY = 15  # Hz
        self.DETECTION_FREQUENCY = 3 # Hz - slower but more reliable
        self.MOVEMENT_STEP_SIZE = 20 # cm per movement step
        
        # Initialize hardware
        self.picarx = None
        self.object_detector = None
        self.stop_event = Event()
        
        # State tracking
        self.last_detection_time = 0
        self.distance_since_detection = 0
        self.is_moving = False
        self.consecutive_failures = 0
        
        logger.info("Fixed self-driving system initialized")

    async def initialize(self):
        """Initialize all hardware components with robust error handling."""
        logger.info("Initializing hardware...")
        
        # Initialize PiCar
        if HW_AVAILABLE:
            try:
                self.picarx = Picarx(servo_pins=["P0", "P1", "P3"])
                # Set initial servo position with offset
                self.picarx.set_dir_servo_angle(self.SERVO_OFFSET)
                await asyncio.sleep(0.5)  # Allow servo to settle
                logger.info("PiCar initialized successfully")
            except Exception as e:
                logger.error(f"PiCar initialization failed: {e}")
                raise
        else:
            logger.warning("Running without PiCar hardware")
        
        # Initialize object detection
        if DETECTION_AVAILABLE:
            try:
                self.object_detector = ObjectDetector(
                    confidence_threshold=0.4,  # Balanced threshold
                    max_detections=5
                )
                
                # Test detection before starting
                if self.object_detector.test_detection():
                    detection_started = self.object_detector.start_detection()
                    if detection_started:
                        logger.info("Object detection initialized successfully")
                    else:
                        logger.warning("Object detection failed to start")
                        self.object_detector = None
                else:
                    logger.warning("Object detection test failed")
                    self.object_detector = None
                    
            except Exception as e:
                logger.error(f"Object detection initialization failed: {e}")
                self.object_detector = None
        else:
            logger.warning("Running without object detection")
        
        logger.info("Initialization complete")

    def update_position(self, distance_moved, turn_angle=0):
        """
        Update position tracking with drift detection.
        CRITICAL: This fixes the coordinate system confusion.
        """
        # Apply drift correction to distance
        corrected_distance = distance_moved * self.DRIFT_CORRECTION_FACTOR
        
        # Update position based on current heading
        self.current_x += corrected_distance * math.sin(self.current_theta)
        self.current_y += corrected_distance * math.cos(self.current_theta)
        
        # Update heading if turn was made
        if turn_angle != 0:
            self.current_theta += math.radians(turn_angle)
            # Normalize to [-π, π]
            while self.current_theta > math.pi:
                self.current_theta -= 2 * math.pi
            while self.current_theta < -math.pi:
                self.current_theta += 2 * math.pi
        
        # Record position for drift analysis
        self.position_history.append({
            'x': self.current_x,
            'y': self.current_y,
            'theta': self.current_theta,
            'time': time.time()
        })
        
        # Keep only recent history (last 10 positions)
        if len(self.position_history) > 10:
            self.position_history.pop(0)
        
        logger.info(f"Position updated: ({self.current_x:.1f}, {self.current_y:.1f}), heading: {math.degrees(self.current_theta):.1f}°")

    def detect_drift(self):
        """
        Detect if car is drifting off course.
        Returns: (is_drifting, drift_amount_cm, suggested_correction)
        """
        # For straight driving, check X deviation from expected path
        if abs(self.current_theta) < 0.2:  # Roughly straight
            drift_amount = abs(self.current_x - self.expected_x)
            
            if drift_amount > self.max_drift_tolerance:
                # Determine correction direction
                if self.current_x > self.expected_x:
                    correction = "steer_left"  # Drifting right, steer left
                else:
                    correction = "steer_right"  # Drifting left, steer right
                
                logger.warning(f"DRIFT DETECTED: {drift_amount:.1f}cm, correction: {correction}")
                return True, drift_amount, correction
        
        return False, 0.0, None

    async def correct_drift(self, correction_type, drift_amount):
        """Apply drift correction maneuver."""
        logger.info(f"Applying drift correction: {correction_type}")
        
        if not self.picarx:
            return
        
        # Calculate correction strength based on drift amount
        correction_angle = min(15, drift_amount * 2)  # Max 15 degrees
        correction_time = 0.3  # Brief correction
        
        try:
            if correction_type == "steer_left":
                self.picarx.set_dir_servo_angle(-correction_angle + self.SERVO_OFFSET)
            else:  # steer_right
                self.picarx.set_dir_servo_angle(correction_angle + self.SERVO_OFFSET)
            
            # Move forward briefly with correction
            self.picarx.forward(self.DRIVE_POWER)
            await asyncio.sleep(correction_time)
            
            # Return to center
            self.picarx.set_dir_servo_angle(self.SERVO_OFFSET)
            await asyncio.sleep(0.2)
            
            # Update position estimate
            correction_distance = self.DRIVE_SPEED * correction_time / 100  # Convert to cm
            self.update_position(correction_distance)
            
            logger.info("Drift correction completed")
            
        except Exception as e:
            logger.error(f"Drift correction failed: {e}")
        finally:
            if self.picarx:
                self.picarx.stop()

    async def safe_forward_movement(self, target_distance):
        """
        Move forward with continuous safety monitoring and drift correction.
        Returns: actual_distance_moved
        """
        if not self.picarx:
            return 0
        
        logger.info(f"Moving forward {target_distance:.1f}cm with safety monitoring")
        
        # Calculate movement time
        movement_time = target_distance / self.DRIVE_SPEED  # seconds
        increment_time = 0.1  # Check every 100ms
        total_time = 0
        distance_moved = 0
        
        try:
            # Ensure steering is centered with offset
            self.picarx.set_dir_servo_angle(self.SERVO_OFFSET)
            await asyncio.sleep(0.1)
            
            self.picarx.forward(self.DRIVE_POWER)
            self.is_moving = True
            
            while total_time < movement_time and not self.stop_event.is_set():
                # Check for obstacles
                if self.picarx.ultrasonic:
                    try:
                        obstacle_distance = self.picarx.ultrasonic.read()
                        if 0 < obstacle_distance <= self.EMERGENCY_STOP_DISTANCE:
                            logger.warning(f"EMERGENCY STOP: Obstacle at {obstacle_distance}cm")
                            break
                    except Exception as e:
                        logger.warning(f"Ultrasonic read error: {e}")
                
                # Check object detection
                if self.object_detector and self.object_detector.is_halt_needed():
                    logger.warning("HALT: Object detected by camera")
                    break
                
                # Check for drift every few increments
                if total_time > 0.5:  # After 0.5 seconds
                    is_drifting, drift_amount, correction = self.detect_drift()
                    if is_drifting:
                        # Apply real-time steering correction
                        try:
                            if correction == "steer_left":
                                self.picarx.set_dir_servo_angle(-5 + self.SERVO_OFFSET)
                            else:
                                self.picarx.set_dir_servo_angle(5 + self.SERVO_OFFSET)
                            await asyncio.sleep(0.1)
                            self.picarx.set_dir_servo_angle(self.SERVO_OFFSET)
                        except Exception as e:
                            logger.warning(f"Real-time correction failed: {e}")
                
                await asyncio.sleep(increment_time)
                total_time += increment_time
                distance_moved = (total_time / movement_time) * target_distance
            
        except Exception as e:
            logger.error(f"Forward movement error: {e}")
        finally:
            self.picarx.stop()
            self.is_moving = False
        
        # Update position with actual distance moved
        actual_distance = min(distance_moved, target_distance)
        self.update_position(actual_distance)
        self.distance_since_detection += actual_distance
        
        logger.info(f"Moved {actual_distance:.1f}cm (planned: {target_distance:.1f}cm)")
        return actual_distance

    async def turn_towards_goal(self):
        """Turn to face the goal with improved accuracy."""
        if not self.picarx:
            return False
        
        # Calculate required heading to goal
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        
        if dy <= 0:  # Already at or past goal
            return True
        
        required_heading = math.atan2(dx, dy)  # atan2(x, y) for our coordinate system
        heading_error = required_heading - self.current_theta
        
        # Normalize heading error
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Only turn if error is significant
        if abs(heading_error) < 0.1:  # ~5.7 degrees
            logger.info("Heading OK, no turn needed")
            return True
        
        turn_angle_deg = math.degrees(heading_error)
        logger.info(f"Turning {turn_angle_deg:.1f} degrees towards goal")
        
        try:
            # Set steering for turn
            if turn_angle_deg > 0:  # Turn left
                self.picarx.set_dir_servo_angle(-25 + self.SERVO_OFFSET)
            else:  # Turn right
                self.picarx.set_dir_servo_angle(25 + self.SERVO_OFFSET)
            
            # Execute turn
            turn_time = abs(turn_angle_deg) / 90 * 0.4  # Calibrated turn time
            turn_time = max(0.2, min(turn_time, 1.0))  # Constrain turn time
            
            self.picarx.forward(self.TURN_POWER)
            await asyncio.sleep(turn_time)
            self.picarx.stop()
            
            # Return steering to center
            self.picarx.set_dir_servo_angle(self.SERVO_OFFSET)
            await asyncio.sleep(0.3)
            
            # Update heading
            self.update_position(0, turn_angle_deg)
            
            return True
            
        except Exception as e:
            logger.error(f"Turn failed: {e}")
            return False

    async def handle_object_detection(self):
        """Handle detected objects (stop signs, people, etc.)."""
        if not self.object_detector:
            return
        
        if self.object_detector.is_halt_needed():
            logger.info("Object detected - handling...")
            
            # Get detection statistics
            stats = self.object_detector.get_detection_stats()
            logger.info(f"Detection stats: {stats}")
            
            # Stop and wait
            if self.picarx:
                self.picarx.stop()
            
            self.is_moving = False
            
            # For stop signs - pause and continue
            # For people - wait until clear
            pause_time = 3.0  # seconds
            logger.info(f"Pausing for {pause_time} seconds due to detected object")
            await asyncio.sleep(pause_time)
            
            # Reset detection timer
            self.last_detection_time = time.time()
            self.distance_since_detection = 0

    async def navigation_step(self):
        """Execute one navigation step with integrated safety and drift correction."""
        try:
            # Check if we need to stop for detection
            if (self.distance_since_detection >= 30 or  # Every 30cm
                time.time() - self.last_detection_time > 5):  # Every 5 seconds
                await self.handle_object_detection()
                self.last_detection_time = time.time()
                self.distance_since_detection = 0
            
            # Calculate distance to goal
            distance_to_goal = math.sqrt(
                (self.target_x - self.current_x)**2 + 
                (self.target_y - self.current_y)**2
            )
            
            logger.info(f"Distance to goal: {distance_to_goal:.1f}cm")
            
            # Check if goal reached
            if distance_to_goal < 15:  # 15cm tolerance
                logger.info("GOAL REACHED!")
                return False  # Stop navigation
            
            # Turn towards goal if needed
            await self.turn_towards_goal()
            
            # Move forward with safety monitoring
            step_distance = min(self.MOVEMENT_STEP_SIZE, distance_to_goal)
            actual_distance = await self.safe_forward_movement(step_distance)
            
            # Check for drift and correct if needed
            is_drifting, drift_amount, correction = self.detect_drift()
            if is_drifting and not self.is_moving:
                await self.correct_drift(correction, drift_amount)
            
            # Log progress
            completion_percent = (self.current_y / self.target_y) * 100
            logger.info(f"Progress: {completion_percent:.1f}% complete")
            
            return True  # Continue navigation
            
        except Exception as e:
            logger.error(f"Navigation step failed: {e}")
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= 3:
                logger.error("Too many consecutive failures - stopping")
                return False
            
            await asyncio.sleep(1)  # Brief pause before retry
            return True

    async def main_control_loop(self):
        """Main control loop with comprehensive error handling."""
        logger.info("Starting main control loop...")
        
        try:
            while not self.stop_event.is_set():
                loop_start = time.time()
                
                # Execute navigation step
                continue_navigation = await self.navigation_step()
                
                if not continue_navigation:
                    logger.info("Navigation complete or stopped")
                    break
                
                # Maintain control frequency
                elapsed = time.time() - loop_start
                sleep_time = max(0, 1.0/self.CONTROL_FREQUENCY - elapsed)
                await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Control loop interrupted by user")
        except Exception as e:
            logger.error(f"Control loop error: {e}")
        finally:
            if self.picarx:
                self.picarx.stop()
            logger.info("Control loop stopped")

    async def run(self):
        """Main run method."""
        try:
            await self.initialize()
            await self.main_control_loop()
        except KeyboardInterrupt:
            logger.info("System stopped by user")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Clean shutdown of all systems."""
        logger.info("Shutting down...")
        self.stop_event.set()
        
        if self.picarx:
            self.picarx.stop()
        
        if self.object_detector:
            self.object_detector.stop_detection()
        
        logger.info("Shutdown complete")


# Test and calibration functions
async def test_drift_detection():
    """Test drift detection and correction."""
    system = FixedSelfDrivingSystem()
    await system.initialize()
    
    logger.info("Testing drift detection...")
    
    # Simulate drift
    system.current_x = 75  # 15cm off center
    is_drifting, amount, correction = system.detect_drift()
    
    logger.info(f"Drift test: drifting={is_drifting}, amount={amount:.1f}cm, correction={correction}")
    
    if is_drifting:
        await system.correct_drift(correction, amount)
    
    await system.shutdown()

async def calibrate_servo_offset():
    """Interactive servo offset calibration."""
    if not HW_AVAILABLE:
        logger.error("Hardware not available for calibration")
        return
    
    logger.info("Starting servo offset calibration...")
    logger.info("The car will drive straight for 3 seconds.")
    logger.info("Observe if it drifts left or right.")
    
    picarx = Picarx(servo_pins=["P0", "P1", "P3"])
    
    for offset in range(-10, 11, 2):
        logger.info(f"Testing servo offset: {offset}")
        
        picarx.set_dir_servo_angle(offset)
        await asyncio.sleep(0.5)
        
        picarx.forward(35)
        await asyncio.sleep(3)
        picarx.stop()
        
        response = input(f"Offset {offset}: Did it go straight? (y/n/q to quit): ")
        if response.lower() == 'y':
            logger.info(f"Optimal servo offset found: {offset}")
            break
        elif response.lower() == 'q':
            break
    
    picarx.stop()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-drift":
            asyncio.run(test_drift_detection())
        elif sys.argv[1] == "--calibrate":
            asyncio.run(calibrate_servo_offset())
        else:
            print("Usage: python fixed_integrated_main.py [--test-drift|--calibrate]")
    else:
        # Run main system
        system = FixedSelfDrivingSystem()
        asyncio.run(system.run())
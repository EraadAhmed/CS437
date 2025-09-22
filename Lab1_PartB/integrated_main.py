# integrated_main.py
# CS 437 Lab 1 Step 7: Integrated Self-Driving System
# Combines ultrasonic mapping, object detection, and car control

import asyncio
import time
import numpy as np
import logging
from threading import Event

from picarx import Picarx
# from computer_vision import ultrasonic_pan_loop, car_pixels, print_map  # Not used in this implementation
from car_control import hybrid_a_star, Coordinate
from object_detection import ObjectDetector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IntegratedSelfDrivingSystem:
    """
    Integrated self-driving system combining:
    - Ultrasonic sensor mapping
    - TensorFlow Lite object detection  
    - Hybrid A* path planning
    - Car control
    """
    
    def __init__(self):
        # Physical constants (from your existing code)
        self.WIDTH = 120  # 120cm width of hallway
        self.LENGTH = 380  # 380cm to the endpoint
        self.X_MID = 60  # Middle of 120cm hallway
        self.CAR_WIDTH = 14
        self.CAR_LENGTH = 23
        self.MAXREAD = 100
        self.SAMPLING = 1  # 1cm per grid unit
        
        # Car control constants
        self.SPEED = 30.5  # Refined speed calibration: was 26.3, but car went 440cm instead of 380cm
        self.POWER = 40
        self.DELTA_T = 0.5  # Increased from 0.25 to make larger steps
        
        # Servo calibration - adjust this if car drifts left/right when going "straight"
        self.SERVO_OFFSET = -2  # Reduced from -3 since car drifted slightly left
        
        # Initialize positions - start at beginning middle, goal 380cm forward
        self.start_pos = Coordinate((60, 0, 0))   # Middle of 120cm width, at 0cm length
        self.goal_pos = Coordinate((60, 380, 0))  # 380cm forward, staying in middle
        
        # System components
        self.picarx = None
        self.object_detector = None
        self.stop_event = Event()
        
        # State variables
        self.current_pos = self.start_pos.state  # Get the (x, y, theta) tuple from Coordinate
        self.map_grid = np.zeros((int(self.LENGTH/self.SAMPLING), int(self.WIDTH/self.SAMPLING)))
        self.planned_path = []
        self.path_index = 0
        self.last_replan_time = 0
        self.halt_duration = 0
        
        # Control loop timing
        self.loop_frequency = 10  # 10 Hz as specified in design
        self.replan_interval = 2.0  # Replan every 2 seconds
        self.max_halt_time = 5.0  # Maximum time to wait for safety objects
        
    async def initialize(self):
        """Initialize all system components."""
        logging.info("Initializing integrated self-driving system...")
        
        # Initialize PiCar
        try:
            self.picarx = Picarx(servo_pins=["P0", "P1", "P3"])
            logging.info("PiCar initialized")
        except Exception as e:
            logging.error(f"Failed to initialize PiCar: {e}")
            raise
        
        # Initialize object detector
        try:
            self.object_detector = ObjectDetector(
                model_path='efficientdet_lite0.tflite',
                confidence_threshold=0.3,
                num_threads=2,
                enable_edge_tpu=False
            )
            self.object_detector.start_detection()
            logging.info("Object detector initialized")
        except Exception as e:
            logging.error(f"Failed to initialize object detector: {e}")
            raise
        
        # Initialize car position in map
        await self._update_car_position()
        
        # Initial calibration scan
        await self._perform_calibration_scan()
        
        logging.info("System initialization complete")

    async def _update_car_position(self):
        """Update car position in the occupancy grid."""
        samp_car_width = int(np.ceil(self.CAR_WIDTH/self.SAMPLING))
        samp_car_length = int(np.ceil(self.CAR_LENGTH/self.SAMPLING))
        
        # Clear previous car position
        self.map_grid[self.map_grid == 2] = 0
        
        # Mark current car position
        x, y, _ = self.current_pos
        for i in range(max(0, int(x - samp_car_width/2)), 
                      min(int(self.WIDTH/self.SAMPLING), int(x + samp_car_width/2))):
            for j in range(max(0, int(y - samp_car_length)), 
                          min(int(self.LENGTH/self.SAMPLING), int(y))):
                self.map_grid[j][i] = 2

    async def _perform_calibration_scan(self):
        """Perform initial 360-degree ultrasonic scan for mapping."""
        logging.info("Performing calibration scan...")
        logging.info(f"Current pos type: {type(self.current_pos)}, value: {self.current_pos}")
        
        self.picarx.set_cam_pan_angle(0)
        await asyncio.sleep(1)
        
        angle = -90
        while angle <= 90 and not self.stop_event.is_set():
            self.picarx.set_cam_pan_angle(angle)
            await asyncio.sleep(0.1)
            
            reading = self.picarx.ultrasonic.read()
            
            if 0 < reading <= self.MAXREAD:
                await self._update_map_with_reading(angle, reading)
            
            angle += 5  # Faster scan for calibration
        
        self.picarx.set_cam_pan_angle(0)  # Return to center
        await asyncio.sleep(1)
        
        logging.info("Calibration scan complete")

    async def _update_map_with_reading(self, angle, reading):
        """Update occupancy grid with ultrasonic reading."""
        x, y, theta = self.current_pos
        
        if angle == 0:
            # Straight ahead
            object_x = x
            object_y = min(int(self.LENGTH/self.SAMPLING) - 1, 
                          y + int(reading / self.SAMPLING))
        else:
            # Angled reading
            object_x = int((x + reading * np.sin(np.radians(angle)))/self.SAMPLING)
            object_y = int((y + reading * np.cos(np.radians(angle)))/self.SAMPLING)
        
        # Bounds checking
        if (0 <= object_x < int(self.WIDTH/self.SAMPLING) and 
            0 <= object_y < int(self.LENGTH/self.SAMPLING)):
            self.map_grid[object_y][object_x] = 1  # Obstacle

    async def _plan_path(self):
        """Plan path from current position to goal."""
        try:
            # Check if this is a simple straight-line case
            if (abs(self.current_pos[0] - self.goal_pos.state[0]) < 5 and  # Same X lane (within 5cm)
                self.current_pos[1] < self.goal_pos.state[1]):             # Moving forward in Y
                
                logging.info("Using simple straight-line path planning")
                path = self._plan_straight_line_path()
                if path:
                    self.planned_path = path
                    self.path_index = 0
                    logging.info(f"Straight-line path planned with {len(path)} waypoints")
                    return True
            
            # Use full Hybrid A* for complex paths
            logging.info("Using Hybrid A* path planning")
            
            # Inflate obstacles for safety
            inflated_map = self._inflate_obstacles(self.map_grid)
            
            # Plan path using Hybrid A*
            path = await hybrid_a_star(
                self.current_pos, 
                self.goal_pos.state,  # Extract the state tuple from Coordinate object
                inflated_map,
                self.WIDTH, 
                self.CAR_WIDTH, 
                self.SPEED, 
                self.DELTA_T,
                self.CAR_LENGTH
            )
            
            if path:
                self.planned_path = path
                self.path_index = 0
                logging.info(f"Path planned with {len(path)} waypoints")
                return True
            else:
                logging.warning("No path found to goal")
                return False
                
        except Exception as e:
            logging.error(f"Path planning failed: {e}")
            return False

    def _plan_straight_line_path(self):
        """Create a simple straight-line path for simple cases."""
        path = []
        current_y = self.current_pos[1]
        goal_y = self.goal_pos.state[1]
        x = self.current_pos[0]  # Stay in same lane
        
        # Create waypoints every 50 units
        step_size = 50
        y = current_y
        
        while y < goal_y:
            y = min(y + step_size, goal_y)
            path.append((x, y, 0))  # (x, y, theta)
            
        return path

    def _inflate_obstacles(self, map_grid, inflation_radius=2):
        """Inflate obstacles to account for car width."""
        inflated = map_grid.copy()
        
        obstacles = np.where(map_grid == 1)
        for oy, ox in zip(obstacles[0], obstacles[1]):
            for dy in range(-inflation_radius, inflation_radius + 1):
                for dx in range(-inflation_radius, inflation_radius + 1):
                    ny, nx = oy + dy, ox + dx
                    if (0 <= ny < inflated.shape[0] and 
                        0 <= nx < inflated.shape[1] and
                        inflated[ny][nx] == 0):  # Don't overwrite car position
                        inflated[ny][nx] = 1
        
        return inflated

    async def _execute_motion_command(self, target_state):
        """Execute motion to reach target state."""
        current_x, current_y, current_theta = self.current_pos
        target_x, target_y, target_theta = target_state
        
        # Calculate required motion
        dx = target_x - current_x
        dy = target_y - current_y
        
        distance = np.sqrt(dx**2 + dy**2) * self.SAMPLING  # Convert to cm
        
        if distance > 1:  # Only move if significant distance
            # Calculate required heading
            # COORDINATE SYSTEM: X=left/right, Y=forward/backward
            # For forward motion (positive dy), heading should be 0
            # For rightward motion (positive dx), heading should be -pi/2  
            if abs(dx) < 1:  # Moving straight forward/backward
                if dy > 0:
                    target_heading = 0  # Forward
                else:
                    target_heading = np.pi  # Backward
            elif abs(dy) < 1:  # Moving straight left/right
                if dx > 0:
                    target_heading = -np.pi/2  # Right
                else:
                    target_heading = np.pi/2   # Left
            else:
                # Diagonal movement - use standard arctan2 but swap coordinates
                # to match our coordinate system (Y=forward, X=right)
                target_heading = np.arctan2(-dx, dy)  # Note: -dx because right is negative angle
            
            heading_error = target_heading - current_theta
            
            # Normalize heading error
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi
            
            # Execute turn if needed
            if abs(heading_error) > 0.1:  # 0.1 radians ~ 6 degrees
                turn_angle = np.degrees(heading_error)
                logging.info(f"Executing turn: {turn_angle:.1f}° (heading error: {np.degrees(heading_error):.1f}°)")
                await self._turn_car(turn_angle)
                self.current_pos = (current_x, current_y, target_heading)
            else:
                logging.info(f"No turn needed (heading error: {np.degrees(heading_error):.1f}°, target: {np.degrees(target_heading):.1f}°)")
            
            # Execute forward motion
            move_time = distance / self.SPEED  # Time in seconds
            logging.info(f"Moving forward: distance={distance:.1f}cm, time={move_time:.1f}s, power={self.POWER}")
            await self._move_forward(move_time)
            
            # Update position based on actual movement (more realistic)
            # Instead of assuming perfect movement to target_state, 
            # update based on the distance we attempted to move
            actual_distance_moved = self.SPEED * move_time / self.SAMPLING  # Convert back to grid units
            
            # Calculate actual new position based on our coordinate system
            # COORDINATE SYSTEM: X=left/right, Y=forward/backward, heading=0 is forward
            # heading=0 → forward (+Y), heading=pi/2 → left (+X), heading=-pi/2 → right (-X)
            new_x = current_x + actual_distance_moved * np.sin(target_heading)   # Left/right movement
            new_y = current_y + actual_distance_moved * np.cos(target_heading)   # Forward/backward movement
            
            # Update to realistic position (not perfect target)
            self.current_pos = (int(round(new_x)), int(round(new_y)), target_heading)
            await self._update_car_position()
            
            # Log the movement for debugging
            logging.info(f"Moved from ({current_x}, {current_y}) to {self.current_pos[:2]}, target was ({target_x}, {target_y})")

    async def _turn_car(self, angle_degrees):
        """Turn car by specified angle."""
        # Convert angle to servo angle and duration
        if angle_degrees > 0:  # Turn right
            self.picarx.set_dir_servo_angle(30 + self.SERVO_OFFSET)
        else:  # Turn left
            self.picarx.set_dir_servo_angle(-30 + self.SERVO_OFFSET)
        
        # Move forward briefly to execute turn
        self.picarx.forward(self.POWER)
        await asyncio.sleep(abs(angle_degrees) / 90 * 0.5)  # Rough timing
        self.picarx.stop()
        
        # Return steering to center with calibration offset
        self.picarx.set_dir_servo_angle(self.SERVO_OFFSET)
        await asyncio.sleep(0.2)  # Extra time for servo to fully center

    async def _move_forward(self, duration):
        """Move car forward for specified duration."""
        # Ensure steering is centered with calibration offset
        self.picarx.set_dir_servo_angle(self.SERVO_OFFSET)
        await asyncio.sleep(0.1)  # Give servo time to center
        
        self.picarx.forward(self.POWER)
        await asyncio.sleep(duration)
        self.picarx.stop()

    async def _continuous_mapping(self):
        """Continuous ultrasonic mapping in background."""
        angle = 0
        direction = 5
        
        while not self.stop_event.is_set():
            # Quick ultrasonic reading
            reading = self.picarx.ultrasonic.read()
            if 0 < reading <= self.MAXREAD:
                await self._update_map_with_reading(angle, reading)
            
            # Update angle for next scan
            angle += direction
            if angle >= 30 or angle <= -30:  # Limited range for forward driving
                direction *= -1
            
            await asyncio.sleep(0.1)

    async def _object_detection_monitor(self):
        """Monitor object detection results."""
        while not self.stop_event.is_set():
            if self.object_detector.is_halt_needed():
                # Check what specific objects were detected
                detection_result = self.object_detector.get_latest_detection()
                stop_sign_detected = False
                
                if detection_result and 'detections' in detection_result:
                    for detection in detection_result['detections']:
                        if 'stop_sign' in detection['class_name'].lower():
                            stop_sign_detected = True
                            logging.info("STOP SIGN DETECTED - Initiating stop and recalibration")
                            break
                
                if stop_sign_detected:
                    logging.warning("STOP SIGN: Stopping for recalibration")
                else:
                    logging.warning("Safety object detected - initiating halt")
                
                self.halt_duration = time.time()
                
                # Stop car immediately
                self.picarx.stop()
                
                # If stop sign, perform recalibration
                if stop_sign_detected:
                    await asyncio.sleep(2.0)  # Stop for 2 seconds at stop sign
                    logging.info("Performing stop sign recalibration scan...")
                    await self._perform_calibration_scan()
                    logging.info("Recalibration complete - resuming")
                else:
                    # Wait for other objects to clear or timeout
                    while (self.object_detector.is_halt_needed() and 
                           time.time() - self.halt_duration < self.max_halt_time and
                           not self.stop_event.is_set()):
                        await asyncio.sleep(0.1)
                    
                    if time.time() - self.halt_duration >= self.max_halt_time:
                        logging.warning("Halt timeout reached - attempting recovery")
                        await self._recovery_maneuver()
                    else:
                        logging.info("Safety object cleared - resuming")
                
                self.halt_duration = 0
            
            await asyncio.sleep(0.1)

    async def _recovery_maneuver(self):
        """Simple recovery maneuver when blocked."""
        logging.info("Executing recovery maneuver")
        
        current_x, current_y, current_theta = self.current_pos
        
        # Back up slightly
        logging.info("Recovery: Backing up 1 second")
        self.picarx.backward(self.POWER)
        await asyncio.sleep(1)
        self.picarx.stop()
        
        # Update position for backup (assume ~20cm backward)
        backup_distance = 20  # cm, approximate
        backup_distance_grid = backup_distance / self.SAMPLING
        
        # Move backward in current heading direction
        new_x = current_x - backup_distance_grid * np.sin(current_theta)
        new_y = current_y - backup_distance_grid * np.cos(current_theta)
        
        logging.info(f"Recovery: Updated position after backup: ({current_x}, {current_y}) -> ({new_x:.0f}, {new_y:.0f})")
        
        # Turn 45 degrees right to avoid obstacle
        turn_angle = 45
        # COORDINATE SYSTEM: positive angle = right turn, negative = left turn
        # heading: 0 = forward, pi/2 = left, -pi/2 = right, pi = backward
        new_theta = (current_theta + np.radians(turn_angle)) % (2 * np.pi)  # Add for right turn
        
        logging.info(f"Recovery: Turning {turn_angle}° right")
        await self._turn_car(turn_angle)
        
        # Update position with new heading
        self.current_pos = (int(round(new_x)), int(round(new_y)), new_theta)
        logging.info(f"Recovery: Position after turn: {self.current_pos}")
        
        # Update car position in map
        await self._update_car_position()
        
        # Force replan
        self.last_replan_time = 0

    async def main_control_loop(self):
        """Main control loop integrating all subsystems."""
        logging.info("Starting main control loop")
        
        # Debug: Log initial and goal positions
        logging.info(f"Start position: {self.current_pos}")
        logging.info(f"Goal position: {self.goal_pos.state}")
        logging.info(f"Expected distance to goal: {np.sqrt(sum((a - b)**2 for a, b in zip(self.current_pos[:2], self.goal_pos.state[:2]))):.1f} units")
        
        # Start background tasks
        mapping_task = asyncio.create_task(self._continuous_mapping())
        detection_task = asyncio.create_task(self._object_detection_monitor())
        
        try:
            while not self.stop_event.is_set():
                loop_start = time.time()
                
                # Check if we need to replan
                if (time.time() - self.last_replan_time > self.replan_interval or 
                    not self.planned_path):
                    await self._plan_path()
                    self.last_replan_time = time.time()
                
                # Execute next waypoint if path available
                if (self.planned_path and self.path_index < len(self.planned_path) and
                    not self.object_detector.is_halt_needed()):
                    
                    target_state = self.planned_path[self.path_index]
                    await self._execute_motion_command(target_state)
                    self.path_index += 1
                    
                    # Check if we've reached the goal
                    if self.path_index >= len(self.planned_path):
                        current_pos = self.current_pos[:2]  # Just x, y
                        goal_pos = self.goal_pos.state[:2]  # Just x, y from Coordinate object
                        distance_to_goal = np.sqrt(sum((a - b)**2 for a, b in zip(current_pos, goal_pos)))
                        
                        if distance_to_goal < 2:  # Within 2 grid cells
                            logging.info("Goal reached!")
                            break
                        else:
                            # Need to replan
                            self.last_replan_time = 0
                
                # Maintain loop frequency
                elapsed = time.time() - loop_start
                sleep_time = max(0, 1.0/self.loop_frequency - elapsed)
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            logging.error(f"Control loop error: {e}")
        finally:
            # Cleanup
            mapping_task.cancel()
            detection_task.cancel()
            self.picarx.stop()
            logging.info("Control loop stopped")

    async def run(self):
        """Main run method."""
        try:
            await self.initialize()
            await self.main_control_loop()
        except KeyboardInterrupt:
            logging.info("System stopped by user")
        except Exception as e:
            logging.error(f"System error: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Shutdown all systems."""
        logging.info("Shutting down system...")
        
        self.stop_event.set()
        
        if self.picarx:
            self.picarx.stop()
        
        if self.object_detector:
            self.object_detector.stop_detection()
        
        logging.info("Shutdown complete")

# Analysis functions for lab report
def analyze_performance():
    """Analyze system performance for lab report."""
    print("=== CS 437 Step 7 Performance Analysis ===")
    
    print("\n1. Hardware Acceleration Analysis:")
    print("   - TensorFlow Lite uses XNNPACK for CPU SIMD acceleration")
    print("   - OpenCV leverages NEON instructions on ARM processors")  
    print("   - Picamera2 uses ISP for hardware-accelerated image processing")
    print("   - Coral EdgeTPU can provide 10-100x inference speedup for supported models")
    print("   - GPU acceleration limited on Pi; EdgeTPU or optimized CPU paths preferred")
    
    print("\n2. Multithreading Analysis:")
    print("   - Beneficial for I/O bound tasks (camera, ultrasonic, motor control)")
    print("   - CPU-bound inference doesn't benefit from threading due to Python GIL")
    print("   - Separate thread/process for inference prevents blocking main control loop")
    print("   - Small bounded queues prevent memory buildup and reduce latency")
    print("   - Threading improves system responsiveness to ~10-15 Hz control rate")
    
    print("\n3. Frame Rate vs Accuracy Trade-offs:")
    print("   - Target: ~1 FPS detection with 10-15 Hz control loop responsiveness")
    print("   - Use 320x320 input size for speed vs larger sizes for accuracy")
    print("   - INT8 quantized models provide 2-4x speedup vs FP32")
    print("   - Lower confidence threshold (0.3) catches more objects but more false positives")
    print("   - Temporal filtering and tracking between detections maintains perception rate")
    print("   - Hardware acceleration allows higher accuracy within latency budget")

if __name__ == "__main__":
    # Option to run analysis or full system
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        analyze_performance()
    else:
        system = IntegratedSelfDrivingSystem()
        asyncio.run(system.run())
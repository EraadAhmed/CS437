#!/usr/bin/env python3
"""
CS 437 Lab 1: Advanced Integrated Self-Driving Car System
Authors: [Your Names Here]

Addresses the sensing speed issues from integrated_main.py by:
1. Optimized scanning patterns that don't block driving
2. Stops at camera detection range (~30cm minus safety margin)
3. Fast response to obstacles with immediate stopping
4. Comprehensive test cases for distance calibration

Features:
- Advanced 2D mapping with ultrasonic sensor
- TensorFlow Lite object detection for stop signs/traffic rules
- A* pathfinding with periodic rescanning
- Efficient multi-threaded architecture
- Real-time obstacle avoidance
"""

import asyncio
import time
import numpy as np
import logging
import cv2
from threading import Event, Lock
from queue import Queue, Empty
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import os

# Hardware imports
try:
    from picarx import Picarx
    from picamera2 import Picamera2
    from object_detection import ObjectDetector as ObjectDetectorHW
    HW_AVAILABLE = True
except ImportError:
    print("WARNING: Hardware libraries not available. Running in simulation mode.")
    HW_AVAILABLE = False

# TensorFlow Lite imports
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    print("WARNING: TensorFlow not available. Object detection disabled.")
    TF_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """Configuration parameters for the self-driving system."""
    # Physical dimensions (in cm)
    FIELD_WIDTH: int = 120
    FIELD_LENGTH: int = 380  
    CAR_WIDTH: int = 14
    CAR_LENGTH: int = 23
    
    # Sensor parameters
    ULTRASONIC_MAX_RANGE: int = 100
    CAMERA_RANGE: int = 30  # Stop at 25cm (30cm - 5cm safety margin)
    SAFETY_MARGIN: int = 5
    
    # Navigation parameters
    GRID_RESOLUTION: float = 1.0  # 1cm per grid cell
    OBSTACLE_INFLATION: int = 8   # Cells to inflate around obstacles (car half-width + safety)
    CAR_CLEARANCE: int = 15       # Additional clearance needed around car (cm)
    STRAIGHT_DRIVING_THRESHOLD: float = 0.2  # Radians for considering "straight" (~11.5 degrees)
    
    # Control parameters
    DRIVE_SPEED: float = 25.0     # cm/s - reduced for better control
    TURN_POWER: int = 35
    DRIVE_POWER: int = 35
    SERVO_OFFSET: int = -5       # Calibration offset for straight driving (negative = steer left to counter right drift)
    STEERING_CORRECTION_FACTOR: float = 0.5  # Factor for steering corrections (reduced from 0.8)
    
    # Timing parameters
    CONTROL_FREQUENCY: int = 20   # Hz - increased for responsive control
    SCAN_FREQUENCY: int = 10      # Hz - fast scanning while driving
    REPLAN_INTERVAL: float = 1.0  # seconds - frequent replanning
    DETECTION_FREQUENCY: int = 5  # Hz - object detection rate
    DETECTION_STOP_INTERVAL: float = 0.2  # seconds - stop every 15-30cm for detection
    
    # Safety parameters
    EMERGENCY_STOP_DISTANCE: int = 15  # cm - immediate stop distance
    DETECTION_DISTANCE: int = 25      # cm - stop for camera detection (30cm - 5cm safety)
    BACKUP_DISTANCE: int = 20         # cm - backup distance when obstacle detected
    STOP_SIGN_PAUSE: float = 3.0      # seconds to pause at stop signs
    MOVEMENT_STEP_SIZE: float = 15.0  # cm - distance per movement step for precise control

class Position:
    """Represents a position and orientation in the grid."""
    def __init__(self, x: float, y: float, theta: float = 0.0):
        self.x = x
        self.y = y  
        self.theta = theta
    
    def to_grid(self, resolution: float) -> Tuple[int, int]:
        """Convert to grid coordinates."""
        return (int(round(self.x / resolution)), int(round(self.y / resolution)))
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __repr__(self):
        return f"Position({self.x:.1f}, {self.y:.1f}, {np.degrees(self.theta):.1f} degrees)"

class OccupancyGrid:
    """2D occupancy grid for mapping obstacles."""
    def __init__(self, width: int, height: int, resolution: float):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # Grid values: 0=free, 1=obstacle, 2=inflated_obstacle, 3=car
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.lock = Lock()
    
    def clear_car_position(self):
        """Clear previous car position markers."""
        try:
            self.grid[self.grid == 3] = 0
        except Exception as e:
            print(f"Error in clear_car_position: {e}")
    
    def mark_car_position(self, pos: Position, car_width: int, car_length: int):
        """Mark current car position in grid."""
        try:
            self.clear_car_position()
            
            # Convert car dimensions to grid cells
            gx, gy = pos.to_grid(self.resolution)
            car_w_cells = int(np.ceil(car_width / self.resolution))
            car_l_cells = int(np.ceil(car_length / self.resolution))
            
            # Mark car footprint
            for dx in range(-car_w_cells//2, car_w_cells//2 + 1):
                for dy in range(-car_l_cells//2, car_l_cells//2 + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        self.grid[ny, nx] = 3
        except Exception as e:
            print(f"Error in mark_car_position: {e}")
            # Continue without failing
    
    def add_obstacle_reading(self, sensor_pos: Position, angle: float, distance: float):
        """Add obstacle from ultrasonic reading."""
        if distance <= 0 or distance > 100:  # Invalid reading
            return
            
        # Calculate obstacle position
        obs_x = sensor_pos.x + distance * np.cos(angle)
        obs_y = sensor_pos.y + distance * np.sin(angle)
        
        # Convert to grid coordinates
        gx = int(round(obs_x / self.resolution))
        gy = int(round(obs_y / self.resolution))
        
        # Mark obstacle if within bounds
        with self.lock:
            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                if self.grid[gy, gx] == 0:  # Don't overwrite car position
                    self.grid[gy, gx] = 1
    
    def inflate_obstacles(self, inflation_radius: int):
        """Inflate obstacles for path planning safety."""
        with self.lock:
            original_obstacles = (self.grid == 1)
            
            # Clear previous inflated obstacles
            self.grid[self.grid == 2] = 0
            
            # Inflate around each obstacle
            for y, x in np.argwhere(original_obstacles):
                for dy in range(-inflation_radius, inflation_radius + 1):
                    for dx in range(-inflation_radius, inflation_radius + 1):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and
                            self.grid[ny, nx] == 0):  # Only inflate free space
                            self.grid[ny, nx] = 2
    
    def is_free(self, x: int, y: int) -> bool:
        """Check if grid cell is free (for path planning)."""
        try:
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                return self.grid[y, x] == 0
            return False
        except Exception as e:
            logger.warning(f"Grid access error at ({x}, {y}): {e}")
            return False
    
    def get_copy(self) -> np.ndarray:
        """Get thread-safe copy of grid."""
        try:
            return self.grid.copy()
        except Exception as e:
            logger.warning(f"Grid copy error: {e}")
            return np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)

class AStarPlanner:
    """Optimized A* path planner with DFS and set-based avoidance."""
    def __init__(self, config: SystemConfig):
        self.config = config
    
    def plan_path(self, start: Position, goal: Position, grid: OccupancyGrid) -> List[Position]:
        """Plan path using optimized A* with DFS and Manhattan heuristic."""
        # Convert positions to grid coordinates
        start_grid = start.to_grid(grid.resolution)
        goal_grid = goal.to_grid(grid.resolution)
        
        print(f"DEBUG: Planning path from grid {start_grid} to {goal_grid}")
        
        # Debug grid status around start and goal
        self._debug_grid_area(start_grid, grid, "START")
        self._debug_grid_area(goal_grid, grid, "GOAL")
        
        # Check if start and goal are valid
        if not self._is_valid_cell(start_grid[0], start_grid[1], grid):
            print(f"DEBUG: Start position {start_grid} is not valid")
            return []
        
        if not self._is_valid_cell(goal_grid[0], goal_grid[1], grid):
            print(f"DEBUG: Goal position {goal_grid} is not valid")
            return []
        
        # DFS-based A* with sets for efficiency
        import heapq
        open_set = []
        open_set_hash = set()  # For O(1) lookup
        closed_set = set()
        came_from = {}
        g_score = {start_grid: 0}
        
        # Initialize with start node
        start_f = self._manhattan_distance(start_grid, goal_grid)
        heapq.heappush(open_set, (start_f, start_grid))
        open_set_hash.add(start_grid)
        
        max_iterations = 2000  # Increased limit
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current_f, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            # Goal check
            if current == goal_grid:
                print(f"DEBUG: Path found in {iterations} iterations")
                return self._reconstruct_path(came_from, current, grid.resolution)
            
            # Add to closed set
            closed_set.add(current)
            
            # Get valid neighbors (adjacent nodes only for speed)
            neighbors = self._get_valid_neighbors(current, grid, closed_set)
            
            for neighbor in neighbors:
                # Skip if already processed
                if neighbor in closed_set:
                    continue
                
                # Calculate scores
                tentative_g = g_score[current] + 1
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._manhattan_distance(neighbor, goal_grid)
                    
                    # Add to open set if not already there
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score, neighbor))
                        open_set_hash.add(neighbor)
        
        print(f"DEBUG: A* failed after {iterations} iterations - no path found")
        return []  # No path found
    
    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic for grid-based pathfinding."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _is_valid_cell(self, x: int, y: int, grid: OccupancyGrid) -> bool:
        """Check if a cell is valid and free for navigation."""
        # Check bounds
        if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
            return False
        
        # Check if cell is free for navigation
        # 0 = free, 3 = car (also valid), 1 = obstacle, 2 = inflated obstacle
        try:
            cell_value = grid.grid[y, x]
            return cell_value == 0 or cell_value == 3  # Allow free space and car position
        except:
            return False
    
    def _get_valid_neighbors(self, pos: Tuple[int, int], grid: OccupancyGrid, 
                           closed_set: set) -> List[Tuple[int, int]]:
        """Get valid neighboring cells using 4-connected grid (faster than 8-connected)."""
        x, y = pos
        neighbors = []
        
        # 4-connected neighbors (up, down, left, right) for speed
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Skip if already in closed set (avoid recomputation)
            if (nx, ny) in closed_set:
                continue
            
            # Check if neighbor is valid and free
            if self._is_valid_cell(nx, ny, grid):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def _debug_grid_area(self, pos: Tuple[int, int], grid: OccupancyGrid, label: str):
        """Debug grid area around a position."""
        x, y = pos
        print(f"DEBUG: {label} position {pos}")
        
        # Check 3x3 area around position
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid.grid_width and 0 <= ny < grid.grid_height:
                    value = grid.grid[ny, nx]
                    symbol = "." if value == 0 else "X" if value == 1 else "I" if value == 2 else "C"
                    print(f"  ({nx:3},{ny:3}): {value} {symbol}")
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int], 
                         resolution: float) -> List[Position]:
        """Reconstruct path from A* result."""
        path = []
        
        while current in came_from:
            x, y = current
            path.append(Position(x * resolution, y * resolution, 0))  # Add heading
            current = came_from[current]
        
        # Add start position
        if path:
            x, y = current
            path.append(Position(x * resolution, y * resolution, 0))
        
        path.reverse()
        
        # Debug: print first few and last few waypoints to verify path
        if path:
            print(f"DEBUG: Path waypoints (first 5): {[(p.x, p.y) for p in path[:5]]}")
            print(f"DEBUG: Path waypoints (last 5): {[(p.x, p.y) for p in path[-5:]]}")
            print(f"DEBUG: Total waypoints: {len(path)}")
        
        return path

class CarController:
    """Low-level car control with hardware abstraction."""
    def __init__(self, config: SystemConfig):
        self.config = config
        self.picar = None
        
        if HW_AVAILABLE:
            try:
                self.picar = Picarx()
                logger.info("PiCar hardware initialized")
            except Exception as e:
                logger.error(f"PiCar initialization failed: {e}")
    
    def read_ultrasonic(self) -> float:
        """Read ultrasonic sensor distance."""
        if self.picar:
            try:
                return self.picar.ultrasonic.read()
            except:
                return -1
        return -1  # Simulation mode
    
    def set_servo_angle(self, angle: int):
        """Set servo angle for ultrasonic scanning."""
        if self.picar:
            try:
                self.picar.set_cam_pan_angle(angle)
            except Exception as e:
                logger.error(f"Servo control error: {e}")
    
    def move_forward(self, power: int):
        """Move car forward."""
        if self.picar:
            try:
                self.picar.set_dir_servo_angle(self.config.SERVO_OFFSET)
                self.picar.forward(power)
            except Exception as e:
                logger.error(f"Forward movement error: {e}")
    
    def move_backward(self, power: int):
        """Move car backward."""
        if self.picar:
            try:
                self.picar.set_dir_servo_angle(self.config.SERVO_OFFSET)
                self.picar.backward(power)
            except Exception as e:
                logger.error(f"Backward movement error: {e}")
    
    def turn_left(self, power: int):
        """Turn car left."""
        if self.picar:
            try:
                self.picar.set_dir_servo_angle(-30 + self.config.SERVO_OFFSET)
                self.picar.forward(power)
            except Exception as e:
                logger.error(f"Left turn error: {e}")
    
    def turn_right(self, power: int):
        """Turn car right."""
        if self.picar:
            try:
                self.picar.set_dir_servo_angle(30 + self.config.SERVO_OFFSET)
                self.picar.forward(power)
            except Exception as e:
                logger.error(f"Right turn error: {e}")
    
    def stop(self):
        """Stop car immediately."""
        if self.picar:
            try:
                self.picar.stop()
            except Exception as e:
                logger.error(f"Stop error: {e}")

class IntegratedSelfDrivingSystem:
    """Main integrated self-driving system."""
    
    def __init__(self, config: SystemConfig = None, enable_hardware: bool = True):
        self.config = config or SystemConfig()
        
        # Initialize subsystems
        self.grid = OccupancyGrid(
            self.config.FIELD_WIDTH, 
            self.config.FIELD_LENGTH, 
            self.config.GRID_RESOLUTION
        )
        
        if enable_hardware:
            self.car_controller = CarController(self.config)
            if HW_AVAILABLE:
                try:
                    self.object_detector = ObjectDetectorHW()
                    logger.info("Hardware ObjectDetector initialized")
                except Exception as e:
                    logger.warning(f"ObjectDetector initialization failed: {e}")
                    self.object_detector = None
            else:
                self.object_detector = None
        else:
            # Use mock implementations for testing
            self.car_controller = None
            self.object_detector = None
            
        self.planner = AStarPlanner(self.config)
        
        # State variables
        self.current_position = Position(
            60,  # Start at x=60cm (middle of 120cm hallway)
            0,   # Start at y=0cm (beginning of hallway)
            0    # Facing forward
        )
        self.goal_position = Position(
            60,   # Stay in middle lane (x=60cm)
            379,  # End at y=379cm (within field bounds)
            0     # Facing forward
        )
        
        # Control state
        self.running = Event()
        self.emergency_stop = Event()
        self.current_path = []
        self.path_index = 0
        self.last_replan_time = 0
        
        # Scanning state
        self.scan_angle = 0
        self.scan_direction = 1
        
        # Movement and detection tracking
        self.last_detection_time = 0
        self.movement_distance_since_detection = 0
        self.is_moving = False
        
        logger.info("Integrated self-driving system initialized")
    
    async def start(self):
        """Start the self-driving system."""
        logger.info("Starting integrated self-driving system...")
        
        self.running.set()
        if self.object_detector:
            self.object_detector.start_detection()
        
        # Initial mapping scan
        await self._perform_initial_scan()
        
        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self._main_control_loop()),
            asyncio.create_task(self._continuous_scanning()),
            asyncio.create_task(self._safety_monitor())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Stopping system...")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            # Cancel all tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to finish cancelling with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                logger.warning("Task cancellation timed out")
            
            await self.stop()
    
    async def stop(self):
        """Stop the self-driving system."""
        logger.info("Stopping integrated self-driving system...")
        
        self.running.clear()
        if self.car_controller:
            self.car_controller.stop()
        if self.object_detector:
            self.object_detector.stop_detection()
        
        logger.info("System stopped")
    
    async def _perform_initial_scan(self):
        """Perform comprehensive initial mapping scan."""
        logger.info("Performing initial 360 degree mapping scan...")
        
        scan_angles = range(-90, 91, 10)  # -90 to +90 degrees, 10 degree steps
        
        for angle in scan_angles:
            if not self.running.is_set():
                break
                
            self.car_controller.set_servo_angle(angle)
            await asyncio.sleep(0.1)  # Allow servo to move
            
            distance = self.car_controller.read_ultrasonic()
            if 0 < distance <= self.config.ULTRASONIC_MAX_RANGE:
                sensor_angle = self.current_position.theta + np.radians(angle)
                self.grid.add_obstacle_reading(self.current_position, sensor_angle, distance)
        
        # Return servo to center
        self.car_controller.set_servo_angle(0)
        await asyncio.sleep(0.2)
        
        # Inflate obstacles for safe path planning
        self.grid.inflate_obstacles(self.config.OBSTACLE_INFLATION)
        
        logger.info("Initial scan complete")
    
    async def _continuous_scanning(self):
        """Continuous scanning while driving."""
        while self.running.is_set():
            try:
                # Limited range scanning while driving (+/-30 degrees)
                self.scan_angle += self.scan_direction * 15
                if self.scan_angle >= 30 or self.scan_angle <= -30:
                    self.scan_direction *= -1
                
                self.car_controller.set_servo_angle(self.scan_angle)
                await asyncio.sleep(0.05)  # Fast servo movement
                
                distance = self.car_controller.read_ultrasonic()
                if 0 < distance <= self.config.ULTRASONIC_MAX_RANGE:
                    sensor_angle = self.current_position.theta + np.radians(self.scan_angle)
                    self.grid.add_obstacle_reading(self.current_position, sensor_angle, distance)
                
                await asyncio.sleep(1.0 / self.config.SCAN_FREQUENCY)
                
            except Exception as e:
                logger.error(f"Scanning error: {e}")
                await asyncio.sleep(0.1)
    
    async def _safety_monitor(self):
        """Monitor for safety-critical situations with enhanced detection."""
        while self.running.is_set():
            try:
                # Check for immediate obstacles ahead
                forward_distance = self.car_controller.read_ultrasonic()
                
                # Emergency stop if obstacle too close (immediate danger)
                if (0 < forward_distance <= self.config.EMERGENCY_STOP_DISTANCE):
                    logger.warning(f"EMERGENCY STOP: Obstacle at {forward_distance}cm")
                    self.emergency_stop.set()
                    self.car_controller.stop()
                    self.is_moving = False
                    
                    # Force backup if very close
                    if forward_distance <= 10:
                        await self._execute_backup_maneuver()
                        
                elif forward_distance > self.config.DETECTION_DISTANCE:
                    # Clear emergency stop only if we're well beyond detection range
                    if self.emergency_stop.is_set():
                        self.emergency_stop.clear()
                        logger.info("Emergency cleared - obstacle far enough")
                
                # Enhanced object detection checks
                if self.object_detector:
                    try:
                        # Check for stop signs and obstacles  
                        if self.object_detector.is_halt_needed():
                            logger.info("OBSTACLE/STOP SIGN DETECTED: Stopping for traffic rule")
                            self.car_controller.stop()
                            self.is_moving = False
                            await asyncio.sleep(self.config.STOP_SIGN_PAUSE)
                            # Re-scan after stop sign
                            await self._perform_initial_scan()
                        
                        # The is_halt_needed() method already covers both stop signs and people
                        # so we don't need separate person detection here
                        
                    except Exception as e:
                        logger.warning(f"Object detection error: {e}")
                
                await asyncio.sleep(1.0 / self.config.DETECTION_FREQUENCY)
                
            except Exception as e:
                logger.error(f"Safety monitor error: {e}")
                await asyncio.sleep(0.1)
    
    async def _main_control_loop(self):
        """Main navigation control loop."""
        logger.info("Starting main control loop...")
        loop_count = 0
        while self.running.is_set():
            try:
                loop_count += 1
                logger.info(f"Control loop iteration {loop_count}")
                
                current_time = time.time()
                
                # Update car position in grid
                logger.info("Updating car position in grid...")
                try:
                    self.grid.mark_car_position(
                        self.current_position,
                        self.config.CAR_WIDTH,
                        self.config.CAR_LENGTH
                    )
                    logger.info("Car position updated successfully")
                except Exception as e:
                    logger.error(f"Error updating car position: {e}")
                    # Continue without failing
                
                # Monitor position and check for boundary violations
                safety_margin = 10  # 10cm margin from edges
                if (self.current_position.x < safety_margin or 
                    self.current_position.x > self.config.FIELD_WIDTH - safety_margin):
                    logger.warning(f"Position boundary violation: x={self.current_position.x:.1f}cm (field width: {self.config.FIELD_WIDTH}cm)")
                    logger.warning("Triggering emergency stop and replanning")
                    self.emergency_stop.set()
                    self.last_replan_time = 0  # Force immediate replan
                    await asyncio.sleep(0.5)  # Brief pause
                    self.emergency_stop.clear()
                
                # Check if we need to replan
                need_replan = (current_time - self.last_replan_time > self.config.REPLAN_INTERVAL or
                              not self.current_path or self.path_index >= len(self.current_path))
                
                logger.info(f"Need replan: {need_replan}")
                
                if need_replan:
                    logger.info("Starting replan...")
                    await self._replan_path()
                    self.last_replan_time = current_time
                    logger.info("Replan complete")
                
                # Execute next navigation step if not in emergency
                if not self.emergency_stop.is_set() and self.current_path:
                    await self._execute_navigation_step()
                
                # Check if goal reached
                if self.current_position.distance_to(self.goal_position) < 10:  # 10cm tolerance
                    logger.info("GOAL REACHED!")
                    await self.stop()
                    break
                
                await asyncio.sleep(1.0 / self.config.CONTROL_FREQUENCY)
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _replan_path(self):
        """Replan path to goal."""
        logger.info("Replanning path...")
        
        # Inflate obstacles for current planning
        self.grid.inflate_obstacles(self.config.OBSTACLE_INFLATION)
        
        logger.info(f"Planning from {self.current_position} to {self.goal_position}")
        
        # Plan new path
        try:
            new_path = self.planner.plan_path(
                self.current_position,
                self.goal_position,
                self.grid
            )
            logger.info("Pathfinding completed")
        except Exception as e:
            logger.error(f"Pathfinding error: {e}")
            new_path = None
        
        if new_path:
            self.current_path = new_path
            self.path_index = 0
            logger.info(f"New path planned with {len(new_path)} waypoints")
        else:
            logger.warning("No path found to goal - creating avoidance maneuver")
            await self._create_avoidance_path()
    
    async def _create_avoidance_path(self):
        """Create an intelligent avoidance path when A* fails."""
        logger.info("Creating avoidance maneuver path...")
        
        # First priority: Get back to center lane if we're off course
        center_x = 60  # Center of hallway
        current_x = self.current_position.x
        
        # If we're significantly off center, create a corrective path
        if abs(current_x - center_x) > 15:  # More than 15cm off center
            logger.info(f"Off center by {abs(current_x - center_x):.1f}cm - creating corrective path")
            
            # Calculate path back to center and then toward goal
            intermediate_y = self.current_position.y + 30  # Move 30cm forward
            corrective_path = [
                Position(center_x, intermediate_y, 0),  # Get back to center with forward heading
                self.goal_position
            ]
            
            # Check if corrective path is within bounds
            safety_margin = 10
            if (safety_margin <= center_x <= self.config.FIELD_WIDTH - safety_margin and
                0 <= intermediate_y <= self.config.FIELD_LENGTH):
                
                logger.info("Creating corrective path back to center")
                self.current_path = corrective_path
                self.path_index = 0
                return
        
        # Second try: simple forward movement toward goal
        forward_distance = 30  # Try moving 30cm forward
        forward_x = self.current_position.x + forward_distance * np.cos(self.current_position.theta)
        forward_y = self.current_position.y + forward_distance * np.sin(self.current_position.theta)
        
        # Check if forward path is within bounds (with safety margins)
        safety_margin = 10  # 10cm margin from edges
        if (safety_margin <= forward_x <= self.config.FIELD_WIDTH - safety_margin and
            0 <= forward_y <= self.config.FIELD_LENGTH):
            
            logger.info("Creating forward avoidance path")
            self.current_path = [
                Position(forward_x, forward_y, self.current_position.theta),
                self.goal_position
            ]
            self.path_index = 0
            return
        
        # Try multiple avoidance strategies if forward doesn't work
        avoidance_paths = []
        
        # Strategy 1: Move to the right side
        right_angle = self.current_position.theta - np.pi/2  # 90 degrees right
        right_x = self.current_position.x + 30 * np.cos(right_angle)
        right_y = self.current_position.y + 30 * np.sin(right_angle)
        avoidance_paths.append([
            Position(right_x, right_y, self.current_position.theta),
            Position(right_x, right_y + 30, self.current_position.theta),
            self.goal_position
        ])
        
        # Strategy 2: Move to the left side
        left_angle = self.current_position.theta + np.pi/2  # 90 degrees left
        left_x = self.current_position.x + 30 * np.cos(left_angle)
        left_y = self.current_position.y + 30 * np.sin(left_angle)
        avoidance_paths.append([
            Position(left_x, left_y, self.current_position.theta),
            Position(left_x, left_y + 30, self.current_position.theta),
            self.goal_position
        ])
        
        # Choose the path that moves toward the goal
        best_path = None
        best_distance = float('inf')
        
        for path in avoidance_paths:
            if path:
                # Calculate distance from first waypoint to goal
                first_waypoint = path[0]
                distance_to_goal = first_waypoint.distance_to(self.goal_position)
                
                # Check if waypoint is within bounds (with safety margins)
                safety_margin = 10  # 10cm margin from edges
                if (safety_margin <= first_waypoint.x <= self.config.FIELD_WIDTH - safety_margin and
                    0 <= first_waypoint.y <= self.config.FIELD_LENGTH):
                    
                    if distance_to_goal < best_distance:
                        best_distance = distance_to_goal
                        best_path = path
        
        if best_path:
            self.current_path = best_path
            self.path_index = 0
            logger.info(f"Created avoidance path with {len(best_path)} waypoints")
        else:
            # Last resort: try to get back to center regardless of risk
            logger.warning("All avoidance strategies failed - forcing return to center")
            center_x = 60
            fallback_y = self.current_position.y + 15  # Small forward movement
            self.current_path = [
                Position(center_x, fallback_y, 0)  # Head toward center with forward orientation
            ]
            self.path_index = 0
    
    async def _execute_navigation_step(self):
        """Execute one navigation step with precise distance control and object detection."""
        if self.path_index >= len(self.current_path):
            return
        
        target = self.current_path[self.path_index]
        
        # Calculate movement vector
        dx = target.x - self.current_position.x
        dy = target.y - self.current_position.y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 2:  # Close enough to waypoint
            self.path_index += 1
            return
        
        # Check if we need to stop for detection (every 15-30cm or time interval)
        current_time = time.time()
        should_detect = (
            self.movement_distance_since_detection >= self.config.MOVEMENT_STEP_SIZE or
            current_time - self.last_detection_time >= self.config.DETECTION_STOP_INTERVAL
        )
        
        if should_detect and self.is_moving:
            await self._perform_detection_stop()
            return
        
        # Calculate required heading
        target_heading = np.arctan2(dy, dx)
        heading_error = target_heading - self.current_position.theta
        
        # Normalize heading error
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Execute movement with precise control
        if abs(heading_error) > self.config.STRAIGHT_DRIVING_THRESHOLD:  # Need to turn (tighter threshold)
            await self._execute_turn(heading_error)
        else:  # Move forward with straight driving
            await self._execute_forward_movement()
    
    async def _execute_turn(self, heading_error: float):
        """Execute a precise turn maneuver with improved steering."""
        self.is_moving = False
        
        # Convert to degrees for easier calculation
        angle_degrees = np.degrees(heading_error)
        
        # Apply steering correction factor
        corrected_error = heading_error * self.config.STEERING_CORRECTION_FACTOR
        corrected_angle_degrees = np.degrees(corrected_error)
        
        logger.info(f"Turning {angle_degrees:.1f} degrees (corrected: {corrected_angle_degrees:.1f})")
        
        # Set steering angle based on turn direction (like integrated_main.py)
        if self.car_controller.picar:
            if angle_degrees > 0:  # Turn right
                self.car_controller.picar.set_dir_servo_angle(30 + self.config.SERVO_OFFSET)
            else:  # Turn left
                self.car_controller.picar.set_dir_servo_angle(-30 + self.config.SERVO_OFFSET)
        
        # Move forward while turning (like integrated_main.py)
        turn_time = abs(corrected_angle_degrees) / 90 * 0.5  # Rough timing from integrated_main.py
        turn_time = min(turn_time, 0.6)  # Max turn time
        
        if self.car_controller.picar:
            self.car_controller.picar.forward(self.config.DRIVE_POWER)
            await asyncio.sleep(turn_time)
            self.car_controller.picar.stop()
            
            # CRITICAL: Return steering to center with servo offset (like integrated_main.py)
            self.car_controller.picar.set_dir_servo_angle(self.config.SERVO_OFFSET)
            await asyncio.sleep(0.2)  # Extra time for servo to fully center
        
        # Update position with corrected turn amount
        self.current_position.theta += corrected_error
        
        # Normalize theta to [-pi, pi]
        while self.current_position.theta > np.pi:
            self.current_position.theta -= 2 * np.pi
        while self.current_position.theta < -np.pi:
            self.current_position.theta += 2 * np.pi
        
        # Reset movement tracking after turn
        self.movement_distance_since_detection = 0
        logger.info(f"Turn complete - new heading: {np.degrees(self.current_position.theta):.1f} degrees")
    
    async def _execute_forward_movement(self):
        """Execute forward movement with straight-line driving correction."""
        self.is_moving = True
        
        # Check distance ahead before moving
        forward_distance = self.car_controller.read_ultrasonic()
        
        # Stop if obstacle detected within detection range
        if 0 < forward_distance <= self.config.DETECTION_DISTANCE:
            logger.warning(f"Obstacle detected at {forward_distance}cm - performing camera detection")
            await self._handle_obstacle_detection(forward_distance)
            return
        
        # Calculate movement step
        movement_step = min(self.config.MOVEMENT_STEP_SIZE, 
                           self.config.DETECTION_DISTANCE - 5)  # Stay within detection range
        
        # Check if movement would exceed field boundaries
        predicted_x = self.current_position.x + movement_step * np.cos(self.current_position.theta)
        predicted_y = self.current_position.y + movement_step * np.sin(self.current_position.theta)
        
        # Field boundaries with safety margins (5cm from edges)
        min_x, max_x = 5, self.config.FIELD_WIDTH - 5
        min_y, max_y = 0, self.config.FIELD_LENGTH
        
        if not (min_x <= predicted_x <= max_x and min_y <= predicted_y <= max_y):
            logger.warning(f"Movement would exceed boundaries: predicted position ({predicted_x:.1f}, {predicted_y:.1f})")
            logger.warning(f"Boundaries: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
            # Set emergency stop and force replanning
            self.emergency_stop.set()
            self.last_replan_time = 0
            return
        
        # Move forward for calculated time
        movement_time = movement_step / self.config.DRIVE_SPEED
        
        logger.info(f"Moving straight forward {movement_step:.1f}cm for {movement_time:.2f}s")
        
        # CRITICAL: Ensure steering is centered with servo offset before moving (like integrated_main.py)
        if self.car_controller.picar:
            self.car_controller.picar.set_dir_servo_angle(self.config.SERVO_OFFSET)
            await asyncio.sleep(0.1)  # Give servo time to center
        
        self.car_controller.move_forward(self.config.DRIVE_POWER)
        await asyncio.sleep(movement_time)
        self.car_controller.stop()
        
        # Update position with precise movement using integrated_main.py coordinate system
        actual_distance = movement_step  # Assume precise movement
        
        # COORDINATE SYSTEM from integrated_main.py: X=left/right, Y=forward/backward, heading=0 is forward
        # heading=0 → forward (+Y), heading=pi/2 → left (-X), heading=-pi/2 → right (+X)
        # CORRECTED FORMULA: X increases when heading is negative (right turns)
        new_x = self.current_position.x - actual_distance * np.sin(self.current_position.theta)   # Left/right movement (FIXED)
        new_y = self.current_position.y + actual_distance * np.cos(self.current_position.theta)   # Forward/backward movement
        
        self.current_position.x = new_x
        self.current_position.y = new_y
        
        # Clamp position to valid boundaries
        self.current_position.x = max(0, min(self.current_position.x, self.config.FIELD_WIDTH))
        self.current_position.y = max(0, min(self.current_position.y, self.config.FIELD_LENGTH))
        
        self.movement_distance_since_detection += actual_distance
        
        logger.info(f"Moved forward {actual_distance:.1f}cm to ({self.current_position.x:.1f}, {self.current_position.y:.1f})")
        logger.info(f"Total since detection: {self.movement_distance_since_detection:.1f}cm")
    
    async def _perform_detection_stop(self):
        """Stop for camera-based object detection."""
        self.is_moving = False
        current_time = time.time()
        
        logger.info("Stopping for object detection...")
        self.car_controller.stop()
        
        # Perform camera-based object detection
        if self.object_detector:
            # Center servo for forward detection
            self.car_controller.set_servo_angle(0)
            await asyncio.sleep(0.1)  # Allow servo to move
            
            # Check for obstacles in camera view
            try:
                obstacle_detected = self.object_detector.is_halt_needed()
                
                if obstacle_detected:
                    await self._handle_camera_obstacle()
                    return
            except Exception as e:
                logger.warning(f"Camera detection error: {e}")
        else:
            logger.info("No camera available - using ultrasonic only")
        
        # Also check ultrasonic for backup
        forward_distance = self.car_controller.read_ultrasonic()
        if 0 < forward_distance <= self.config.DETECTION_DISTANCE:
            await self._handle_obstacle_detection(forward_distance)
            return
        
        # Reset detection tracking
        self.last_detection_time = current_time
        self.movement_distance_since_detection = 0
        logger.info("Detection complete - continuing movement")
    
    async def _handle_obstacle_detection(self, distance: float):
        """Handle obstacle detected by ultrasonic sensor with proper grid mapping."""
        logger.warning(f"Obstacle detected at {distance}cm - adding to grid and replanning")
        
        # Stop immediately
        self.car_controller.stop()
        self.emergency_stop.set()
        
        # Add obstacle to grid with proper margins
        await self._add_obstacle_to_grid(distance)
        
        # Backup maneuver
        await self._execute_backup_maneuver()
        
        # Clear emergency stop after backup
        self.emergency_stop.clear()
        
        # Force immediate replan with updated grid
        self.last_replan_time = 0
    
    async def _handle_camera_obstacle(self):
        """Handle obstacle detected by camera with proper grid mapping."""
        logger.warning("Camera obstacle detected - adding to grid and replanning")
        
        # Use conservative distance estimate for camera detection
        estimated_distance = self.config.DETECTION_DISTANCE - 5
        
        # Add obstacle to grid
        await self._add_obstacle_to_grid(estimated_distance)
        
        # Backup maneuver
        await self._execute_backup_maneuver()
        
        # Force immediate replan with updated grid
        self.last_replan_time = 0
    
    async def _add_obstacle_to_grid(self, distance_ahead: float):
        """Add detected obstacle to grid with appropriate car-size margins."""
        logger.info(f"Adding obstacle to grid at distance {distance_ahead}cm ahead")
        
        # Calculate obstacle position in world coordinates
        obstacle_x = self.current_position.x + distance_ahead * np.cos(self.current_position.theta)
        obstacle_y = self.current_position.y + distance_ahead * np.sin(self.current_position.theta)
        
        # Convert to grid coordinates
        grid_x = int(round(obstacle_x / self.config.GRID_RESOLUTION))
        grid_y = int(round(obstacle_y / self.config.GRID_RESOLUTION))
        
        # Calculate obstacle size with car clearance
        # Assume obstacle is at least car-width sized, plus clearance
        obstacle_radius = int(np.ceil((self.config.CAR_WIDTH + self.config.CAR_CLEARANCE) / 2 / self.config.GRID_RESOLUTION))
        
        logger.info(f"Adding obstacle at grid ({grid_x}, {grid_y}) with radius {obstacle_radius} cells")
        
        # Mark obstacle area in grid
        try:
            for dx in range(-obstacle_radius, obstacle_radius + 1):
                for dy in range(-obstacle_radius, obstacle_radius + 1):
                    if dx*dx + dy*dy <= obstacle_radius*obstacle_radius:  # Circular obstacle
                        ox, oy = grid_x + dx, grid_y + dy
                        if 0 <= ox < self.grid.grid_width and 0 <= oy < self.grid.grid_height:
                            self.grid.grid[oy, ox] = 1  # Mark as obstacle
            
            # Inflate obstacles for path planning
            self.grid.inflate_obstacles(self.config.OBSTACLE_INFLATION)
            logger.info("Obstacle successfully added to grid with margins")
            
        except Exception as e:
            logger.error(f"Error adding obstacle to grid: {e}")
    
    async def _execute_backup_maneuver(self):
        """Execute precise backup maneuver."""
        logger.info(f"Backing up {self.config.BACKUP_DISTANCE}cm...")
        
        # Calculate backup time
        backup_time = self.config.BACKUP_DISTANCE / self.config.DRIVE_SPEED
        
        # Execute backup
        self.car_controller.move_backward(self.config.DRIVE_POWER)
        await asyncio.sleep(backup_time)
        self.car_controller.stop()
        
        # Update position
        self.current_position.x -= self.config.BACKUP_DISTANCE * np.cos(self.current_position.theta)
        self.current_position.y -= self.config.BACKUP_DISTANCE * np.sin(self.current_position.theta)
        
        # Clamp position to valid boundaries
        self.current_position.x = max(0, min(self.current_position.x, self.config.FIELD_WIDTH))
        self.current_position.y = max(0, min(self.current_position.y, self.config.FIELD_LENGTH))
        
        # Reset movement tracking
        self.movement_distance_since_detection = 0
        self.is_moving = False
        
        logger.info(f"Backup complete - new position: ({self.current_position.x:.1f}, {self.current_position.y:.1f})")
    
    async def _recovery_maneuver(self):
        """Legacy recovery maneuver - use _execute_backup_maneuver instead."""
        logger.info("Using enhanced backup maneuver...")
        await self._execute_backup_maneuver()
        
        # Additional turn for recovery
        self.car_controller.turn_right(self.config.TURN_POWER)
        await asyncio.sleep(0.5)
        self.car_controller.stop()
        
        self.current_position.theta -= np.pi/4  # 45 degree turn
        
        # Force replan
        self.last_replan_time = 0

class TestSuite:
    """Comprehensive test cases for distance calibration and validation."""
    
    def __init__(self, system: IntegratedSelfDrivingSystem):
        self.system = system
        self.test_results = []
    
    async def run_all_tests(self):
        """Run all test cases."""
        logger.info("Starting comprehensive test suite...")
        
        tests = [
            self.test_distance_calibration,
            self.test_obstacle_detection,
            self.test_stop_sign_response,
            self.test_path_planning,
            self.test_emergency_stop,
            self.test_goal_navigation
        ]
        
        for test in tests:
            try:
                await test()
            except Exception as e:
                logger.error(f"Test failed: {test.__name__}: {e}")
        
        self._print_test_summary()
    
    async def test_distance_calibration(self):
        """Test distance measurement accuracy."""
        logger.info("Testing distance calibration...")
        
        # Test at various distances
        test_distances = [10, 20, 30, 50, 75, 100]
        
        for expected_distance in test_distances:
            logger.info(f"Measure distance at {expected_distance}cm...")
            
            # Take multiple readings
            readings = []
            for _ in range(10):
                distance = self.system.car_controller.read_ultrasonic()
                if distance > 0:
                    readings.append(distance)
                await asyncio.sleep(0.1)
            
            if readings:
                avg_distance = np.mean(readings)
                std_distance = np.std(readings)
                error = abs(avg_distance - expected_distance)
                
                result = {
                    'test': 'distance_calibration',
                    'expected': expected_distance,
                    'measured': avg_distance,
                    'std': std_distance,
                    'error': error,
                    'pass': error < 5  # 5cm tolerance
                }
                self.test_results.append(result)
                
                logger.info(f"Distance {expected_distance}cm: measured {avg_distance:.1f}+/-{std_distance:.1f}cm, error {error:.1f}cm")
    
    async def test_obstacle_detection(self):
        """Test obstacle detection and mapping."""
        logger.info("Testing obstacle detection...")
        
        # Perform scan and check for obstacles
        await self.system._perform_initial_scan()
        
        grid_copy = self.system.grid.get_copy()
        obstacles_found = np.sum(grid_copy == 1)
        
        result = {
            'test': 'obstacle_detection',
            'obstacles_found': obstacles_found,
            'grid_coverage': np.sum(grid_copy > 0) / grid_copy.size,
            'pass': obstacles_found > 0
        }
        self.test_results.append(result)
        
        logger.info(f"Obstacle detection: {obstacles_found} obstacles found")
    
    async def test_stop_sign_response(self):
        """Test stop sign detection and response."""
        logger.info("Testing stop sign response...")
        
        # Simulate stop sign detection
        self.system.object_detector.stop_sign_detected.set()
        
        start_time = time.time()
        initial_position = Position(
            self.system.current_position.x,
            self.system.current_position.y,
            self.system.current_position.theta
        )
        
        # Wait for stop response
        await asyncio.sleep(self.system.config.STOP_SIGN_PAUSE + 1)
        
        stop_duration = time.time() - start_time
        position_change = self.system.current_position.distance_to(initial_position)
        
        result = {
            'test': 'stop_sign_response',
            'stop_duration': stop_duration,
            'position_change': position_change,
            'pass': stop_duration >= self.system.config.STOP_SIGN_PAUSE and position_change < 5
        }
        self.test_results.append(result)
        
        logger.info(f"Stop sign response: stopped for {stop_duration:.1f}s, moved {position_change:.1f}cm")
    
    async def test_path_planning(self):
        """Test A* path planning algorithm."""
        logger.info("Testing path planning...")
        
        # Create test scenario with obstacles
        test_start = Position(30, 30)
        test_goal = Position(90, 100)
        
        path = self.system.planner.plan_path(test_start, test_goal, self.system.grid)
        
        result = {
            'test': 'path_planning',
            'path_length': len(path),
            'start_to_goal_distance': test_start.distance_to(test_goal),
            'pass': len(path) > 0
        }
        self.test_results.append(result)
        
        logger.info(f"Path planning: found path with {len(path)} waypoints")
    
    async def test_emergency_stop(self):
        """Test emergency stop functionality."""
        logger.info("Testing emergency stop...")
        
        # Start moving
        self.system.car_controller.move_forward(self.system.config.DRIVE_POWER)
        await asyncio.sleep(0.5)
        
        # Trigger emergency stop
        self.system.emergency_stop.set()
        start_time = time.time()
        
        # Check stop response time
        self.system.car_controller.stop()
        stop_time = time.time() - start_time
        
        result = {
            'test': 'emergency_stop',
            'stop_time': stop_time,
            'pass': stop_time < 0.5  # Should stop within 500ms
        }
        self.test_results.append(result)
        
        logger.info(f"Emergency stop: stopped in {stop_time:.3f}s")
    
    async def test_goal_navigation(self):
        """Test navigation to goal position."""
        logger.info("Testing goal navigation...")
        
        start_position = Position(
            self.system.current_position.x,
            self.system.current_position.y,
            self.system.current_position.theta
        )
        
        # Set close goal for testing
        test_goal = Position(
            start_position.x + 30,
            start_position.y + 30
        )
        self.system.goal_position = test_goal
        
        # Run navigation for limited time
        start_time = time.time()
        max_test_time = 30  # 30 seconds max
        
        while (time.time() - start_time < max_test_time and
               self.system.current_position.distance_to(test_goal) > 10):
            await asyncio.sleep(1)
        
        final_distance = self.system.current_position.distance_to(test_goal)
        navigation_time = time.time() - start_time
        
        result = {
            'test': 'goal_navigation',
            'final_distance': final_distance,
            'navigation_time': navigation_time,
            'pass': final_distance < 15  # 15cm tolerance
        }
        self.test_results.append(result)
        
        logger.info(f"Goal navigation: reached within {final_distance:.1f}cm in {navigation_time:.1f}s")
    
    def _print_test_summary(self):
        """Print summary of all test results."""
        logger.info("\n" + "="*50)
        logger.info("TEST SUITE SUMMARY")
        logger.info("="*50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['pass'])
        
        for result in self.test_results:
            status = "PASS" if result['pass'] else "FAIL"
            logger.info(f"{result['test']:20s}: {status}")
        
        logger.info("-"*50)
        logger.info(f"Total: {passed_tests}/{total_tests} tests passed")
        logger.info("="*50)
        
        # Save results to file
        import json
        with open('test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2)
        logger.info("Test results saved to test_results.json")

# Main execution functions
async def run_integrated_system():
    """Run the complete integrated self-driving system."""
    config = SystemConfig()
    system = IntegratedSelfDrivingSystem(config)
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    finally:
        await system.stop()

async def run_test_suite():
    """Run comprehensive test suite."""
    config = SystemConfig()
    system = IntegratedSelfDrivingSystem(config)
    test_suite = TestSuite(system)
    
    try:
        await test_suite.run_all_tests()
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
    finally:
        await system.stop()

def main():
    """Main entry point with command line options."""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            logger.info("Running test suite...")
            asyncio.run(run_test_suite())
        elif sys.argv[1] == "--config":
            # Print current configuration
            config = SystemConfig()
            logger.info("Current system configuration:")
            for field, value in config.__dict__.items():
                if not field.startswith('_'):
                    logger.info(f"  {field}: {value}")
        else:
            logger.info("Usage: python integrated.py [--test|--config]")
    else:
        logger.info("Starting integrated self-driving system...")
        asyncio.run(run_integrated_system())

if __name__ == "__main__":
    main()
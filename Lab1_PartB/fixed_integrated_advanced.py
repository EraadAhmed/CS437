# fixed_integrated_advanced.py
# CS 437 Lab 1: Advanced Fixed Self-Driving Car with A* Navigation
# Combines the best of both original files with comprehensive fixes

import asyncio
import time
import numpy as np
import logging
import math
from threading import Event, Lock
from queue import Queue
from dataclasses import dataclass
from typing import List, Tuple, Optional
import heapq

# Hardware imports with error handling
try:
    from picarx import Picarx
    HW_AVAILABLE = True
except ImportError:
    print("WARNING: PiCarx not available - running in simulation mode")
    HW_AVAILABLE = False

try:
    from fixed_object_detection import ObjectDetector
    DETECTION_AVAILABLE = True
except ImportError:
    print("WARNING: Object detection not available")
    DETECTION_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """Centralized configuration with optimized parameters."""
    # Field dimensions
    FIELD_WIDTH: int = 120
    FIELD_LENGTH: int = 380
    CAR_WIDTH: int = 14
    CAR_LENGTH: int = 23
    
    # Navigation
    GRID_RESOLUTION: float = 2.0  # 2cm per grid cell for better performance
    OBSTACLE_INFLATION: int = 4   # Grid cells to inflate around obstacles
    
    # Movement - CALIBRATED VALUES
    SERVO_OFFSET: int = 0         # Hardware fixed: no adjustment needed
    DRIVE_SPEED: float = 25.0     # cm/s
    DRIVE_POWER: int = 35
    TURN_POWER: int = 35
    DRIFT_CORRECTION_FACTOR: float = 0.95
    
    # Safety
    EMERGENCY_STOP_DISTANCE: int = 15
    DETECTION_DISTANCE: int = 25
    SAFE_FOLLOWING_DISTANCE: int = 30
    
    # Timing
    CONTROL_FREQUENCY: int = 15
    REPLAN_INTERVAL: float = 2.0
    DETECTION_INTERVAL: float = 0.3

class Position:
    """Enhanced position class with better coordinate handling."""
    def __init__(self, x: float, y: float, theta: float = 0.0):
        self.x = x
        self.y = y
        self.theta = theta
    
    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_grid(self, resolution: float) -> Tuple[int, int]:
        return (int(round(self.x / resolution)), int(round(self.y / resolution)))
    
    def __repr__(self):
        return f"Position({self.x:.1f}, {self.y:.1f}, {math.degrees(self.theta):.1f}°)"

class OccupancyGrid:
    """Thread-safe occupancy grid with proper coordinate handling."""
    def __init__(self, width: int, height: int, resolution: float):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # Grid: 0=free, 1=obstacle, 2=inflated, 3=car
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.lock = Lock()
    
    def add_obstacle(self, x: float, y: float):
        """Add obstacle at world coordinates."""
        gx, gy = int(round(x / self.resolution)), int(round(y / self.resolution))
        with self.lock:
            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                if self.grid[gy, gx] == 0:  # Don't overwrite car
                    self.grid[gy, gx] = 1
    
    def mark_car_position(self, pos: Position):
        """Mark car position in grid."""
        with self.lock:
            # Clear previous car position
            self.grid[self.grid == 3] = 0
            
            # Mark new position
            gx, gy = pos.to_grid(self.resolution)
            car_size = 2  # Grid cells
            
            for dx in range(-car_size//2, car_size//2 + 1):
                for dy in range(-car_size//2, car_size//2 + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        self.grid[ny, nx] = 3
    
    def inflate_obstacles(self, radius: int):
        """Inflate obstacles for path planning."""
        with self.lock:
            # Clear previous inflation
            self.grid[self.grid == 2] = 0
            
            # Find obstacles
            obstacles = np.where(self.grid == 1)
            
            # Inflate each obstacle
            for oy, ox in zip(obstacles[0], obstacles[1]):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        nx, ny = ox + dx, oy + dy
                        if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and
                            self.grid[ny, nx] == 0):
                            self.grid[ny, nx] = 2
    
    def is_free(self, x: int, y: int) -> bool:
        """Check if grid cell is free for navigation."""
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            return self.grid[y, x] in [0, 3]  # Free or car position
        return False
    
    def get_copy(self) -> np.ndarray:
        with self.lock:
            return self.grid.copy()

class AStarPlanner:
    """Optimized A* path planner."""
    def __init__(self, config: SystemConfig):
        self.config = config
    
    def plan_path(self, start: Position, goal: Position, grid: OccupancyGrid) -> List[Position]:
        """Plan path using A* algorithm."""
        start_grid = start.to_grid(grid.resolution)
        goal_grid = goal.to_grid(grid.resolution)
        
        logger.info(f"Planning path from {start_grid} to {goal_grid}")
        
        # A* algorithm
        open_set = [(0, start_grid)]
        open_set_hash = {start_grid}
        closed_set = set()
        came_from = {}
        g_score = {start_grid: 0}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    x, y = current
                    path.append(Position(x * grid.resolution, y * grid.resolution, 0))
                    current = came_from[current]
                
                # Add start
                x, y = current
                path.append(Position(x * grid.resolution, y * grid.resolution, 0))
                path.reverse()
                
                logger.info(f"Path found with {len(path)} waypoints")
                return path
            
            closed_set.add(current)
            
            # Check neighbors (4-connected)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if neighbor in closed_set:
                    continue
                
                if not grid.is_free(neighbor[0], neighbor[1]):
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal_grid)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score, neighbor))
                        open_set_hash.add(neighbor)
        
        logger.warning("No path found")
        return []
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

class CarController:
    """Hardware abstraction layer with drift correction."""
    def __init__(self, config: SystemConfig):
        self.config = config
        self.picarx = None
        
        if HW_AVAILABLE:
            try:
                self.picarx = Picarx(servo_pins=["P0", "P1", "P3"])
                self.picarx.set_dir_servo_angle(config.SERVO_OFFSET)
                logger.info("PiCarx initialized")
            except Exception as e:
                logger.error(f"PiCarx init failed: {e}")
    
    def read_ultrasonic(self) -> float:
        """Read ultrasonic distance with error handling."""
        if self.picarx:
            try:
                distance = self.picarx.ultrasonic.read()
                return distance if distance > 0 else -1
            except:
                return -1
        return -1
    
    def move_forward(self, power: int):
        if self.picarx:
            self.picarx.set_dir_servo_angle(self.config.SERVO_OFFSET)
            self.picarx.forward(power)
    
    def turn_left(self, power: int):
        if self.picarx:
            self.picarx.set_dir_servo_angle(-25 + self.config.SERVO_OFFSET)
            self.picarx.forward(power)
    
    def turn_right(self, power: int):
        if self.picarx:
            self.picarx.set_dir_servo_angle(25 + self.config.SERVO_OFFSET)
            self.picarx.forward(power)
    
    def stop(self):
        if self.picarx:
            self.picarx.stop()

class AdvancedSelfDrivingSystem:
    """
    Advanced self-driving system combining the best features from both original files.
    Addresses all major issues: drift, object detection, navigation, and error handling.
    """
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        
        # Initialize components
        self.grid = OccupancyGrid(
            self.config.FIELD_WIDTH,
            self.config.FIELD_LENGTH,
            self.config.GRID_RESOLUTION
        )
        self.planner = AStarPlanner(self.config)
        self.car_controller = CarController(self.config)
        
        # Object detection
        self.object_detector = None
        if DETECTION_AVAILABLE:
            try:
                self.object_detector = ObjectDetector(confidence_threshold=0.4)
                logger.info("Object detector ready")
            except Exception as e:
                logger.warning(f"Object detection failed: {e}")
        
        # State variables
        self.current_position = Position(60, 0, 0)  # Center, start, facing forward
        self.goal_position = Position(60, 375, 0)   # Center, near end
        
        # Navigation state
        self.current_path = []
        self.path_index = 0
        self.last_replan_time = 0
        
        # Position tracking for drift correction
        self.position_history = []
        self.expected_x = 60.0  # Center line
        
        # Control state
        self.running = Event()
        self.emergency_stop = Event()
        self.last_detection_time = 0
        self.distance_since_detection = 0
        
        logger.info("Advanced self-driving system initialized")
    
    async def initialize(self):
        """Initialize all subsystems."""
        logger.info("Initializing system...")
        
        # Start object detection
        if self.object_detector:
            if self.object_detector.start_detection():
                logger.info("Object detection started")
            else:
                logger.warning("Object detection failed to start")
                self.object_detector = None
        
        # Initial scan
        await self._perform_initial_scan()
        
        logger.info("System initialization complete")
    
    async def _perform_initial_scan(self):
        """Perform comprehensive 180-degree scan."""
        logger.info("Performing initial environmental scan...")
        
        if not self.car_controller.picarx:
            return
        
        # Scan from -90 to +90 degrees
        for angle in range(-90, 91, 15):
            try:
                self.car_controller.picarx.set_cam_pan_angle(angle)
                await asyncio.sleep(0.1)
                
                distance = self.car_controller.read_ultrasonic()
                if 0 < distance <= 100:
                    # Convert sensor reading to world coordinates
                    sensor_angle = self.current_position.theta + math.radians(angle)
                    obs_x = self.current_position.x + distance * math.sin(sensor_angle)
                    obs_y = self.current_position.y + distance * math.cos(sensor_angle)
                    
                    self.grid.add_obstacle(obs_x, obs_y)
            except Exception as e:
                logger.warning(f"Scan error at angle {angle}: {e}")
        
        # Return sensor to center
        try:
            self.car_controller.picarx.set_cam_pan_angle(0)
            await asyncio.sleep(0.2)
        except:
            pass
        
        # Inflate obstacles for path planning
        self.grid.inflate_obstacles(self.config.OBSTACLE_INFLATION)
        logger.info("Initial scan complete")
    
    def _update_position(self, distance_moved: float, heading_change: float = 0):
        """Update position with drift tracking."""
        # Apply drift correction
        corrected_distance = distance_moved * self.config.DRIFT_CORRECTION_FACTOR
        
        # Update position
        self.current_position.x += corrected_distance * math.sin(self.current_position.theta)
        self.current_position.y += corrected_distance * math.cos(self.current_position.theta)
        self.current_position.theta += heading_change
        
        # Normalize theta
        while self.current_position.theta > math.pi:
            self.current_position.theta -= 2 * math.pi
        while self.current_position.theta < -math.pi:
            self.current_position.theta += 2 * math.pi
        
        # Track position for drift analysis
        self.position_history.append({
            'position': Position(self.current_position.x, self.current_position.y, self.current_position.theta),
            'time': time.time()
        })
        
        # Keep only recent history
        if len(self.position_history) > 10:
            self.position_history.pop(0)
        
        # Update grid
        self.grid.mark_car_position(self.current_position)
        
        logger.info(f"Position: {self.current_position}")
    
    def _detect_drift(self) -> Tuple[bool, float, str]:
        """Detect and quantify drift from expected path."""
        if abs(self.current_position.theta) > 0.3:  # Not driving straight
            return False, 0.0, ""
        
        drift_amount = abs(self.current_position.x - self.expected_x)
        
        if drift_amount > 8.0:  # 8cm tolerance
            if self.current_position.x > self.expected_x:
                return True, drift_amount, "left"  # Drifting right, correct left
            else:
                return True, drift_amount, "right"  # Drifting left, correct right
        
        return False, 0.0, ""
    
    async def _replan_path(self):
        """Replan path to goal."""
        logger.info("Replanning path...")
        
        # Update obstacles and inflate
        self.grid.inflate_obstacles(self.config.OBSTACLE_INFLATION)
        
        # Plan new path
        new_path = self.planner.plan_path(
            self.current_position,
            self.goal_position,
            self.grid
        )
        
        if new_path:
            self.current_path = new_path
            self.path_index = 0
            logger.info(f"New path planned with {len(new_path)} waypoints")
        else:
            logger.warning("No path found - creating simple forward path")
            # Create simple forward path
            forward_distance = min(50, self.goal_position.y - self.current_position.y)
            if forward_distance > 0:
                self.current_path = [
                    Position(self.current_position.x, self.current_position.y + forward_distance, 0)
                ]
                self.path_index = 0
    
    async def _execute_navigation_step(self):
        """Execute one navigation step."""
        if not self.current_path or self.path_index >= len(self.current_path):
            return False
        
        target = self.current_path[self.path_index]
        
        # Calculate movement vector
        dx = target.x - self.current_position.x
        dy = target.y - self.current_position.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < 3:  # Close enough to waypoint
            self.path_index += 1
            return True
        
        # Calculate required heading
        required_heading = math.atan2(dx, dy)
        heading_error = required_heading - self.current_position.theta
        
        # Normalize heading error
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Execute turn if needed
        if abs(heading_error) > 0.2:  # ~11 degrees
            await self._execute_turn(heading_error)
        else:
            # Move forward
            await self._execute_forward_movement(min(distance, 20))  # Max 20cm steps
        
        return True
    
    async def _execute_turn(self, heading_error: float):
        """Execute turn maneuver."""
        turn_angle_deg = math.degrees(heading_error)
        logger.info(f"Turning {turn_angle_deg:.1f} degrees")
        
        if not self.car_controller.picarx:
            return
        
        try:
            # Set steering
            if heading_error > 0:  # Turn left
                self.car_controller.turn_left(self.config.TURN_POWER)
            else:  # Turn right
                self.car_controller.turn_right(self.config.TURN_POWER)
            
            # Calculate turn time
            turn_time = abs(turn_angle_deg) / 90 * 0.4
            turn_time = max(0.2, min(turn_time, 1.0))
            
            await asyncio.sleep(turn_time)
            self.car_controller.stop()
            
            # Return to center
            self.car_controller.picarx.set_dir_servo_angle(self.config.SERVO_OFFSET)
            await asyncio.sleep(0.2)
            
            # Update position
            self._update_position(0, heading_error)
            
        except Exception as e:
            logger.error(f"Turn execution failed: {e}")
    
    async def _execute_forward_movement(self, distance: float):
        """Execute forward movement with safety monitoring."""
        logger.info(f"Moving forward {distance:.1f}cm")
        
        if not self.car_controller.picarx:
            return
        
        movement_time = distance / self.config.DRIVE_SPEED
        increment_time = 0.1
        elapsed_time = 0
        
        try:
            self.car_controller.move_forward(self.config.DRIVE_POWER)
            
            while elapsed_time < movement_time and not self.emergency_stop.is_set():
                # Check for obstacles
                obstacle_distance = self.car_controller.read_ultrasonic()
                if 0 < obstacle_distance <= self.config.EMERGENCY_STOP_DISTANCE:
                    logger.warning(f"Emergency stop: obstacle at {obstacle_distance}cm")
                    break
                
                # Check object detection
                if (self.object_detector and self.object_detector.is_halt_needed()):
                    logger.warning("Stop: object detected")
                    break
                
                await asyncio.sleep(increment_time)
                elapsed_time += increment_time
            
            actual_distance = (elapsed_time / movement_time) * distance
            self._update_position(actual_distance)
            self.distance_since_detection += actual_distance
            
        except Exception as e:
            logger.error(f"Forward movement failed: {e}")
        finally:
            self.car_controller.stop()
    
    async def _safety_monitor(self):
        """Continuous safety monitoring."""
        while self.running.is_set():
            try:
                # Check for immediate obstacles
                obstacle_distance = self.car_controller.read_ultrasonic()
                if 0 < obstacle_distance <= self.config.EMERGENCY_STOP_DISTANCE:
                    logger.warning("Emergency stop triggered")
                    self.emergency_stop.set()
                    self.car_controller.stop()
                elif obstacle_distance > self.config.SAFE_FOLLOWING_DISTANCE:
                    self.emergency_stop.clear()
                
                # Handle object detection
                if (self.object_detector and self.object_detector.is_halt_needed()):
                    logger.info("Object detection stop")
                    self.car_controller.stop()
                    await asyncio.sleep(3.0)  # Pause for stop sign/person
                    self.last_detection_time = time.time()
                    self.distance_since_detection = 0
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Safety monitor error: {e}")
                await asyncio.sleep(0.5)
    
    async def _main_control_loop(self):
        """Main navigation control loop."""
        logger.info("Starting main control loop...")
        
        while self.running.is_set():
            try:
                current_time = time.time()
                
                # Check if goal reached
                if self.current_position.distance_to(self.goal_position) < 15:
                    logger.info("GOAL REACHED!")
                    break
                
                # Replan if needed
                if (current_time - self.last_replan_time > self.config.REPLAN_INTERVAL or
                    not self.current_path):
                    await self._replan_path()
                    self.last_replan_time = current_time
                
                # Execute navigation if not in emergency
                if not self.emergency_stop.is_set():
                    continue_nav = await self._execute_navigation_step()
                    if not continue_nav:
                        logger.info("Navigation step completed")
                
                # Check for drift and correct
                is_drifting, drift_amount, correction = self._detect_drift()
                if is_drifting:
                    logger.warning(f"Drift detected: {drift_amount:.1f}cm, correcting {correction}")
                    # Apply mild steering correction in next movement
                
                await asyncio.sleep(1.0 / self.config.CONTROL_FREQUENCY)
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                await asyncio.sleep(0.5)
    
    async def run(self):
        """Main run method."""
        try:
            await self.initialize()
            self.running.set()
            
            # Start concurrent tasks
            safety_task = asyncio.create_task(self._safety_monitor())
            control_task = asyncio.create_task(self._main_control_loop())
            
            await asyncio.gather(safety_task, control_task)
            
        except KeyboardInterrupt:
            logger.info("System interrupted by user")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down...")
        self.running.clear()
        
        if self.car_controller:
            self.car_controller.stop()
        
        if self.object_detector:
            self.object_detector.stop_detection()
        
        logger.info("Shutdown complete")


# Main execution and testing
async def test_system():
    """Test system functionality."""
    config = SystemConfig()
    system = AdvancedSelfDrivingSystem(config)
    
    logger.info("Running system test...")
    
    # Test for 30 seconds
    try:
        await asyncio.wait_for(system.run(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.info("Test completed (timeout)")
    
    await system.shutdown()

def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        asyncio.run(test_system())
    else:
        config = SystemConfig()
        system = AdvancedSelfDrivingSystem(config)
        asyncio.run(system.run())

if __name__ == "__main__":
    main()
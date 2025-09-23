# fixed_integrated_advanced.py
# CS 437 Lab 1: Advanced Fixed Self-Driving Car with A* Navigation
# Combines the best of both original files with comprehensive fixes

import asyncio
import time
import numpy as np
import logging
import math
import os
import sys
from threading import Event, Lock
from queue import Queue
from dataclasses import dataclass
from typing import List, Tuple, Optional
import heapq
from picarx import Picarx
from fixed_object_detection import ObjectDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MapVisualizer:
    """Terminal-based map visualization for obstacle detection and navigation."""
    
    def __init__(self, width: int, height: int, resolution: float):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # Colors for terminal display
        self.RESET = '\033[0m'
        self.RED = '\033[91m'      # Obstacles
        self.GREEN = '\033[92m'    # Free space
        self.BLUE = '\033[94m'     # Car position
        self.YELLOW = '\033[93m'   # Goal
        self.CYAN = '\033[96m'     # Path
        self.MAGENTA = '\033[95m'  # Inflated obstacles\

                #pan stuff
        self.PAN_ANGLE = 60  # degrees
        self.SENSOR_REFRESH = 0.2  # seconds between sensor reads
        self.MAXREAD = 100  # cm - max valid ultrasonic reading

        # Map and sensor parameters
        self.stop_event = asyncio.Event()
        self.map_lock = asyncio.Lock()
        self.map_dirty = asyncio.Event()
        self.plan_lock = asyncio.Lock()
        self.halt_event = asyncio.Event()
        self.SENSOR_REFRESH = 0.10
        self.DISPLAY_REFRESH = 0.20
        self.CAR_DISPLAY_REFRESH = 0.10
        self.PLAN_REFRESH_MIN = 0.5
        self.map_ = np.zeros((self.FIELD_LENGTH, self.FIELD_WIDTH), dtype=np.uint8)
        self.flag = 0 
        
    def clear_screen(self):
        """Clear terminal screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def display_map(self, grid: np.ndarray, car_pos: 'Position', goal_pos: 'Position', 
                   path: List['Position'] = None, scan_angle: float = None):
        """Display colored map in terminal with legend and statistics.
        
        **VIDEO DEMO POINT**: This method creates the real-time obstacle map visualization
        Line numbers: ~35-95 in MapVisualizer class
        """
        self.clear_screen()
        
        # Create display grid
        display_grid = grid.copy()
        
        # Mark car position
        car_grid_x, car_grid_y = int(car_pos.x / self.resolution), int(car_pos.y / self.resolution)
        if 0 <= car_grid_x < self.grid_width and 0 <= car_grid_y < self.grid_height:
            display_grid[car_grid_y, car_grid_x] = 3  # Car marker
        
        # Mark goal position  
        goal_grid_x, goal_grid_y = int(goal_pos.x / self.resolution), int(goal_pos.y / self.resolution)
        if 0 <= goal_grid_x < self.grid_width and 0 <= goal_grid_y < self.grid_height:
            display_grid[goal_grid_y, goal_grid_x] = 4  # Goal marker
            
        # Mark path if provided
        if path:
            for waypoint in path[:10]:  # Show first 10 waypoints
                path_x, path_y = int(waypoint.x / self.resolution), int(waypoint.y / self.resolution)
                if (0 <= path_x < self.grid_width and 0 <= path_y < self.grid_height and
                    display_grid[path_y, path_x] == 0):
                    display_grid[path_y, path_x] = 5  # Path marker
        
        # Print header with statistics
        obstacle_count = np.sum(grid == 1)
        free_count = np.sum(grid == 0)
        print(f"\n🗺️  AUTONOMOUS CAR MAPPING SYSTEM")
        print(f"📍 Car: ({car_pos.x:.1f}, {car_pos.y:.1f}) @ {math.degrees(car_pos.theta):.1f}°")
        print(f"🎯 Goal: ({goal_pos.x:.1f}, {goal_pos.y:.1f})")
        print(f"📊 Map: {obstacle_count} obstacles, {free_count} free cells")
        if scan_angle is not None:
            print(f"📡 Scanning: {scan_angle:.0f}° (360° environmental scan)")
        print("=" * 60)
        
        # Print map with colors
        for y in range(min(25, self.grid_height)):  # Limit display size
            for x in range(min(60, self.grid_width)):
                cell_value = display_grid[y, x] if y < self.grid_height and x < self.grid_width else 0
                
                if cell_value == 1:    # Obstacle
                    print(f"{self.RED}██{self.RESET}", end="")
                elif cell_value == 2:  # Inflated obstacle
                    print(f"{self.MAGENTA}▓▓{self.RESET}", end="")
                elif cell_value == 3:  # Car
                    print(f"{self.BLUE}🚗{self.RESET}", end="")
                elif cell_value == 4:  # Goal
                    print(f"{self.YELLOW}🎯{self.RESET}", end="")
                elif cell_value == 5:  # Path
                    print(f"{self.CYAN}··{self.RESET}", end="")
                else:                  # Free space
                    print(f"{self.GREEN}  {self.RESET}", end="")
            print()  # New line
        
        # Print legend
        print("=" * 60)
        print(f"Legend: {self.RED}██{self.RESET}=Obstacles {self.MAGENTA}▓▓{self.RESET}=Danger Zone " +
              f"{self.BLUE}🚗{self.RESET}=Car {self.YELLOW}🎯{self.RESET}=Goal {self.CYAN}··{self.RESET}=Path")
        print("=" * 60)

@dataclass
class SystemConfig:
    """Centralized configuration with optimized parameters."""
    # Field dimensions - CORRECTED VALUES
    FIELD_WIDTH: int = 120  # Real field width in cm
    FIELD_LENGTH: int = 200  # Real field length in cm
    CAR_WIDTH: int = 14
    CAR_LENGTH: int = 23
    
    # Navigation
    GRID_RESOLUTION: float = 2.0  # 2cm per grid cell - match real coordinates
    OBSTACLE_INFLATION: int = 8   # Grid cells to inflate around obstacles (16cm radius)
    
    # Movement - CALIBRATED VALUES
    SERVO_OFFSET: int = -1.0
    DRIVE_SPEED: float = 100.0     # cm/s - increased to match actual movement
    DRIVE_POWER: int = 35
    TURN_POWER: int = 25
    DRIFT_CORRECTION_FACTOR: float = 1.0  # No distance reduction - track actual movement
    
    # Turn angles (in degrees)
    OBSTACLE_AVOIDANCE_ANGLE: float = 30.0  # Larger turn for obstacle avoidance
    MAX_DRIFT_CORRECTION_ANGLE: float = 5.0  # Max steering adjustment for drift
    NAVIGATION_TURN_THRESHOLD: float = 11.0  # Turn threshold in degrees (0.2 rad)
    
    # Distances (in cm)
    OBSTACLE_BACKUP_DISTANCE: float = 15.0   # How far to back up when obstacle detected
    OBSTACLE_BACKUP_TIME: float = 1.5       # How long to back up
    
    # Safety
    EMERGENCY_STOP_DISTANCE: int = 15
    DETECTION_DISTANCE: int = 15
    SAFE_FOLLOWING_DISTANCE: int = 30
    
    # Timing
    CONTROL_FREQUENCY: int = 15
    REPLAN_INTERVAL: float = 2.0
    DETECTION_INTERVAL: float = 0.05  # Check for objects every 50ms - more aggressive detection

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
        self.picarx = Picarx(servo_pins=["P0", "P1", "P3"])
        self.picarx.set_dir_servo_angle(config.SERVO_OFFSET)
        logger.info("PiCarx initialized")
    
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
        
        # Object detection with corrected settings to avoid false positives
        self.object_detector = ObjectDetector(confidence_threshold=0.8)
        logger.info("Object detector ready")
        
        # Add map visualizer for terminal display
        self.map_visualizer = MapVisualizer(
            self.config.FIELD_WIDTH,
            self.config.FIELD_LENGTH, 
            self.config.GRID_RESOLUTION
        )
        logger.info("Map visualizer ready")
        
        # State variables
        self.current_position = Position(60, 0, 0)  # Center, start, facing forward
        self.goal_position = Position(60, 190, 0)   # Adjusted for new field length
        
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
        self.stop_sign_pause = Event()  # New: for stop sign pauses
        self.last_detection_time = 0
        self.distance_since_detection = 0
        self.stop_sign_detected_time = 0
        self.stop_sign_pause_duration = 3.0  # seconds to pause for stop sign
        
        # Obstacle avoidance
        self.obstacle_detected_time = 0
        self.obstacle_avoidance_timeout = 10.0  # Try to avoid obstacle for 10 seconds
        self.last_obstacle_position = None
        
        # Detection timing - ensure we check detection at configured intervals
        self.last_detection_check = 0
        self.detection_check_interval = self.config.DETECTION_INTERVAL
        
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
        """Perform comprehensive 180-degree scan with real-time map visualization.
        
        **VIDEO DEMO POINT**: Environmental scanning and obstacle detection
        Line numbers: ~370-420 in AdvancedSelfDrivingSystem class
        Shows: Real-time map building, obstacle detection, 360° scanning
        """
        logger.info("Performing initial environmental scan...")
        
        if not self.car_controller.picarx:
            return
        
        # Display initial empty map
        self.map_visualizer.display_map(
            self.grid.get_copy(), 
            self.current_position, 
            self.goal_position,
            scan_angle=0
        )
        await asyncio.sleep(1)  # Pause for video
        
        # Scan from -90 to +90 degrees with visualization
        scan_angles = list(range(-90, 91, 15))
        for i, angle in enumerate(scan_angles):
            try:
                self.car_controller.picarx.set_cam_pan_angle(angle)
                await asyncio.sleep(0.2)  # Allow time for camera positioning
                
                # Update display to show current scan angle
                self.map_visualizer.display_map(
                    self.grid.get_copy(),
                    self.current_position,
                    self.goal_position, 
                    scan_angle=angle
                )
                
                distance = self.car_controller.read_ultrasonic()
                if 0 < distance <= 100:
                    # Convert sensor reading to world coordinates
                    sensor_angle = self.current_position.theta + math.radians(angle)
                    obs_x = self.current_position.x + distance * math.sin(sensor_angle)
                    obs_y = self.current_position.y + distance * math.cos(sensor_angle)
                    
                    self.grid.add_obstacle(obs_x, obs_y)
                    logger.info(f"📡 Scan {angle:+3d}°: Obstacle detected at {distance:.1f}cm -> ({obs_x:.1f}, {obs_y:.1f})")
                    
                    # Update display immediately when obstacle is found
                    await asyncio.sleep(0.5)  # Pause to show detection
                else:
                    logger.info(f"📡 Scan {angle:+3d}°: Clear ({distance:.1f}cm)")
                    
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
        
        # Display final map with obstacles and inflated zones
        self.map_visualizer.display_map(
            self.grid.get_copy(),
            self.current_position,
            self.goal_position
        )
        
        logger.info("🗺️ Initial scan complete - map generated!")
        await asyncio.sleep(2)  # Pause for video
    
    def _update_position(self, distance_moved: float, heading_change: float = 0):
        """Update position with proper heading consideration and drift tracking.
        
        **VIDEO DEMO POINT**: Position tracking and coordinate system
        Line numbers: ~380-410 in AdvancedSelfDrivingSystem class
        Shows: Heading-aware position updates, coordinate transformation
        """
        # Apply heading change first
        self.current_position.theta += heading_change
        
        # Normalize theta
        while self.current_position.theta > math.pi:
            self.current_position.theta -= 2 * math.pi
        while self.current_position.theta < -math.pi:
            self.current_position.theta += 2 * math.pi
        
        # Apply drift correction
        corrected_distance = distance_moved * self.config.DRIFT_CORRECTION_FACTOR
        
        # Update position based on CURRENT heading (after any turn)
        # cos(theta) = forward/backward along Y axis (north/south)
        # sin(theta) = left/right along X axis (west/east)
        self.current_position.x += corrected_distance * math.sin(self.current_position.theta)
        self.current_position.y += corrected_distance * math.cos(self.current_position.theta)
        
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
        
        # Update map visualization periodically (every few movements)
        if hasattr(self, '_movement_count'):
            self._movement_count += 1
        else:
            self._movement_count = 1
            
        if self._movement_count % 3 == 0:  # Update map every 3 movements
            try:
                self.map_visualizer.display_map(
                    self.grid.get_copy(),
                    self.current_position,
                    self.goal_position,
                    path=self.current_path
                )
            except:
                pass  # Don't let visualization errors stop navigation
        
        logger.info(f"📍 Position: {self.current_position}")
    
    def _add_detected_obstacle(self, distance: float):
        """Add detected obstacle to the grid for path planning with proper heading consideration.
        
        **VIDEO DEMO POINT**: Real-time obstacle detection and mapping
        Line numbers: ~460-490 in AdvancedSelfDrivingSystem class
        Shows: Dynamic obstacle detection, map updates, safety margins
        """
        # Calculate obstacle position based on current position and heading
        # Add safety margin to the detected distance
        safety_margin = 5.0  # Add 5cm safety margin
        obstacle_distance = distance + safety_margin
        
        # Use current heading to properly place obstacle
        obstacle_x = self.current_position.x + obstacle_distance * math.sin(self.current_position.theta)
        obstacle_y = self.current_position.y + obstacle_distance * math.cos(self.current_position.theta)
        
        # Ensure obstacle is within field bounds
        obstacle_x = max(0, min(self.config.FIELD_WIDTH, obstacle_x))
        obstacle_y = max(0, min(self.config.FIELD_LENGTH, obstacle_y))
        
        # Add obstacle to grid - create a larger obstacle area for safety
        obstacle_size = 15  # 15cm x 15cm obstacle area
        for dx in range(-obstacle_size, obstacle_size + 1, 5):
            for dy in range(-obstacle_size, obstacle_size + 1, 5):
                obs_x = obstacle_x + dx
                obs_y = obstacle_y + dy
                if (0 <= obs_x <= self.config.FIELD_WIDTH and 
                    0 <= obs_y <= self.config.FIELD_LENGTH):
                    self.grid.add_obstacle(obs_x, obs_y)
        
        logger.info(f"🚨 Added {obstacle_size*2}x{obstacle_size*2}cm obstacle area at ({obstacle_x:.1f}, {obstacle_y:.1f}) based on detection at {distance:.1f}cm, heading {math.degrees(self.current_position.theta):.1f}°")
        
        # Show updated map with new obstacle
        try:
            self.map_visualizer.display_map(
                self.grid.get_copy(),
                self.current_position,
                self.goal_position,
                path=self.current_path
            )
        except:
            pass  # Don't let visualization errors stop navigation
        
        # Force path replanning
        self.last_replan_time = 0
        
        return (obstacle_x, obstacle_y)
    
    async def _handle_obstacle_avoidance(self, obstacle_distance: float):
        """Handle obstacle avoidance using proper A* replanning."""
        current_time = time.time()
        
        # Add obstacle to grid if not already added recently
        if (self.last_obstacle_position is None or 
            current_time - self.obstacle_detected_time > 5.0):
            self.last_obstacle_position = self._add_detected_obstacle(obstacle_distance)
            self.obstacle_detected_time = current_time
            logger.info(f"Added obstacle to grid at {self.last_obstacle_position}")
        
        try:
            logger.info("Executing systematic obstacle avoidance with A* replanning")
            
            # Step 1: Back up to create maneuvering space
            if self.car_controller.picarx:
                self.car_controller.picarx.backward(self.config.DRIVE_POWER)
                await asyncio.sleep(self.config.OBSTACLE_BACKUP_TIME)
                self.car_controller.stop()
                
                # Update position (moved backward)
                self._update_position(-self.config.OBSTACLE_BACKUP_DISTANCE, 0)
                logger.info(f"Backed up {self.config.OBSTACLE_BACKUP_DISTANCE}cm to position {self.current_position}")
                
                # Step 2: Update grid with obstacles and replan path
                await self._replan_path()
                
                # Step 3: If replanning failed, try a more aggressive avoidance maneuver
                if not self.current_path or len(self.current_path) <= self.path_index:
                    logger.warning("A* replanning failed, trying aggressive avoidance maneuver")
                    
                    # Do a wider turn to get around obstacle - use config values
                    turn_angle_deg = self.config.OBSTACLE_AVOIDANCE_ANGLE
                    turn_angle_rad = math.radians(turn_angle_deg)
                    
                    # Try to determine better direction by checking both sides
                    # For now, alternate direction to avoid getting stuck
                    if hasattr(self, '_last_turn_direction'):
                        self._last_turn_direction = not self._last_turn_direction
                    else:
                        self._last_turn_direction = True
                    
                    # Make a more significant turn
                    if self._last_turn_direction:
                        self.car_controller.turn_left(self.config.TURN_POWER)
                        # Apply heading change first, then forward movement from turn
                        self._update_position(0, turn_angle_rad)  
                        self._update_position(2.0, 0)  # ~2cm forward during turn
                        logger.info(f"Aggressive left turn: {turn_angle_deg}°")
                    else:
                        self.car_controller.turn_right(self.config.TURN_POWER)
                        # Apply heading change first, then forward movement from turn
                        self._update_position(0, -turn_angle_rad)
                        self._update_position(2.0, 0)  # ~2cm forward during turn
                        logger.info(f"Aggressive right turn: {turn_angle_deg}°")
                    
                    await asyncio.sleep(0.8)  # Longer turn time for wider angle
                    self.car_controller.stop()
                    
                    # Move forward a bit to get clear of the obstacle area
                    self.car_controller.move_forward(self.config.DRIVE_POWER)
                    await asyncio.sleep(0.5)  # Move forward 25cm
                    self.car_controller.stop()
                    
                    # Update position for the forward movement
                    forward_distance = 0.5 * self.config.DRIVE_SPEED
                    self._update_position(forward_distance, 0)
                    logger.info(f"Moved forward {forward_distance:.1f}cm to clear obstacle area")
                    
                    # Return to center steering
                    self.car_controller.picarx.set_dir_servo_angle(self.config.SERVO_OFFSET)
                    await asyncio.sleep(0.2)
                    
                    # Try replanning again after the maneuver
                    await self._replan_path()
                
        except Exception as e:
            logger.error(f"Obstacle avoidance failed: {e}")
            # Emergency fallback - just stop
            self.car_controller.stop()
    
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
    
    async def _apply_drift_correction(self, drift_amount: float, correction_direction: str):
        """Apply gentle steering adjustment to fix drift - NO active turning."""
        if not self.car_controller.picarx:
            return
            
        try:
            # Only apply very small steering adjustments, don't do active turns
            if drift_amount > 15 and drift_amount < 40:  # Only for moderate drift
                # Use config constant for max correction angle
                correction_angle = min(drift_amount * 0.1, self.config.MAX_DRIFT_CORRECTION_ANGLE)
                
                logger.info(f"Gentle steering adjustment: {correction_angle:.1f}° {correction_direction}")
                
                # Just adjust steering angle, don't do active turning
                if correction_direction == "right":
                    self.car_controller.picarx.set_dir_servo_angle(
                        self.config.SERVO_OFFSET + correction_angle)
                else:
                    self.car_controller.picarx.set_dir_servo_angle(
                        self.config.SERVO_OFFSET - correction_angle)
                        
                # No heading update since this is just steering adjustment, not a turn
                await asyncio.sleep(0.1)  # Brief pause
                
            elif drift_amount >= 40:
                logger.warning(f"Large drift {drift_amount:.1f}cm - may be calculation error, no correction")
            
            # For small drift or large drift, just go straight
            else:
                self.car_controller.picarx.set_dir_servo_angle(self.config.SERVO_OFFSET)
                
        except Exception as e:
            logger.error(f"Drift correction failed: {e}")
    
    def _apply_continuous_drift_correction(self):
        """Apply drift correction during forward movement by adjusting steering angle."""
        if not self.car_controller.picarx:
            return
            
        # Calculate drift from goal path - use current path target if available
        if self.current_path and self.path_index < len(self.current_path):
            target_x = self.current_path[self.path_index].x
        else:
            target_x = 60.0  # Default center line
            
        drift_amount = abs(self.current_position.x - target_x)
        
        if drift_amount > 3:  # Apply correction for any significant drift
            # More aggressive correction for larger drifts
            if drift_amount > 20:
                correction_angle = self.config.MAX_DRIFT_CORRECTION_ANGLE  # Max correction
            else:
                correction_angle = min(drift_amount * 0.4, self.config.MAX_DRIFT_CORRECTION_ANGLE)
            
            if self.current_position.x < target_x:
                # Drifted left, steer right
                self.car_controller.picarx.set_dir_servo_angle(
                    self.config.SERVO_OFFSET + correction_angle)
            else:
                # Drifted right, steer left
                self.car_controller.picarx.set_dir_servo_angle(
                    self.config.SERVO_OFFSET - correction_angle)
                self.car_controller.picarx.set_dir_servo_angle(
                    self.config.SERVO_OFFSET - correction_angle)
        else:
            # Small drift or large drift - go straight
            self.car_controller.picarx.set_dir_servo_angle(self.config.SERVO_OFFSET)
    
    async def _replan_path(self):
        """Replan path to goal with map visualization.
        
        **VIDEO DEMO POINT**: A* pathfinding algorithm in action
        Line numbers: ~630-680 in AdvancedSelfDrivingSystem class
        Shows: A* algorithm, path planning, obstacle avoidance routing
        """
        logger.info("🧭 Replanning path with A* algorithm...")
        
        # Update obstacles and inflate
        self.grid.inflate_obstacles(self.config.OBSTACLE_INFLATION)
        
        # Plan new path using A* algorithm
        new_path = self.planner.plan_path(
            self.current_position,
            self.goal_position,
            self.grid
        )
        
        if new_path:
            self.current_path = new_path
            self.path_index = 0
            logger.info(f"✅ A* path found with {len(new_path)} waypoints")
            
            # Display map with new path
            self.map_visualizer.display_map(
                self.grid.get_copy(),
                self.current_position,
                self.goal_position,
                path=new_path
            )
            await asyncio.sleep(1)  # Show path for video
            
        else:
            logger.warning("❌ A* failed - creating corrective path toward goal")
            # Create path that heads toward the goal, not just forward
            dx = self.goal_position.x - self.current_position.x
            dy = self.goal_position.y - self.current_position.y
            
            # Calculate a waypoint that moves toward the goal
            if abs(dx) > abs(dy):
                # More horizontal correction needed
                step_x = 20 if dx > 0 else -20
                step_y = dy * (20 / abs(dx)) if dx != 0 else 0
            else:
                # More vertical correction needed  
                step_y = 20 if dy > 0 else -20
                step_x = dx * (20 / abs(dy)) if dy != 0 else 0
                
            target_x = self.current_position.x + step_x
            target_y = self.current_position.y + step_y
            
            # Ensure we stay within field bounds
            target_x = max(10, min(110, target_x))
            target_y = max(0, min(190, target_y))
            
            self.current_path = [Position(target_x, target_y, 0)]
            self.path_index = 0
            logger.info(f"🔄 Created corrective path to ({target_x:.1f}, {target_y:.1f}) toward goal")
            
            # Display corrective path
            self.map_visualizer.display_map(
                self.grid.get_copy(),
                self.current_position,
                self.goal_position,
                path=self.current_path
            )
    
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
        if abs(heading_error) > math.radians(self.config.NAVIGATION_TURN_THRESHOLD):
            await self._execute_turn(heading_error)
        else:
            # Move forward
            await self._execute_forward_movement(min(distance, 20))  # Max 20cm steps
        
        return True
    
    async def _execute_turn(self, heading_error: float):
        """Execute turn maneuver with proper position tracking."""
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
            
            # Update position: turns cause some forward movement + heading change
            # PiCar-X moves forward while turning, estimate ~2cm per turn
            forward_distance_during_turn = min(2.0, abs(turn_angle_deg) / 45.0 * 2.0)
            
            # Apply heading change first, then forward movement
            self._update_position(0, heading_error)  # Apply heading change
            self._update_position(forward_distance_during_turn, 0)  # Apply forward movement
            
            logger.info(f"Turn complete: {turn_angle_deg:.1f}° + {forward_distance_during_turn:.1f}cm forward")
            
        except Exception as e:
            logger.error(f"Turn execution failed: {e}")
    
    async def _execute_forward_movement(self, distance: float):
        """Execute forward movement with safety monitoring and drift correction."""
        logger.info(f"Moving forward {distance:.1f}cm")
        
        if not self.car_controller.picarx:
            return

        movement_time = distance / self.config.DRIVE_SPEED
        increment_time = min(0.05, self.detection_check_interval / 2)  # Check more frequently than detection interval
        elapsed_time = 0
        start_time = time.time()
        
        try:
            self.car_controller.move_forward(self.config.DRIVE_POWER)
            
            while elapsed_time < movement_time and not self.emergency_stop.is_set() and not self.stop_sign_pause.is_set():
                current_time = time.time()
                
                # Check for obstacles (but not during stop sign pause)
                if not self.stop_sign_pause.is_set():
                    obstacle_distance = self.car_controller.read_ultrasonic()
                    if 0 < obstacle_distance <= self.config.EMERGENCY_STOP_DISTANCE:
                        logger.warning(f"Emergency stop: obstacle at {obstacle_distance}cm")
                        self.car_controller.stop()  # Stop immediately
                        break  # Exit movement loop immediately
                
                # Apply drift correction during movement (maintain straight path)
                self._apply_continuous_drift_correction()
                
                # CRITICAL: Check object detection during movement at configured intervals
                if (current_time - self.last_detection_check >= self.detection_check_interval):
                    self.last_detection_check = current_time
                    if (self.object_detector and self.object_detector.is_halt_needed() and 
                        not self.stop_sign_pause.is_set()):
                        logger.warning("Stop sign detected during movement - stopping immediately")
                        break
                
                await asyncio.sleep(increment_time)
                elapsed_time += increment_time
            
            # Calculate actual distance based on actual movement time
            actual_movement_time = time.time() - start_time
            actual_distance = min(actual_movement_time * self.config.DRIVE_SPEED, distance)
            
            # Update position with actual distance moved
            self._update_position(actual_distance)
            self.distance_since_detection += actual_distance
            
            # Log movement tracking for debugging
            logger.info(f"Movement complete: planned={distance:.1f}cm, actual_time={actual_movement_time:.2f}s, actual_distance={actual_distance:.1f}cm")
            
        except Exception as e:
            logger.error(f"Forward movement failed: {e}")
        finally:
            self.car_controller.stop()
    
    async def _safety_monitor(self):
        """Continuous safety monitoring with proper detection intervals."""
        while self.running.is_set():
            try:
                current_time = time.time()
                
                # Handle stop sign pause
                if self.stop_sign_pause.is_set():
                    if current_time - self.stop_sign_detected_time >= self.stop_sign_pause_duration:
                        logger.info("Stop sign pause completed, resuming navigation")
                        self.stop_sign_pause.clear()
                        self.emergency_stop.clear()  # Clear emergency stop after stop sign pause
                    else:
                        remaining = self.stop_sign_pause_duration - (current_time - self.stop_sign_detected_time)
                        if int(remaining) != int(remaining + 0.1):  # Log only on second changes
                            logger.info(f"Stop sign pause: {remaining:.1f}s remaining")
                    await asyncio.sleep(0.05)  # Faster check during stop sign pause
                    continue
                
                # Check for immediate obstacles (only if not in stop sign pause)
                obstacle_distance = self.car_controller.read_ultrasonic()
                if 0 < obstacle_distance <= self.config.EMERGENCY_STOP_DISTANCE:
                    logger.warning(f"Emergency stop: obstacle at {obstacle_distance}cm")
                    self.emergency_stop.set()
                    self.car_controller.stop()
                    
                    # Trigger obstacle avoidance if stuck for too long
                    if (current_time - self.obstacle_detected_time > 2.0 or 
                        self.obstacle_detected_time == 0):
                        await self._handle_obstacle_avoidance(obstacle_distance)
                        
                elif obstacle_distance > self.config.SAFE_FOLLOWING_DISTANCE:
                    self.emergency_stop.clear()
                    # Reset obstacle detection timer when clear
                    if self.obstacle_detected_time > 0:
                        logger.info("Obstacle cleared, resuming normal navigation")
                        self.obstacle_detected_time = 0
                        self.last_obstacle_position = None
                
                # Handle object detection at configured intervals - CRITICAL for catching objects
                if (current_time - self.last_detection_check >= self.detection_check_interval):
                    self.last_detection_check = current_time
                    
                    if (self.object_detector and self.object_detector.is_halt_needed() and 
                        not self.stop_sign_pause.is_set()):
                        logger.info("Stop sign detected - initiating pause")
                        self.car_controller.stop()
                        self.stop_sign_pause.set()
                        self.stop_sign_detected_time = current_time
                        self.last_detection_time = current_time
                        self.distance_since_detection = 0
                
                # Sleep for a short time but not longer than detection interval
                sleep_time = min(0.05, self.detection_check_interval / 2)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Safety monitor error: {e}")
                await asyncio.sleep(0.1)
    
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
                
                # Execute navigation if not in emergency or stop sign pause
                if not self.emergency_stop.is_set() and not self.stop_sign_pause.is_set():
                    continue_nav = await self._execute_navigation_step()
                    if not continue_nav:
                        logger.info("Navigation step completed")
                elif self.stop_sign_pause.is_set():
                    # Do nothing during stop sign pause - safety monitor handles timing
                    pass
                
                # Drift correction is now handled continuously during forward movement
                # No need for separate drift correction here
                
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

async def demo_multiple_destinations():
    """
    **VIDEO DEMO POINT**: Full autonomous navigation demo
    Line numbers: ~1000-1080 in main demo function
    Shows: Multiple destinations, obstacle avoidance, stop sign detection
    """
    config = SystemConfig()
    system = AdvancedSelfDrivingSystem(config)
    
    logger.info("🚗 STARTING FULL AUTONOMOUS DRIVING DEMO")
    logger.info("🎯 Destination 1: (30, 100) - Left side")
    
    try:
        # Initialize system
        await system.initialize()
        system.running.set()
        
        # First destination: Left side of field
        system.goal_position = Position(30, 100, 0)
        logger.info(f"🎯 Navigating to Destination 1: {system.goal_position}")
        
        # Start navigation tasks
        safety_task = asyncio.create_task(system._safety_monitor())
        control_task = asyncio.create_task(system._main_control_loop())
        
        # Run until first goal reached or timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(safety_task, control_task, return_when=asyncio.FIRST_COMPLETED),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            logger.info("⏰ First destination timeout - switching to destination 2")
        
        # Cancel current tasks
        safety_task.cancel()
        control_task.cancel()
        
        # Second destination: Right side of field
        system.goal_position = Position(90, 150, 0)
        logger.info(f"🎯 Navigating to Destination 2: {system.goal_position}")
        
        # Restart navigation tasks for second destination
        safety_task = asyncio.create_task(system._safety_monitor())
        control_task = asyncio.create_task(system._main_control_loop())
        
        # Run until second goal reached or timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(safety_task, control_task),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            logger.info("⏰ Demo completed - both destinations attempted")
            
    except KeyboardInterrupt:
        logger.info("🛑 Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo error: {e}")
    finally:
        await system.shutdown()
        logger.info("🏁 DEMO COMPLETE")

def main():
    """Main entry point with multiple demo modes."""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            asyncio.run(test_system())
        elif sys.argv[1] == "--demo":
            asyncio.run(demo_multiple_destinations())
        else:
            print("Usage: python3 fixed_integrated_advanced.py [--test|--demo]")
    else:
        # Default single destination mode
        config = SystemConfig()
        system = AdvancedSelfDrivingSystem(config)
        asyncio.run(system.run())

if __name__ == "__main__":
    main()
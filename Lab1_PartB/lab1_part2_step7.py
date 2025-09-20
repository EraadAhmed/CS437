"""
Lab 1 Part 2 Step 7: Object Detection Integration
Complete self-driving car with TensorFlow Lite object detection, ultrasonic mapping, and A* pathfinding

This implementation combines:
- Real-time object detection using TensorFlow Lite
- Ultrasonic sensor mapping
- A* pathfinding for navigation
- Autonomous car control with obstacle avoidance
"""

import numpy as np
import cv2
import time
import threading
import argparse
from queue import Queue, PriorityQueue
from collections import deque

# TensorFlow Lite imports (with fallback for older versions)
try:
    # Try newer tflite-support API (0.4.x)
    from tflite_support.task import core
    from tflite_support.task import processor
    from tflite_support.task import vision
    TFLITE_SUPPORT_NEW = True
except ImportError:
    try:
        # Try older tflite-support API (0.1.x)
        from tflite_support.task import vision
        TFLITE_SUPPORT_NEW = False
        print("Using older tflite-support API (0.1.x)")
    except ImportError:
        # Fallback to basic TensorFlow Lite
        try:
            import tensorflow as tf
            TFLITE_SUPPORT_NEW = False
            print("Using basic TensorFlow Lite interpreter")
        except ImportError:
            print("Warning: No TensorFlow Lite support available")

# Hardware imports (conditional for development)
try:
    from picamera2 import Picamera2
    from picarx import Picarx
    HARDWARE_AVAILABLE = True
except ImportError:
    print("Warning: Hardware libraries not available. Running in simulation mode.")
    HARDWARE_AVAILABLE = False

# Physical Constants (from computer_vision.py)
WIDTH = 120  # width of map in cm
LENGTH = 380  # length of map in cm
X_MID = 60  # midpoint on x axis
CAR_Width = 14
CAR_Length = 23

# Scaled dimensions (5cm per grid cell)
width_scaled = int(WIDTH / 5)
length_scaled = int(LENGTH / 5)
x_mid_scaled = int(width_scaled / 2)
CAR_Width_scaled = int(np.ceil(CAR_Width / 5))
CAR_Length_scaled = int(np.ceil(CAR_Length / 5))

# Vehicle positioning constants
MAX_READ = 100  # max ultrasonic reading considered for mapping
MAXREAD_SCALED = int(MAX_READ / 5)
SPEED = 10  # cm/sec
POWER = 40  # 0-100
delta_t = 0.25  # time step
SafeDistance = 25  # in cm
DangerDistance = 10  # in cm

# Start and End positions
start_pos = (x_mid_scaled - 1, 0)
end_pos = (x_mid_scaled - 1, length_scaled - 1)

# Object detection constants
WATCH_CLASSES = {"stop sign", "person"}
DETECTION_CONFIDENCE = 0.5
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
INFERENCE_PERIOD = 1.0  # seconds between inferences (~1 FPS)


class Coordinate:
    """Node class for A* pathfinding algorithm"""
    def __init__(self, state, g=0, h=0, parent=None):
        self.state = state        # (x, y, theta)
        self.g = g                # cost from start
        self.h = h                # heuristic to goal
        self.parent = parent      # backpointer
    
    def f(self):
        return self.g + self.h
    
    def __lt__(self, other):
        return self.f() < other.f()


class ObjectDetector:
    """TensorFlow Lite object detection with halt functionality (compatible with multiple TF Lite versions)"""
    
    def __init__(self, model_path='efficientdet_lite0.tflite', score_threshold=DETECTION_CONFIDENCE):
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.detector = None
        self.interpreter = None  # For basic TF Lite fallback
        self.last_detection_time = 0
        self.detection_period = INFERENCE_PERIOD
        
        # Detection results
        self.current_detections = []
        self.halt_triggered = False
        self.halt_object = None
        
        self.initialize_detector()
    
    def initialize_detector(self):
        """Initialize TensorFlow Lite object detector with version compatibility"""
        try:
            if TFLITE_SUPPORT_NEW:
                # Use newer API (0.4.x)
                base_options = core.BaseOptions(
                    file_name=self.model_path, 
                    use_coral=False, 
                    num_threads=2
                )
                detection_options = processor.DetectionOptions(
                    max_results=3,
                    score_threshold=self.score_threshold
                )
                options = vision.ObjectDetectorOptions(
                    base_options=base_options, 
                    detection_options=detection_options
                )
                self.detector = vision.ObjectDetector.create_from_options(options)
                print("Object detector initialized with newer API")
            else:
                # Try older API (0.1.x) or basic TF Lite
                try:
                    # Attempt older tflite-support
                    self.detector = vision.ObjectDetector.create_from_file(self.model_path)
                    print("Object detector initialized with older API")
                except:
                    # Fall back to basic TensorFlow Lite interpreter
                    self.initialize_basic_interpreter()
                    
        except Exception as e:
            print(f"Failed to initialize object detector: {e}")
            print("Continuing without object detection...")
            self.detector = None
            self.interpreter = None
    
    def initialize_basic_interpreter(self):
        """Initialize basic TensorFlow Lite interpreter as fallback"""
        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            print("Using basic TensorFlow Lite interpreter")
        except Exception as e:
            print(f"Failed to initialize basic interpreter: {e}")
            self.interpreter = None
    
    def detect_objects(self, image):
        """Perform object detection on input image (compatible with multiple TF Lite versions)"""
        if self.detector is None and self.interpreter is None:
            return []
        
        now = time.time()
        if (now - self.last_detection_time) < self.detection_period:
            return self.current_detections
        
        self.last_detection_time = now
        
        try:
            if self.detector is not None:
                # Use tflite-support API
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if TFLITE_SUPPORT_NEW:
                    # Newer API
                    input_tensor = vision.TensorImage.create_from_array(rgb_image)
                    detection_result = self.detector.detect(input_tensor)
                    self.current_detections = detection_result.detections
                else:
                    # Older API
                    detection_result = self.detector.detect(rgb_image)
                    self.current_detections = getattr(detection_result, 'detections', [])
                
            elif self.interpreter is not None:
                # Use basic TensorFlow Lite interpreter
                self.current_detections = self.detect_with_basic_interpreter(image)
            
            # Check for halt conditions
            self.check_halt_conditions()
            
            return self.current_detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def detect_with_basic_interpreter(self, image):
        """Basic detection using TensorFlow Lite interpreter"""
        try:
            # This is a simplified implementation
            # For full implementation, you'd need to handle model-specific preprocessing
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            # Resize image to model input size
            height, width = input_details[0]['shape'][1:3]
            resized_image = cv2.resize(image, (width, height))
            input_data = np.expand_dims(resized_image, axis=0)
            
            # Normalize if needed (model-dependent)
            if input_details[0]['dtype'] == np.float32:
                input_data = (input_data.astype(np.float32) - 127.5) / 127.5
            
            self.interpreter.set_tensor(input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # For simplicity, return empty list
            # Full implementation would parse output tensors
            print("Basic interpreter detection completed (simplified)")
            return []
            
        except Exception as e:
            print(f"Basic interpreter error: {e}")
            return []
    
    def check_halt_conditions(self):
        """Check if detected objects require halting"""
        self.halt_triggered = False
        self.halt_object = None
        
        for detection in self.current_detections:
            if detection.categories:
                category = detection.categories[0]
                name = getattr(category, "category_name", "")
                score = getattr(category, "score", 0.0)
                
                if score >= self.score_threshold and name in WATCH_CLASSES:
                    self.halt_triggered = True
                    self.halt_object = f"{name} ({score:.2f})"
                    print(f"HALT: Detected {self.halt_object}")
                    break


class UltrasonicMapper:
    """Ultrasonic sensor mapping system (sensor mounted on camera, no pan needed)"""
    
    def __init__(self, picar):
        self.picar = picar
        self.map = np.zeros((width_scaled, length_scaled))
        self.current_pos = list(start_pos)
        self.heading_angle = 0  # radians
        
    def scan_surroundings(self):
        """Scan surroundings with ultrasonic sensor (now mounted on camera, no pan needed)"""
        if not HARDWARE_AVAILABLE:
            return self.map
        
        # Single reading straight ahead since ultrasonic is now mounted on camera
        reading = self.picar.ultrasonic.read()
        reading_scaled = int(np.ceil(reading / 5.0) * 5)
        
        if reading_scaled <= MAX_READ:
            self.update_map_with_reading(0, reading_scaled)  # angle = 0 (straight ahead)
        
        return self.map
    
    def update_map_with_reading(self, angle, reading):
        """Update map with ultrasonic reading (simplified for forward-facing sensor)"""
        if reading > MAXREAD_SCALED:
            return
        
        x, y = self.current_pos
        
        # Since ultrasonic is now mounted straight ahead on the camera
        if angle == 0:
            # Straight ahead - this is the primary case now
            if y + reading < length_scaled:
                self.map[x][y + reading] = 1
        elif angle < 0:
            # Left side (keeping for compatibility if pan is added back later)
            if reading <= x / np.cos(np.radians(abs(angle))):
                object_x = int(x - reading * np.cos(np.radians(abs(angle))))
                object_y = int(y + reading * np.sin(np.radians(abs(angle))))
                if 0 <= object_x < width_scaled and 0 <= object_y < length_scaled:
                    self.map[object_x][object_y] = 1
        else:
            # Right side (keeping for compatibility if pan is added back later)
            if reading <= (width_scaled - x) / np.cos(np.radians(angle)):
                object_x = int(x + reading * np.cos(np.radians(angle)))
                object_y = int(y + reading * np.sin(np.radians(angle)))
                if 0 <= object_x < width_scaled and 0 <= object_y < length_scaled:
                    self.map[object_x][object_y] = 1
    
    def add_clearance_to_obstacles(self, clearance=1):
        """Add clearance around obstacles to prevent collision"""
        original_map = self.map.copy()
        for i in range(width_scaled):
            for j in range(length_scaled):
                if original_map[i][j] == 1:
                    for di in range(-clearance, clearance + 1):
                        for dj in range(-clearance, clearance + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < width_scaled and 0 <= nj < length_scaled:
                                self.map[ni][nj] = 1
    
    def update_position(self, velocity, dt, steer_angle):
        """Update car position based on movement"""
        d = velocity * dt
        beta = d * np.tan(np.radians(steer_angle)) / CAR_Length
        
        if np.abs(beta) < 0.001:  # straight line approximation
            dx = int(d * np.cos(self.heading_angle))
            dy = int(d * np.sin(self.heading_angle))
        else:
            R = d / beta  # radius of curvature
            dx = int(R * np.sin(self.heading_angle + beta))
            dy = int(-R * np.cos(self.heading_angle + beta))
        
        new_x = self.current_pos[0] + dx
        new_y = self.current_pos[1] + dy
        
        # Boundary check
        if 0 <= new_x < width_scaled and 0 <= new_y < length_scaled:
            self.current_pos = [new_x, new_y]
            self.heading_angle = (self.heading_angle + beta) % (2 * np.pi)


class PathPlanner:
    """A* pathfinding algorithm implementation"""
    
    @staticmethod
    def cost(state1, state2):
        """Calculate cost between two states"""
        p1 = np.array(state1[:2])
        p2 = np.array(state2[:2])
        return np.linalg.norm(p1 - p2)
    
    @staticmethod
    def heuristic(state, goal):
        """Heuristic function for A*"""
        return PathPlanner.cost(state, goal)
    
    @staticmethod
    def reconstruct_path(node):
        """Reconstruct path from goal to start"""
        path = []
        while node is not None:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))
    
    @staticmethod
    def collision_check(state, grid):
        """Check if state collides with obstacles"""
        x, y = int(state[0]), int(state[1])
        
        if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
            return True  # outside map
        
        return grid[x, y] == 1  # obstacle
    
    @staticmethod
    def find_path(start_state, goal_state, grid):
        """Find path using A* algorithm"""
        open_set = PriorityQueue()
        closed_set = set()
        g_costs = {}
        
        start_node = Coordinate(start_state, 0, PathPlanner.heuristic(start_state, goal_state))
        g_costs[start_state] = 0
        open_set.put(start_node)
        
        while not open_set.empty():
            current = open_set.get()
            
            if tuple(current.state[:2]) == tuple(goal_state[:2]):
                return PathPlanner.reconstruct_path(current)
            
            closed_set.add(tuple(current.state))
            
            # Generate neighbors (4-directional movement)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_x = current.state[0] + dx
                next_y = current.state[1] + dy
                next_state = (next_x, next_y, current.state[2])
                
                if tuple(next_state) in closed_set:
                    continue
                
                if PathPlanner.collision_check(next_state, grid):
                    continue
                
                tentative_g = current.g + 1
                
                if (tuple(next_state) not in g_costs or 
                    tentative_g < g_costs[tuple(next_state)]):
                    
                    g_costs[tuple(next_state)] = tentative_g
                    h_cost = PathPlanner.heuristic(next_state, goal_state)
                    next_node = Coordinate(next_state, tentative_g, h_cost, current)
                    open_set.put(next_node)
        
        return None  # No path found


class SelfDrivingCar:
    """Main self-driving car controller"""
    
    def __init__(self, goal_position=None):
        self.goal_position = goal_position or end_pos
        self.picar = None
        self.camera = None
        self.running = False
        
        # Initialize subsystems
        self.object_detector = ObjectDetector()
        
        if HARDWARE_AVAILABLE:
            self.picar = Picarx()  # Simplified initialization without servo pins for pan
            self.mapper = UltrasonicMapper(self.picar)
            self.initialize_camera()
        else:
            self.mapper = UltrasonicMapper(None)
        
        # Navigation state
        self.current_path = []
        self.path_index = 0
        self.last_replan_time = 0
        self.replan_interval = 3.0  # seconds
        
        # Threading
        self.detection_thread = None
        self.frame_queue = Queue(maxsize=2)
        
    def initialize_camera(self):
        """Initialize Picamera2 for object detection"""
        if not HARDWARE_AVAILABLE:
            return
        
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(2)  # Camera warm-up
            print("Camera initialized successfully")
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            self.camera = None
    
    def capture_frame(self):
        """Capture frame from camera"""
        if self.camera is None:
            # Return dummy frame for testing
            return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        
        try:
            frame = self.camera.capture_array()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Frame capture error: {e}")
            return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    
    def detection_worker(self):
        """Worker thread for object detection"""
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    detections = self.object_detector.detect_objects(frame)
                    # Processing happens in detect_objects method
                time.sleep(0.1)  # Small delay to prevent busy waiting
            except Exception as e:
                print(f"Detection worker error: {e}")
    
    def plan_path(self):
        """Plan path to goal using A*"""
        current_state = (*self.mapper.current_pos, self.mapper.heading_angle)
        goal_state = (*self.goal_position, 0)
        
        print(f"Planning path from {current_state[:2]} to {goal_state[:2]}")
        
        path = PathPlanner.find_path(current_state, goal_state, self.mapper.map)
        
        if path:
            self.current_path = path
            self.path_index = 0
            print(f"Path found with {len(path)} waypoints")
            return True
        else:
            print("No path found to goal")
            return False
    
    def execute_movement(self, target_pos):
        """Execute movement towards target position"""
        if not HARDWARE_AVAILABLE:
            # Simulate movement
            self.mapper.current_pos = list(target_pos[:2])
            return True
        
        current_x, current_y = self.mapper.current_pos
        target_x, target_y = target_pos[:2]
        
        # Calculate movement direction
        dx = target_x - current_x
        dy = target_y - current_y
        
        # Simple movement logic
        if abs(dx) > abs(dy):
            # Move horizontally
            if dx > 0:
                steer_angle = 30  # Turn right
            else:
                steer_angle = -30  # Turn left
        else:
            # Move forward/backward
            steer_angle = 0
        
        try:
            self.picar.set_dir_servo_angle(steer_angle)
            if dy > 0:
                self.picar.forward(POWER)
            else:
                self.picar.backward(POWER)
            
            time.sleep(delta_t)
            self.picar.forward(0)  # Stop
            
            # Update position
            self.mapper.update_position(SPEED, delta_t, steer_angle)
            return True
            
        except Exception as e:
            print(f"Movement error: {e}")
            return False
    
    def navigate_to_goal(self):
        """Main navigation loop"""
        print(f"Starting navigation to goal: {self.goal_position}")
        
        while self.running:
            try:
                # Capture and queue frame for detection
                frame = self.capture_frame()
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                # Check for halt condition
                if self.object_detector.halt_triggered:
                    print(f"Halting due to detection: {self.object_detector.halt_object}")
                    if HARDWARE_AVAILABLE:
                        self.picar.forward(0)  # Stop
                    time.sleep(2.0)  # Wait 2 seconds
                    continue
                
                # Check if we've reached the goal
                current_pos = self.mapper.current_pos
                goal_distance = np.linalg.norm(np.array(current_pos) - np.array(self.goal_position))
                if goal_distance < 2:  # Within 2 grid cells
                    print("Goal reached!")
                    break
                
                # Periodic replanning
                now = time.time()
                if (now - self.last_replan_time) > self.replan_interval:
                    print("Rescanning environment and replanning...")
                    # Note: ultrasonic sensor now only scans straight ahead (mounted on camera)
                    self.mapper.scan_surroundings()
                    self.mapper.add_clearance_to_obstacles()
                    
                    if self.plan_path():
                        self.last_replan_time = now
                    else:
                        print("Unable to find path, waiting...")
                        time.sleep(1.0)
                        continue
                
                # Execute next movement
                if self.path_index < len(self.current_path):
                    next_waypoint = self.current_path[self.path_index]
                    print(f"Moving to waypoint {self.path_index}: {next_waypoint[:2]}")
                    
                    if self.execute_movement(next_waypoint):
                        self.path_index += 1
                    else:
                        print("Movement failed, replanning...")
                        self.last_replan_time = 0  # Force replan
                
                time.sleep(0.1)  # Small delay
                
            except KeyboardInterrupt:
                print("Navigation interrupted by user")
                break
            except Exception as e:
                print(f"Navigation error: {e}")
                time.sleep(1.0)
        
        self.stop()
    
    def start(self):
        """Start the self-driving car system"""
        print("Starting self-driving car system...")
        self.running = True
        
        # Start detection thread
        if HARDWARE_AVAILABLE:
            self.detection_thread = threading.Thread(target=self.detection_worker)
            self.detection_thread.daemon = True
            self.detection_thread.start()
        
        # Initial mapping and planning
        print("Performing initial environment scan (forward-facing ultrasonic)...")
        self.mapper.scan_surroundings()
        self.mapper.add_clearance_to_obstacles()
        
        if self.plan_path():
            self.navigate_to_goal()
        else:
            print("Failed to find initial path to goal")
    
    def stop(self):
        """Stop the self-driving car system"""
        print("Stopping self-driving car system...")
        self.running = False
        
        if HARDWARE_AVAILABLE and self.picar:
            self.picar.forward(0)  # Stop motors
        
        if self.camera:
            self.camera.stop()
        
        print("System stopped successfully")


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Self-Driving Car with Object Detection')
    parser.add_argument('--goal-x', type=int, default=end_pos[0], 
                        help='Goal X coordinate (grid cells)')
    parser.add_argument('--goal-y', type=int, default=end_pos[1], 
                        help='Goal Y coordinate (grid cells)')
    parser.add_argument('--model', type=str, default='efficientdet_lite0.tflite',
                        help='TensorFlow Lite model path')
    parser.add_argument('--confidence', type=float, default=DETECTION_CONFIDENCE,
                        help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    goal_position = (args.goal_x, args.goal_y)
    
    print("="*50)
    print("CS 437 Lab 1 Part 2 Step 7: Object Detection")
    print("="*50)
    print(f"Hardware available: {HARDWARE_AVAILABLE}")
    print(f"Goal position: {goal_position}")
    print(f"Detection model: {args.model}")
    print(f"Confidence threshold: {args.confidence}")
    print("="*50)
    
    try:
        # Create and start self-driving car
        car = SelfDrivingCar(goal_position)
        car.object_detector.model_path = args.model
        car.object_detector.score_threshold = args.confidence
        car.object_detector.initialize_detector()
        
        car.start()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Program error: {e}")
    finally:
        print("Program ended")


if __name__ == "__main__":
    main()

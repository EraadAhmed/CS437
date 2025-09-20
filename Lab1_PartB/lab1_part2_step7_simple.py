#!/usr/bin/env python3
"""
Simplified Step 7 implementation for Raspberry Pi with limited TensorFlow Lite support
This version works with the available tflite-support 0.1.0a1 on your Pi
"""

import numpy as np
import time
import threading
import argparse
from queue import Queue, PriorityQueue

# Try to import what's available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available")

# TensorFlow Lite with graceful degradation
TFLITE_AVAILABLE = False
try:
    from tflite_support.task import vision
    TFLITE_AVAILABLE = True
    print("TensorFlow Lite Support available")
except ImportError:
    try:
        import tensorflow as tf
        TFLITE_AVAILABLE = True
        print("Basic TensorFlow available")
    except ImportError:
        print("Warning: No TensorFlow support - object detection disabled")

# Hardware imports (conditional)
try:
    from picamera2 import Picamera2
    from picarx import Picarx
    HARDWARE_AVAILABLE = True
    print("Hardware libraries available")
except ImportError:
    print("Warning: Hardware libraries not available. Running in simulation mode.")
    HARDWARE_AVAILABLE = False

# Constants
WIDTH = 120
LENGTH = 380
CAR_Width = 14
CAR_Length = 23

width_scaled = int(WIDTH / 5)
length_scaled = int(LENGTH / 5)
x_mid_scaled = int(width_scaled / 2)

MAX_READ = 100
SPEED = 10
POWER = 40
delta_t = 0.25

start_pos = (x_mid_scaled - 1, 0)
end_pos = (x_mid_scaled - 1, length_scaled - 1)

WATCH_CLASSES = {"stop sign", "person"}
DETECTION_CONFIDENCE = 0.5
FRAME_WIDTH = 320
FRAME_HEIGHT = 240


class SimpleObjectDetector:
    """Simplified object detector that works with available TF Lite versions"""
    
    def __init__(self, model_path='efficientdet_lite0.tflite'):
        self.model_path = model_path
        self.detector = None
        self.halt_triggered = False
        self.halt_object = None
        self.last_detection_time = 0
        self.detection_period = 1.0
        
        if TFLITE_AVAILABLE:
            self.initialize_detector()
    
    def initialize_detector(self):
        """Initialize with available TF Lite version"""
        if not TFLITE_AVAILABLE:
            return
            
        try:
            # Try the 0.1.x API first
            self.detector = vision.ObjectDetector.create_from_file(self.model_path)
            print("Object detector initialized with 0.1.x API")
        except Exception as e:
            print(f"Could not initialize object detector: {e}")
            print("Continuing without object detection...")
    
    def detect_objects(self, image):
        """Simple object detection"""
        if not TFLITE_AVAILABLE or self.detector is None or not CV2_AVAILABLE:
            return []
        
        now = time.time()
        if (now - self.last_detection_time) < self.detection_period:
            return []
        
        self.last_detection_time = now
        
        try:
            # Convert image format if needed
            if image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Run detection with 0.1.x API
            detection_result = self.detector.detect(rgb_image)
            
            # Check for halt conditions
            self.check_halt_conditions(detection_result)
            
            return getattr(detection_result, 'detections', [])
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def check_halt_conditions(self, detection_result):
        """Check for objects that require halting"""
        self.halt_triggered = False
        self.halt_object = None
        
        detections = getattr(detection_result, 'detections', [])
        
        for detection in detections:
            if hasattr(detection, 'categories') and detection.categories:
                category = detection.categories[0]
                name = getattr(category, 'category_name', '')
                score = getattr(category, 'score', 0.0)
                
                if score >= DETECTION_CONFIDENCE and name in WATCH_CLASSES:
                    self.halt_triggered = True
                    self.halt_object = f"{name} ({score:.2f})"
                    print(f"HALT: Detected {self.halt_object}")
                    break


class UltrasonicMapper:
    """Simplified ultrasonic mapping (forward-facing only)"""
    
    def __init__(self, picar):
        self.picar = picar
        self.map = np.zeros((width_scaled, length_scaled))
        self.current_pos = list(start_pos)
        self.heading_angle = 0
    
    def scan_surroundings(self):
        """Single forward scan"""
        if not HARDWARE_AVAILABLE or self.picar is None:
            return self.map
        
        try:
            reading = self.picar.ultrasonic.read()
            reading_scaled = int(np.ceil(reading / 5.0) * 5)
            
            if reading_scaled <= MAX_READ:
                self.update_map_with_reading(reading_scaled)
        except Exception as e:
            print(f"Ultrasonic error: {e}")
        
        return self.map
    
    def update_map_with_reading(self, reading):
        """Update map with forward reading"""
        max_read_scaled = int(MAX_READ / 5)
        if reading > max_read_scaled:
            return
        
        x, y = self.current_pos
        obstacle_y = y + int(reading / 5)
        
        if 0 <= x < width_scaled and 0 <= obstacle_y < length_scaled:
            self.map[x][obstacle_y] = 1
    
    def add_clearance_to_obstacles(self, clearance=1):
        """Add safety clearance around obstacles"""
        original_map = self.map.copy()
        for i in range(width_scaled):
            for j in range(length_scaled):
                if original_map[i][j] == 1:
                    for di in range(-clearance, clearance + 1):
                        for dj in range(-clearance, clearance + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < width_scaled and 0 <= nj < length_scaled:
                                self.map[ni][nj] = 1


class SimplifiedCar:
    """Simplified self-driving car for Raspberry Pi"""
    
    def __init__(self, goal_position=None):
        self.goal_position = goal_position or end_pos
        self.picar = None
        self.camera = None
        self.running = False
        
        # Initialize components
        self.object_detector = SimpleObjectDetector()
        
        if HARDWARE_AVAILABLE:
            try:
                self.picar = Picarx()
                self.mapper = UltrasonicMapper(self.picar)
                self.initialize_camera()
            except Exception as e:
                print(f"Hardware initialization error: {e}")
                HARDWARE_AVAILABLE = False
                self.mapper = UltrasonicMapper(None)
        else:
            self.mapper = UltrasonicMapper(None)
    
    def initialize_camera(self):
        """Initialize camera for object detection"""
        if not HARDWARE_AVAILABLE:
            return
        
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(2)
            print("Camera initialized")
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            self.camera = None
    
    def capture_frame(self):
        """Capture frame from camera"""
        if self.camera is None or not CV2_AVAILABLE:
            return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        
        try:
            frame = self.camera.capture_array()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Frame capture error: {e}")
            return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    
    def basic_navigation(self):
        """Basic navigation loop"""
        print(f"Starting basic navigation to goal: {self.goal_position}")
        
        step_count = 0
        max_steps = 100  # Prevent infinite loop
        
        while self.running and step_count < max_steps:
            try:
                step_count += 1
                
                # Capture frame for object detection
                frame = self.capture_frame()
                detections = self.object_detector.detect_objects(frame)
                
                # Check halt condition
                if self.object_detector.halt_triggered:
                    print(f"HALT: {self.object_detector.halt_object}")
                    if HARDWARE_AVAILABLE and self.picar:
                        self.picar.forward(0)
                    time.sleep(2.0)
                    continue
                
                # Check if at goal
                current_pos = self.mapper.current_pos
                goal_distance = np.linalg.norm(
                    np.array(current_pos) - np.array(self.goal_position)
                )
                
                if goal_distance < 2:
                    print("Goal reached!")
                    break
                
                # Simple forward movement
                print(f"Step {step_count}: Moving forward...")
                
                # Scan for obstacles
                self.mapper.scan_surroundings()
                
                # Simple obstacle avoidance
                if HARDWARE_AVAILABLE and self.picar:
                    try:
                        distance = self.picar.ultrasonic.read()
                        
                        if distance > 25:  # Safe distance
                            self.picar.set_dir_servo_angle(0)
                            self.picar.forward(POWER)
                            time.sleep(delta_t)
                            self.picar.forward(0)
                        elif distance > 10:  # Caution
                            self.picar.set_dir_servo_angle(30)
                            self.picar.forward(POWER//2)
                            time.sleep(delta_t)
                            self.picar.forward(0)
                        else:  # Too close
                            self.picar.set_dir_servo_angle(-30)
                            self.picar.backward(POWER//2)
                            time.sleep(delta_t)
                            self.picar.forward(0)
                        
                        # Update position (simplified)
                        if distance > 10:
                            self.mapper.current_pos[1] += 1
                            
                    except Exception as e:
                        print(f"Movement error: {e}")
                        time.sleep(1.0)
                else:
                    # Simulation mode
                    print("Simulation: Moving forward")
                    self.mapper.current_pos[1] += 1
                    time.sleep(0.5)
                
            except KeyboardInterrupt:
                print("Navigation interrupted")
                break
            except Exception as e:
                print(f"Navigation error: {e}")
                time.sleep(1.0)
        
        self.stop()
    
    def start(self):
        """Start the simplified car system"""
        print("Starting simplified car system...")
        self.running = True
        self.basic_navigation()
    
    def stop(self):
        """Stop the car system"""
        print("Stopping car system...")
        self.running = False
        
        if HARDWARE_AVAILABLE and self.picar:
            self.picar.forward(0)
        
        if self.camera:
            self.camera.stop()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simplified Self-Driving Car for Pi')
    parser.add_argument('--goal-x', type=int, default=end_pos[0], help='Goal X coordinate')
    parser.add_argument('--goal-y', type=int, default=end_pos[1], help='Goal Y coordinate')
    parser.add_argument('--model', type=str, default='efficientdet_lite0.tflite', help='Model path')
    
    args = parser.parse_args()
    goal_position = (args.goal_x, args.goal_y)
    
    print("="*50)
    print("CS 437 Lab 1 Step 7: Simplified Implementation")
    print("="*50)
    print(f"Hardware available: {HARDWARE_AVAILABLE}")
    print(f"TensorFlow Lite available: {TFLITE_AVAILABLE}")
    print(f"OpenCV available: {CV2_AVAILABLE}")
    print(f"Goal position: {goal_position}")
    print("="*50)
    
    try:
        car = SimplifiedCar(goal_position)
        car.object_detector.model_path = args.model
        car.start()
    except KeyboardInterrupt:
        print("\nProgram interrupted")
    except Exception as e:
        print(f"Program error: {e}")
    finally:
        print("Program ended")


if __name__ == "__main__":
    main()
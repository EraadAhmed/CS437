# fixed_object_detection.py
# CS 437 Lab 1: Improved Object Detection System

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from threading import Thread, Lock
import time
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Default halt objects - adjust based on your labelmap.txt
HALT_OBJECT_IDS = [0, 12]  # person=0, stop sign=12 (adjust based on your model)

class ObjectDetector:
    """
    Robust TensorFlow Lite object detection with improved reliability
    and better error handling for IoT hardware constraints.
    """
    
    def __init__(self, model_path='efficientdet_lite0.tflite', 
                 confidence_threshold=0.4, max_detections=10):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        
        # Load model with error handling
        try:
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            logger.info("TensorFlow Lite model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TensorFlow model: {e}")
            self.interpreter = None
            return
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        
        # Camera setup with better error handling
        self.cap = None
        self._init_camera()
        
        # Threading and state management
        self._lock = Lock()
        self._is_halt_needed = False
        self._last_detection_time = 0
        self._detection_stale_timeout = 3.0  # seconds
        self._is_running = False
        self._thread = None
        self._frame_count = 0
        self._detection_count = 0
        
    def _init_camera(self):
        """Initialize camera with multiple fallback attempts."""
        camera_configs = [
            {'index': 0, 'width': 640, 'height': 480},
            {'index': 0, 'width': 320, 'height': 240},  # Lower resolution fallback
            {'index': 1, 'width': 640, 'height': 480},  # Try different camera
        ]
        
        for config in camera_configs:
            try:
                self.cap = cv2.VideoCapture(config['index'])
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
                    self.cap.set(cv2.CAP_PROP_FPS, 10)  # Limit FPS for stability
                    
                    # Test capture
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        logger.info(f"Camera initialized: {config['width']}x{config['height']}")
                        return
                    else:
                        self.cap.release()
                        self.cap = None
            except Exception as e:
                logger.warning(f"Camera config {config} failed: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
        
        logger.error("All camera initialization attempts failed")

    def _detection_thread(self):
        """Main detection loop with improved error handling."""
        consecutive_failures = 0
        max_failures = 10
        
        while self._is_running:
            try:
                # Check if camera and model are available
                if not self.cap or not self.interpreter:
                    time.sleep(0.5)
                    continue
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.error("Too many camera failures - reinitializing")
                        self._init_camera()
                        consecutive_failures = 0
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0  # Reset on successful capture
                self._frame_count += 1
                
                # Skip processing every other frame for performance
                if self._frame_count % 2 == 0:
                    time.sleep(0.1)
                    continue
                
                # Preprocess image
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_resized = cv2.resize(image_rgb, (self.input_width, self.input_height))
                input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)
                
                # Run inference
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                
                # Get detection results
                scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                classes = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
                
                # Check for halt objects
                halt_detected = False
                detection_info = []
                
                for i in range(min(len(scores), self.max_detections)):
                    if scores[i] > self.confidence_threshold:
                        class_id = int(classes[i])
                        detection_info.append({
                            'class_id': class_id,
                            'confidence': float(scores[i])
                        })
                        
                        if class_id in HALT_OBJECT_IDS:
                            halt_detected = True
                            logger.info(f"HALT OBJECT DETECTED: class_id={class_id}, confidence={scores[i]:.2f}")
                
                # Update shared state
                with self._lock:
                    self._is_halt_needed = halt_detected
                    self._last_detection_time = time.time()
                    if halt_detected:
                        self._detection_count += 1
                
                # Control frame rate
                time.sleep(0.2)  # ~5 FPS for object detection
                
            except Exception as e:
                logger.error(f"Detection thread error: {e}")
                consecutive_failures += 1
                time.sleep(0.1)

    def start_detection(self):
        """Start background detection thread."""
        if self._is_running or not self.cap or not self.interpreter:
            logger.warning("Cannot start detection - missing camera or model")
            return False
            
        self._is_running = True
        self._thread = Thread(target=self._detection_thread, daemon=True)
        self._thread.start()
        logger.info("Object detection started")
        return True

    def stop_detection(self):
        """Stop detection thread and cleanup."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Object detection stopped")

    def is_halt_needed(self):
        """
        Thread-safe check for halt requirement.
        Returns False if detections are stale.
        """
        with self._lock:
            current_time = time.time()
            
            # Consider detections stale if too old
            if current_time - self._last_detection_time > self._detection_stale_timeout:
                return False
                
            return self._is_halt_needed

    def get_detection_stats(self):
        """Get detection statistics for debugging."""
        with self._lock:
            return {
                'frame_count': self._frame_count,
                'detection_count': self._detection_count,
                'last_detection_age': time.time() - self._last_detection_time,
                'is_running': self._is_running,
                'camera_available': self.cap is not None,
                'model_available': self.interpreter is not None
            }

    def test_detection(self):
        """Test object detection functionality."""
        logger.info("Testing object detection...")
        
        if not self.cap or not self.interpreter:
            logger.error("Camera or model not available for testing")
            return False
        
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                logger.info(f"Camera test successful - frame shape: {frame.shape}")
                return True
            else:
                logger.error("Camera test failed - no frame captured")
                return False
        except Exception as e:
            logger.error(f"Detection test failed: {e}")
            return False


# Test function for standalone usage
def test_object_detector():
    """Test the object detector independently."""
    logging.basicConfig(level=logging.INFO)
    
    detector = ObjectDetector()
    
    if not detector.test_detection():
        print("Object detector test failed!")
        return
    
    print("Starting detection test for 10 seconds...")
    detector.start_detection()
    
    try:
        for i in range(20):  # Test for 10 seconds (0.5s intervals)
            time.sleep(0.5)
            halt_needed = detector.is_halt_needed()
            stats = detector.get_detection_stats()
            
            print(f"Test {i+1}/20: Halt={halt_needed}, "
                  f"Frames={stats['frame_count']}, "
                  f"Detections={stats['detection_count']}")
            
    except KeyboardInterrupt:
        print("Test interrupted")
    finally:
        detector.stop_detection()
        print("Test completed")


if __name__ == "__main__":
    test_object_detector()
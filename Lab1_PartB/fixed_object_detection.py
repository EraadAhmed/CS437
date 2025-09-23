# fixed_object_detection.py
# CS 437 Lab 1: Improved Object Detection System

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from threading import Thread, Lock
import time
import logging

# Import vilib for better camera support on PiCar
try:
    import vilib
    VILIB_AVAILABLE = True
    print("vilib imported successfully")
except ImportError:
    VILIB_AVAILABLE = False
    print("vilib not available, falling back to OpenCV")

# Configure logging
logger = logging.getLogger(__name__)

# Default halt objects - adjust based on your labelmap.txt
# Note: person=0 can cause false positives if user is in view
HALT_OBJECT_IDS = [12]  # Only stop sign=12, temporarily removing person=0 to avoid false positives

class ObjectDetector:
    """
    Robust TensorFlow Lite object detection with improved reliability
    and better error handling for IoT hardware constraints.
    """
    
    def __init__(self, model_path='efficientdet_lite0.tflite', 
                 confidence_threshold=0.8, max_detections=10):
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
        self.vilib_instance = None
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
        
        # False positive reduction
        self._consecutive_detections = 0
        self._min_consecutive_detections = 3  # Require 3 consecutive detections
        self._recent_detections = []  # Track recent detection confidence scores
        
    def _init_camera(self):
        """Initialize camera using vilib for better Raspberry Pi compatibility."""
        logger.info("Initializing camera with vilib...")
        
        if not VILIB_AVAILABLE:
            logger.error("vilib not available, cannot initialize camera")
            self.cap = None
            return
        
        try:
            # Create vilib instance and start camera
            self.vilib_instance = vilib.Vilib()
            self.vilib_instance.camera_start(vflip=False, hflip=False)
            time.sleep(2)  # Give camera time to initialize
            
            # Test frame capture
            for attempt in range(10):
                frame = self.vilib_instance.img
                if frame is not None and hasattr(frame, 'size') and frame.size > 0:
                    logger.info(f"vilib camera initialized successfully: frame shape {frame.shape}")
                    self.cap = 'vilib'  # Mark that we're using vilib
                    return
                time.sleep(0.5)
            
            logger.error("vilib camera test failed - no frames captured")
            self.vilib_instance.camera_close()
            self.cap = None
            
        except Exception as e:
            logger.error(f"vilib camera initialization failed: {e}")
            self.cap = None

    def _detection_thread(self):
        """Main detection loop with improved error handling."""
        consecutive_failures = 0
        max_failures = 20  # Increase tolerance before reinitializing
        
        while self._is_running:
            try:
                # Check if camera and model are available
                if not self.cap or not self.interpreter:
                    time.sleep(0.5)
                    continue
                
                # Capture frame using vilib or OpenCV
                if self.cap == 'vilib' and self.vilib_instance:
                    frame = self.vilib_instance.img
                    ret = frame is not None and hasattr(frame, 'size') and frame.size > 0
                else:
                    # Fallback to OpenCV (shouldn't happen with current setup)
                    ret, frame = None, None
                    for attempt in range(3):
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            break
                        time.sleep(0.05)
                
                if not ret or frame is None or frame.size == 0:
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
                if self._frame_count % 4 == 0:  # Process only every 4th frame
                    time.sleep(0.1)
                    continue
                
                # Preprocess image
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_resized = cv2.resize(image_rgb, (self.input_width, self.input_height))
                input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)
                
                # Run inference
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                
                # Get detection results from EfficientDet model
                # Output 0: boxes [1, 25, 4]
                # Output 1: scores [1, 25] 
                # Output 2: classes [1, 25]
                # Output 3: num_detections [1]
                
                boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                scores = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
                classes = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
                num_detections = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])
                
                # Ensure arrays are properly shaped
                scores = np.array(scores).flatten()
                classes = np.array(classes).flatten()
                
                # Debug: log tensor info occasionally
                if self._frame_count % 100 == 0:
                    logger.debug(f"Detections: {num_detections}, scores shape: {scores.shape}, classes shape: {classes.shape}")
                
                # Check for halt objects with false positive reduction
                halt_detected = False
                detection_info = []
                current_frame_detections = []
                
                # Use actual number of detections, limited by tensor size
                max_to_check = min(num_detections, len(scores), len(classes), self.max_detections)
                
                for i in range(max_to_check):
                    # Extract scalar values to avoid array comparison ambiguity
                    score_val = float(scores[i])
                    class_val = int(classes[i])
                    
                    if score_val > self.confidence_threshold:
                        detection_info.append({
                            'class_id': class_val,
                            'confidence': score_val
                        })
                        
                        if class_val in HALT_OBJECT_IDS:
                            current_frame_detections.append({
                                'class_id': class_val,
                                'confidence': score_val
                            })
                
                # Implement consecutive detection logic to reduce false positives
                if current_frame_detections:
                    self._consecutive_detections += 1
                    self._recent_detections.append(current_frame_detections)
                    
                    # Keep only recent detections (last 5 frames)
                    if len(self._recent_detections) > 5:
                        self._recent_detections.pop(0)
                    
                    # Only trigger halt if we have enough consecutive detections
                    if self._consecutive_detections >= self._min_consecutive_detections:
                        halt_detected = True
                        # Log only when we first detect or every 10th consecutive detection
                        if self._consecutive_detections == self._min_consecutive_detections or self._consecutive_detections % 10 == 0:
                            for det in current_frame_detections:
                                logger.info(f"PERSISTENT HALT OBJECT: class_id={det['class_id']}, confidence={det['confidence']:.2f}, consecutive={self._consecutive_detections}")
                else:
                    # Reset consecutive count if no detections
                    self._consecutive_detections = 0
                    self._recent_detections.clear()
                
                # Update shared state
                with self._lock:
                    self._is_halt_needed = halt_detected
                    self._last_detection_time = time.time()
                    if halt_detected:
                        self._detection_count += 1
                
                # Control frame rate
                time.sleep(0.5)  # ~2 FPS for object detection to reduce load
                
            except Exception as e:
                logger.error(f"Detection thread error: {e}")
                consecutive_failures += 1
                time.sleep(0.1)

    def start_detection(self):
        """Start background detection thread."""
        if self._is_running:
            logger.warning("Detection already running")
            return True
            
        if not self.interpreter:
            logger.error("Cannot start detection - TensorFlow model not loaded")
            return False
            
        if not self.cap:
            logger.warning("Camera not available - attempting to reinitialize")
            self._init_camera()
            if not self.cap:
                logger.error("Cannot start detection - camera initialization failed")
                return False
            
        self._is_running = True
        self._thread = Thread(target=self._detection_thread, daemon=True)
        self._thread.start()
        logger.info("Object detection started successfully")
        return True

    def stop_detection(self):
        """Stop detection thread and cleanup."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        if self.cap == 'vilib' and self.vilib_instance:
            try:
                self.vilib_instance.camera_close()
            except Exception as e:
                logger.warning(f"Error closing vilib camera: {e}")
        elif self.cap and hasattr(self.cap, 'release'):
            self.cap.release()
        
        self.cap = None
        self.vilib_instance = None
        logger.info("Object detection stopped")

    def is_halt_needed(self):
        """
        Thread-safe check for halt requirement.
        Returns False if detections are stale or detection is not running.
        """
        if not self._is_running:
            return False
            
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
                'consecutive_detections': self._consecutive_detections,
                'last_detection_age': time.time() - self._last_detection_time,
                'is_running': self._is_running,
                'camera_available': self.cap is not None,
                'model_available': self.interpreter is not None,
                'min_consecutive_required': self._min_consecutive_detections
            }

    def test_detection(self):
        """Test object detection functionality."""
        logger.info("Testing object detection...")
        
        if not self.cap or not self.interpreter:
            logger.error("Camera or model not available for testing")
            return False
        
        try:
            if self.cap == 'vilib' and self.vilib_instance:
                frame = self.vilib_instance.img
                ret = frame is not None and hasattr(frame, 'size') and frame.size > 0
            else:
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
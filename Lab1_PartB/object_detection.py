# object_detection.py
# CS 437 Lab 1 Step 7: TensorFlow Lite Object Detection
# Integrates with existing ultrasonic mapping and car control system

import cv2
import numpy as np
import time
import asyncio
from threading import Thread, Event
from queue import Queue, Empty
import logging

# TensorFlow Lite imports with fallback
try:
    from tflite_support.task import core
    from tflite_support.task import processor
    from tflite_support.task import vision
    TFLITE_SUPPORT_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        TFLITE_SUPPORT_AVAILABLE = False
    except ImportError:
        print("ERROR: Neither tflite-support nor tensorflow available!")
        print("Please run: pip3 install tflite-support --break-system-packages")
        exit(1)

from picamera2 import Picamera2

class ObjectDetector:
    """
    TensorFlow Lite object detection for CS 437 IoT self-driving car.
    Designed to work with constrained Raspberry Pi resources and integrate
    with ultrasonic mapping system.
    """
    
    def __init__(self, model_path='efficientdet_lite0.tflite', 
                 confidence_threshold=0.3, max_results=5,
                 num_threads=2, enable_edge_tpu=False):
        """
        Initialize the object detector with optimized settings for Raspberry Pi.
        
        Args:
            model_path: Path to TFLite model file
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            max_results: Maximum number of detections per frame
            num_threads: Number of CPU threads (keep low for Pi)
            enable_edge_tpu: Whether to use Coral EdgeTPU if available
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.max_results = max_results
        self.num_threads = num_threads
        self.enable_edge_tpu = enable_edge_tpu
        
        # Detection state
        self.detector = None
        self.camera = None
        self.detection_active = Event()
        self.halt_detected = Event()
        self.detection_queue = Queue(maxsize=3)  # Small queue to prevent memory buildup
        
        # Safety-critical object classes (COCO dataset indices)
        self.safety_classes = {
            'person': 0,
            'stop_sign': 11,
            'traffic_light': 9,
            'car': 2,
            'truck': 7,
            'bus': 5
        }
        
        # Performance metrics
        self.fps = 0.0
        self.inference_time = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Initialize detection system
        self._initialize_detector()
        self._initialize_camera()
        
        logging.info("Object detector initialized successfully")

    def _initialize_detector(self):
        """Initialize TensorFlow Lite detector with fallback options."""
        if TFLITE_SUPPORT_AVAILABLE:
            self._init_tflite_support()
        else:
            self._init_basic_tensorflow()

    def _init_tflite_support(self):
        """Initialize using tflite-support (preferred method)."""
        try:
            base_options = core.BaseOptions(
                file_name=self.model_path,
                use_coral=self.enable_edge_tpu,
                num_threads=self.num_threads
            )
            detection_options = processor.DetectionOptions(
                max_results=self.max_results,
                score_threshold=self.confidence_threshold
            )
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                detection_options=detection_options
            )
            self.detector = vision.ObjectDetector.create_from_options(options)
            self.use_tflite_support = True
            logging.info("Using tflite-support for object detection")
        except Exception as e:
            logging.error(f"Failed to initialize tflite-support: {e}")
            self._init_basic_tensorflow()

    def _init_basic_tensorflow(self):
        """Fallback initialization using basic TensorFlow Lite."""
        try:
            self.interpreter = tf.lite.Interpreter(
                model_path=self.model_path,
                num_threads=self.num_threads
            )
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.use_tflite_support = False
            
            logging.info("Using basic TensorFlow Lite for object detection")
        except Exception as e:
            logging.error(f"Failed to initialize TensorFlow Lite: {e}")
            raise

    def _initialize_camera(self):
        """Initialize Picamera2 with optimized settings for object detection."""
        try:
            self.camera = Picamera2()
            
            # Configure for 320x320 input (matches most TFLite models)
            config = self.camera.create_preview_configuration(
                main={"size": (320, 320), "format": "RGB888"},
                lores={"size": (160, 160), "format": "YUV420"}  # For ISP processing
            )
            self.camera.configure(config)
            
            # Set controls for better detection performance
            controls = {
                "AwbEnable": True,       # Auto white balance
                "AeEnable": True,        # Auto exposure
                "NoiseReductionMode": 1,  # Minimal noise reduction for speed
                "Sharpness": 1.0,        # Slight sharpening
                "ExposureTime": 10000,   # 10ms exposure (100fps theoretical)
                "AnalogueGain": 1.0      # Low gain for less noise
            }
            self.camera.set_controls(controls)
            
            logging.info("Camera initialized for object detection")
        except Exception as e:
            logging.error(f"Failed to initialize camera: {e}")
            raise

    def start_detection(self):
        """Start the object detection pipeline."""
        if not self.camera:
            raise RuntimeError("Camera not initialized")
        
        self.detection_active.set()
        self.camera.start()
        
        # Start detection thread
        self.detection_thread = Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        logging.info("Object detection started")

    def stop_detection(self):
        """Stop the object detection pipeline."""
        self.detection_active.clear()
        if self.camera:
            self.camera.stop()
        
        # Clear detection queue
        while not self.detection_queue.empty():
            try:
                self.detection_queue.get_nowait()
            except Empty:
                break
        
        logging.info("Object detection stopped")

    def _detection_loop(self):
        """Main detection loop running in separate thread."""
        while self.detection_active.is_set():
            try:
                start_time = time.time()
                
                # Capture frame from camera
                frame = self.camera.capture_array()
                
                # Run detection
                if self.use_tflite_support:
                    detections = self._detect_with_support(frame)
                else:
                    detections = self._detect_with_basic_tf(frame)
                
                # Check for safety-critical objects
                halt_needed = self._check_safety_objects(detections)
                
                if halt_needed:
                    self.halt_detected.set()
                    logging.warning("HALT: Safety-critical object detected!")
                else:
                    self.halt_detected.clear()
                
                # Add to detection queue (non-blocking)
                try:
                    detection_result = {
                        'detections': detections,
                        'halt_needed': halt_needed,
                        'timestamp': time.time(),
                        'frame': frame
                    }
                    self.detection_queue.put_nowait(detection_result)
                except:
                    pass  # Queue full, skip this frame
                
                # Update performance metrics
                self.inference_time = time.time() - start_time
                self._update_fps()
                
                # Control detection rate (~1 FPS as specified)
                time.sleep(max(0, 1.0 - self.inference_time))
                
            except Exception as e:
                logging.error(f"Detection loop error: {e}")
                time.sleep(0.1)

    def _detect_with_support(self, frame):
        """Run detection using tflite-support."""
        # Convert numpy array to TensorImage
        tensor_image = vision.TensorImage.create_from_array(frame)
        
        # Run detection
        detection_result = self.detector.detect(tensor_image)
        
        return self._parse_support_results(detection_result)

    def _detect_with_basic_tf(self, frame):
        """Run detection using basic TensorFlow Lite."""
        # Preprocess frame
        input_data = np.expand_dims(frame.astype(np.uint8), axis=0)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensors
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        return self._parse_basic_results(boxes, classes, scores)

    def _parse_support_results(self, detection_result):
        """Parse results from tflite-support detector."""
        detections = []
        for detection in detection_result.detections:
            for classification in detection.classes:
                if classification.score >= self.confidence_threshold:
                    detections.append({
                        'class_id': classification.index,
                        'class_name': classification.class_name,
                        'confidence': classification.score,
                        'bbox': {
                            'x': detection.bounding_box.origin_x,
                            'y': detection.bounding_box.origin_y,
                            'width': detection.bounding_box.width,
                            'height': detection.bounding_box.height
                        }
                    })
        return detections

    def _parse_basic_results(self, boxes, classes, scores):
        """Parse results from basic TensorFlow Lite."""
        detections = []
        for i in range(len(scores)):
            if scores[i] >= self.confidence_threshold:
                y1, x1, y2, x2 = boxes[i]
                detections.append({
                    'class_id': int(classes[i]),
                    'class_name': self._get_class_name(int(classes[i])),
                    'confidence': float(scores[i]),
                    'bbox': {
                        'x': int(x1 * 320),  # Scale to image size
                        'y': int(y1 * 320),
                        'width': int((x2 - x1) * 320),
                        'height': int((y2 - y1) * 320)
                    }
                })
        return detections

    def _get_class_name(self, class_id):
        """Get class name from COCO dataset class ID."""
        coco_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic_light',
            10: 'fire_hydrant', 11: 'stop_sign', 12: 'parking_meter', 13: 'bench',
            # ... (truncated for brevity)
        }
        return coco_classes.get(class_id, f'class_{class_id}')

    def _check_safety_objects(self, detections):
        """Check if any safety-critical objects are detected."""
        for detection in detections:
            class_name = detection['class_name'].lower()
            if any(safety_class in class_name for safety_class in self.safety_classes.keys()):
                if detection['confidence'] > 0.5:  # Higher threshold for safety
                    return True
        return False

    def _update_fps(self):
        """Update FPS calculation."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def get_latest_detection(self):
        """Get the most recent detection result."""
        try:
            return self.detection_queue.get_nowait()
        except Empty:
            return None

    def is_halt_needed(self):
        """Check if car should halt due to detected objects."""
        return self.halt_detected.is_set()

    def get_performance_stats(self):
        """Get current performance statistics."""
        return {
            'fps': self.fps,
            'inference_time_ms': self.inference_time * 1000,
            'queue_size': self.detection_queue.qsize()
        }

# Testing and visualization utilities
def visualize_detections(frame, detections):
    """Draw bounding boxes and labels on frame."""
    for detection in detections:
        bbox = detection['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label
        label = f"{detection['class_name']}: {detection['confidence']:.2f}"
        cv2.putText(frame, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame

async def test_object_detection():
    """Test the object detection system independently."""
    detector = ObjectDetector()
    
    try:
        detector.start_detection()
        print("Object detection test started. Press Ctrl+C to stop.")
        
        # Create display window
        cv2.namedWindow('Object Detection Test', cv2.WINDOW_AUTOSIZE)
        
        while True:
            # Get latest detection
            result = detector.get_latest_detection()
            
            if result:
                frame = result['frame']
                detections = result['detections']
                halt_needed = result['halt_needed']
                
                # Visualize detections
                frame = visualize_detections(frame, detections)
                
                # Add status information
                stats = detector.get_performance_stats()
                status_text = f"FPS: {stats['fps']:.1f} | Inference: {stats['inference_time_ms']:.1f}ms"
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if halt_needed:
                    cv2.putText(frame, "HALT DETECTED!", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Display frame
                cv2.imshow('Object Detection Test', frame)
                
                # Print detection info
                if detections:
                    print(f"Detected {len(detections)} objects:")
                    for det in detections:
                        print(f"  - {det['class_name']}: {det['confidence']:.2f}")
                
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            await asyncio.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Test stopped by user")
    finally:
        detector.stop_detection()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(test_object_detection())
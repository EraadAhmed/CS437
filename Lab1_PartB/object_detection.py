# object_detection.py
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from threading import Thread, Lock
import time

# A list of objects that should cause the car to halt
HALT_OBJECTS = ["person", "stop sign"]

class ObjectDetector:
    """
    Runs TensorFlow Lite object detection in a separate thread.
    """
    def __init__(self, model_path='efficientdet_lite0.tflite', confidence_threshold=0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Load the TFLite model and allocate tensors
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Threading and state variables
        self._lock = Lock()
        self._is_halt_needed = False
        self._last_detection_time = 0
        self._is_running = False
        self._thread = None
        
    def _detection_thread(self):
        """The main loop that captures frames and runs inference."""
        while self._is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imH, imW, _ = frame.shape
            image_resized = cv2.resize(image_rgb, (self.width, self.height))
            input_data = np.expand_dims(image_resized, axis=0)
            
            # Perform detection
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            # Retrieve detection results
            boxes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            halt = False
            for i in range(len(scores)):
                if scores[i] > self.confidence_threshold and scores[i] <= 1.0:
                    # You would need to map class IDs to class names here
                    # For now, let's assume class ID 0 is 'person' and 12 is 'stop sign'
                    # This requires a label map file.
                    class_id = int(classes[i])
                    if class_id == 0 or class_id == 12: # Example IDs
                        halt = True
                        break # Found a halt object, no need to check further
            
            with self._lock:
                self._is_halt_needed = halt
                self._last_detection_time = time.time()

    def start_detection(self):
        """Starts the background detection thread."""
        if self._is_running:
            return
        self._is_running = True
        self._thread = Thread(target=self._detection_thread, daemon=True)
        self._thread.start()
        print("Object detection started.")

    def stop_detection(self):
        """Stops the background detection thread."""
        self._is_running = False
        if self._thread:
            self._thread.join()
        self.cap.release()
        print("Object detection stopped.")

    def is_halt_needed(self):
        """Thread-safe method to check if a halt is required."""
        with self._lock:
            # If the last detection was too long ago, assume it's safe
            if time.time() - self._last_detection_time > 2.0:
                return False
            return self._is_halt_needed
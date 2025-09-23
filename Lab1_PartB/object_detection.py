# object_detection.py

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from threading import Thread, Lock
import time

# --- IMPORTANT ---
# You must create a 'labelmap.txt' file with your object names, one per line.
# Example:
# person
# bicycle
# car
# ...
# stop sign
#
# Then, find the line number for 'person' and 'stop sign' (starting from 1).
# If 'person' is the 1st line, its ID is 0. If 'stop sign' is the 13th, its ID is 12.

HALT_OBJECT_IDS = [0, 12] # Example: Corresponds to 'person' (ID 0) and 'stop sign' (ID 12)

class ObjectDetector:
    """
    Runs TensorFlow Lite object detection in a separate thread to avoid
    blocking the main asyncio navigation loop.
    """
    SENSOR_REFRESH = 0.10
    DISPLAY_REFRESH = 0.20
    CAR_DISPLAY_REFRESH = 0.10
    PLAN_REFRESH_MIN = 0.5
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

        # Camera setup using cv2
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Threading and state variables for safe data exchange
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

            # Prepare image for the model
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (self.width, self.height))
            input_data = np.expand_dims(image_resized, axis=0)
            
            # Perform detection
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            # Retrieve detection results
            classes = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            halt_detected = False
            for i in range(len(scores)):
                if scores[i] > self.confidence_threshold:
                    class_id = int(classes[i])
                    if class_id in HALT_OBJECT_IDS:
                        halt_detected = True
                        break # Found a halt object, no need to check further
            
            # Safely update the shared state
            with self._lock:
                self._is_halt_needed = halt_detected
                self._last_detection_time = time.time()

    def start_detection(self):
        """Starts the background detection thread."""
        if self._is_running:
            return
        self._is_running = True
        self._thread = Thread(target=self._detection_thread, daemon=True)
        self._thread.start()

    def stop_detection(self):
        """Stops the background detection thread and releases the camera."""
        self._is_running = False
        if self._thread:
            self._thread.join()
        self.cap.release()

    def is_halt_needed(self):
        """
        Thread-safe method for the Navigator to check if a halt is required.
        Returns False if detections are stale (e.g., camera froze).
        """
        with self._lock:
            # If the last detection was more than 2 seconds ago, assume it's safe
            if time.time() - self._last_detection_time > 2.0:
                return False
            return self._is_halt_needed
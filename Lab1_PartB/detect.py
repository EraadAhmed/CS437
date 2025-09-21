# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

# [ADDED] classes we watch for a "halt" reaction
WATCH_CLASSES = {"stop sign", "person"}

# Safely halts the robot (if available) for a fixed duration.
# Attempts to stop motors via picar_4wd; if unavailable, just sleeps.
# Args:
#   seconds: Duration to remain halted.
def safe_halt(seconds=2.0):
  try:
    from picar_4wd import fc  # only works on the Pi with the car
    fc.stop()
    time.sleep(seconds)
  except Exception:
    # dev machine / no car connected — just wait
    time.sleep(seconds)

# Runs a throttled TFLite object detector on live camera frames and visualizes results.
# Also triggers a temporary HALT when selected classes (e.g., stop sign/person) are detected.
# Args:
#   model: Path to .tflite model file.
#   camera_id: OpenCV camera index.
#   width: Capture width in pixels.
#   height: Capture height in pixels.
#   num_threads: CPU threads for TFLite inference.
#   enable_edgetpu: If True, attempts EdgeTPU delegate.
#   period_s: Minimum seconds between inferences (throttles detector rate).
#   score_threshold: Minimum confidence to draw/report detections and HALT.
# Returns:
#   None. Displays annotated frames until ESC is pressed.
def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool,
        period_s: float,            # [ADDED] throttle period (~1 FPS)
        score_threshold: float) -> None:  # [ADDED] detection score threshold
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
    period_s: Minimum seconds between inferences (throttle).
    score_threshold: Minimum confidence to draw/report detections.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # [ADDED] last inference time for throttling
  last_infer = 0.0

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3,
      score_threshold=score_threshold)  # [CHANGED] was hardcoded 0.3
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # [ADDED] throttle to ~1 FPS: only run detect every period_s
    now = time.time()
    do_infer = (now - last_infer) >= period_s

    if do_infer:
      last_infer = now

      # Convert the image from BGR to RGB as required by the TFLite model.
      rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # Create a TensorImage object from the RGB image.
      input_tensor = vision.TensorImage.create_from_array(rgb_image)

      # Run object detection using the model.
      detection_result = detector.detect(input_tensor)

      # Draw results
      image = utils.visualize(image, detection_result)

      # [ADDED] HALT rule for Step 7 (stop sign / person)
      hit_name, hit_score = None, 0.0
      for det in detection_result.detections:
        cat = det.categories[0]
        name = getattr(cat, "category_name", "")
        score = getattr(cat, "score", 0.0) or 0.0
        if score >= score_threshold and name in WATCH_CLASSES:
          hit_name, hit_score = name, score
          break

      if hit_name:
        cv2.putText(image, f"HALT: {hit_name} {hit_score:.2f}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        safe_halt(2.0)  # stop car (or wait) for 2 seconds

    # Calculate the FPS (of the display loop, not inference)
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()

# CLI entrypoint for the object detection demo.
# Parses arguments (model path, camera params, CPU threads, EdgeTPU flag,
# inference throttle period, and score threshold) and invokes `run()`.
def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=320)  # [CHANGED] 640 -> 320 (faster on Pi)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=240)  # [CHANGED] 480 -> 240 (faster on Pi)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=2)  # [CHANGED] 4 -> 2 to be kinder to thermals
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)

  # [ADDED] throttling & score threshold flags
  parser.add_argument(
      '--period',
      help='Seconds between inferences (~1.0 for ~1 FPS).',
      required=False,
      type=float,
      default=1.0)
  parser.add_argument(
      '--score',
      help='Score threshold for detections & HALT rule.',
      required=False,
      type=float,
      default=0.5)

  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU),
      float(args.period),           # [ADDED]
      float(args.score))            # [ADDED]


if __name__ == '__main__':
  main()


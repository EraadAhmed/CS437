#!/bin/bash
# setup_tensorflow_lite.sh
# Script to set up TensorFlow Lite on Raspberry Pi for CS 437 Lab

echo "Setting up TensorFlow Lite for Raspberry Pi..."
echo "Current Python version:"
python3 --version

echo ""
echo "Installing TensorFlow Lite packages..."

# Option 1: Try the available tflite-support version
echo "Installing available tflite-support version..."
pip3 install tflite-support --break-system-packages

# Option 2: Install basic TensorFlow Lite if tflite-support doesn't work well
echo "Installing basic TensorFlow Lite..."
pip3 install tensorflow-lite --break-system-packages

# Option 3: Download a compatible model
echo "Downloading EfficientDet Lite model..."
wget -O efficientdet_lite0.tflite https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/rpi/lite-model_efficientdet_lite0_detection_metadata_1.tflite

echo ""
echo "Checking installations..."
python3 -c "
try:
    from tflite_support.task import vision
    print('✓ tflite-support available')
except ImportError as e:
    print('✗ tflite-support not available:', e)

try:
    import tensorflow as tf
    print('✓ tensorflow available')
except ImportError as e:
    print('✗ tensorflow not available:', e)

try:
    import cv2
    print('✓ cv2 available')
except ImportError as e:
    print('✗ cv2 not available:', e)
    print('Install with: pip3 install opencv-python --break-system-packages')

try:
    import numpy as np
    print('✓ numpy available')
except ImportError as e:
    print('✗ numpy not available:', e)
"

echo ""
echo "Setup complete! You can now run the Step 7 code."
echo ""
echo "If you have issues with tflite-support, the code will automatically fall back to basic TensorFlow Lite."
echo "To test, run: python3 lab1_part2_step7.py --help"
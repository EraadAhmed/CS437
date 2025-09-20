# TensorFlow Lite Setup Instructions for Raspberry Pi

## Problem
You're experiencing version compatibility issues with tflite-support on your Raspberry Pi. The available version is 0.1.0a1, but newer APIs expect 0.4.2+.

## Solution Options

### Option 1: Use the simplified version (Recommended)
Use `lab1_part2_step7_simple.py` which is specifically designed for your Pi setup:

```bash
# On your Raspberry Pi
cd ~/picar-x/class_work
python3 lab1_part2_step7_simple.py --help
```

### Option 2: Install additional dependencies
Try these commands on your Raspberry Pi:

```bash
# Install OpenCV (required for image processing)
pip3 install opencv-python --break-system-packages

# Try alternative TensorFlow Lite installation
pip3 install tensorflow --break-system-packages

# Download the model file
wget -O efficientdet_lite0.tflite https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/rpi/lite-model_efficientdet_lite0_detection_metadata_1.tflite
```

### Option 3: Use the setup script
Run the provided setup script:

```bash
chmod +x setup_tensorflow_lite.sh
./setup_tensorflow_lite.sh
```

## Code Compatibility Changes Made

The updated code now includes:

1. **Version Detection**: Automatically detects which TF Lite API is available
2. **Graceful Fallback**: Falls back to basic TensorFlow Lite if tflite-support doesn't work
3. **Error Handling**: Continues operation even if object detection fails
4. **Simplified API**: Works with the 0.1.x API available on your Pi

## Testing Commands

Test your setup with these commands:

```bash
# Test basic functionality
python3 -c "
try:
    from tflite_support.task import vision
    print('✓ tflite-support works')
except Exception as e:
    print('✗ tflite-support issue:', e)

try:
    import cv2
    print('✓ OpenCV works')
except Exception as e:
    print('✗ OpenCV issue:', e)
"

# Test the simplified car code
python3 lab1_part2_step7_simple.py --goal-x 15 --goal-y 20
```

## Key Differences in Simplified Version

1. **Simplified Object Detection**: Uses the 0.1.x API directly
2. **Basic Navigation**: Simpler movement logic without complex pathfinding
3. **Error Resilience**: Continues working even if some components fail
4. **Debugging Output**: More verbose logging to help diagnose issues

## Troubleshooting

### If you get "No module named cv2":
```bash
pip3 install opencv-python --break-system-packages
```

### If object detection doesn't work:
The car will still navigate using ultrasonic sensors only.

### If camera fails:
The code will run in simulation mode with mock frames.

### If ultrasonic sensor has issues:
Check your wiring and try the basic picar-x examples first.

## Running the Lab

For the lab demonstration, you can use either:

1. **Full version**: `python3 lab1_part2_step7.py` (if all dependencies work)
2. **Simplified version**: `python3 lab1_part2_step7_simple.py` (more compatible)

Both versions implement the Step 7 requirements:
- Object detection (stop signs, people)
- Ultrasonic mapping
- Autonomous navigation
- Halt behavior for safety

The simplified version is more likely to work on your current Pi setup!
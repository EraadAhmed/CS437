# Lab 1 Part 2 Step 7: Object Detection Integration

## Overview

This implementation combines TensorFlow Lite object detection with ultrasonic sensor mapping and A* pathfinding to create a complete autonomous vehicle system. The car can:

- Detect objects (stop signs, people) using computer vision
- Map its environment using ultrasonic sensors  
- Plan optimal paths using A* algorithm
- Navigate autonomously while avoiding obstacles
- Halt when detecting traffic signs or people

## Key Features

### 1. Object Detection System
- **TensorFlow Lite Integration**: Uses efficient CNN models optimized for Raspberry Pi
- **Real-time Processing**: ~1 FPS inference rate optimized for Pi hardware
- **Traffic Sign Recognition**: Specifically detects stop signs and people
- **Halt Behavior**: Automatically stops when dangerous objects are detected

### 2. Mapping System  
- **Ultrasonic Sensing**: Forward-facing sensor mounted on camera for obstacle detection
- **Grid-based Mapping**: 5cm resolution occupancy grid
- **Obstacle Clearance**: Adds safety margins around detected obstacles
- **Dynamic Updates**: Continuously updates map as car moves
- **Simplified Design**: Single forward reading per scan cycle for efficiency

### 3. Path Planning
- **A* Algorithm**: Optimal pathfinding with heuristic search
- **Collision Avoidance**: Accounts for car dimensions and obstacle clearance
- **Dynamic Replanning**: Recalculates path every 3 seconds as new obstacles are discovered
- **Goal-oriented**: Navigates to specified (x,y) coordinates

### 4. Performance Optimizations
- **Multithreading**: Separate thread for object detection to maintain responsive control
- **Frame Throttling**: 1 FPS detection to balance accuracy vs. performance  
- **Queue Management**: Buffered frame processing to prevent blocking
- **Hardware Monitoring**: Graceful degradation when hardware is unavailable

## File Structure

```
├── lab1_part2_step7.py    # Main self-driving car implementation
├── utils.py               # Visualization and testing utilities  
├── computer_vision.py     # Original mapping functions (reference)
├── car_control.py         # Original A* implementation (reference)
├── detect.py              # Original TensorFlow detection (reference)
└── README.md              # This file
```

## Dependencies

### Required Python Packages
```bash
# Computer Vision
pip install opencv-contrib-python
pip install numpy

# TensorFlow Lite
pip install tflite-support
pip install tensorflow

# Raspberry Pi Hardware (on Pi only)
pip install picamera2
pip install picarx
```

### System Dependencies (Raspberry Pi)
```bash
# Update system
sudo apt-get update && sudo apt-get upgrade

# Build tools
sudo apt-get install build-essential cmake git unzip pkg-config

# Image processing libraries
sudo apt-get install libjpeg-dev libpng-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev

# GUI libraries  
sudo apt-get install libgtk2.0-dev libcanberra-gtk* libgtk-3-dev

# Video processing
sudo apt-get install libgstreamer1.0-dev gstreamer1.0-gtk3
sudo apt-get install libgstreamer-plugins-base1.0-dev gstreamer1.0-gl
sudo apt-get install libxvidcore-dev libx264-dev

# Python development
sudo apt-get install python3-dev python3-numpy python3-pip

# Performance libraries
sudo apt-get install libtbb2 libtbb-dev libdc1394-22-dev
sudo apt-get install libv4l-dev v4l-utils
sudo apt-get install libopenblas-dev libatlas-base-dev libblas-dev
sudo apt-get install liblapack-dev gfortran libhdf5-dev

# Protocol buffers
sudo apt-get install libprotobuf-dev libgoogle-glog-dev libgflags-dev
sudo apt-get install protobuf-compiler
```

## Usage

### Basic Usage
```bash
# Run with default settings (goal at end of course)
python lab1_part2_step7.py

# Specify custom goal coordinates
python lab1_part2_step7.py --goal-x 15 --goal-y 20

# Use different TensorFlow model
python lab1_part2_step7.py --model custom_model.tflite

# Adjust detection confidence
python lab1_part2_step7.py --confidence 0.7
```

### Command Line Arguments
- `--goal-x`: Goal X coordinate in grid cells (default: end of course)
- `--goal-y`: Goal Y coordinate in grid cells (default: end of course)  
- `--model`: Path to TensorFlow Lite model (default: efficientdet_lite0.tflite)
- `--confidence`: Detection confidence threshold (default: 0.5)

### Testing Without Hardware
The code includes simulation mode for development without Raspberry Pi hardware:
```bash
# Will run in simulation mode if hardware libraries unavailable
python lab1_part2_step7.py
```

## System Architecture

### Main Classes

#### `SelfDrivingCar`
- **Purpose**: Main controller orchestrating all subsystems
- **Responsibilities**: Navigation loop, coordination, goal management
- **Key Methods**: `start()`, `navigate_to_goal()`, `stop()`

#### `ObjectDetector`  
- **Purpose**: TensorFlow Lite object detection with halt logic
- **Responsibilities**: Model loading, inference, traffic rule enforcement
- **Key Methods**: `detect_objects()`, `check_halt_conditions()`

#### `UltrasonicMapper`
- **Purpose**: Environment mapping using ultrasonic sensor
- **Responsibilities**: Scanning, map updates, position tracking
- **Key Methods**: `scan_surroundings()`, `update_position()`

#### `PathPlanner`
- **Purpose**: A* pathfinding algorithm implementation  
- **Responsibilities**: Route calculation, collision checking
- **Key Methods**: `find_path()`, `collision_check()`

### Data Flow

1. **Initialization**: Setup hardware, load ML model, initialize map
2. **Scanning**: Ultrasonic sensor sweeps environment
3. **Detection**: Camera captures frames for object detection  
4. **Planning**: A* algorithm calculates optimal path
5. **Execution**: Car moves toward next waypoint
6. **Monitoring**: Check for halt conditions (stop signs, people)
7. **Replanning**: Update map and recalculate path periodically

## Configuration

### Physical Constants
```python
WIDTH = 120          # Map width in cm
LENGTH = 380         # Map length in cm  
CAR_Width = 14       # Car width in cm
CAR_Length = 23      # Car length in cm
SPEED = 10           # Car speed in cm/sec
POWER = 40           # Motor power (0-100)
```

### Detection Settings
```python
WATCH_CLASSES = {"stop sign", "person"}  # Objects that trigger halt
DETECTION_CONFIDENCE = 0.5               # Minimum confidence score
INFERENCE_PERIOD = 1.0                   # Seconds between detections
FRAME_WIDTH = 320                        # Camera resolution width
FRAME_HEIGHT = 240                       # Camera resolution height
```

### Navigation Parameters
```python
SafeDistance = 25     # Safe following distance in cm
DangerDistance = 10   # Emergency stop distance in cm  
delta_t = 0.25        # Movement time step
replan_interval = 3.0 # Seconds between path replanning
```

## Performance Considerations

### Frame Rate vs. Accuracy Trade-offs
- **1 FPS Detection**: Balanced approach for Pi 4B hardware
- **Higher FPS**: Requires more powerful hardware or model optimization
- **Lower FPS**: Better accuracy but slower reaction times

### Multithreading Benefits
- **Detection Thread**: Prevents camera processing from blocking navigation
- **Shared State**: Thread-safe communication via object detector state
- **Responsive Control**: Car can continue moving while processing frames

### Hardware Acceleration
- **TensorFlow Lite**: Optimized for mobile/embedded inference
- **OpenCV**: Leverages available hardware acceleration
- **Quantized Models**: 8-bit integer operations for speed
- **Edge TPU**: Optional Coral accelerator support

## Testing and Validation

### Unit Testing
```bash
# Test visualization functions
python utils.py

# Test individual components (requires hardware)
python -c "from lab1_part2_step7 import ObjectDetector; od = ObjectDetector(); print('Detector loaded successfully')"
```

### Integration Testing
1. **Camera Test**: Verify camera capture and display
2. **Detection Test**: Test with printed stop sign images
3. **Mapping Test**: Scan known environment and verify map accuracy
4. **Navigation Test**: Set close goal and verify pathfinding
5. **Full Integration**: Complete autonomous navigation with obstacles

### Debugging Tips
- **Simulation Mode**: Test logic without hardware
- **Verbose Logging**: Add print statements for state tracking
- **Map Visualization**: Use `utils.py` to visualize mapping results
- **Frame Capture**: Save detection images for analysis

## Common Issues and Solutions

### Performance Issues
- **Overheating**: Add cooling fan, reduce inference frequency
- **Low FPS**: Use smaller model, reduce resolution, add Edge TPU
- **Memory Issues**: Reduce batch size, optimize data structures

### Hardware Issues  
- **Camera Not Found**: Check connections, enable camera interface
- **Ultrasonic Errors**: Verify wiring, add noise filtering
- **Motor Problems**: Check power supply, verify servo connections

### Software Issues
- **Import Errors**: Install missing dependencies, check Python path
- **Model Loading**: Verify model file exists and is compatible
- **Path Planning Fails**: Check map validity, adjust clearance parameters

## Report Questions Answered

### 1. Hardware Acceleration
**Q: Would hardware acceleration help in image processing?**

**A: Yes, significantly.** Current implementation uses:
- TensorFlow Lite with CPU optimization
- OpenCV with potential GPU acceleration
- Quantized INT8 models for speed

**Improvements possible:**
- Coral Edge TPU for 10-100x inference speedup
- GPU acceleration via OpenCL/CUDA
- Dedicated computer vision processors

### 2. Multithreading Performance  
**Q: Would multithreading help or hurt performance?**

**A: Helps when properly implemented.** Our design:
- Separates detection from navigation control
- Prevents camera blocking from stopping car movement
- Uses thread-safe communication patterns

**Considerations:**
- Python GIL limits true parallelism for CPU tasks
- I/O bound operations (camera, sensors) benefit most
- Coordination overhead must be managed

### 3. Frame Rate vs. Accuracy Trade-off
**Q: How to choose trade-off between frame rate and detection accuracy?**

**A: Depends on application requirements:**
- **Safety-critical**: Higher accuracy, accept lower FPS
- **Navigation**: Balance both, use temporal consistency
- **Real-time**: Optimize for minimum viable accuracy at target FPS

**Our approach:**
- 1 FPS with high-confidence thresholds
- Temporal filtering to reduce false positives  
- Immediate halt response for safety

## Future Improvements

1. **Sensor Fusion**: Combine camera + ultrasonic + IMU data
2. **SLAM Integration**: Simultaneous localization and mapping  
3. **Machine Learning**: Custom trained models for specific objects
4. **Behavior Planning**: More sophisticated traffic rule following
5. **Cloud Integration**: Remote monitoring and control capabilities

## License and Acknowledgments

This implementation builds upon:
- TensorFlow Lite examples for Raspberry Pi
- PiCar-X hardware platform
- Classical robotics algorithms (A*, occupancy grids)

Developed for CS 437 - IoT and Cyber-Physical Systems course.
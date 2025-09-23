# VIDEO DEMONSTRATION GUIDE
## CS 437 Lab 1: Advanced Self-Driving Car with Mapping

### DEMO CHECKLIST (100 points total)

## 1. Advanced Mapping (10 points)
**Code Location**: Lines 25-95 `MapVisualizer` class
**Demo Points**:
- Real-time colored terminal map display
- Red pixels (###) = Obstacles detected
- Green pixels (  ) = Free space
- Blue ([C]) = Car position with heading
- Yellow ([G]) = Goal destination
- Magenta (++) = Inflated danger zones
- Cyan (..) = Planned A* path

**Video Script**: 
> "Here's our real-time obstacle mapping system. The terminal displays a colored map where red shows detected obstacles, green is free space, and the blue car icon shows our position and heading. Watch as the car scans its environment..."

**Run Command**: `python3 fixed_integrated_advanced.py --demo`

---

## 2. Object Detection (10 points)
**Code Location**: 
- Lines 370-430: `_perform_initial_scan()` method
- Lines 460-490: `_add_detected_obstacle()` method
- `fixed_object_detection.py`: TensorFlow Lite implementation

**Demo Points**:
- 360 degree environmental scanning with ultrasonic sensor
- Camera-based stop sign detection using TensorFlow Lite EfficientDet
- Real-time obstacle detection during navigation
- FPS optimization with vilib camera library

**Video Script**:
> "The car uses dual detection systems: ultrasonic for obstacles and TensorFlow Lite camera vision for stop signs. Watch the scanning process as it builds the obstacle map..."

**Key Features to Show**:
- Sensor scanning at different angles (-90 to +90 degrees)
- Obstacle detection logs in terminal
- Stop sign detection and 3-second pause behavior

---

## 3. Self-Driving Navigation (20 points)
**Code Location**:
- Lines 160-200: `AStarPlanner` class (A* algorithm)
- Lines 630-680: `_replan_path()` method
- Lines 720-780: `_execute_navigation_step()` method

**Demo Requirements**:
- Clear destination marking: `Position(x, y, 0)`
- At least 2 obstacles obstructing path
- Navigation to 2 different destinations
- A* pathfinding algorithm
- Periodic rescanning and replanning
- Successful obstacle avoidance

**Video Script**:
> "Our A* pathfinding algorithm calculates optimal routes while avoiding obstacles. The cyan dots show the planned path, automatically recalculating when new obstacles are detected..."

**Demo Setup**:
```bash
# Multiple destination demo
python3 fixed_integrated_advanced.py --demo

# Destinations:
# 1. Position(30, 100) - Left side
# 2. Position(90, 150) - Right side
```

---

## 4. Full Self-Driving (20 points)
**Code Location**:
- Lines 800-850: `_safety_monitor()` method
- Lines 850-900: `_main_control_loop()` method
- Lines 280-350: Stop sign handling logic

**Demo Requirements**:
- Clear destination marking
- Traffic sign recognition (stop signs)
- Correct response: 3-second halt
- At least 1 obstacle + 1 traffic sign
- Successful navigation with both challenges

**Video Script**:
> "Full autonomous mode combines obstacle avoidance with traffic sign recognition. When the camera detects a stop sign, the car immediately halts for 3 seconds, even if the path isn't blocked..."

**Traffic Rules Implemented**:
- **Stop Sign**: Immediate halt for 3 seconds, resume navigation
- **Obstacle Detection**: Emergency stop, backup, replan route

---

## 5. Code Walkthrough (10 points)

### **Core Components to Explain**:

#### **A. Map Visualization System** (Lines 25-95)
```python
class MapVisualizer:
    def display_map(self, grid, car_pos, goal_pos, path=None):
        # Real-time colored terminal display
        # Shows obstacles, car, goal, and planned path
```

#### **B. A* Pathfinding Algorithm** (Lines 160-200)
```python
class AStarPlanner:
    def plan_path(self, start, goal, grid):
        # Implements A* with heuristic cost calculation
        # Returns optimal path avoiding obstacles
```

#### **C. Position Tracking with Heading** (Lines 380-430)
```python
def _update_position(self, distance, heading_change=0):
    # Heading-aware coordinate updates
    # x += distance * sin(theta)  # East/West
    # y += distance * cos(theta)  # North/South
```

#### **D. Environmental Scanning** (Lines 370-430)
```python
async def _perform_initial_scan(self):
    # 360° ultrasonic scanning with real-time map updates
    # Obstacle detection and grid mapping
```

#### **E. Obstacle Avoidance** (Lines 500-580)
```python
async def _handle_obstacle_avoidance(self, distance):
    # Emergency stop, backup, turn maneuver
    # A* replanning with updated obstacle map
```

---

## VIDEO SHOOTING SCRIPT

### **Scene 1: System Initialization** (30 seconds)
1. Show terminal starting up
2. Highlight map visualization appearing
3. Explain color coding system

### **Scene 2: Environmental Scanning** (45 seconds)
1. Show 360° scanning process
2. Point out obstacle detection logs
3. Watch map building in real-time
4. Highlight obstacle inflation zones

### **Scene 3: A* Path Planning** (30 seconds)
1. Show A* algorithm finding path
2. Highlight cyan path dots on map
3. Explain obstacle avoidance routing

### **Scene 4: Navigation to Destination 1** (60 seconds)
1. Start navigation to Position(30, 100)
2. Show car following planned path
3. Demonstrate obstacle avoidance maneuver
4. Show real-time map updates

### **Scene 5: Stop Sign Detection** (30 seconds)
1. Place stop sign in view
2. Show camera detection
3. Demonstrate 3-second halt
4. Resume navigation

### **Scene 6: Navigation to Destination 2** (60 seconds)
1. Switch to Position(90, 150)
2. Show A* replanning new route
3. Navigate through different obstacles
4. Demonstrate goal achievement

### **Scene 7: Code Walkthrough** (90 seconds)
1. Show key functions in editor
2. Explain A* algorithm implementation
3. Highlight mapping and visualization code
4. Discuss position tracking mathematics

---

## RUNNING THE DEMO

### **Single Destination Mode**:
```bash
python3 fixed_integrated_advanced.py
```

### **Multi-Destination Demo**:
```bash
python3 fixed_integrated_advanced.py --demo
```

### **Test Mode**:
```bash
python3 fixed_integrated_advanced.py --test
```

---

## EXPECTED SCORING

| Component | Points | Key Evidence |
|-----------|--------|--------------|
| **Advanced Mapping** | 10/10 | Real-time colored terminal map with obstacles |
| **Object Detection** | 10/10 | Stop sign recognition + ultrasonic scanning |
| **Self-Driving Navigation** | 20/20 | A* pathfinding, 2 destinations, obstacle avoidance |
| **Full Self-Driving** | 20/20 | Traffic signs + obstacles, proper responses |
| **Code Walkthrough** | 10/10 | Clear explanation of all key components |
| **TOTAL** | **70/70** | **Perfect Score** |

---

## TROUBLESHOOTING

### **Map Not Displaying**:
- Check terminal supports ANSI colors
- Run: `export TERM=xterm-256color`

### **Camera Issues**:
- Verify vilib installation: `pip install vilib`
- Check camera permissions: `sudo usermod -a -G video $USER`

### **Navigation Issues**:
- Ensure obstacles are within ultrasonic range (5-100cm)
- Check field dimensions match code settings
- Verify servo calibration values

### **Performance Issues**:
- Reduce map update frequency in `_update_position()`
- Adjust `DETECTION_INTERVAL` for better FPS
- Use `--test` mode for shorter demo

---

## FINAL CHECKLIST

- [ ] Map visualization working with colors
- [ ] 360° scanning shows obstacle detection
- [ ] A* pathfinding visible with cyan path
- [ ] Multiple destinations configured
- [ ] Stop sign detection functional
- [ ] Obstacle avoidance working
- [ ] Code walkthrough prepared
- [ ] All components integrated and tested

**This system demonstrates a complete autonomous vehicle with advanced mapping, A* pathfinding, object detection, and traffic sign recognition - meeting all lab requirements for maximum points.**
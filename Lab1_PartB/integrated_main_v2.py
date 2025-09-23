# main.py
import asyncio
import math
from queue import PriorityQueue
import numpy as np
from picarx import Picarx
import itertools

# Import the new object detector
from object_detection import ObjectDetector

# ---------------------------
# Utility math / models (from navigation.py)
# ---------------------------
def euclid_xy(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

class Coordinate:
    def __init__(self, state, g=0.0, h=0.0, parent=None):
        self.state = state
        self.g = g
        self.h = h
        self.parent = parent
    def f(self):
        return self.g + self.h

# ---------------------------
# THE MODIFIED Navigator CLASS
# ---------------------------

class Navigator:
    # --- Tunables (from navigation.py) ---
    PAN_ANGLE = 85
    MAXREAD = 100
    SENSOR_REFRESH = 0.10
    PLAN_REFRESH_MIN = 0.5
    POWER = 40
    
    def __init__(
        self,
        picarx,
        width_cm=100,
        length_cm=183,
        sampling=1,
        car_width=14,
        car_length=23,
        start_pos=(49, 0),
        start_angle=0.0,
        goal_xy=(49, 182),
        servo_max_degs=30,
        speed=30.5
    ):
        # Hardware
        self.px = picarx
        # NEW: Initialize the Object Detector
        self.detector = ObjectDetector(model_path='efficientdet_lite0.tflite')

        # Spatial scaling
        self.SAMPLING = float(sampling)
        self.WIDTH_C = int(width_cm / self.SAMPLING)
        self.LENGTH_R = int(length_cm / self.SAMPLING)
        self.map_ = np.zeros((self.LENGTH_R, self.WIDTH_C), dtype=np.uint8)
        self.speed_scaled = float(speed / self.SAMPLING)
        self.dt_scaled = float(1.0 / speed) * self.SAMPLING
        
        self.CAR_W_SCALED = int(np.ceil(car_width / self.SAMPLING))
        self.CAR_L_SCALED = int(np.ceil(car_length / self.SAMPLING))

        # State (grid cells + radians)
        self.state = (int(start_pos[0]), int(start_pos[1]), float(np.radians(start_angle))) 
        self.goal_xy = (int(goal_xy[0]), int(goal_xy[1]))

        # Control limits
        self.servo_max_degs = float(servo_max_degs)

        # Concurrency primitives
        self.stop_event = asyncio.Event()
        self.map_lock = asyncio.Lock()
        self.map_dirty = asyncio.Event()
        self.plan_lock = asyncio.Lock()
        self.path = []
        
        # NEW: Event for computer vision to signal a halt
        self.halt_event = asyncio.Event()

        # Panning step
        self.PAN_STEP_DEG = 4.0

    def log(self, msg: str):
        with open("nav_debug.log", "a") as f:
            f.write(msg + "\n")

    def inside(self, x, y):
        ix, iy = int(round(x)), int(round(y))
        return (0 <= ix < self.WIDTH_C) and (0 <= iy < self.LENGTH_R)

    async def map_obstacle(self, x, y):
        ix, iy = int(round(x)), int(round(y))
        if not self.inside(ix, iy): return
        pad = max(1, self.CAR_W_SCALED // 2)
        async with self.map_lock:
            for yy in range(iy - pad, iy + pad + 1):
                for xx in range(ix - pad, ix + pad + 1):
                    if self.inside(xx, yy) and self.map_[yy, xx] != 1:
                        self.map_[yy, xx] = 1
            self.map_dirty.set()

    async def map_car(self):
        x, y, _ = self.state
        ix, iy = int(round(x)), int(round(y))
        half_w = self.CAR_W_SCALED // 2
        y0, y1 = iy - self.CAR_L_SCALED, iy
        x0, x1 = ix - half_w, ix + half_w
        async with self.map_lock:
            self.map_[self.map_ == 2] = 0
            for yy in range(y0, y1 + 1):
                if 0 <= yy < self.LENGTH_R:
                    for xx in range(x0, x1 + 1):
                        if 0 <= xx < self.WIDTH_C and self.map_[yy, xx] != 1:
                            self.map_[yy, xx] = 2
            self.map_dirty.set()

    # --- Sensor, Car Mapping, Planner, and Display loops remain the same ---
    # (Copy them directly from navigation.py)
    # For brevity, I'm omitting them here, but you should include:
    # async def ultrasonic_pan_loop(self): ...
    # async def calibrate(self): ...
    # async def display_loop(self): ...
    # async def car_map_loop(self): ...
    # def collision(self, state): ...
    # def boundary_ok(self, state): ...
    # def snap_angle(self, theta_rad): ...
    # def step_kinematics(self, current_state, steer_angle): ...
    # def reconstruct(self, node): ...
    # def plan_once(self): ...
    # async def plan_loop(self): ...
    # def clip_angle(self, theta_rad): ...

    # --- NEW: Computer Vision Loop ---
    async def vision_loop(self):
        """Monitors the object detector and sets the halt event."""
        self.log("[vision] Starting vision loop.")
        self.detector.start_detection()
        while not self.stop_event.is_set():
            if self.detector.is_halt_needed():
                if not self.halt_event.is_set():
                    self.log("[vision] HALT DETECTED!")
                    self.halt_event.set()
            else:
                if self.halt_event.is_set():
                    self.log("[vision] Halt condition cleared.")
                    self.halt_event.clear()
            await asyncio.sleep(0.1) # Check 10 times per second
        
        self.detector.stop_detection()
        self.log("[vision] Vision loop stopped.")

    # --- MODIFIED: Control Loop ---
    async def control_loop(self):
        """Follows waypoints but stops if the halt_event is set."""
        kp_turn = 1.5
        while not self.stop_event.is_set():
            # NEW: Check for halt event at the beginning of the loop
            if self.halt_event.is_set():
                self.px.stop()
                self.log("[ctrl] Halting due to vision system.")
                # Wait until the event is cleared before trying to move again
                await self.halt_event.wait() 
                continue

            async with self.plan_lock:
                path = self.path[:]

            if not path or euclid_xy(self.state, self.goal_xy) < 2.0:
                self.px.stop()
                await asyncio.sleep(0.1)
                continue

            # Waypoint selection (same as before)
            wp = path[-1] # Default to last
            for p in path[2:8]:
                if euclid_xy(self.state, p) > 0.5:
                    wp = p
                    break
            
            x, y, th = self.state
            dx, dy = wp[0] - x, wp[1] - y
            target_th = math.atan2(dx, dy)
            err = (target_th - th + np.pi) % (2 * np.pi) - np.pi
            
            steer_cmd = self.clip_angle(kp_turn * err)

            self.log(f"[ctrl] state=({x:.1f},{y:.1f},{np.degrees(th):.1f}°) "
                     f"wp=({wp[0]:.1f},{wp[1]:.1f}) steer={np.degrees(steer_cmd):.1f}°")
            
            self.px.set_dir_servo_angle(np.degrees(steer_cmd))
            self.px.forward(self.POWER)

            next_state = self.step_kinematics(self.state, steer_cmd)
            if self.inside(next_state[0], next_state[1]):
                self.state = next_state

            self.map_dirty.set()
            await asyncio.sleep(0.1)

    # --- MODIFIED: Start and Stop Methods ---
    async def start(self):
        open("nav_debug.log", "w").close()
        self.px.set_cam_pan_angle(0)
        self.px.set_dir_servo_angle(0)
        await self.calibrate()
        self.map_dirty.set()
        
        # Add the new vision_loop to the tasks
        tasks = [  
            asyncio.create_task(self.ultrasonic_pan_loop(), name="sensor"),
            asyncio.create_task(self.car_map_loop(), name="carstamp"),
            asyncio.create_task(self.plan_loop(), name="planner"),
            asyncio.create_task(self.control_loop(), name="controller"),
            asyncio.create_task(self.vision_loop(), name="vision"), # NEW
            # asyncio.create_task(self.display_loop(), name="display"), # Optional
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.stop() # Ensure stop is called

    def stop(self):
        self.stop_event.set()
        self.px.stop()
        self.px.set_cam_pan_angle(0)
        # Ensure detector is stopped if not already
        if self.detector._is_running:
            self.detector.stop_detection()
        print("Navigator stopped.")

if __name__ == "__main__":
    px = Picarx(servo_pins=["P0", "P1", "P3"])
    nav = Navigator(px)
    try:
        asyncio.run(nav.start())
    except KeyboardInterrupt:
        print("Stopping via KeyboardInterrupt...")
        nav.stop()
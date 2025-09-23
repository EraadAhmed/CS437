# main.py
import asyncio
import math
from queue import PriorityQueue
import numpy as np
from picarx import Picarx
import itertools
from scipy.ndimage import zoom # You will need this for the display_loop

# Import the new object detector
from object_detection import ObjectDetector

# ---------------------------
# Utility math / models
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

# main.py

# ... (keep all your existing imports and helper classes)

class Navigator:
    # --- Tunables ---
    PAN_ANGLE = 85
    MAXREAD = 100
    SENSOR_REFRESH = 0.10
    DISPLAY_REFRESH = 0.20
    CAR_DISPLAY_REFRESH = 0.10
    PLAN_REFRESH_MIN = 0.5
    POWER = 40
    
    def __init__(
        self,
        picarx,
        width_cm=100,
        length_cm=183,
        sampling=4,
        car_width=14,
        car_length=23,
        start_pos_cm=(49, 0),
        start_angle=90.0,
        goal_xy_cm=(49, 178),
        servo_max_degs=30,
        speed=30.5
    ):
        self.px = picarx
        self.detector = ObjectDetector(model_path='efficientdet_lite0.tflite')
        
        ### FIX 1: RESTORE THE SCALED GRID COORDINATE SYSTEM ###
        self.SAMPLING = float(sampling)
        self.WIDTH_C = int(width_cm / self.SAMPLING)
        self.LENGTH_R = int(length_cm / self.SAMPLING)
        self.map_ = np.zeros((self.LENGTH_R, self.WIDTH_C), dtype=np.uint8)
        
        self.speed_scaled = float(speed / self.SAMPLING)
        self.dt_scaled = float(1.0 / speed) * self.SAMPLING
        
        self.CAR_W_SCALED = int(np.ceil(car_width / self.SAMPLING))
        self.CAR_L_SCALED = int(np.ceil(car_length / self.SAMPLING))
        
        start_pos_scaled = (start_pos_cm[0] / self.SAMPLING, start_pos_cm[1] / self.SAMPLING)
        goal_xy_scaled = (goal_xy_cm[0] / self.SAMPLING, goal_xy_cm[1] / self.SAMPLING)

        self.start_state = (start_pos_scaled[0], start_pos_scaled[1], np.radians(start_angle))
        self.state = self.start_state
        self.goal_xy = (goal_xy_scaled[0], goal_xy_scaled[1])
        
        self.servo_max_degs = float(servo_max_degs)
        self.stop_event = asyncio.Event()
        self.map_lock = asyncio.Lock()
        self.map_dirty = asyncio.Event()
        self.plan_lock = asyncio.Lock()
        self.path = []
        self.halt_event = asyncio.Event()
        self.PAN_STEP_DEG = 4.0

    def log(self, msg: str):
        with open("nav_debug.log", "a") as f:
            f.write(msg + "\n")

    def inside(self, x, y):
        ix, iy = int(round(x)), int(round(y))
        return (0 <= ix < self.WIDTH_C) and (0 <= iy < self.LENGTH_R)

    async def map_obstacle(self, x, y):
        ix, iy = int(round(x)), int(round(y))
        if not self.inside(ix, iy):
            return
        start_x, start_y, _ = self.start_state
        safe_zone_radius = self.CAR_L_SCALED * 1.5
        if math.hypot(ix - start_x, iy - start_y) < safe_zone_radius:
            return
        async with self.map_lock:
            if self.map_[iy, ix] != 1:
                self.map_[iy, ix] = 1
                self.map_dirty.set()

    async def map_car(self):
        x, y, theta = self.current_x, self.current_y, self.current_theta
        ix, iy = int(round(x)), int(round(y))
        half_l = self.CAR_L_SCALED / 2.0
        half_w = self.CAR_W_SCALED / 2.0
        corners = [(-half_w, -half_l), (half_w, -half_l), (half_w, half_l), (-half_w, half_l)]
        rotated_corners = []
        for c_x, c_y in corners:
            rot_x = c_x * math.cos(theta) - c_y * math.sin(theta)
            rot_y = c_x * math.sin(theta) + c_y * math.cos(theta)
            rotated_corners.append((ix + rot_x, iy + rot_y))
        
        min_x, max_x = int(round(min(c[0] for c in rotated_corners))), int(round(max(c[0] for c in rotated_corners)))
        min_y, max_y = int(round(min(c[1] for c in rotated_corners))), int(round(max(c[1] for c in rotated_corners)))

        async with self.map_lock:
            self.map_[self.map_ == 2] = 0
            for r in range(min_y, max_y + 1):
                for c in range(min_x, max_x + 1):
                    if self.inside(c, r) and self.map_[r, c] == 0:
                        self.map_[r, c] = 2
            self.map_dirty.set()

    async def ultrasonic_pan_loop(self):
        angle, dir_ = 0.0, self.PAN_STEP_DEG
        pan_limit = self.PAN_ANGLE
        self.px.set_cam_pan_angle(0)
        await asyncio.sleep(0.3)
        while not self.stop_event.is_set():
            self.px.set_cam_pan_angle(angle)
            await asyncio.sleep(self.SENSOR_REFRESH)
            reading_cm = self.px.ultrasonic.read()
            if not (0 < reading_cm <= self.MAXREAD):
                angle += dir_
                if abs(angle) >= pan_limit: dir_ *= -1
                continue
            
            x, y, theta = self.state
            theta_ray = theta + np.radians(angle)
            dx = (reading_cm * math.cos(theta_ray)) / self.SAMPLING
            dy = (reading_cm * math.sin(theta_ray)) / self.SAMPLING
            await self.map_obstacle(x + dx, y + dy)
            angle += dir_
            if abs(angle) >= pan_limit: dir_ *= -1
        self.px.set_cam_pan_angle(0)

    async def calibrate(self):
        angle, step, pan_limit = -90, self.PAN_STEP_DEG, 90
        self.px.set_cam_pan_angle(0)
        await asyncio.sleep(0.5)
        while angle <= pan_limit:
            self.px.set_cam_pan_angle(angle)
            await asyncio.sleep(self.SENSOR_REFRESH)
            reading_cm = self.px.ultrasonic.read()
            if (0 < reading_cm <= self.MAXREAD):
                x, y, theta = self.state
                theta_ray = theta + np.radians(angle)
                dx = (reading_cm * math.cos(theta_ray)) / self.SAMPLING
                dy = (reading_cm * math.sin(theta_ray)) / self.SAMPLING
                await self.map_obstacle(x + dx, y + dy)
            angle += step
        self.px.set_cam_pan_angle(0)
        await asyncio.sleep(0.5)

    async def display_loop(self):
        log_path = "display.log"
        factor = min(0.5, 1.0/self.SAMPLING)
        with open(log_path, "w") as f:
            while not self.stop_event.is_set():
                async with self.map_lock:
                    small_map = zoom(self.map_, zoom=factor, order=0)
                lines = ["".join(". " if val == 0 else "X " if val == 1 else "C " for val in row) for row in small_map]
                f.seek(0)
                f.truncate(0)
                f.write("\n".join(lines) + "\n")
                f.flush()
                await asyncio.sleep(self.DISPLAY_REFRESH)

    async def car_map_loop(self):
        while not self.stop_event.is_set():
            await self.map_car()
            await asyncio.sleep(self.CAR_DISPLAY_REFRESH)

    def collision(self, state):
        x, y, _ = state
        ix, iy = int(round(x)), int(round(y))
        if not self.inside(ix, iy):
            return True
        return self.map_[iy, ix] == 1

    ### FIX 2: THE DEFINITIVE, CORRECT BOUNDARY CHECK ###
    def boundary_ok(self, state):
        x, y, theta = state
        half_l = self.CAR_L_SCALED / 2.0
        half_w = self.CAR_W_SCALED / 2.0
        corners = [(-half_w, -half_l), (half_w, -half_l), (half_w, half_l), (-half_w, half_l)]
        for c_x, c_y in corners:
            rot_x = c_x * math.cos(theta) - c_y * math.sin(theta)
            rot_y = c_x * math.sin(theta) + c_y * math.cos(theta)
            map_x = x + rot_x
            map_y = y + rot_y
            if not (0 <= map_x < self.WIDTH_C and 0 <= map_y < self.LENGTH_R):
                return False
        return True

    def step_kinematics(self, current_state, steer_angle):
        x, y, theta = current_state
        d = self.speed_scaled * self.dt_scaled # This is now ~1.0 grid cell
        if abs(steer_angle) < 1e-3:
            new_x = x + d * math.cos(theta)
            new_y = y + d * math.sin(theta)
            new_theta = theta
        else:
            R = self.CAR_L_SCALED / math.tan(steer_angle)
            beta = d / R
            cx = x - math.sin(theta) * R
            cy = y + math.cos(theta) * R
            new_x = cx + math.sin(theta + beta) * R
            new_y = cy - math.cos(theta + beta) * R
            new_theta = theta + beta
        return (new_x, new_y, new_theta % (2 * math.pi))

    def reconstruct(self, node):
        path = []
        while node is not None:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))

    async def plan_once(self):
        start, goal_xy = tuple(self.state), self.goal_xy
        openq, counter = PriorityQueue(), itertools.count()
        g_cost = { (int(round(start[0])), int(round(start[1]))): 0.0 }
        
        h_weight = 2.0 
        def h_of(state): return h_weight * math.hypot(state[0] - goal_xy[0], state[1] - goal_xy[1])

        start_node = Coordinate(start, 0.0, h=h_of(start))
        openq.put((start_node.f(), next(counter), start_node))
        
        iterations = 0
        while not openq.empty() and iterations < 20000:
            iterations += 1
            _, _, cur = openq.get()

            if euclid_xy(cur.state, goal_xy) < (10.0 / self.SAMPLING):
                self.log(f"[planner] SUCCESS: Found path after {iterations} iterations.")
                return self.reconstruct(cur)

            for control in np.radians([-25, -15, 0, 15, 25]):
                nxt = self.step_kinematics(cur.state, control)
                if not self.boundary_ok(nxt) or self.collision(nxt):
                    continue
                
                nkey = (int(round(nxt[0])), int(round(nxt[1])))
                ng = cur.g + euclid_xy(cur.state, nxt)

                if nkey not in g_cost or ng < g_cost[nkey]:
                    g_cost[nkey] = ng
                    node = Coordinate(nxt, ng, h_of(nxt), parent=cur)
                    openq.put((node.f(), next(counter), node))
        
        self.log(f"[planner] ABORT: Open queue emptied or max iterations hit after {iterations} iterations.")
        return []

    async def plan_loop(self):
        while not self.stop_event.is_set():
            try: await asyncio.wait_for(self.map_dirty.wait(), timeout=self.PLAN_REFRESH_MIN)
            except asyncio.TimeoutError: pass
            self.map_dirty.clear()
            new_path = await self.plan_once()
            if new_path:
                async with self.plan_lock: self.path = new_path

    def clip_angle(self, theta_rad):
        max_rad = np.radians(self.servo_max_degs)
        return max(-max_rad, min(max_rad, theta_rad))

    async def vision_loop(self):
        self.detector.start_detection()
        while not self.stop_event.is_set():
            if self.detector.is_halt_needed(): self.halt_event.set()
            else: self.halt_event.clear()
            await asyncio.sleep(0.1)
        self.detector.stop_detection()

    async def control_loop(self):
        kp_turn = 1.5
        while not self.stop_event.is_set():
            if self.halt_event.is_set():
                self.px.stop()
                await asyncio.sleep(0.1) 
                continue
            
            async with self.plan_lock:
                path = self.path[:]
            
            if not path or euclid_xy(self.state, self.goal_xy) < (10.0 / self.SAMPLING):
                self.px.stop()
                await asyncio.sleep(0.1)
                continue

            lookahead_idx = min(len(path) - 1, 10)
            wp = path[lookahead_idx]
            x, y, th = self.state
            dx, dy = wp[0] - x, wp[1] - y
            target_th = math.atan2(dy, dx)
            
            err = (target_th - th + np.pi) % (2 * np.pi) - np.pi
            steer_cmd = self.clip_angle(kp_turn * err)
            
            self.px.set_dir_servo_angle(np.degrees(steer_cmd))
            self.px.forward(self.POWER)
            self.state = self.step_kinematics(self.state, steer_cmd)
            self.map_dirty.set()
            await asyncio.sleep(0.1)

    async def start(self):
        open("nav_debug.log", "w").close()
        self.px.set_cam_pan_angle(0)
        self.px.set_dir_servo_angle(0)
        await self.calibrate()
        self.map_dirty.set()
        
        self.path = await self.plan_once()
        if not self.path:
            self.log("[start] Initial planning failed. Stopping.")
            self.stop()
            return

        tasks = [
            asyncio.create_task(self.ultrasonic_pan_loop(), name="sensor"),
            asyncio.create_task(self.car_map_loop(), name="carstamp"),
            asyncio.create_task(self.control_loop(), name="controller"),
            asyncio.create_task(self.vision_loop(), name="vision"),
            asyncio.create_task(self.display_loop(), name="display"),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.stop()

    def stop(self):
        self.stop_event.set()
        self.px.stop()
        self.px.set_cam_pan_angle(0)
        if hasattr(self.detector, '_is_running') and self.detector._is_running:
            self.detector.stop_detection()
        print("Navigator stopped.")

# ... (keep the __main__ block the same)
if __name__ == "__main__":
    px = Picarx(servo_pins=["P0", "P1", "P3"])
    nav = Navigator(px)
    try:
        asyncio.run(nav.start())
    except KeyboardInterrupt:
        print("Stopping via KeyboardInterrupt...")
        nav.stop()
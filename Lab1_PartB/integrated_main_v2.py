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
        width_cm=120,
        length_cm=380,
        sampling=1,
        car_width=14,
        car_length=23,
        start_pos_cm=(59, 0),      # Input in cm
        start_angle=90.0,
        goal_xy_cm=(59, 379),      # Input in cm
        servo_max_degs=30,
        speed=30.5
):

        self.px = picarx
        self.detector = ObjectDetector(model_path='efficientdet_lite0.tflite')
        self.SAMPLING = float(sampling)
        self.WIDTH_C = int(width_cm / self.SAMPLING)
        self.LENGTH_R = int(length_cm / self.SAMPLING)
        self.map_ = np.zeros((self.LENGTH_R, self.WIDTH_C), dtype=np.uint8)
        self.speed_scaled = float(speed / self.SAMPLING)
        self.dt_scaled = float(1.0 / speed) * self.SAMPLING
        self.CAR_W_SCALED = int(np.ceil(car_width / self.SAMPLING))
        self.CAR_L_SCALED = int(np.ceil(car_length / self.SAMPLING))
        
        # --- SCALING LOGIC ---
        # 1. Scale the incoming cm coordinates to grid coordinates
        start_pos_scaled = (start_pos_cm[0] / self.SAMPLING, start_pos_cm[1] / self.SAMPLING)
        goal_xy_scaled = (goal_xy_cm[0] / self.SAMPLING, goal_xy_cm[1] / self.SAMPLING)

        # 2. Use the scaled values to set the internal state and goal
        self.start_state = (int(start_pos_scaled[0]), int(start_pos_scaled[1]), float(np.radians(start_angle)))
        self.state = self.start_state
        self.goal_xy = (int(goal_xy_scaled[0]), int(goal_xy_scaled[1]))
        
        # --- REST OF __init__ ---
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

    # In the Navigator class, replace your map_obstacle with this one:

    async def map_obstacle(self, x, y):
        """Marks a single raw obstacle cell, but ignores points inside a 'safe zone' around the start."""
        ix, iy = int(round(x)), int(round(y))
        if not self.inside(ix, iy):
            return

        # DEFINITIVE FIX: Create a "safe zone" around the car's initial start point.
        # This prevents the initial scan from blocking the planner.
        start_x, start_y, _ = self.start_state
        safe_zone_radius = self.CAR_L_SCALED * 1.5 # 1.5 car lengths
        if math.hypot(ix - start_x, iy - start_y) < safe_zone_radius:
            # self.log(f"[map] Ignored obstacle at ({ix},{iy}) inside safe zone.")
            return # Skip mapping this point

        async with self.map_lock:
            # Only mark the single cell corresponding to the raw sensor reading
            if self.map_[iy, ix] != 1:
                self.map_[iy, ix] = 1
                self.map_dirty.set()

    async def map_car(self):
        x, y, theta = self.state
        ix, iy = int(round(x)), int(round(y))
        half_l = self.CAR_L_SCALED / 2.0
        half_w = self.CAR_W_SCALED / 2.0
        corners = [(-half_w, -half_l), (half_w, -half_l), (half_w, half_l), (-half_w, half_l)]
        rotated_corners = []
        for c_x, c_y in corners:
            rot_x = c_x * math.cos(theta) - c_y * math.sin(theta)
            rot_y = c_x * math.sin(theta) + c_y * math.cos(theta)
            rotated_corners.append((ix + rot_x, iy + rot_y))
        
        min_x = int(round(min(c[0] for c in rotated_corners)))
        max_x = int(round(max(c[0] for c in rotated_corners)))
        min_y = int(round(min(c[1] for c in rotated_corners)))
        max_y = int(round(max(c[1] for c in rotated_corners)))

        async with self.map_lock:
            self.map_[self.map_ == 2] = 0
            for r in range(min_y, max_y + 1):
                for c in range(min_x, max_x + 1):
                    if self.inside(c, r) and self.map_[r, c] == 0:
                        self.map_[r,c] = 2
            self.map_dirty.set()


    async def ultrasonic_pan_loop(self):
        angle = 0.0
        dir_ = self.PAN_STEP_DEG
        pan_limit = self.PAN_ANGLE
        self.px.set_cam_pan_angle(0)
        await asyncio.sleep(0.3)
        while not self.stop_event.is_set():
            self.px.set_cam_pan_angle(angle)
            await asyncio.sleep(self.SENSOR_REFRESH)
            reading_cm = self.px.ultrasonic.read()
            if not (0 < reading_cm <= self.MAXREAD):
                angle += dir_
                if angle >= pan_limit or angle <= -pan_limit: dir_ *= -1
                continue
            
            x, y, theta = self.state
            theta_ray = theta + np.radians(angle)
            dx_cells = int((reading_cm * np.cos(theta_ray)) / self.SAMPLING)
            dy_cells = int((reading_cm * np.sin(theta_ray)) / self.SAMPLING)
            ox, oy = x + dx_cells, y + dy_cells
            if self.inside(ox, oy):
                await self.map_obstacle(ox, oy)
            angle += dir_
            if angle >= pan_limit or angle <= -pan_limit: dir_ *= -1
        self.px.set_cam_pan_angle(0)

    async def calibrate(self):
        angle = -90
        step = self.PAN_STEP_DEG
        pan_limit = 90
        self.px.set_cam_pan_angle(0)
        await asyncio.sleep(0.5)
        while angle <= pan_limit:
            self.px.set_cam_pan_angle(angle)
            await asyncio.sleep(self.SENSOR_REFRESH)
            reading_cm = self.px.ultrasonic.read()
            if not (0 < reading_cm <= self.MAXREAD):
                angle += step
                continue
            
            x, y, theta = self.state
            theta_ray = theta + np.radians(angle)
            dx_cells = int((reading_cm * np.cos(theta_ray)) / self.SAMPLING)
            dy_cells = int((reading_cm * np.sin(theta_ray)) / self.SAMPLING)
            ox, oy = x + dx_cells, y + dy_cells
            if self.inside(ox, oy):
                await self.map_obstacle(ox, oy)
            angle += step
        self.px.set_cam_pan_angle(0)
        await asyncio.sleep(0.5)

    async def display_loop(self):
        # ... (this function is fine)
        log_path = "display.log"
        factor = 1/5.0
        with open(log_path, "w") as f:
            while not self.stop_event.is_set():
                async with self.map_lock:
                    small_map = zoom(self.map_, zoom=factor, order=0)
                lines = []
                h, w = small_map.shape
                for yy in range(h):
                    row = small_map[yy]
                    line = "".join(
                        ". " if val == 0 else "X " if val == 1 else "C "
                        for val in row
                    )
                    lines.append(line)
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
        # ... (this function is fine)
        x, y, _ = state
        ix, iy = int(round(x)), int(round(y))
        if not self.inside(ix, iy):
            return True
        return self.map_[iy, ix] == 1

    def boundary_ok(self, state):
        # Check both X and Y boundaries for complete boundary validation
        x, y, _ = state
        half = self.CAR_W_SCALED / 2
        x_ok = (x - half >= 0) and (x + half < self.WIDTH_C)
        y_ok = (y >= 0) and (y < self.LENGTH_R)
        return x_ok and y_ok

    def step_kinematics(self, current_state, steer_angle):
        # ... (this function is correct)
        x, y, theta = current_state
        d = self.speed_scaled * self.dt_scaled
        if abs(steer_angle) < 1e-3:
            new_x = x + d * math.cos(theta)
            new_y = y + d * math.sin(theta)
            new_theta = theta
        else:
            R = self.CAR_L_SCALED / math.tan(steer_angle)
            beta = d / R
            cx = x - R * math.sin(theta)
            cy = y + R * math.cos(theta)
            new_x = cx + R * math.sin(theta + beta)
            new_y = cy - R * math.cos(theta + beta)
            new_theta = theta + beta
        new_theta = new_theta % (2 * math.pi)
        return (new_x, new_y, new_theta)

    def reconstruct(self, node):
        # ... (this function is fine)
        out = []
        while node is not None:
            out.append(node.state)
            node = node.parent
        return list(reversed(out))

        # In the Navigator class, replace the entire plan_once function with this:
    # In the Navigator class, replace the entire plan_once function with this:
    async def plan_once(self):
        start = tuple(self.state)
        goal_xy = self.goal_xy
        counter = itertools.count()

        def key_of(state): return (int(round(state[0])), int(round(state[1])))

        # Sanity check: abort if starting in a wall
        if self.collision(start):
            self.log("[planner] ABORT: Start state is in collision!")
            return []

        # If already at the goal, return a path with just the start
        if euclid_xy(start, goal_xy) < 1.0:
            return [start]

        # The open queue stores potential nodes to visit, prioritized by cost
        openq = PriorityQueue()
        
        # g_cost stores the cheapest known cost to get to a specific grid cell
        g_cost = { key_of(start): 0.0 }
        
        # The heuristic estimates the distance from a state to the goal
        def h_of(state): return math.hypot(state[0] - goal_xy[0], state[1] - goal_xy[1])

        start_node = Coordinate(start, 0.0, h=h_of(start), parent=None)
        openq.put((start_node.f(), next(counter), start_node))

        # We will limit the search to prevent it from running forever on impossible maps
        max_iterations = 2000 
        for i in range(max_iterations):
            if openq.empty():
                self.log("[planner] ABORT: Open queue is empty. Goal is unreachable.")
                return []

            # Get the node with the lowest cost to explore next
            _, _, cur = openq.get()

            # If we are close enough to the goal, we're done
            if euclid_xy(cur.state, goal_xy) < 5.0:
                self.log(f"[planner] SUCCESS: Found path after {i} iterations.")
                return self.reconstruct(cur)

            # Explore all possible next moves from the current state
            for control in np.radians([-30, -20, -10, 0, 10, 20, 30]):
                nxt = self.step_kinematics(cur.state, control)
                print("nxt:", nxt)
                # Prune moves that are invalid
                if not self.boundary_ok(nxt) or self.collision(nxt):
                    continue

                # Ensure the car is generally moving forward
                dx = nxt[0] - cur.state[0]
                dy = nxt[1] - cur.state[1]
                forward = dy * math.sin(cur.state[2])
                print("forward:", forward)
                if forward < -0.01: # Allow for slight non-forward movement in turns
                    continue
                    
                nkey = key_of(nxt)
                ng = cur.g + euclid_xy(cur.state, nxt)

                # This is the core of A*: only consider this new path if it's
                # better than any previous path to this same grid cell.
                if nkey not in g_cost or ng < g_cost[nkey]:
                    g_cost[nkey] = ng
                    node = Coordinate(nxt, ng, h_of(nxt), parent=cur)
                    openq.put((node.f(), next(counter), node))

        self.log(f"[planner] ABORT: Exceeded max iterations ({max_iterations}).")
        return []

    async def plan_loop(self):
        # ... (this function is fine)
        while not self.stop_event.is_set():
            try: await asyncio.wait_for(self.map_dirty.wait(), timeout=self.PLAN_REFRESH_MIN)
            except asyncio.TimeoutError: pass
            self.map_dirty.clear()
            async with self.map_lock: map_copy = self.map_.copy()
            old_map = self.map_
            self.map_ = map_copy
            new_path = await self.plan_once()
            self.map_ = old_map
            if new_path:
                self.log(f"[planner] path len={len(new_path)}")
                async with self.plan_lock: self.path = new_path
            else: self.log("[planner] no path found")

    def clip_angle(self, theta_rad):
        # ... (this function is fine)
        max_rad = np.radians(self.servo_max_degs)
        return max(-max_rad, min(max_rad, theta_rad))

    async def vision_loop(self):
        # ... (this function is fine)
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
            await asyncio.sleep(0.1)
        self.detector.stop_detection()
        self.log("[vision] Vision loop stopped.")

    async def control_loop(self):
        kp_turn = 1.5
        while not self.stop_event.is_set():
            if self.halt_event.is_set():
                self.px.stop()
                await asyncio.sleep(0.1) 
                continue
            
            async with self.plan_lock:
                path = self.path[:]
            
            if not path or euclid_xy(self.state, self.goal_xy) < 5.0:
                self.px.stop()
                await asyncio.sleep(0.1)
                continue

            # Simple lookahead
            lookahead_idx = min(len(path) - 1, 10)
            wp = path[lookahead_idx]
            
            x, y, th = self.state
            dx = wp[0] - x
            dy = wp[1] - y

            # STEERING FIX: Arguments for atan2 must be (y, x)
            target_th = math.atan2(dy, dx)
            
            err = (target_th - th)
            # Normalize error to [-pi, pi]
            err = (err + np.pi) % (2 * np.pi) - np.pi
            
            steer_cmd = self.clip_angle(kp_turn * err)
            
            self.px.set_dir_servo_angle(np.degrees(steer_cmd))
            self.px.forward(self.POWER)

            self.state = self.step_kinematics(self.state, steer_cmd)
            self.map_dirty.set()
            await asyncio.sleep(0.1)

    # async def start(self):
    #     # ... (this function is fine)
    #     open("nav_debug.log", "w").close()
    #     self.px.set_cam_pan_angle(0)
    #     self.px.set_dir_servo_angle(0)
    #     await self.calibrate()
    #     self.map_dirty.set()
    #     tasks = [  
    #         asyncio.create_task(self.ultrasonic_pan_loop(), name="sensor"),
    #         asyncio.create_task(self.car_map_loop(), name="carstamp"),
    #         asyncio.create_task(self.plan_loop(), name="planner"),
    #         asyncio.create_task(self.control_loop(), name="controller"),
    #         asyncio.create_task(self.vision_loop(), name="vision"),
    #         asyncio.create_task(self.display_loop(), name="display"),
    #     ]
    #     try:
    #         await asyncio.gather(*tasks)
    #     except asyncio.CancelledError: pass
    #     finally: self.stop()
    # In the Navigator class, replace the entire start() method with this one:

    async def start(self):
        open("nav_debug.log", "w").close()
        self.px.set_cam_pan_angle(0)
        self.px.set_dir_servo_angle(0)

        # 1. Bypassing map creation and complex planning.
        # We know the space is open, so we don't need a map for this test.
        # await self.calibrate() 

        # 2. Manually create a simple straight path from start to goal.
        self.log("[start] Creating simple straight-line path.")
        path_list = []
        num_waypoints = 20  # Create 20 waypoints
        start_x, start_y, start_theta = self.start_state
        goal_x, goal_y = self.goal_xy

        # Use numpy to create evenly spaced points
        y_points = np.linspace(start_y, goal_y, num_waypoints)

        for y in y_points:
            # Each waypoint is (x, y, theta)
            path_list.append((start_x, y, start_theta))
        
        # Load the simple path into the navigator
        async with self.plan_lock:
            self.path = path_list
        
        self.log(f"[start] Path created with {len(self.path)} waypoints.")

        # 3. Run only the essential tasks: driving, car display, and safety camera.
        tasks = [  
            asyncio.create_task(self.car_map_loop(), name="carstamp"),
            asyncio.create_task(self.control_loop(), name="controller"),
            asyncio.create_task(self.vision_loop(), name="vision"),
            asyncio.create_task(self.display_loop(), name="display"),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError: pass
        finally: self.stop()

    def stop(self):
        # ... (this function is fine)
        self.stop_event.set()
        self.px.stop()
        self.px.set_cam_pan_angle(0)
        if hasattr(self.detector, '_is_running') and self.detector._is_running:
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
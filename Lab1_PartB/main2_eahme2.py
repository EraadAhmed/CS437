import asyncio
import math
import sys
from queue import PriorityQueue

import numpy as np


# ---------------------------
# Utility math / models
# ---------------------------

def euclid_xy(a, b):
    """Euclidean distance on (x,y)."""
    ax, ay = a[:2]
    bx, by = b[:2]
    return math.hypot(ax - bx, ay - by)

def wrap_angle(a):
    """Wrap radians to [0, 2π)."""
    twopi = 2.0 * math.pi
    a = a % twopi
    return a

def rad(deg):
    return deg * math.pi / 180.0

def deg(rad_):
    return rad_ * 180.0 / math.pi


class Coordinate:
    def __init__(self, state, g=0.0, h=0.0, parent=None):
        self.state = state      # (x, y, theta)
        self.g = g              # cost from start
        self.h = h              # heuristic to goal
        self.parent = parent

    def f(self):
        return self.g + self.h


# ---------------------------
# THE Navigator CLASS
# ---------------------------

class Navigator:
    """
    Owns:
      - map grid (numpy, indexed [y, x])
      - car state (x, y, theta) in grid units (cells)
      - hardware (picarx)
      - async tasks: sensing, car footprint, replanning, control, display
      - a single global stop_event
    """

    # ---- Tunables you will likely tweak ----
    PAN_ANGLE = 85
    MAXREAD = 100
    SENSOR_REFRESH = 0.10
    DISPLAY_REFRESH = 0.20
    CAR_DISPLAY_REFRESH = 0.10 #refreshes car position on map
    PLAN_REFRESH_MIN = 0.10      # debounce: minimum time between plan attempts
    WAYPOINT_REACHED_DIST = 1.0     # in grid cells
    REPLAN_ON_MAP = True
    POWER = 40
    def __init__(
        self,
        picarx,
        width_cm=120,
        length_cm=380,
        sampling = 1,
        car_width=14,
        car_length=23,
        start_pos=(60, 0),
        start_angle=0.0,
        goal_xy=(60, 380),
        servo_max_degs=30,
        speed=30.5,
        dt = float(1/30.5)
    ):
        # Hardware
        self.px = picarx

        # Spatial scaling
        self.SAMPLING = float(sampling)  # cm per cell
        self.WIDTH_C = int(width_cm / self.SAMPLING)
        self.LENGTH_R = int(length_cm / self.SAMPLING)  # rows (y)
        self.map_ = np.zeros((self.LENGTH_R, self.WIDTH_C), dtype=np.uint8)  # 0 free, 1 obstacle, 2 car
        self.speed_scaled = float(speed / self.SAMPLING)  # cells per second
        self.dt_scaled = float(dt*self.SAMPLING)  # seconds per control step
        # Car geometry (in cm, but we stamp on grid cells)
        self.CAR_W_CM = float(car_width)
        self.CAR_L_CM = float(car_length)
        self.CAR_W_SCALED = int(np.ceil(self.CAR_W_CM / self.SAMPLING))
        self.CAR_L_SCALED = int(np.ceil(self.CAR_L_CM / self.SAMPLING))


        # State (grid cells + radians)
        x0, y0 = start_pos
        self.state = (int(x0), int(y0), float(np.radians(start_angle))) 
        self.goal_xy = (int(goal_xy[0]), int(goal_xy[1]))

        # Control limits
        self.servo_max_degs = float(servo_max_degs)  # max steering

        # Concurrency primitives
        self.stop_event = asyncio.Event()
        self.map_lock = asyncio.Lock()
        self.map_dirty = asyncio.Event()  # set whenever mapping updates obstacles/car cells

        # Planning state
        self.path = []           # list[(x,y,theta)] in cells
        self.plan_lock = asyncio.Lock()
        self.last_plan_ts = 0.0

        # Panning config based on resolution (rule-of-thumb)
        # Δθ ≈ arcsin(cell_size / max_range). Clamp sensible step.
        # Panning step based on sampling resolution
        if self.SAMPLING == 1:
            self.PAN_STEP_DEG = 2.0
        elif self.SAMPLING == 2:
            self.PAN_STEP_DEG = 4.0
        elif self.SAMPLING == 5:
            self.PAN_STEP_DEG = 6.0
        else:
            # fallback: arcsin heuristic
            cell_cm = self.SAMPLING
            step = np.degrees(math.asin(max(0.01, min(0.99, cell_cm / self.MAXREAD))))
            self.PAN_STEP_DEG = max(1.0, min(5.0, step))


    # ---------------------------
    # Map helpers (indexing [y,x])
    # ---------------------------

    def inside(self, x, y):
        return (0 <= x < self.WIDTH_C) and (0 <= y < self.LENGTH_R)

    async def map_obstacle(self, x, y):
        if not self.inside(x, y):
            return
        async with self.map_lock:
            if self.map_[y, x] != 1:
                self.map_[y, x] = 1
                self.map_dirty.set()

    async def clear_car(self):
        async with self.map_lock:
            self.map_[self.map_ == 2] = 0
            self.map_dirty.set()

    async def map_car(self):
        """Stamp a rectangle with center at car's (x,y) extending backward in +y direction."""
        x, y, _ = self.state
        half_w = self.CAR_W_SCALED // 2
        # Car rectangle: width along x, length along +y behind current (simple top-down footprint)
        y0 = y - self.CAR_L_SCALED
        y1 = y
        x0 = x - half_w
        x1 = x + half_w
        async with self.map_lock:
            # Clear old car first (optional—display task also clears)
            self.map_[self.map_ == 2] = 0
            for yy in range(y0, y1 + 1):
                if 0 <= yy < self.LENGTH_R:
                    for xx in range(x0, x1 + 1):
                        if 0 <= xx < self.WIDTH_C:
                            # Don’t overwrite obstacles
                            if self.map_[yy, xx] != 1:
                                self.map_[yy, xx] = 2
            self.map_dirty.set()

    # ---------------------------
    # Sensor loop (ultrasonic pan)
    # ---------------------------

    async def ultrasonic_pan_loop(self):
        """Continuously pan camera/ultrasonic and mark obstacles in map."""
        angle = 0.0
        dir_ = self.PAN_STEP_DEG  # degrees/step
        pan_limit = self.PAN_ANGLE

        # Center sensor
        self.px.set_cam_pan_angle(0)
        await asyncio.sleep(0.3)

        while not self.stop_event.is_set():
            self.px.set_cam_pan_angle(angle)
            await asyncio.sleep(self.SENSOR_REFRESH)
            reading_cm = self.px.ultrasonic.read()  # in cm
            if not (0 < reading_cm <= self.MAXREAD):
                # advance and continue
                angle += dir_
                if angle >= pan_limit or angle <= -pan_limit:
                    dir_ *= -1
                continue
            
            await asyncio.sleep((reading_cm * 1e-2) / 343.0) #343 speed of sound

            # Convert sensor ray to map cell (relative to car)
            # Car's heading affects forward (cos) & sideways (sin).
            # Here we treat angle as pan relative to car's forward, not world.
            x, y, theta = self.state
            theta_ray = theta + np.radians(angle)
            dx_cells = int((reading_cm * np.sin(theta_ray) / self.SAMPLING))
            dy_cells = int((reading_cm * np.cos(theta_ray) / self.SAMPLING))

            ox = x + dx_cells
            oy = y + dy_cells

            # Clamp inside and stamp obstacle
            if self.inside(ox, oy):
                await self.map_obstacle(ox, oy)

            # Advance sweep
            angle += dir_
            if angle >= pan_limit or angle <= -pan_limit:
                dir_ *= -1

        # recentre when stopping
        self.px.set_cam_pan_angle(0)
        await asyncio.sleep(0.2)

    async def calibrate(self):
        """
        Do one sweep from -90° to +90° before starting async loops.
        Marks obstacles in map once at startup.
        """
        angle = -90
        step = self.PAN_STEP_DEG
        pan_limit = 90

        # Center sensor briefly
        self.px.set_cam_pan_angle(0)
        await asyncio.sleep(0.5)

        while angle <= pan_limit:
            self.px.set_cam_pan_angle(angle)
            await asyncio.sleep(self.SENSOR_REFRESH)

            reading_cm = self.px.ultrasonic.read()  # distance in cm

            if not (0 < reading_cm <= self.MAXREAD):
                angle += step
                continue

            # small wait proportional to distance
            await asyncio.sleep((reading_cm * 1e-2) / 343.0)

            # Same coordinate math as ultrasonic_pan_loop
            x, y, theta = self.state
            theta_ray = theta + np.radians(angle)
            dx_cells = int((reading_cm * np.sin(theta_ray)) / self.SAMPLING)
            dy_cells = int((reading_cm * np.cos(theta_ray)) / self.SAMPLING)

            ox = x + dx_cells
            oy = y + dy_cells

            if self.inside(ox, oy):
                await self.map_obstacle(ox, oy)

            angle += step

        # Reset pan to center
        self.px.set_cam_pan_angle(0)
        await asyncio.sleep(0.5)

    # ---------------------------
    # Display loop (ANSI refresh)
    # ---------------------------

    async def display_loop(self):
        while not self.stop_event.is_set():
            sys.stdout.write("\033[H")  # move cursor to top-left
            async with self.map_lock:
                h, w = self.map_.shape
                for yy in range(h):
                    row = self.map_[yy]
                    # 0='.', 1='X', 2='C'
                    line = []
                    for val in row:
                        if val == 0:
                            line.append(". ")
                        elif val == 1:
                            line.append("X ")
                        else:
                            line.append("C ")
                    sys.stdout.write("".join(line) + "\n")
            sys.stdout.flush()
            await asyncio.sleep(self.DISPLAY_REFRESH)

    # ---------------------------
    # Car “footprint” loop
    # ---------------------------

    async def car_map_loop(self):
        while not self.stop_event.is_set():
            await self.map_car()
            await asyncio.sleep(self.CAR_DISPLAY_REFRESH)

    # ---------------------------
    # Planning – Hybrid A* (grid kinematics)
    # ---------------------------

    def collision(self, state):
        """Return True if collides (outside map or into obstacle)."""
        x, y, _ = state
        if not self.inside(x, y):
            return True
        # Treat any obstacle cell as collision
        return self.map_[y, x] == 1

    def boundary_ok(self, state):
        """Simple lateral clearance check using car width (optional)."""
        x, y, _ = state
        half = self.CAR_W_SCALED // 2
        return (x - half >= 0) and (x + half < self.WIDTH_C)
    def snap_angle(self, theta_rad):
        """Snap theta (radians) to nearest allowed angle in degrees."""
        theta_deg = np.degrees(theta_rad)
        STEER = np.array([-30, -20, -10, 0, 10, 20, 30])
        nearest = STEER[np.argmin(np.abs(STEER - theta_deg))]
        return np.radians(nearest)

    def step_kinematics(self, current_state, steer_angle):
    # Kinematic update for a bicycle model; returns (x, y, theta) after dt.
        """
        Update position based on velocity and heading.
        current_pos: [x, y]
        velocity: units/sec (e.g. cm/sec)
        heading_angle: radians (0 = facing east, pi/2 = north)
        dt: time since last update
        """
        d = self.speed_scaled * self.dt_scaled
        beta = d  * np.tan(steer_angle)/self.CAR_L_SCALED  # angular change
        if np.abs(beta) < 0.001: #straight line approx
            dx = int(d * np.cos(current_state[2]))
            dy = int(d * np.sin(current_state[2]))
        else:
            R = d / beta  # radius of curvature
            dx = int(R * np.sin(current_state[2] + beta))
            dy = int(-1*R * np.cos(current_state[2] + beta))
        new_x = int(current_state[0]  + dx)
        new_y = int(current_state[1] + dy)
        
        new_head_angle = (current_state[2] + beta) % (2 * np.pi)  # Normalize angle
        nearest = self.snap_angle(new_head_angle)
        return (new_x, new_y, nearest)

    def reconstruct(self, node):
        out = []
        while node is not None:
            out.append(node.state)
            node = node.parent
        return list(reversed(out))

    def plan_once(self):
        """Hybrid A* with heading bins limited to ±30°."""
        start = tuple(self.state)                   # (x, y, theta)
        goal = (self.goal_xy[0], self.goal_xy[1], 0.0)

        # Early exit
        if (start[0], start[1]) == self.goal_xy:
            return [start]

        g_cost_key = {(start): 0.0}
        openq = PriorityQueue()
        g = 0
        h = euclid_xy(start, goal)
        start_node = Coordinate(start, g, h= h, parent=None)
        openq.put((start_node.f(), start_node))

        closed = set()

        # Steering controls in degrees
        STEERS = [-30, -20, -10, 0, 10, 20, 30]

        while not openq.empty():
            current_f , current_node = openq.get()
            if current_node.state[:2] ==  self.goal_xy:
                return self.reconstruct(current_node)
            closed.add(current_node.state)
            if current_node in closed:
                continue
            closed.add(current_node)
            for control in [-30, -20, -10, 0, 10, 20, 30]:
                next_state = self.step_kinematics(current_node.state, control)

                if next_state in closed:
                    continue
                if not self.boundary_ok(next_state) or self.collision(next_state):
                    continue

                ng = current_node.g + euclid_xy(current_node.state, next_state)
                nh = euclid_xy(next_state, goal)
                if (next_state not in g_cost_key) or (ng < g_cost_key[next_state]):
                    g_cost_key[next_state] = ng
                    next_node = Coordinate(next_state, ng, nh, parent= current_node)
                    openq.put((next_node.f(),next_node))


        return []

    async def plan_loop(self):
        """Continuously recompute path when map changes or at a minimum interval."""
        while not self.stop_event.is_set():
            # Wait for either map change or small periodic trigger
            try:
                await asyncio.wait_for(self.map_dirty.wait(), timeout=self.PLAN_REFRESH_MIN)
            except asyncio.TimeoutError:
                pass
            self.map_dirty.clear()

            # Snapshot map to avoid locking during planning
            async with self.map_lock:
                map_copy = self.map_.copy()
            # Temporarily swap map for collision checks
            old_map = self.map_
            self.map_ = map_copy
            new_path = self.plan_once()
            self.map_ = old_map

            async with self.plan_lock:
                self.path = new_path

    # ---------------------------
    # Controller – follow waypoints
    # ---------------------------

    async def control_loop(self):
        """
        Very simple waypoint follower:
          - reads current planned path
          - picks next waypoint
          - steers toward it with a proportional controller
          - sends commands to picarx
        """
        kp_turn = 1.5  # deg per rad of heading error (tune)
        base_speed = self.speed_scaled

        while not self.stop_event.is_set():
            # If at goal or no path, stop
            async with self.plan_lock:
                path = self.path[:]

            if not path or (self.state[0], self.state[1]) == self.goal_xy:
                self.px.stop()
                await asyncio.sleep(0.1)
                continue

            # Choose a lookahead waypoint (skip the very first if it's current cell)
            wp = None
            for p in path[1:4]:  # small lookahead window
                if euclid_xy(self.state, p) > 0.5:
                    wp = p
                    break
            if wp is None:
                wp = path[-1]

            # Heading to waypoint
            x, y, th = self.state
            dx = wp[0] - x
            dy = wp[1] - y
            target_th = math.atan2(dy, dx)
            err = (target_th - th)
            # Normalize error to [-pi, pi]
            err = (err + np.pi) % (2 * np.pi) - np.pi
            raw_cmd = kp_turn * err  # still in radians
            steer_cmd = self.snap_angle(raw_cmd)
            # Send commands to car
            self.px.set_dir_servo_angle(np.degrees(steer_cmd))
            self.px.forward(self.POWER)

            next_state = self.step_kinematics(self.state, steer_cmd)
            # Only accept if inside bounds
            if self.inside(next_state[0], next_state[1]):
                self.state = next_state

            # Mark map as dirty so car stamp updates position
            self.map_dirty.set()

            await asyncio.sleep(0.1)

    # ---------------------------
    # Public API
    # ---------------------------

    def set_goal(self, x, y):
        self.goal_xy = (int(x), int(y))
        self.map_dirty.set()

    async def start(self):
        # Center camera
        self.px.set_cam_pan_angle(0)
        # Do one initial calibration sweep
        await self.calibrate()
        tasks = [  
            asyncio.create_task(self.ultrasonic_pan_loop(), name="sensor"),
            asyncio.create_task(self.car_map_loop(), name="carstamp"),
            asyncio.create_task(self.plan_loop(), name="planner"),
            asyncio.create_task(self.control_loop(), name="controller"),
            asyncio.create_task(self.display_loop(), name="display"),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.px.stop()
            self.px.set_cam_pan_angle(0)

    def stop(self):
        self.stop_event.set()

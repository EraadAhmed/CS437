import asyncio
import math
import sys
from queue import PriorityQueue
import numpy as np
from picamera2 import Picamera2
from picarx import Picarx
import itertools
import os 
from scipy.ndimage import zoom


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
        width_cm=100,
        length_cm=183,
        sampling = 1,
        car_width=14,
        car_length=23,
        start_pos=(49, 0),
        start_angle=0.0,
        goal_xy=(49, 182),
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

    def log(self, msg: str):
        """Append debug messages to nav_debug.log."""
        with open("nav_debug.log", "a") as f:
            f.write(msg + "\n")


    def inside(self, x, y):
        ix, iy = int(round(x)), int(round(y))
        return (0 <= ix < self.WIDTH_C) and (0 <= iy < self.LENGTH_R)



    async def map_obstacle(self, x, y):
        """Mark a single obstacle cell if inside map bounds."""
        ix, iy = int(round(x)), int(round(y))
        if not self.inside(ix, iy):
            return
        async with self.map_lock:
            if self.map_[iy, ix] != 1:
                self.map_[iy, ix] = 1
                self.map_dirty.set()


    async def clear_car(self):
        async with self.map_lock:
            self.map_[self.map_ == 2] = 0
            self.map_dirty.set()

    async def map_car(self):
        """Stamp a rectangle with center at car's (x,y) extending backward in +y direction."""
        x, y, _ = self.state
        ix, iy = int(round(x)), int(round(y))
        half_w = self.CAR_W_SCALED // 2
        # Car rectangle: width along x, length along +y behind current (simple top-down footprint)
        y0 = iy - self.CAR_L_SCALED
        y1 = iy
        x0 = ix - half_w
        x1 = ix + half_w
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
        """Write downsampled map to a log file for easier viewing."""
        log_path = "display.log"

        factor = 1/2.0  # shrink by 5x
        with open(log_path, "w") as f:
            while not self.stop_event.is_set():
                async with self.map_lock:
                    # downsample using nearest-neighbor
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

                # Rewind and overwrite file
                f.seek(0)
                f.truncate(0)
                f.write("\n".join(lines) + "\n")
                f.flush()

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
        ix, iy = int(round(x)), int(round(y))
        if not self.inside(ix, iy):
            return True
        # Treat any obstacle cell as collision
        return self.map_[iy, ix] == 1

    def boundary_ok(self, state):
        """Simple lateral clearance check using car width (optional)."""
        x, y, _ = state
        ix, iy = int(round(x)), int(round(y))
        half = self.CAR_W_SCALED // 2
        return (ix - half >= 0) and (ix + half < self.WIDTH_C)
    def snap_angle(self, theta_rad):
        """Snap theta (radians) to nearest allowed angle in degrees."""
        theta_deg = np.degrees(theta_rad)
        STEER = np.array([-30, -20, -10, 0, 10, 20, 30])
        nearest = STEER[np.argmin(np.abs(STEER - theta_deg))]
        return np.radians(nearest)

    def step_kinematics(self, current_state, steer_angle):
        """
        Bicycle model kinematics.
        state = (x, y, theta)
        steer_angle: radians
        Returns (new_x, new_y, new_theta).
        """
        x, y, theta = current_state
        d = self.speed_scaled * self.dt_scaled  # distance traveled in this step

        if abs(steer_angle) < 1e-3:  # ~straight line
            new_x = x + d * math.cos(theta)
            new_y = y + d * math.sin(theta)
            new_theta = theta
        else:
            R = self.CAR_L_SCALED / math.tan(steer_angle)   # turning radius
            beta = d / R                                   # heading change

            # Instantaneous center of rotation (ICC)
            cx = x - R * math.sin(theta)
            cy = y + R * math.cos(theta)

            # Rotate around ICC by beta
            new_x = cx + R * math.sin(theta + beta)
            new_y = cy - R * math.cos(theta + beta)
            new_theta = theta + beta

        # Normalize and snap heading
        new_theta = new_theta % (2 * math.pi)
        nearest = self.snap_angle(new_theta)

        self.log(
            f"[step] from=({x:.1f},{y:.1f},{math.degrees(theta):.1f}°) "
            f"-> to=({new_x:.1f},{new_y:.1f},{math.degrees(nearest):.1f}°) "
            f"d={d:.2f}, steer={math.degrees(steer_angle):.1f}°"
        )
        return (new_x, new_y, nearest)


    def reconstruct(self, node):
        out = []
        while node is not None:
            out.append(node.state)
            node = node.parent
        return list(reversed(out))

    async def plan_once(self):
        """Hybrid A* with heading bins limited to ±30°, forward-only, cell-keyed."""
        start = tuple(self.state)                   # (x, y, th) floats
        goal_xy = self.goal_xy                      # (gx, gy)
        counter = itertools.count()

        def key_of(state):
            x, y, _ = state
            return (int(round(x)), int(round(y)))

        # already at goal cell?
        if key_of(start) == goal_xy:
            return [start]

        openq = PriorityQueue()
        g_cost = { key_of(start): 0.0 }

        # simple Euclidean heuristic to goal cell-center
        def h_of(state):
            sx, sy, _ = state
            gx, gy = goal_xy
            return math.hypot(sx - gx, sy - gy)

        start_node = Coordinate(start, 0.0, h=h_of(start), parent=None)
        openq.put((start_node.f(), next(counter), start_node))

        closed = set()
        STEERS = [-30, -20, -10, 0, 10, 20, 30]

        while not openq.empty():
            _, _, cur = openq.get()
            ckey = key_of(cur.state)

            if ckey in closed:
                continue
            closed.add(ckey)

            gx, gy = goal_xy
            if abs(cur.state[0] - gx) <= 3.0 and abs(cur.state[1] - gy) <= 3.0:
                return self.reconstruct(cur)

            for control in STEERS:
                nxt = self.step_kinematics(cur.state, np.radians(control))

                # forward-only filter (dot product with heading)
                dx = nxt[0] - cur.state[0]
                dy = nxt[1] - cur.state[1]
                forward = dx * math.sin(cur.state[2]) + dy * math.cos(cur.state[2])
                self.log(f"[plan] control={control}°, dx={dx:.2f}, dy={dy:.2f}, fwd={forward:.2f}")
                if forward < 1e-3:   # instead of <=0
                    continue

                nkey = key_of(nxt)

                # prune invalid or blocked
                if not self.boundary_ok(nxt) or self.collision(nxt):
                    continue

                ng = cur.g + euclid_xy(cur.state, nxt)
                if nkey not in g_cost or ng < g_cost[nkey]:
                    g_cost[nkey] = ng
                    node = Coordinate(nxt, ng, h_of(nxt), parent=cur)
                    openq.put((node.f(), next(counter), node))

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
            if new_path:
                self.log(f"[planner] path len={len(new_path)} "
                    f"first=({new_path[0][0]:.1f},{new_path[0][1]:.1f}) "
                    f"last=({new_path[-1][0]:.1f},{new_path[-1][1]:.1f})")
            else:
                self.log("[planner] no path found")

            async with self.plan_lock:
                self.path = new_path

    # ---------------------------
    # Controller – follow waypoints
    # ---------------------------

    def clip_angle(self, theta_rad):
        max_rad = np.radians(self.servo_max_degs)
        return max(-max_rad, min(max_rad, theta_rad))

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
            for p in path[2:8]:  # small lookahead window
                if euclid_xy(self.state, p) > 0.5:
                    wp = p
                    break
            if wp is None:
                wp = path[-1]

            # Heading to waypoint
            x, y, th = self.state
            dx = wp[0] - x
            dy = wp[1] - y
            target_th = math.atan2(dx, dy)
            err = (target_th - th)
            # Normalize error to [-pi, pi]
            err = (err + np.pi) % (2 * np.pi) - np.pi
            raw_cmd = kp_turn * err  # still in radians
            steer_cmd = self.clip_angle(raw_cmd)

            self.log(f"[ctrl] state=({self.state[0]:.1f},{self.state[1]:.1f},"
            f"{np.degrees(self.state[2]):.1f}°) "
            f"wp=({wp[0]:.1f},{wp[1]:.1f}) "
            f"steer={np.degrees(steer_cmd):.1f}° "
            f"err={np.degrees(err):.1f}°")
            
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
        open("nav_debug.log", "w").close() # clear log
        open("display.log", "w").close() # clear log
        self.px.set_cam_pan_angle(0)
        self.px.set_dir_servo_angle(0)
        # Do one initial calibration sweep
        await self.calibrate()
        self.map_dirty.set()
            # ---- Plan once here ----
        self.log("[start] Planning initial path...")
        new_path = self.plan_once()
        if new_path:
            self.log(f"[start] path len={len(new_path)}")
            async with self.plan_lock:
                self.path = new_path
        else:
            self.log("[start] no path found")
        tasks = [  
            #asyncio.create_task(self.ultrasonic_pan_loop(), name="sensor"),
            asyncio.create_task(self.car_map_loop(), name="carstamp"),
            #asyncio.create_task(self.plan_loop(), name="planner"),
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


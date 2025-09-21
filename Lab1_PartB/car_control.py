import numpy as np
import time
from picamera2 import Picamera2
from picarx import Picarx
from queue import PriorityQueue




# Represents a node/state in search: (x, y, theta) with costs and a backpointer.
class Coordinate:
    def __init__(self, state, g=0, h=0, parent=None):
        self.state = state        # (x, y, theta)
        self.g = g                # cost from start
        self.h = h                # heuristic to goal
        self.parent = parent      # backpointer
    def f(self):
        return self.g + self.h
# Euclidean distance between two (x, y) positions used as motion cost/heuristic.
def cost(state1, state2):
    p1 = np.array(state1[:2])  # take (x, y)
    p2 = np.array(state2[:2])
    return np.linalg.norm(p1 - p2)
# Reconstructs a path by following parent pointers from goal back to start.
def reconstruct_path(node):
    path = []
    while node is not None:
        path.append(node.state)
        node = node.parent
    return list(reversed(path))

# Kinematic update for a bicycle model; returns (x, y, theta) after dt.
def next_state_gen(current_state, velocity, dt, steer_angle, Car_Length):
    """
    Update position based on velocity and heading.
    current_pos: [x, y]
    velocity: units/sec (e.g. cm/sec)
    heading_angle: radians (0 = facing east, pi/2 = north)
    dt: time since last update
    """
    d = velocity * dt
    beta = d  * np.tan(steer_angle)/Car_Length  # angular change
    if np.abs(beta) < 0.001: #straight line approx
        dx = int(d * np.cos(current_state[2]))
        dy = int(d * np.sin(current_state[2]))
    else:
        R = d / beta  # radius of curvature
        dx = int(R * np.sin(current_state[2] + beta))
        dy = int(-1*R * np.cos(current_state[2] + beta))
    new_x = current_state[0]  + dx
    new_y = current_state[1] + dy
    
    new_head_angle = (current_state[2] + beta) % (2 * np.pi)  # Normalize angle

    return (new_x, new_y, new_head_angle)
# Returns True if the state collides (outside map or in an occupied cell).
def collision_check(state, grid):
    x, y, theta = state
    
    # Make sure it's inside the map
    if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
        return True   # outside map = collision
    
    # Check occupancy
    if grid[x, y] == 1:
        return True   # obstacle
    
    return False
# Checks lateral bounds with a vehicle width margin; True if within bounds.
def boundary_check(state, WIDTH, Car_Width):
    if state[0]+(Car_Width/2) >= WIDTH:
        return False
    elif state[0]-(Car_Width/2) < 0:
        return False
    else: 
        return True
    

# Hybrid A* planner: searches over (x, y, theta) with simple kinematics.
# Returns a path (list of states) from start_state to final_state or raises on failure.
async def hybrid_a_star(start_state, final_state, map, WIDTH, Car_Width, SPEED, delta_t):
    open = PriorityQueue()
    closed = set()
    g_start = 0
    g_cost_key = {}
    h_start = cost(start_state, final_state)
    start_node = Coordinate(start_state, g_start, h= h_start, parent=None)
    g_cost_key[start_state] = g_start
    open.put((start_node.f(), start_node))
    while not open.empty():
        current_f , current_node = open.get()
        if current_node.state[:2] == final_state[:2]:
            return reconstruct_path(current_node)
        closed.append(current_node.state)
        for control in [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]:  # e.g., steering angles, velocities
            next_state = next_state_gen(current_node.state,SPEED, delta_t, control)
            if collision_check(next_state, map) and boundary_check(next_state, WIDTH, Car_Width):
                continue

            next_g = current_node.g + cost(current_node.state, next_state)
            next_h = cost(next_state, final_state)
            

            if next_state in closed:
                continue

            if (next_state not in g_cost_key) or (next_g < g_cost_key[next_state]):
                g_cost_key[next_state] = next_g
                next_node = Coordinate(next_state, next_g, next_h, parent= current_node)
                open.put((next_node.f,next_node))

    raise RuntimeError("Hybrid A* failed: no path found")


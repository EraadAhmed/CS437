import numpy as np
import time
from picamera2 import Picamera2
from picarx import Picarx
from queue import PriorityQueue

width_scaled = WIDTH/5
length_scaled = LENGTH/5
x_mid_scaled = int(width_scaled/2)
CAR_Width_scaled = int(np.ceil(CAR_Width/5))
CAR_Length_scaled = int(np.ceil(CAR_Length/5))

#vehicle positioning constants
MAX_READ = 100 #max ultrasonic reading considered for mapping to be 1 
MAXREAD_SCALED = MAX_READ/5
SPEED = 10 #cm/sec
POWER = 40 #0-100
delta_t = 0.25 #1 cm / velocity 
SafeDistance = 25 #in cm
DangerDistance = 10 #in cm



class Coordinate:
    def __init__(self, state, g=0, h=0, parent=None):
        self.state = state        # (x, y, theta)
        self.g = g                # cost from start
        self.h = h                # heuristic to goal
        self.parent = parent      # backpointer
    def f(self):
        return self.g + self.h

def heuristic(state1, state2):
    p1 = np.array(state1[:2])  # take (x, y)
    p2 = np.array(state2[:2])
    return np.linalg.norm(p1 - p2)

def hybrid_a_star(start_state, goal_state, map):
    open = PriorityQueue()
    closed = []
    h_start = heuristic(start_state, goal_state)
    start_node = Coordinate(start_state, 0, h= h_start, parent=None)
    
    open.put(start_node)
    while open:
        current_node = open.get()
        if current_node.state[:2] == goal_state[:2]:
            return path(current_node)
        closed.append(current_node)
        for control in [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]:  # e.g., steering angles, velocities
            next_node = next_state_gen(current_node.state, )

            # if COLLISION(next_state, map):
            #     continue

            # next_g ← current_node.g + cost(current_node.state, next_state)
            # next_h ← heuristic(next_state, goal_state)
            # next_node ← Node(state=next_state, g=next_g, h=next_h, parent=current_node)

            # if discretize(next_state) ∈ closed_list:
            #     continue

            # if not open_list.contains_better(next_node):
            #     open_list.push(next_node)

    raise RuntimeError("Hybrid A* failed: no path found")
#copilot generated below:
    # while open:
    #     open.sort(key=lambda node: node.f())
    #     current = open.pop(0)
        
    #     if current.state[:2] == goal_state[:2]:  # goal check
    #         path = []
    #         while current:
    #             path.append(current.state)
    #             current = current.parent
    #         return path[::-1]  # return reversed path
        
    #     close.append(current.state)
        
    #     # Generate successors (this is a placeholder, actual motion model needed)
    #     for delta_theta in [-15, 0, 15]:  # example steering angles
    #         new_theta = (current.state[2] + delta_theta) % 360
    #         new_x = current.state[0] + np.cos(np.radians(new_theta))
    #         new_y = current.state[1] + np.sin(np.radians(new_theta))
    #         new_state = (new_x, new_y, new_theta)
            
    #         if (0 <= new_x < map.shape[0] and 0 <= new_y < map.shape[1] and 
    #             map[int(new_x)][int(new_y)] == 0 and new_state not in close):
                
    #             g_new = current.g + 1  # assuming uniform cost
    #             h_new = heuristic(new_state, goal_state)
    #             successor = Coordinate(new_state, g_new, h_new, parent=current)
                
    #             if not any(node.state == new_state and node.g <= g_new for node in open):
    #                 open.append(successor)

def next_state_gen(current_state, velocity, dt, steer_angle):
    """
    Update position based on velocity and heading.
    current_pos: [x, y]
    velocity: units/sec (e.g. cm/sec)
    heading_angle: radians (0 = facing east, pi/2 = north)
    dt: time since last update
    """
    d = velocity * dt
    beta = d  * np.tan(steer_angle)/ CAR_Length_scaled  # angular change
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

################################################################################
#  PsuedoCode for A star algorithm
# function HYBRID_A_STAR(start_state, goal_state, map):
#     open_list ← priority queue ordered by f = g + h
#     closed_list ← empty set
    

#     start_node ← Node(state=start_state, g=0, h=heuristic(start_state, goal_state), parent=null)
#     open_list.push(start_node)

#     while open_list is not empty:
#         current_node ← open_list.pop_lowest_f()

#         if is_goal(current_node.state, goal_state):
#             return RECONSTRUCT_PATH(current_node)

#         closed_list.add(discretize(current_node.state))

#         for control in feasible_controls:  # e.g., steering angles, velocities
#             next_state ← MOTION_MODEL(current_node.state, control)

#             if COLLISION(next_state, map):
#                 continue

#             next_g ← current_node.g + cost(current_node.state, next_state)
#             next_h ← heuristic(next_state, goal_state)
#             next_node ← Node(state=next_state, g=next_g, h=next_h, parent=current_node)

#             if discretize(next_state) ∈ closed_list:
#                 continue

#             if not open_list.contains_better(next_node):
#                 open_list.push(next_node)

#     raise RuntimeError("Hybrid A* failed: no path found")
###############################################################################



function RECONSTRUCT_PATH(node):
    path ← empty list
    while node ≠ null:
        path.prepend(node.state)
        node ← node.parent
    return path

import threading
import time
import numpy as np
from picarx import Picarx
from computer_vision import calibrate, ultrasonic_pan_loop
from car_control import hybrid_a_star, reconstruct_path

#PHYSICAL CONSTANTS
WIDTH = 120 #width of map
LENGTH = 380 #length of map
X_MID = 60 #midpoint on x axis
CAR_Width = 14
CAR_Length = 23

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


# Shared occupancy grid
grid = None
grid_lock = threading.Lock()

def sensing_thread(px, start_pos, current_pos):
    global grid
    # First do an initial calibration sweep
    with grid_lock:
        grid = calibrate(px, start_pos, grid)

    # Then keep sweeping left-right forever
    ultrasonic_pan_loop(px, current_pos, grid, MAXREAD=100, WIDTH=120)

def planning_thread(px, start_state, goal_state):
    global grid
    while True:
        if grid is None:
            time.sleep(0.1)
            continue
        
        # Copy the grid safely
        with grid_lock:
            local_grid = np.copy(grid)

        try:
            path = hybrid_a_star(start_state, goal_state, local_grid)
            print("Path found:", path)

            # Follow the path step by step
            for (x, y, theta) in path:
                px.set_dir_servo_angle(np.degrees(theta))
                px.forward(40)
                time.sleep(0.25)

                # if an obstacle shows up, break and replan
                with grid_lock:
                    if local_grid[x, y] == 1:
                        print("Obstacle detected! Replanning...")
                        px.forward(0)
                        break
        except RuntimeError:
            print("No path found, retrying...")

def main():
    px = Picarx()

    # Initialize grid
    global grid
    grid = np.zeros((24, 76))   # example: scaled map 5cm/cell

    start_state = (12, 0, 0)    # (x, y, θ)
    goal_state = (12, 75, 0)

    # Start sensing and planning threads
    t1 = threading.Thread(target=sensing_thread, args=(px, start_state[:2]))
    t2 = threading.Thread(target=planning_thread, args=(px, start_state, goal_state))
    t1.daemon = True
    t2.daemon = True
    t1.start()
    t2.start()

    # Keep main alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
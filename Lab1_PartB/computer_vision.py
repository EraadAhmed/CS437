import numpy as np
from picamera2 import Picamera2
from picarx import Picarx
import time

WIDTH = 117
HEIGHT = 380
X_MID = 53.5
MAX_reading = 100
SPEED = 10

CAR_Width = 14
CAR_Length = 23



map = np.zeros((WIDTH, HEIGHT)) #makes a 2d grid
start_pos = (X_MID-1, 0)
end_pos = (X_MID-1, HEIGHT-1)
current_pos = start_pos


def calibrate(picarx, current_pos, map):
    #calibrate positioning of ultrasonic sensor relative velocity
    
    angle = -60
    while angle <= 60:
        time.sleep(0.1)
        picarx.set_cam_pan_angle(angle)
        reading = picarx.ultrasonic.read()
        time.sleep(0.01)
        if angle == 0 and reading <= MAX_reading:
            map[current_pos[0]][(reading-1)+current_pos[1]] = 1
        elif angle < 0:
            if(reading <= current_pos[0]/np.cos(angle) and reading <= MAX_reading):
                object_xval = int(current_pos[0] - reading*np.cos(angle))
                object_yval = int(current_pos[1] + reading*np.sin(np.abs(angle)))
                map[object_xval][object_yval] = 1
        elif angle > 0:
            if(reading <= (WIDTH - current_pos[0])/np.cos(angle) and reading <= MAX_reading):
                object_xval = int(current_pos[0] + reading*np.cos(angle))
                object_yval = int(current_pos[1] + reading*np.sin(np.abs(angle)))
                map[object_xval][object_yval] = 1
        angle += 10
    picarx.set_cam_pan_angle(0)
    return map

import time
import numpy as np

def ultrasonic_pan_loop(picarx, current_pos, map_grid, MAX_reading=100, WIDTH=200):
    """
    Continuously pan ultrasonic sensor between -60 and +60 degrees.
    Updates map_grid with detected objects.
    If a detected cell is already occupied, updates current_pos instead.
    """
    angle = -60
    direction = 10   # step size (positive = sweeping right, negative = sweeping left)

    while True:   # keep running while car is on
        time.sleep(0.1)
        picarx.set_cam_pan_angle(angle)
        reading = picarx.ultrasonic.read()
        time.sleep(0.01)

        object_xval, object_yval = None, None

        # --- Straight ahead ---
        if angle == 0 and reading <= MAX_reading:
            object_xval = current_pos[0]
            object_yval = (reading - 1) + current_pos[1]

        # --- Left side (< 0) ---
        elif angle < 0:
            if reading <= current_pos[0]/np.cos(np.radians(angle)) and reading <= MAX_reading:
                object_xval = int(current_pos[0] - reading * np.cos(np.radians(angle)))
                object_yval = int(current_pos[1] + reading * np.sin(np.radians(abs(angle))))

        # --- Right side (> 0) ---
        elif angle > 0:
            if reading <= (WIDTH - current_pos[0])/np.cos(np.radians(angle)) and reading <= MAX_reading:
                object_xval = int(current_pos[0] + reading * np.cos(np.radians(angle)))
                object_yval = int(current_pos[1] + reading * np.sin(np.radians(abs(angle))))

        # # --- If we got valid coordinates ---
        # if object_xval is not None and object_yval is not None:
        #     try:
        #         if map_grid[object_xval][object_yval] == 1:
        #             # Already marked → treat as localization update
        #             current_pos = [object_xval, object_yval]
        #         else:
        #             # Mark obstacle
        #             map_grid[object_xval][object_yval] = 1
        #     except IndexError:
        #         # Skip if out of map bounds
        #         pass doesnt work 

        # --- Update angle sweep direction ---
        angle += direction
        if angle >= 60 or angle <= -60:
            direction *= -1   # reverse sweep direction

    # Reset pan angle when loop breaks (if ever)
    picarx.set_cam_pan_angle(0)
    return map_grid, current_pos

def car_pixels(map, current_pos):
    for i in range(int(current_pos[0]-CAR_Width/2), int(current_pos[0]+CAR_Width/2)):
        for j in range(int(current_pos[1]), int(current_pos[1]+CAR_Length)):
            map[i][j] = 2 #denotes its the car
    return map

def location_update(picarx, current_pos):
    #updates the current position of the car
    distance = picarx.ultrasonic.read()
    if distance > 20:
        current_pos = (current_pos[0], current_pos[1] + SPEED)
    return current_pos



def main():
    try:
        px = Picarx()
        px = Picarx(ultrasonic_pins=['D2','D3']) # tring, echo
       
        while True:
            distance = round(px.ultrasonic.read(), 2)
            print("distance: ",distance)
            if distance >= SafeDistance:
                px.set_dir_servo_angle(0)
                px.forward(POWER)
            elif distance >= DangerDistance:
                px.set_dir_servo_angle(30)
                px.forward(POWER)
                time.sleep(0.1)
            else:
                px.set_dir_servo_angle(-30)
                px.backward(POWER)
                time.sleep(0.5)

    finally:
        px.forward(0)


if __name__ == "__main__":
    main()
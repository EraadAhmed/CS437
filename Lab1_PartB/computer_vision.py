import numpy as np
from picamera2 import Picamera2
from picarx import Picarx
import time
import sys, asyncio
from threading import Event
#PHYSICAL Realworld Constants (not used)
# Width = 120 cm
# Length = 380 cm
# midpoint_width = 60 cm
# car width = 14 cm
# car length = 23 
# max ultrasonic read distance = 100 cm 

WIDTH =  60
LENGTH = 150
X_MID = 30
CAR_WIDTH = 14
CAR_LENGTH = 23
MAXREAD = 100 #wont change since we will convert x and y in post processing
SAMPLING = 1 #everything scaled down by factor of 5


async def calibrate(picarx, current_pos, map_grid):
    #calibrate positioning of ultrasonic sensor relative velocity
    await asyncio.sleep(0.5)
    picarx.set_cam_pan_angle(0)
    time.sleep(3)
    angle = -90
    while angle <= 90:
        picarx.set_cam_pan_angle(angle)
        await asyncio.sleep(0.1)
        reading = picarx.ultrasonic.read() #gets distance reading in cm
        # print("reading:", reading)
        if 0 < reading <= MAXREAD:
            await asyncio.sleep((reading*1e-2)/343)
            await asyncio.sleep(0.1)
        else: 
            angle += 2
            continue

        # print("actually passing through")
        
        zero = np.nonzero(map_grid)
        print(angle, reading, list(zip(*zero)))
        object_xval, object_yval = None, None

            # --- Straight ahead ---
        if angle == 0:
            object_xval = current_pos[0]
            reading_0 =  int(reading / SAMPLING) #scales it toto nearest divisible by 5
            object_yval = (reading_0) + current_pos[1]
            if object_yval < int(LENGTH/SAMPLING):
                # print("tuple: ", (object_xval, object_yval), end="\n\n\n\n")
                map_grid[object_yval][object_xval] = 1
                #print(object_xval, object_yval)

        # --- Left side (< 0)  or right side (>0) ---
        else:
            object_xval = int((current_pos[0] + reading * np.sin(np.radians(angle)))/SAMPLING) # np.sign will decide to add to subtract from the x coord
            object_yval = int((current_pos[1] + reading * np.cos(np.radians(angle)))/SAMPLING)
            if (0 <= object_xval < int(WIDTH/SAMPLING) and 0 <= object_yval < int(LENGTH/SAMPLING)):
                # print("tuple: ", (object_xval, object_yval), end="\n\n\n\n")
                map_grid[object_yval][object_xval] = 1
                #print(object_xval, object_yval)
            
        angle += 2
    # print("fin angle:", angle)
    picarx.set_cam_pan_angle(0)
    await asyncio.sleep(2)
    return map_grid



async def ultrasonic_pan_loop(picarx, current_pos, map_grid, stop_event):
    """
    Continuously pan ultrasonic sensor between -60 and +60 degrees.
    Updates map_grid with detected objects.
    If a detected cell is already occupied, updates current_pos instead.
    """
    angle = 0
    direction = 2   # step size (positive = sweeping right, negative = sweeping left)
    pan_angle = 90
    while not stop_event.is_set():   # keep running while car is on
        picarx.set_cam_pan_angle(angle)
        await asyncio.sleep(0.1)
        reading = picarx.ultrasonic.read() #gets distance reading in cm
        # print("reading:", reading)
        if 0 < reading <= MAXREAD:
            await asyncio.sleep((reading*1e-2)/343)
            await asyncio.sleep(0.1)
        else: 
            angle += 2
            continue

        # print("actually passing through")
        
        # zero = np.nonzero(map_grid)
        # print(angle, reading, list(zip(*zero)))
        # object_xval, object_yval = None, None

            # --- Straight ahead ---
        if angle == 0:
            object_xval = current_pos[0]
            reading_0 =  int(reading / SAMPLING) #scales it toto nearest divisible by 5
            object_yval = (reading_0) + current_pos[1]
            if object_yval < int(LENGTH/SAMPLING):
                # print("tuple: ", (object_xval, object_yval), end="\n\n\n\n")
                map_grid[object_yval][object_xval] = 1
                #print(object_xval, object_yval)

        # --- Left side (< 0)  or right side (>0) ---
        else:
            object_xval = int((current_pos[0] + reading * np.sin(np.radians(angle)))/SAMPLING) # np.sign will decide to add to subtract from the x coord
            object_yval = int((current_pos[1] + reading * np.cos(np.radians(angle)))/SAMPLING)
            if (0 <= object_xval < int(WIDTH/SAMPLING) and 0 <= object_yval < int(LENGTH/SAMPLING)):
                # print("tuple: ", (object_xval, object_yval), end="\n\n\n\n")
                map_grid[object_yval][object_xval] = 1
                #print(object_xval, object_yval)
        # --- Update angle sweep direction ---
        angle += direction
        if angle >= pan_angle or angle <= -1*pan_angle:
            direction *= -1   # reverse sweep direction

    # Reset pan angle when loop breaks (if ever)
    picarx.set_cam_pan_angle(0)
    await asyncio.sleep(3)
    return map_grid   


async def car_pixels(map_grid, current_pos, stop_event):
    while not stop_event.is_set():
        samp_car_width = int(np.ceil(CAR_WIDTH/SAMPLING))
        samp_car_length = int(np.ceil(CAR_LENGTH/SAMPLING))
        for i in range(int(current_pos[0]-samp_car_width/2), int(current_pos[0]+samp_car_width/2)):
            for j in range(int(current_pos[1]-samp_car_length), int(current_pos[1])):
                if 0 <= j < int(LENGTH/SAMPLING) and 0 <= i < int(WIDTH/SAMPLING):
                    map_grid[j][i] = 2
        await asyncio.sleep(0.1)


async def print_map(map_grid, stop_event):
    while not stop_event.is_set():
        sys.stdout.write("\033[H")  # Move cursor to top-left
        height, width = map_grid.shape
        for i in range(height):
            for j in range(width):
                if map_grid[i][j] == 0:
                    sys.stdout.write(". ")
                elif map_grid[i][j] == 1:
                    sys.stdout.write("X ")
                elif map_grid[i][j] == 2:
                    sys.stdout.write("C ")
            sys.stdout.write("\n")
        sys.stdout.flush()

        await asyncio.sleep(0.2)  # refresh rate
# if __name__ == "__main__":
#     import asyncio
#     picar = Picarx(servo_pins=["P0", "P1", "P3"])
#     samp_width = int(WIDTH/SAMPLING)
#     samp_length = int(LENGTH/SAMPLING)
#     map_ = np.zeros((samp_length, samp_width))
#     sam_mid = int(X_MID/SAMPLING)
#     current_pos = (int(30/SAMPLING), 0)
#     map_ = calibrate(picar, current_pos, map_)
#     print_map(map_)
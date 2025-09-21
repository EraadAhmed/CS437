import numpy as np
from picamera2 import Picamera2
from picarx import Picarx
import time

#PHYSICAL Realworld Constants (not used)
# Width = 120 cm
# Length = 380 cm
# midpoint_width = 60 cm
# car width = 14 cm
# car length = 23 
# max ultrasonic read distance = 100 cm 

WIDTH = 24 
LENGTH = 76 
X_MID = 12 
CAR_WIDTH = 14
CAR_LENGTH = 23
MAXREAD = 50 #wont change since we will convert x and y in post processing
SAMPLING = 1 #everything scaled down by factor of 5


def calibrate(picarx, current_pos, map_grid):
    #calibrate positioning of ultrasonic sensor relative velocity
    angle = -60
    while angle <= 60:
        time.sleep(0.1)
        picarx.set_cam_pan_angle(angle) 
        reading = int(picarx.ultrasonic.read()) #gets distance reading in cm
        
        time.sleep(0.01)
        if angle == 0 and reading <= MAXREAD:
            reading_0 =  int(np.ceil(reading / 5.0) * 5) #scales it toto nearest divisible by 5
            map[current_pos[0]][reading_0+current_pos[1]] = 1
        elif angle < 0:
            if(reading <= current_pos[0]/np.cos(angle) and reading <= MAXREAD):
                object_xval = int(np.ceil((current_pos[0] - reading * np.cos(np.radians(angle)))/5)*5)
                object_yval = int(np.ceil((current_pos[1] + reading * np.sin(np.radians(abs(angle))))/5)*5)
                map[object_xval][object_yval] = 1
        elif angle > 0:
            if(reading <= (WIDTH - current_pos[0])/np.cos(angle) and reading <= MAXREAD):
                object_xval = int(np.ceil((current_pos[0] + reading * np.cos(np.radians(angle)))/5)*5)
                object_yval = int(np.ceil((current_pos[1] + reading * np.sin(np.radians(abs(angle))))/5)*5)
                map[object_xval][object_yval] = 1
        
        angle += 5
    picarx.set_cam_pan_angle(0)
    return map



async def ultrasonic_pan_loop(picarx, current_pos, map_grid):
    """
    Continuously pan ultrasonic sensor between -60 and +60 degrees.
    Updates map_grid with detected objects.
    If a detected cell is already occupied, updates current_pos instead.
    """
    angle = 0
    direction = 2   # step size (positive = sweeping right, negative = sweeping left)
    pan_angle = 70
    while True:   # keep running while car is on
        time.sleep(0.1)
        picarx.set_cam_pan_angle(angle)
        reading = picarx.ultrasonic.read() #gets distance reading in cm
        zero = np.nonzero(map_grid)
        print(angle, reading, list(zip(*zero)))
        time.sleep(current_pos[0]/np.cos(np.radians(90-angle))/343)

        object_xval, object_yval = None, None

        # --- Straight ahead ---
        if 0 < reading <= MAXREAD:
            if angle == 0:
                object_xval = current_pos[0]
                reading_0 =  int(np.ceil(reading / SAMPLING)) #scales it toto nearest divisible by 5
                object_yval = (reading_0) + current_pos[1]
                if object_yval < LENGTH:
                    print("tuple: ", (object_xval, object_yval), end="\n\n\n\n")
                    map_grid[object_xval][object_yval] = 1

            # --- Left side (< 0)  or right side (>0) ---
            else:
                object_xval = int(np.ceil((current_pos[0] + int(np.sign(angle)) * reading * np.sin(np.radians(90-abs(angle))))/SAMPLING)) # np.sign will decide to add to subtract from the x coord
                object_yval = int(np.ceil((current_pos[1] + reading * np.cos(np.radians(90 - abs(angle))))/SAMPLING))
                if (0 <= object_xval < WIDTH and 0 <= object_yval < LENGTH):
                    print("tuple: ", (object_xval, object_yval), end="\n\n\n\n")
                    map_grid[object_xval][object_yval] = 1

            print(object_xval, object_yval)
        # --- Update angle sweep direction ---
        angle += direction
        if angle >= pan_angle or angle <= -1*pan_angle:
            direction *= -1   # reverse sweep direction

    # Reset pan angle when loop breaks (if ever)
    picarx.set_cam_pan_angle(0)
    return map_grid, current_pos   


async def car_pixels(map_grid, current_pos):
    samp_car_width = int(np.ceil(CAR_WIDTH/SAMPLING))
    samp_car_length = int(np.ceil(CAR_LENGTH/SAMPLING))
    for i in range(int(current_pos[0]-samp_car_width/2), int(current_pos[0]+samp_car_width/2)):
        for j in range(int(current_pos[1]-samp_car_length), int(current_pos[1])):
            if(j >= 0 and j < LENGTH/SAMPLING):
                map_grid[i][j] = 2 #denotes its the car
    return map_grid

async def print_map(map_grid):
    samp_width = int(np.ceil(WIDTH/SAMPLING))
    samp_length = int(np.ceil(LENGTH/SAMPLING))
    for i in range(samp_width):
        for j in range(samp_length):
            if map_grid[i][j] == 0:
                print(".", end=" ")
            elif map_grid[i][j] == 1:
                print("X", end=" ")
            elif map_grid[i][j] == 2:
                print("C", end=" ")
        print()
    print()

if __name__ == "__main__":
    import asyncio
    picar = Picarx(servo_pins=["P0", "P1", "P3"])
    map_ = np.zeros((WIDTH/SAMPLING, LENGTH/SAMPLING))
    current_pos = (X_MID/SAMPLING, 0)
    asyncio.run(ultrasonic_pan_loop(picar, current_pos, map_))
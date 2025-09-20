import numpy as np
from picamera2 import Picamera2
from picarx import Picarx
import time



def calibrate(picarx, current_pos, map, MAXREAD, WIDTH):
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



async def ultrasonic_pan_loop(picarx, current_pos, map_grid, MAXREAD, WIDTH):
    """
    Continuously pan ultrasonic sensor between -60 and +60 degrees.
    Updates map_grid with detected objects.
    If a detected cell is already occupied, updates current_pos instead.
    """
    angle = -60
    direction = 5   # step size (positive = sweeping right, negative = sweeping left)

    while True:   # keep running while car is on
        time.sleep(0.1)
        picarx.set_cam_pan_angle(angle)
        reading = picarx.ultrasonic.read() #gets distance reading in cm
        time.sleep(0.01)

        object_xval, object_yval = None, None

        # --- Straight ahead ---
        if angle == 0 and reading <= MAXREAD:
            object_xval = current_pos[0]
            reading_0 =  int(np.ceil(reading / 5.0) * 5) #scales it toto nearest divisible by 5
            object_yval = (reading_0) + current_pos[1]

        # --- Left side (< 0) ---
        elif angle < 0:
            if reading <= current_pos[0]/np.cos(np.radians(angle)) and reading <= MAXREAD:
                object_xval = int(np.ceil((current_pos[0] - reading * np.cos(np.radians(angle)))/5)*5)
                object_yval = int(np.ceil((current_pos[1] + reading * np.sin(np.radians(abs(angle))))/5)*5)

        # --- Right side (> 0) ---
        elif angle > 0:
            if reading <= (WIDTH - current_pos[0])/np.cos(np.radians(angle)) and reading <= MAXREAD:
                object_xval = int(np.ceil((current_pos[0] + reading * np.cos(np.radians(angle)))/5)*5)
                object_yval = int(np.ceil((current_pos[1] + reading * np.sin(np.radians(abs(angle))))/5)*5)
                

        map[object_xval][object_yval] = 1
        # --- Update angle sweep direction ---
        angle += direction
        if angle >= 60 or angle <= -60:
            direction *= -1   # reverse sweep direction

    # Reset pan angle when loop breaks (if ever)
    picarx.set_cam_pan_angle(0)
    return map_grid, current_pos   


async def car_pixels(map, current_pos,Car_Width, Car_Length, LENGTH):
    for i in range(int(current_pos[0]-Car_Width/2), int(current_pos[0]+Car_Width/2)):
        for j in range(int(current_pos[1]-Car_Length), int(current_pos[1])):
            if(j >= 0 and j < LENGTH):
                map[i][j] = 2 #denotes its the car
    return map

async def print_map(map, WIDTH, LENGTH):
    for i in range(WIDTH):
        for j in range(LENGTH):
            if map[i][j] == 0:
                print(".", end=" ")
            elif map[i][j] == 1:
                print("X", end=" ")
            elif map[i][j] == 2:
                print("C", end=" ")
        print()
    print()

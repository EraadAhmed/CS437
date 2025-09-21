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


    #Performs a one-time ultrasonic sensor sweep from -60° to +60°.
    #Updates the occupancy grid (map_grid) with detected obstacle positions 
    #based on current car position. Returns the updated map.
    
def calibrate(picarx, current_pos, map_grid):
    #calibrate positioning of ultrasonic sensor relative velocity
    picarx.set_cam_pan_angle(angle)
    time.sleep(3)
    angle = -60
    while angle <= 60:
        time.sleep(0.1)
        picarx.set_cam_pan_angle(angle)
        reading = picarx.ultrasonic.read() #gets distance reading in cm
        if 0 < reading <= MAXREAD:
            time.sleep((reading*1e-2)/343)
        else: continue
        
        zero = np.nonzero(map_grid)
        print(angle, reading, list(zip(*zero)))
        object_xval, object_yval = None, None

         # --- Straight ahead ---
        if angle == 0:
            object_xval = current_pos[0]
            reading_0 =  int(np.ceil(reading / SAMPLING)) #scales it toto nearest divisible by 5
            object_yval = (reading_0) + current_pos[1]
            if object_yval < LENGTH:
                print("tuple: ", (object_xval, object_yval), end="\n\n\n\n")
                map_grid[object_xval][object_yval] = 1
                print(object_xval, object_yval)

        # --- Left side (< 0)  or right side (>0) ---
        else:
            object_xval = int(np.ceil((current_pos[0] + int(np.sign(angle)) * reading * np.sin(np.radians(90-abs(angle))))/SAMPLING)) # np.sign will decide to add to subtract from the x coord
            object_yval = int(np.ceil((current_pos[1] + reading * np.cos(np.radians(90 - abs(angle))))/SAMPLING))
            if (0 <= object_xval < WIDTH and 0 <= object_yval < LENGTH):
                print("tuple: ", (object_xval, object_yval), end="\n\n\n\n")
                map_grid[object_xval][object_yval] = 1
                print(object_xval, object_yval)
            
        angle += 5
    picarx.set_cam_pan_angle(0)
    time.sleep(2)
    return map


  #Continuously sweeps the ultrasonic sensor left and right between -pan_angle and +pan_angle.
    #Reads distances, converts them into grid coordinates, and updates map_grid.
    #Runs indefinitely while the car is on. Also updates current_pos if needed.
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

        if 0 < reading <= MAXREAD:
            time.sleep((reading*1e-2)/343)
        else: continue

        zero = np.nonzero(map_grid)
        print(angle, reading, list(zip(*zero)))
        object_xval, object_yval = None, None

         # --- Straight ahead ---
        if angle == 0:
            object_xval = current_pos[0]
            reading_0 =  int(np.ceil(reading / SAMPLING)) #scales it toto nearest divisible by 5
            object_yval = (reading_0) + current_pos[1]
            if object_yval < LENGTH:
                print("tuple: ", (object_xval, object_yval), end="\n\n\n\n")
                map_grid[object_xval][object_yval] = 1
                print(object_xval, object_yval)

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
    time.sleep(3)
    return map_grid, current_pos   
 
    #Marks the area occupied by the car on the occupancy grid.
    #Uses CAR_WIDTH and CAR_LENGTH (scaled by SAMPLING) to determine which cells 
    #are filled with the 'car' marker (value = 2). Returns updated map_grid.
    
async def car_pixels(map_grid, current_pos):
    samp_car_width = int(np.ceil(CAR_WIDTH/SAMPLING))
    samp_car_length = int(np.ceil(CAR_LENGTH/SAMPLING))
    for i in range(int(current_pos[0]-samp_car_width/2), int(current_pos[0]+samp_car_width/2)):
        for j in range(int(current_pos[1]-samp_car_length), int(current_pos[1])):
            if(j >= 0 and j < LENGTH/SAMPLING):
                map_grid[i][j] = 2 #denotes its the car
    return map_grid
  #Prints a visual representation of the occupancy grid.
    #Symbols:
        #'.' = empty cell
        #'X' = detected obstacle
        #'C' = car position/footprint
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
  #Program entry point.
   # - Initializes Picarx hardware.
    #- Creates an empty occupancy grid.
    #- Sets the car's starting position at the map midpoint.
    #- Runs calibration sweep.
    #- Prints the resulting map.
if __name__ == "__main__":
    import asyncio
    picar = Picarx(servo_pins=["P0", "P1", "P3"])
    map_ = np.zeros((WIDTH/SAMPLING, LENGTH/SAMPLING))
    current_pos = (X_MID/SAMPLING, 0)
    map_ = calibrate(picar, current_pos, map_)
    print_map(map_)

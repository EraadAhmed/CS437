import time
import numpy as np
from picarx import Picarx
from computer_vision import calibrate, ultrasonic_pan_loop
from car_control import hybrid_a_star, reconstruct_path

#PHYSICAL Realworld Constants (not used)
# Width = 120 cm
# Length = 380 cm
# midpoint_width = 60 cm
# car width = 14 cm
# car length = 23 
# max ultrasonic read distance = 100 cm 


#Scaled constants used for programming (used) everything scaled down by factor of 5 and rounded up if need be
WIDTH = 24 
LENGTH = 76 
X_MID = 12 
CAR_Width = int(np.ceil(15/5))
CAR_Length = int(np.ceil(23/5))

#vehicle positioning constants also scaled for fitting with 25 cm^2 unit grid
MAX_READ = 20 #max ultrasonic reading considered for mapping to be 1 
SPEED = 10 #5 cm/sec
POWER = 40 #will be the value that gives us 5 cm a second 
delta_t = 0.25 #time needed to move 5cm 


#Start and End CONSTANTS
start_pos = (X_MID-1, 0)
end_pos = (X_MID-1, X_MID-1)

#changing fields
map = np.zeros((WIDTH, LENGTH)) #makes a 2d grid scaled at 5cm^2 per cell
current_pos = start_pos
heading_angle = 0



def main():


    # Keep main alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
import numpy as np
from picamera2 import Picamera2
from picarx import Picarx
import time


#Start and End CONSTANTS
start_pos = (x_mid_scaled-1, 0)
end_pos = (x_mid_scaled-1, length_scaled-1)

#changing fields
map = np.zeros((width_scaled, length_scaled)) #makes a 2d grid scaled at 5cm^2 per cell
current_pos = start_pos
heading_angle = 0


def calibrate(picarx, current_pos, map):
    #calibrate positioning of ultrasonic sensor relative velocity
    angle = -60
    while angle <= 60:
        time.sleep(0.1)
        picarx.set_cam_pan_angle(angle) 
        reading = int(picarx.ultrasonic.read()) #gets distance reading in cm
        reading =  int(np.ceil(reading / 5.0) * 5) #scales it toto nearest divisible by 5
        time.sleep(0.01)
        if angle == 0 and reading <= MAXREAD:
            map[current_pos[0]][reading+current_pos[1]] = 1
        elif angle < 0:
            if(reading <= current_pos[0]/np.cos(angle) and reading <= MAXREAD):
                object_xval = int(current_pos[0] - reading*np.cos(np.radians(angle)))
                object_yval = int(current_pos[1] + reading*np.sin(np.radians(np.abs(angle))))
                map[object_xval][object_yval] = 1
        elif angle > 0:
            if(reading <= (WIDTH - current_pos[0])/np.cos(angle) and reading <= MAXREAD):
                object_xval = int(current_pos[0] + reading*np.cos(np.radians(angle)))
                object_yval = int(current_pos[1] + reading*np.sin(np.abs(np.radians(angle))))
                map[object_xval][object_yval] = 1
        
        angle += 5
    picarx.set_cam_pan_angle(0)
    return map



def ultrasonic_pan_loop(picarx, current_pos, map_grid, MAXREAD, WIDTH):
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
        reading =  int(np.ceil(reading / 5.0) * 5) #scales it toto nearest divisible by 5
        time.sleep(0.01)

        object_xval, object_yval = None, None

        # --- Straight ahead ---
        if angle == 0 and reading <= MAXREAD:
            object_xval = current_pos[0]
            object_yval = (reading - 1) + current_pos[1]

        # --- Left side (< 0) ---
        elif angle < 0:
            if reading <= current_pos[0]/np.cos(np.radians(angle)) and reading <= MAXREAD:
                object_xval = int(current_pos[0] - reading * np.cos(np.radians(angle)))
                object_yval = int(current_pos[1] + reading * np.sin(np.radians(abs(angle))))

        # --- Right side (> 0) ---
        elif angle > 0:
            if reading <= (WIDTH - current_pos[0])/np.cos(np.radians(angle)) and reading <= MAXREAD:
                object_xval = int(current_pos[0] + reading * np.cos(np.radians(angle)))
                object_yval = int(current_pos[1] + reading * np.sin(np.radians(abs(angle))))
                
        # --- Update angle sweep direction ---
        angle += direction
        if angle >= 60 or angle <= -60:
            direction *= -1   # reverse sweep direction

    # Reset pan angle when loop breaks (if ever)
    picarx.set_cam_pan_angle(0)
    return map_grid, current_pos   


def car_pixels(map, current_pos):
    for i in range(int(current_pos[0]-CAR_Width_scaled/2), int(current_pos[0]+CAR_Width_scaled/2)):
        for j in range(int(current_pos[1]-CAR_Length_scaled), int(current_pos[1])):
            if(j >= 0 and j < LENGTH):
                map[i][j] = 2 #denotes its the car
    return map





def update_position(current_pos, velocity, heading_angle, dt, steer_angle):
    """
    Update position based on velocity and heading.
    current_pos: [x, y]
    velocity: units/sec (e.g. cm/sec)
    heading_angle: radians (0 = facing east, pi/2 = north)
    dt: time since last update
    """
    d = velocity * dt
    beta = d  * np.tan(steer_angle)/ CAR_Length  # angular change
    if np.abs(beta) < 0.001: #straight line approx
        dx = int(d * np.cos(heading_angle))
        dy = int(d * np.sin(heading_angle))
    else:
        R = d / beta  # radius of curvature
        dx = int(R * np.sin(heading_angle + beta))
        dy = int(-1*R * np.cos(heading_angle + beta))
    new_x = current_pos[0] + dx
    new_y = current_pos[1] + dy
    
    heading_angle = (heading_angle + beta) % (2 * np.pi)  # Normalize angle
    return [new_x, new_y], heading_angle

def boundary_check(current_pos, WIDTH=WIDTH, LENGTH=LENGTH):
    if current_pos[0]+(CAR_Width/2) >= WIDTH:
        return False
    elif current_pos[0]-(CAR_Width/2) < 0:
        return False
    else: 
        return True
    



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
                time.sleep(0.1)

    finally:
        px.forward(0)


if __name__ == "__main__":
    main()
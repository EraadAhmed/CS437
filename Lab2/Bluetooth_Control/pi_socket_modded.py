import socket
import threading
from collections import deque
import signal
import time
from robot_hat import ADC
from picarx import Picarx

server_addr = '88:A2:9E:2B:1A:4F'
server_port = 1

buf_size = 1024
PRECISION = 12 #12 bit precision of the values in the registers

battery_adcport = ADC('A4')

client_sock = None
server_sock = None
sock = None

exit_event = threading.Event()

message_queue = deque([])
output = ""

dq_lock = threading.Lock()
output_lock = threading.Lock()

power = 0
steer_angle = 0
ploss = 3 # loss due to environmental factors
pmax = 100 #max power for motor
max_speed = 65 #cm/s

px = Picarx(servo_pins=['P0','P1','P3'])

def handler(signum, frame):
    exit_event.set()

signal.signal(signal.SIGINT, handler)

def get_pi_temperature():
    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
        temp_millideg = int(f.readline().strip())
    return temp_millideg / 1000.0   # °C

def increase_velocity():
    global power
    if power < 100:
        power += 10
    power = min(power, 100)

def decrease_velocity():
    global power
    if power > -100:
        power -= 10
    power = max(power,-100)

def turn_right():
    global steer_angle
    if steer_angle < 30:
        steer_angle += 5
    steer_angle = min(steer_angle, 30)
    px.set_dir_servo_angle(steer_angle)
        
def turn_left():
    global steer_angle
    if steer_angle > -30:
        steer_angle -= 5
    steer_angle = max(steer_angle, -30)
    px.set_dir_servo_angle(steer_angle)

def calculate_speed():
    global ploss
    global max_speed
    global pmax
    global power
    
    abs_power = abs(power)
    if abs_power == 0:
        return 0.0
    
    speed = max_speed * (abs_power - ploss) / (pmax - ploss)
    speed = min(speed, max_speed)
    # Apply sign of power (positive for forward, negative for backward)
    return speed if power >= 0 else -speed


def start_client():
    global server_addr
    global server_port
    global server_sock
    global sock
    global exit_event
    global message_queue
    global output
    global dq_lock
    global output_lock
    global battery_adcport
    global steer_angle
    global power
    server_sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    server_sock.bind((server_addr, server_port))
    server_sock.listen(1)
    server_sock.settimeout(10)
    sock, address = server_sock.accept()
    print("Connected")
    server_sock.settimeout(None)
    sock.setblocking(0)
    while not exit_event.is_set():
        if dq_lock.acquire(blocking=False):
            if(len(message_queue) > 0):
                try:
                    sent = sock.send(bytes(message_queue[0], 'utf-8'))
                except Exception as e:
                    exit_event.set()
                    continue
                if sent < len(message_queue[0]):
                    message_queue[0] = message_queue[0][sent:]
                else:
                    message_queue.popleft()
            dq_lock.release()
        
        if output_lock.acquire(blocking=False):
            data = ""
            try:
                try:
                    data = sock.recv(1024).decode('utf-8')
                    px.set_dir_servo_angle(0)
                    if data == "TEMP\r\n":
                        temp = get_pi_temperature()
                        dq_lock.acquire()
                        message_queue.append("pi temp:" + str(temp) + "C " + " \r\n")
                        dq_lock.release()
                    elif(data == "BATLEV\r\n"):
                        raw_read = battery_adcport.read()
                        bat_level = float(raw_read/((2**PRECISION)-1))*100
                        dq_lock.acquire()
                        message_queue.append("battery percent: " + str(round(bat_level, 2)) + "% " + " \r\n")
                        dq_lock.release()
                    elif(data == "SPD\r\n"):
                        spd = calculate_speed()
                        dq_lock.acquire()
                        message_queue.append("Curr SPD (cm/s): " + str(spd) + " cm/s" + " \r\n")
                        dq_lock.release()
                    elif(data == "RT\r\n"):
                        turn_right()
                        dq_lock.acquire()
                        message_queue.append("STEER RT, CURR ANG: " + str(steer_angle) + " degs" + " \r\n")
                        dq_lock.release()
                        px.forward(power)
                    elif(data == "LT\r\n"):
                        turn_left()
                        dq_lock.acquire()
                        message_queue.append("STEER LT, CURR ANG: " + str(steer_angle) + " degs" + " \r\n")
                        dq_lock.release()
                        px.forward(power)

                    elif(data == "FWD\r\n"):
                        steer_angle = 0
                        px.set_dir_servo_angle(steer_angle)
                        increase_velocity()
                        if power == 0:
                            px.forward(power)
                            dq_lock.acquire()
                            message_queue.append("Vehicle Stopped" + "\r\n")
                            dq_lock.release()
                        elif power > 0:
                            px.forward(power)
                            dq_lock.acquire()
                            message_queue.append("SPEEDING UP, POWER: " + str(power) +  " \r\n")
                            dq_lock.release()
                        elif power < 0:
                            px.backward(abs(power))
                            dq_lock.acquire()
                            message_queue.append("SLOWING DOWN, POWER: " + str(power) +  " \r\n")
                            dq_lock.release()

                    elif(data == "BWD\r\n"):
                        steer_angle = 0
                        px.set_dir_servo_angle(steer_angle)
                        decrease_velocity()
                        if power == 0:
                            px.forward(power)
                            dq_lock.acquire()
                            message_queue.append("Vehicle Stopped" + "\r\n")
                            dq_lock.release()
                        elif power > 0:
                            px.forward(power)
                            dq_lock.acquire()
                            message_queue.append("SLOWING DOWN, POWER: " + str(power) +  " \r\n")
                            dq_lock.release()
                        elif power < 0:
                            px.backward(abs(power))
                            dq_lock.acquire()
                            message_queue.append("SPEEDING UP, POWER: " + str(power) +  " \r\n")
                            dq_lock.release()

                except socket.error as e:
                    assert(1==1)
                    #no data

            except Exception as e:
                exit_event.set()
                continue
            output += data
            output_split = output.split("\r\n")
            for i in range(len(output_split) - 1):
                print(output_split[i])
            output = output_split[-1]
            output_lock.release()
    
    px.set_dir_servo_angle(0)
    px.stop()
    server_sock.close()
    sock.close()
    print("client thread end")
    print("BLUETOOTH OPS CLOSED")

cth = threading.Thread(target=start_client)

cth.start()

    

# print("Disconnected.")


# print("All done.")



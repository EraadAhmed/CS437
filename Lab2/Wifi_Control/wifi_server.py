import socket
import threading
from collections import deque
import signal
import time
from robot_hat import ADC
from picarx import Picarx


HOST = "192.168.125.22" # IP address of your Raspberry PI
PORT = 65432          # Port to listen on (non-privileged ports are > 1023)

buf_size = 1024
PRECISION = 12 #12 bit precision of the values in the registers

battery_adcport = ADC('A4')

exit_event = threading.Event()

control_queue = deque([])
output = ""

send_lock = threading.Lock()
queue_lock = threading.Lock()
recieve_lock = threading.Lock()

power = 0
steer_angle = 0
ploss = 3 # loss due to environmental factors
pmax = 100 #max power for motor
max_speed = 65 #cm/s

px = Picarx(servo_pins=['P0','P1','P3'])

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

def telemetry_loop(client):
    global battery_adcport
    global send_lock 
    global queue_lock
    global exit_event
    global power
    """Continuously send telemetry to client."""
    while not exit_event.is_set():
        try:
            temp = get_pi_temperature()
            spd = calculate_speed()
            raw_read = battery_adcport.read()
            bat_level = float(raw_read / ((2**PRECISION)-1)) * 100

            msg = (
                f"TEMP:{temp:.2f}C "
                f"SPD:{spd:.2f}cm/s "
                f"BAT:{bat_level:.2f}%\r\n"
                f"PWR:{power:.2f}%\r\n"
            )
            send_lock.acquire()
            client.sendall(msg.encode("utf-8"))
            send_lock.release()
        except Exception as e:
            print("Telemetry loop error:", e)
            break
        time.sleep(1)  # send once per second

def control_loop(client):
    global send_lock
    global queue_lock
    global output
    global exit_event
    global px
    while not exit_event.is_set():
        if send_lock.acquire(blocking=False):
            queue_lock.acquire()
            if(len(control_queue) > 0):
                try:
                    sent = client.send(bytes(control_queue[0], 'utf-8'))
                except Exception as e:
                    exit_event.set()
                    continue
                if sent < len(control_queue[0]):
                    control_queue[0] = control_queue[0][sent:]
                else:
                    control_queue.popleft()
            queue_lock.release()
            send_lock.release()
        
        if queue_lock.acquire(blocking=False):
            data = ""
            try:
                try:
                    data = client.recv(1024).decode('utf-8')
                    px.set_dir_servo_angle(0)
                    if(data == "RT\r\n"):
                        turn_right()
                        control_queue.append("STEER RT, CURR ANG: " + str(steer_angle) + " degs" + " \r\n")
                        px.forward(power)
                    elif(data == "LT\r\n"):
                        turn_left()
                        control_queue.append("STEER LT, CURR ANG: " + str(steer_angle) + " degs" + " \r\n")
                        px.forward(power)
                    elif(data == "FWD\r\n"):
                        steer_angle = 0
                        px.set_dir_servo_angle(steer_angle)
                        increase_velocity()
                        if power == 0:
                            px.forward(power)
                            control_queue.append("Vehicle Stopped" + "\r\n")
                        elif power > 0:
                            px.forward(power)
                            control_queue.append("SPEEDING UP, POWER: " + str(power) +  " \r\n")
                        elif power < 0:
                            px.backward(abs(power))
                            control_queue.append("SLOWING DOWN, POWER: " + str(power) +  " \r\n")
                    elif(data == "BWD\r\n"):
                        steer_angle = 0
                        px.set_dir_servo_angle(steer_angle)
                        decrease_velocity()
                        if power == 0:
                            px.forward(power)
                            control_queue.append("Vehicle Stopped" + "\r\n")
                        elif power > 0:
                            px.forward(power)
                            control_queue.append("SLOWING DOWN, POWER: " + str(power) +  " \r\n")
                        elif power < 0:
                            px.backward(abs(power))
                            control_queue.append("SPEEDING UP, POWER: " + str(power) +  " \r\n")
                    queue_lock.release()
                except socket.error as e:
                    queue_lock.release()
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


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    
    s.bind((HOST, PORT))
    s.listen(1)
    s.settimeout(10)
    client, clientInfo = s.accept()
    print("Connected")
    s.settimeout(None)
    client.setblocking(0)

    tloop = threading.Thread(target=telemetry_loop, args=(client,), daemon=True)
    tloop.start()
    cloop = threading.Thread(target=control_loop, args=(client,), daemon=True)
    cloop.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        exit_event.set()
    finally:
        px.set_dir_servo_angle(0)
        px.stop()
        print("Closing socket")
        client.close()
        s.close()    


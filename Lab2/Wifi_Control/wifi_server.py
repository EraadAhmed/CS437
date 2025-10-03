import json
import socket
import threading
import time

from robot_hat import ADC
from picarx import Picarx


HOST = "192.168.125.22" # IP address of your Raspberry PI
PORT = 65432          # Port to listen on (non-privileged ports are > 1023)

buf_size = 1024
PRECISION = 12   # 12 bit precision of the values in the registers

battery_adcport = ADC("A4")

exit_event = threading.Event()

send_lock = threading.Lock()

power = 0
steer_angle = 0
ploss = 3  # loss due to environmental factors
pmax = 100  # max power for motor
max_speed = 65  # cm/s

STEER_MAX = 30
STEER_STEP = 5
POWER_STEP = 10

px = Picarx(servo_pins=['P0','P1','P3'])


def apply_motion():
    """Drive the car using the current power value."""
    if power > 0:
        px.forward(power)
    elif power < 0:
        px.backward(abs(power))
    else:
        px.stop()


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))

def get_pi_temperature():
    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
        temp_millideg = int(f.readline().strip())
    return temp_millideg / 1000.0   # °C

def increase_velocity():
    global power
    power = clamp(power + POWER_STEP, -100, 100)
    apply_motion()

def decrease_velocity():
    global power
    power = clamp(power - POWER_STEP, -100, 100)
    apply_motion()

def turn_right():
    global steer_angle
    steer_angle = clamp(steer_angle + STEER_STEP, -STEER_MAX, STEER_MAX)
    px.set_dir_servo_angle(steer_angle)
        
def turn_left():
    global steer_angle
    steer_angle = clamp(steer_angle - STEER_STEP, -STEER_MAX, STEER_MAX)
    px.set_dir_servo_angle(steer_angle)


def straighten():
    global steer_angle
    steer_angle = 0
    px.set_dir_servo_angle(steer_angle)


def stop_vehicle():
    global power
    power = 0
    apply_motion()


def send_packet(client, payload):
    message = json.dumps(payload) + "\n"
    encoded = message.encode("utf-8")
    with send_lock:
        client.sendall(encoded)

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
    global exit_event
    global power
    global steer_angle
    """Continuously send telemetry to client."""
    while not exit_event.is_set():
        try:
            temp = get_pi_temperature()
            spd = calculate_speed()
            raw_read = battery_adcport.read()
            bat_level = float(raw_read / ((2**PRECISION)-1)) * 100

            payload = {
                "type": "telemetry",
                "temperature_c": round(temp, 2),
                "speed_cm_s": round(spd, 2),
                "battery_percent": round(bat_level, 2),
                "power_percent": float(power),
                "steering_deg": float(steer_angle),
                "timestamp": time.time(),
            }
            send_packet(client, payload)
        except Exception as e:
            print("Telemetry loop error:", e)
            exit_event.set()
            break
        time.sleep(1)  # send once per second

def control_loop(client):
    buffer = ""
    while not exit_event.is_set():
        try:
            chunk = client.recv(buf_size)
            if not chunk:
                exit_event.set()
                break
            buffer += chunk.decode("utf-8")
        except socket.timeout:
            continue
        except BlockingIOError:
            time.sleep(0.05)
            continue
        except Exception as exc:
            print("Control loop error:", exc)
            exit_event.set()
            break

        while True:
            newline_index = buffer.find("\n")
            carriage_index = buffer.find("\r")

            # choose earliest line break if present
            candidates = [idx for idx in (newline_index, carriage_index) if idx != -1]
            if not candidates:
                break

            idx = min(candidates)
            line = buffer[:idx]
            buffer = buffer[idx + 1 :]

            command = line.strip()
            if not command:
                continue

            handle_command(command, client)


def handle_command(command, client):
    global power

    normalized = command.upper()

    if normalized in ("FWD", "FORWARD"):
        straighten()
        increase_velocity()
        status = "accelerating" if power > 0 else "slowing" if power < 0 else "stopped"
    elif normalized in ("BWD", "BACK", "BACKWARD", "REV"):
        straighten()
        decrease_velocity()
        status = "accelerating" if power < 0 else "slowing" if power > 0 else "stopped"
    elif normalized in ("STOP", "HALT"):
        stop_vehicle()
        status = "stopped"
    elif normalized in ("RT", "RIGHT"):
        turn_right()
        status = "steering"
    elif normalized in ("LT", "LEFT"):
        turn_left()
        status = "steering"
    elif normalized in ("CENTER", "STRAIGHT"):
        straighten()
        status = "steering"
    else:
        send_packet(
            client,
            {
                "type": "error",
                "command": command,
                "message": "Unknown command",
                "timestamp": time.time(),
            },
        )
        return

    send_packet(
        client,
        {
            "type": "ack",
            "command": normalized,
            "status": status,
            "power_percent": float(power),
            "steering_deg": float(steer_angle),
            "timestamp": time.time(),
        },
    )


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

    s.bind((HOST, PORT))
    s.listen(1)
    s.settimeout(10)
    client, clientInfo = s.accept()
    print("Connected")
    s.settimeout(None)
    client.settimeout(0.5)

    stop_vehicle()
    straighten()

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


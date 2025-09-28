import socket
import threading
from collections import deque
import signal
import time

server_addr = '88:A2:9E:2B:1A:4F'
server_port = 1

buf_size = 1024

client_sock = None
server_sock = None
sock = None

exit_event = threading.Event()

message_queue = deque([])
output = ""

dq_lock = threading.Lock()
output_lock = threading.Lock()

def handler(signum, frame):
    exit_event.set()

signal.signal(signal.SIGINT, handler)

def get_pi_temperature():
    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
        temp_millideg = int(f.readline().strip())
    return temp_millideg / 1000.0   # °C

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
                    temp = get_pi_temperature()
                    if data == "TEMP\r\n":
                        dq_lock.acquire()
                        message_queue.append("pi temperature:" + str(temp) + "C " + " \r\n")
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
    server_sock.close()
    sock.close()
    print("client thread end")


cth = threading.Thread(target=start_client)

cth.start()

    

print("Disconnected.")


print("All done.")



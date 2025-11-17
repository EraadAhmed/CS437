import csv
import json
import time
import ssl
import threading
import paho.mqtt.client as mqtt


NUM_CARS = 5
START_CAR_NUM = 1
CSV_PATH = "data/vehicle{0}.csv"
certificate_formatter = "certs/IOTCar{0}/Car{0}-cert.pem"
key_formatter = "certs/IOTCar{0}/Car{0}-priv.key"
CA_PATH = "AmazonRootCA1.pem"
LOCAL_HOST = "192.168.0.10"
LOCAL_PORT = 8883

# Store results
results = {}
results_lock = threading.Lock()


def create_client(vehicle_id, car_num):
    client = mqtt.Client(client_id=vehicle_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    
    cert_path = certificate_formatter.format(car_num)
    key_path = key_formatter.format(car_num)
    
    client.tls_set(ca_certs=CA_PATH, certfile=cert_path, keyfile=key_path, 
                   cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)
    
    def on_result(client, userdata, message):
        result_data = json.loads(message.payload.decode())
        with results_lock:
            if vehicle_id not in results or result_data['max_CO2'] > results[vehicle_id].get('max_CO2', 0):
                results[vehicle_id] = result_data
        print(f"[{vehicle_id} RECEIVED] {result_data}")
    
    client.on_message = on_result
    client.connect(LOCAL_HOST, LOCAL_PORT)
    client.loop_start()
    
    result_topic = f"clients/{vehicle_id}/emission/result"
    client.subscribe(result_topic)
    print(f"{vehicle_id} subscribed to {result_topic}")
    return client


def run_all():
    clients = {}
    print("\nConnecting clients...")
    
    for i in range(NUM_CARS):
        car_num = START_CAR_NUM + i
        vid = f"IOTCar{car_num}"
        clients[vid] = create_client(vid, car_num)
        print(f"Connected: {vid}")
    
    time.sleep(2)  # Let subscriptions settle
    
    for i in range(NUM_CARS):
        car_num = START_CAR_NUM + i
        vid = f"IOTCar{car_num}"
        csv_path = CSV_PATH.format(i % 5)
        
        print(f"\nSending data for {vid} from {csv_path}")
        
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                payload = {
                    "vehicle_id": vid,
                    "vehicle_CO2": row.get("CO2") or row.get("vehicle_CO2")
                }
                clients[vid].publish(f"clients/{vid}/emission/data", json.dumps(payload))
                time.sleep(0.05)
    
    print("\n=== All data sent. Waiting for results... ===")
    
    # Wait until all results received or timeout
    timeout = 15
    start = time.time()
    while len(results) < NUM_CARS and (time.time() - start) < timeout:
        time.sleep(0.5)
        print(f"Received {len(results)}/{NUM_CARS} results...")
    
    # Print final results
    print("\n=== FINAL RESULTS ===")
    for i in range(NUM_CARS):
        vid = f"IOTCar{START_CAR_NUM + i}"
        if vid in results:
            print(f"{vid}: Max CO2 = {results[vid].get('max_CO2', 'N/A')}")
        else:
            print(f"{vid}: NO RESULT RECEIVED")
    
    for client in clients.values():
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    run_all()

import csv
import json
import time
import ssl
import paho.mqtt.client as mqtt

NUM_CARS = 5
CSV_PATH = "data/vehicle{0}.csv"
certificate_formatter = "certs/IOTCar{0}/Car{0}-cert.pem"
key_formatter = "certs/IOTCar{0}/Car{0}-priv.key"
CA_PATH = "AmazonRootCA1.pem"
LOCAL_HOST = "localhost"
LOCAL_PORT = 8883

def create_client(vehicle_id, car_num):
    client = mqtt.Client(client_id=vehicle_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    
    cert_path = certificate_formatter.format(car_num)
    key_path = key_formatter.format(car_num)
    
    # Disable certificate verification for local Greengrass
    client.tls_set(ca_certs=CA_PATH, certfile=cert_path, keyfile=key_path, 
                   cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)
    
    def on_result(client, userdata, message):
        print(f"[{vehicle_id} RECEIVED RESULT] {message.payload.decode()}")
    
    client.on_message = on_result
    client.connect(LOCAL_HOST, LOCAL_PORT)
    
    result_topic = f"clients/{vehicle_id}/emission/result"
    client.subscribe(result_topic)
    print(f"{vehicle_id} subscribed to {result_topic}")
    return client

def run_all():
    clients = {}
    print("\nWaiting for results...")
    
    for i in range(1, NUM_CARS + 1):
        vid = f"IOTCar{i}"
        clients[vid] = create_client(vid, i)
        print(f"Connected: {vid}")
    
    for i in range(1, NUM_CARS + 1):
        vid = f"IOTCar{i}"
        csv_path = CSV_PATH.format(i%5)
        
        print(f"\nSending CSV rows for {vid} from {csv_path}")
        
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                payload = {
                    "vehicle_id": vid,
                    "vehicle_CO2": row.get("CO2") or row.get("vehicle_CO2")
                }
                
                topic = f"clients/{vid}/emission/data"
                print(f"{vid} ---> {topic}: {payload}")
                
                clients[vid].publish(topic, json.dumps(payload))
                time.sleep(0.1)
    
    print("\nAll CSV data sent.")
    while True:
        for c in clients.values():
            c.loop(timeout=0.1)

if __name__ == "__main__":
    run_all()

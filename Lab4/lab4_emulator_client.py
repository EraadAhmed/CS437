# Import SDK packages
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import time
import json
import pandas as pd
import numpy as np


#TODO 1: modify the following parameters
#Starting and end index, modify this
device_st = 1
device_end = 11



#Path to the dataset, modify this
data_path = "data/vehicle{0}.csv"

#Path to your certificates, modify this
certificate_formatter = "certs/IOTCar{0}/Car{0}-cert.pem"
key_formatter = "certs/IOTCar{0}/Car{0}-priv.key"


class MQTTClient:
    def __init__(self, device_id, cert, key, num):
        # For certificate based connection
        self.device_id = str(device_id)
        self.state = 0
        self.carnum = num
        self.thingname = "IOTCar{0}"
        self.client = AWSIoTMQTTClient(self.device_id)
        #TODO 2: modify your broker address
        self.client.configureEndpoint("a1asbxpmh5fugn-ats.iot.us-east-2.amazonaws.com", 8883)
        self.client.configureCredentials("AmazonRootCA1.pem", key, cert)
        self.client.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
        self.client.configureDrainingFrequency(2)  # Draining: 2 Hz
        self.client.configureConnectDisconnectTimeout(10)  # 10 sec
        self.client.configureMQTTOperationTimeout(5)  # 5 sec
        # self.client.onMessage = self.customOnMessage
        self.client.on_message = self.customOnMessage


    # def customOnMessage(self,message):
    #     #TODO 3: fill in the function to show your received message
    def customOnMessage(self, client, userdata, message):
        print(f"[{self.device_id}] RECEIVED on {message.topic}: {message.payload.decode()}")



    # Suback callback
    def customSubackCallback(self,mid, data):
        #You don't need to write anything here
        pass


    # Puback callback
    def customPubackCallback(self,mid):
        #You don't need to write anything here
        pass


    def publish(self, topic="vehicle/emission/data/part1_6"):
        df = pd.read_csv(data_path.format(self.carnum % 5))
        row = df.iloc[0].to_dict()           # FIRST ROW ONLY
        row["thing_name"] = self.thingname.format(self.carnum)
        payload = json.dumps(row)
        print(f"Publishing: {payload} to {topic}")
        self.client.publishAsync(topic, payload, 0, ackCallback=self.customPubackCallback)

            
            # Sleep to simulate real-time data publishing
    #subscribe
    def subscribe(self, topic="vehicle/emission/admin/part1_6", qos=1):

        def ack_callback(mid, data):
            print(f"Subscription for {self.device_id} acknowledged (mid={mid})")

        self.client.subscribeAsync(
            topic,
            qos,
            messageCallback=self.customOnMessage,
            ackCallback=ack_callback
        )



print("Loading vehicle data...")
data = []
for i in range(5):
    a = pd.read_csv(data_path.format(i))
    data.append(a)

print("Initializing MQTTClients...")
clients = []
for device_id in range(device_st, device_end):
    client = MQTTClient(device_id,certificate_formatter.format(device_id,device_id) ,key_formatter.format(device_id,device_id), num = device_id)
    client.client.connect()
    clients.append(client)


while True:
    print("send now?")
    x = input()
    if x == "s":
        for i,c in enumerate(clients):
            c.publish()
    
    elif  x == "w":
        for i,c in enumerate(clients):
            c.subscribe()

    elif x == "d":
        for c in clients:
            c.client.disconnect()
        print("All devices disconnected")
        exit()
    else:
        print("wrong key pressed")

    time.sleep(3)







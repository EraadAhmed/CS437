import json
import logging
import sys
import greengrasssdk

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# SDK Client
client = greengrasssdk.client("iot-data")

def lambda_handler(event, context):
    # TODO1: Get your data
    if not event or not isinstance(event, list):
        logger.error("Invalid event format: expected a list of records.")
        return {"error": "Invalid input format"}
    
    # TODO2: Calculate max CO2 emission
    maxCounter = max(float(record['vehicle_CO2']) for record in event)
    vehicle_stat = event[-1]['vehicle_id']
        
    # TODO3: Return the result
    client.publish(
        topic="iot/Vehicle_" + vehicle_stat,
        queueFullPolicy="AllOrException",
        payload=json.dumps({"max_CO2": maxCounter}),
    )

    return 'Success'

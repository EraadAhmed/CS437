import json
import logging
import sys
from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2



# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# SDK Client
ipc = GreengrassCoreIPCClientV2()

def lambda_handler(event, context):
    # TODO1: Get your data
    if not event or not isinstance(event, list):
        logger.error("Invalid event format: expected a list of records.")
        return {"error": "Invalid input format"}
    
    # TODO2: Calculate max CO2 emission
    maxCounter = max(float(record['vehicle_CO2']) for record in event)
    vehicle_id = event[-1]['vehicle_id']
        
    # TODO3: Return the result
    payload = json.dumps({
        "vehicle_id": vehicle_id,
        "max_CO2": maxCounter
    })

    ipc.publish_to_topic(
        topic=f"clients/{vehicle_id}/emission/result",
        publish_message={
            "json_message": {
                "data": payload
            }
        }
    )

    return 'Success'

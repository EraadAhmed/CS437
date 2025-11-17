import json
import logging
import sys
from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2
from awsiot.greengrasscoreipc.model import (
    PublishMessage,
    JsonMessage
)

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# SDK Client
ipc = GreengrassCoreIPCClientV2()

def lambda_handler(event, context):
    """
    Process vehicle emission data and publish max CO2 result.
    
    Args:
        event: List of records with 'vehicle_id' and 'vehicle_CO2'
        context: Lambda context (unused in Greengrass)
    """
    if not event or not isinstance(event, list):
        logger.error("Invalid event format: expected a list of records.")
        return {"error": "Invalid input format"}
    
    try:
        # Calculate max CO2 emission from all records
        max_co2 = max(float(record['vehicle_CO2']) for record in event)
        vehicle_id = event[-1]['vehicle_id']
        
        # Build result topic and message
        result_topic = f"clients/{vehicle_id}/emission/result"
        result_data = {"vehicle_id": vehicle_id, "max_CO2": max_co2}
        result_msg = json.dumps(result_data)
        
        # Publish to IPC topic
        ipc.publish_to_topic(
            topic=result_topic,
            publish_message=PublishMessage(
                json_message=JsonMessage(message=result_data)
            )
        )
        
        logger.info(f"Published result to {result_topic}: {result_msg}")
        return 'Success'
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return {"error": str(e)}

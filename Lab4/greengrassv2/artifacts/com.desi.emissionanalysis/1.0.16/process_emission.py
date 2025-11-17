import json
import traceback
import logging
from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2
from awsiot.greengrasscoreipc.model import PublishMessage, JsonMessage

logger = logging.getLogger(__name__)
ipc = GreengrassCoreIPCClientV2()

# Store max CO2 per vehicle (persists across calls)
vehicle_max_co2 = {}

def lambda_handler(event, context):
    try:
        vehicle_id = event.get('vehicle_id')
        co2_value = float(event.get('vehicle_CO2', 0))
        
        logger.info(f"Received: {vehicle_id}, CO2: {co2_value}")
        
        # Update running max for this vehicle
        if vehicle_id not in vehicle_max_co2:
            vehicle_max_co2[vehicle_id] = co2_value
        else:
            vehicle_max_co2[vehicle_id] = max(vehicle_max_co2[vehicle_id], co2_value)
        
        # Publish current max
        result_topic = f"clients/{vehicle_id}/emission/result"
        result_data = {"vehicle_id": vehicle_id, "max_CO2": vehicle_max_co2[vehicle_id]}
        
        ipc.publish_to_topic(
            topic=result_topic, 
            publish_message=PublishMessage(json_message=JsonMessage(message=result_data))
        )
        
        logger.info(f"Published to {result_topic}: {result_data}")
        return result_data
        
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}


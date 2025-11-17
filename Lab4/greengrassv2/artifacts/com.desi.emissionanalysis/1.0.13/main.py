import json
import logging
import sys
import time
import process_emission
from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2

# Config the logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(name)s.%(funcName)s():%(lineno)d] - [%(levelname)s] - %(message)s", 
                    stream=sys.stdout, 
                    level=logging.INFO)

# IPC Client
ipc_client = GreengrassCoreIPCClientV2()

def on_stream_event(event):
    try:
        message = str(event.binary_message.message, 'utf-8')
        logger.info(f"Received message: {message}")
        
        # Process the emission data
        data = json.loads(message)
        process_emission.lambda_handler([data], None)
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)

def on_stream_error(error):
    logger.error(f"Stream error: {error}")

def on_stream_closed():
    logger.info("Stream closed")

if __name__ == "__main__":
    try:
        # Hardcode config - no command line args
        config = {
            'input_topic': 'clients/+/emission/data',
            'output_topic': 'clients/+/emission/result',
            'vehicle_id': 'CS437Car1'
        }
        
        logger.info(f'Component Config: {config}')
        
        # Get subscribe topic from config
        subscribe_topic = config.get('input_topic', 'clients/+/emission/data')
        logger.info(f'Subscribing to topic: {subscribe_topic}')
        
        # Subscribe to the topic
        operation = ipc_client.subscribe_to_topic(
            topic=subscribe_topic,
            on_stream_event=on_stream_event,
            on_stream_error=on_stream_error,
            on_stream_closed=on_stream_closed
        )
        
        logger.info("Successfully subscribed. Waiting for messages...")
        
        # Keep the component running
        while True:
            time.sleep(1)
            
    except Exception as e:
        logger.error(f'Error in main: {e}', exc_info=True)
        sys.exit(1)

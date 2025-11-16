import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def build_record(index: int) -> dict:
    payload = {
        "device_id": f"vehicle_{index:05d}",
        "vehicle_co2": random.randint(200, 499),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return {"Data": (json.dumps(payload) + "\n").encode("utf-8")}


def send_records(stream_name: str, region: str, total: int, batch_size: int, pause: float) -> None:
    client = boto3.client("firehose", region_name=region)
    sent = 0

    while sent < total:
        batch_count = min(batch_size, total - sent)
        batch = [build_record(sent + i + 1) for i in range(batch_count)]

        try:
            response = client.put_record_batch(DeliveryStreamName=stream_name, Records=batch)
        except (BotoCoreError, ClientError) as exc:
            print(f"Error sending batch at record {sent + 1}: {exc}", file=sys.stderr)
            raise

        failed = response.get("FailedPutCount", 0)
        if failed:
            print(f"Warning: {failed} records failed in batch starting at {sent + 1}", file=sys.stderr)

        sent += batch_count
        print(f"Sent {sent}/{total} records")

        if pause and sent < total:
            time.sleep(pause)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send test vehicle emission records to an AWS Firehose delivery stream.",
    )
    parser.add_argument(
        "--stream-name",
        default="vehicle-emissions-data-stream-goober",
        help="Firehose delivery stream name",
    )
    parser.add_argument(
        "--region",
        default="us-east-2",
        help="AWS region for the Firehose client",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=100,
        help="Total number of records to send",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of records per Firehose batch (max 500)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.1,
        help="Pause in seconds between batches",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size < 1 or args.batch_size > 500:
        print("batch-size must be between 1 and 500", file=sys.stderr)
        sys.exit(1)

    if args.total < 1:
        print("total must be at least 1", file=sys.stderr)
        sys.exit(1)

    print(
        f"Sending {args.total} records to stream '{args.stream_name}' in region '{args.region}'",
    )

    send_records(args.stream_name, args.region, args.total, args.batch_size, args.pause)

    print("Finished sending records. Wait for Firehose buffers to flush (default 5 minutes).")


if __name__ == "__main__":
    main()

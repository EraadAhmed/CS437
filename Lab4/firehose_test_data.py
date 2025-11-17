import argparse
import csv
import glob
import json
import random
import sys
import time
from datetime import datetime, timezone
from itertools import islice
from typing import Generator, Iterable, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def build_random_record(index: int) -> dict:
    payload = {
        "device_id": f"vehicle_{index:05d}",
        "vehicle_co2": random.randint(200, 499),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return {"Data": (json.dumps(payload) + "\n").encode("utf-8")}


def normalize_csv_row(row: dict) -> dict:
    payload = {k: v for k, v in row.items() if v not in (None, "")}
    co2_value = row.get("vehicle_CO2") or row.get("vehicle_co2")

    if co2_value is not None:
        try:
            payload["vehicle_co2"] = float(co2_value)
        except ValueError:
            payload["vehicle_co2"] = co2_value

    if "vehicle_id" in row:
        payload.setdefault("device_id", row["vehicle_id"])

    payload.setdefault(
        "timestamp",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return {"Data": (json.dumps(payload) + "\n").encode("utf-8")}


def csv_record_source(pattern: str) -> Generator[dict, None, None]:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No CSV files matched pattern '{pattern}'")

    for path in matches:
        with open(path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield normalize_csv_row(row)


def random_record_source() -> Generator[dict, None, None]:
    index = 0
    while True:
        index += 1
        yield build_random_record(index)


def batched_records(records: Iterable[dict], batch_size: int, limit: Optional[int]):
    sent = 0
    iterator = iter(records)

    while True:
        remaining = None if limit is None else max(limit - sent, 0)
        if remaining == 0:
            break

        take = batch_size if remaining is None else min(batch_size, remaining)
        batch = list(islice(iterator, take))
        if not batch:
            break

        sent += len(batch)
        yield batch


def send_records(
    stream_name: str,
    region: str,
    batch_size: int,
    pause: float,
    records: Iterable[dict],
    limit: Optional[int],
) -> int:
    client = boto3.client("firehose", region_name=region)
    sent = 0

    for batch in batched_records(records, batch_size, limit):
        try:
            response = client.put_record_batch(DeliveryStreamName=stream_name, Records=batch)
        except (BotoCoreError, ClientError) as exc:
            print(f"Error sending batch after {sent} records: {exc}", file=sys.stderr)
            raise

        failed = response.get("FailedPutCount", 0)
        if failed:
            print(
                f"Warning: {failed} records failed in most recent batch (records {sent + 1}-{sent + len(batch)})",
                file=sys.stderr,
            )

        sent += len(batch)
        total_display = "all" if limit is None else limit
        print(f"Sent {sent}/{total_display} records")

        if pause and (limit is None or sent < limit):
            time.sleep(pause)

    return sent


def parse_total(raw_total: Optional[str], using_csv: bool) -> Optional[int]:
    if raw_total is None:
        return None if using_csv else 100

    if raw_total.lower() in {"all", "*"}:
        return None

    try:
        value = int(raw_total)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("total must be an integer or 'all'") from exc

    if value < 1:
        raise argparse.ArgumentTypeError("total must be at least 1")

    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send vehicle emission records to an AWS Firehose delivery stream.",
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
        default=None,
        help="Total number of records to send (integer or 'all'). Defaults to 100 for random data and all rows for CSV input.",
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
    parser.add_argument(
        "--csv-pattern",
        default=None,
        help="Glob pattern for CSV files to stream (e.g. 'data/vehicle*.csv'). If provided, rows from these files are sent instead of synthetic data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size < 1 or args.batch_size > 500:
        print("batch-size must be between 1 and 500", file=sys.stderr)
        sys.exit(1)

    using_csv = args.csv_pattern is not None

    try:
        total = parse_total(args.total, using_csv)
    except argparse.ArgumentTypeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if using_csv:
        try:
            records = csv_record_source(args.csv_pattern)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        source_description = f"CSV rows matching '{args.csv_pattern}'"
    else:
        records = random_record_source()
        source_description = "synthetic test data"

    total_display = "all available" if total is None else str(total)
    print(
        f"Sending {total_display} records from {source_description} to stream '{args.stream_name}' in region '{args.region}'",
    )

    sent = send_records(
        args.stream_name,
        args.region,
        args.batch_size,
        args.pause,
        records,
        total,
    )

    print(
        f"Finished sending {sent} record(s). Wait for Firehose buffers to flush (default 5 minutes).",
    )


if __name__ == "__main__":
    main()

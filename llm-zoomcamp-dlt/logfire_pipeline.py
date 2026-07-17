import argparse
import os
from datetime import datetime, timedelta, timezone

import dlt
import requests
from dotenv import load_dotenv

from main import DEFAULT_QUESTION, SERVICE_NAME


load_dotenv(override=True)

DB_PATH = "llm_zoomcamp_dlt.duckdb"
DATASET_NAME = "agent_traces"


def require_read_token():
    token = os.getenv("LOGFIRE_READ_TOKEN")
    if not token:
        raise RuntimeError("LOGFIRE_READ_TOKEN is not set in .env")
    return token


def logfire_base_url():
    return os.getenv("LOGFIRE_BASE_URL", "https://logfire-us.pydantic.dev").rstrip("/")


def query_logfire(sql, hours=24, limit=10000):
    token = require_read_token()
    min_timestamp = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
    response = requests.post(
        f"{logfire_base_url()}/v2/query",
        json={
            "sql": sql,
            "min_timestamp": min_timestamp.isoformat(),
            "limit": limit,
        },
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(
            "Logfire query failed: "
            f"{response.status_code} {response.reason}\n{response.text}"
        )
    payload = response.json()

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("rows", "data", "result"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return rows
        columns = payload.get("columns")
        if isinstance(columns, dict):
            names = list(columns)
            values = [columns[name] for name in names]
            return [dict(zip(names, row)) for row in zip(*values)]

    raise ValueError(f"Unexpected Logfire response shape: {type(payload).__name__}")


@dlt.resource(name="records", write_disposition="replace")
def logfire_records(hours=24):
    sql = f"""
    SELECT
      *
    FROM records
    WHERE service_name = '{SERVICE_NAME}'
    ORDER BY start_timestamp DESC
    LIMIT 1000
    """

    yield query_logfire(sql, hours=hours)


def run_pipeline(hours=24):
    pipeline = dlt.pipeline(
        pipeline_name="llm_zoomcamp_dlt_logfire",
        destination=dlt.destinations.duckdb(DB_PATH),
        dataset_name=DATASET_NAME,
    )
    return pipeline.run(logfire_records(hours=hours))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()

    if not os.getenv("LOGFIRE_READ_TOKEN"):
        print("LOGFIRE_READ_TOKEN is not set in .env; add a Logfire read token first.")
        return

    print(f"Loading Logfire spans for service_name={SERVICE_NAME!r}")
    print(f"Expected Q1 query: {DEFAULT_QUESTION!r}")
    load_info = run_pipeline(hours=args.hours)
    print(load_info)


if __name__ == "__main__":
    main()

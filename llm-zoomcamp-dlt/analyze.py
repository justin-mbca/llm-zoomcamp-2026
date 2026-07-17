import argparse
import duckdb

from logfire_pipeline import DATASET_NAME, DB_PATH
from main import DEFAULT_QUESTION, SERVICE_NAME


def token_bucket(total):
    if 100 <= total <= 500:
        return "100 - 500"
    if 1500 <= total <= 5000:
        return "1500 - 5000"
    if 10000 <= total <= 20000:
        return "10000 - 20000"
    if 50000 <= total <= 100000:
        return "50000 - 100000"
    return "outside listed ranges"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DB_PATH)
    args = parser.parse_args()

    conn = duckdb.connect(args.db)

    table_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = ?
        """,
        [DATASET_NAME],
    ).fetchone()[0]

    tables = [
        row[0]
        for row in conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = ?
            ORDER BY table_name
            """,
            [DATASET_NAME],
        ).fetchall()
    ]

    target_trace_id = conn.execute(
        f"""
        SELECT trace_id
        FROM {DATASET_NAME}.records
        WHERE service_name = ?
        ORDER BY start_timestamp DESC
        LIMIT 1
        """,
        [SERVICE_NAME],
    ).fetchone()

    if target_trace_id is None:
        raise RuntimeError(f"No records found for service_name={SERVICE_NAME!r}")

    target_trace_id = target_trace_id[0]

    rows = conn.execute(
        f"""
        SELECT
          span_name,
          message,
          attributes__gen_ai_usage_input_tokens,
          attributes__gen_ai_aggregated_usage_input_tokens
        FROM {DATASET_NAME}.records
        WHERE service_name = ?
          AND trace_id = ?
        ORDER BY start_timestamp
        """,
        [SERVICE_NAME, target_trace_id],
    ).fetchall()
    conn.close()

    span_names = [row[0] for row in rows]
    llm_input_tokens = [row[2] for row in rows if row[2] is not None]
    aggregated_tokens = [row[3] for row in rows if row[3] is not None]

    total_input_tokens = aggregated_tokens[-1] if aggregated_tokens else sum(llm_input_tokens)

    print("Homework answers")
    print(f"Question: {DEFAULT_QUESTION}")
    print(f"Trace id used: {target_trace_id}")
    print(f"Q1 span count: {len(rows)}")
    print(f"Q1 span names: {', '.join(span_names)}")
    print(f"Q2 dlt table count: {table_count}")
    print(f"Q2 tables: {', '.join(tables)}")
    print(f"Q3 LLM input tokens by call: {llm_input_tokens}")
    print(f"Q3 total input tokens: {total_input_tokens}")
    print(f"Q3 answer bucket: {token_bucket(total_input_tokens)}")


if __name__ == "__main__":
    main()

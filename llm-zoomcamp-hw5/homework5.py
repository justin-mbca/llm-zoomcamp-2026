import argparse
import os
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


load_dotenv(override=True)

QUERY = "How does the agentic loop keep calling the model until it stops?"
DB_PATH = Path(__file__).with_name("traces.db")

# Cost is not a graded answer here, but Q2 asks us to store it as a span metric.
# Adjust these if you use a model with different pricing.
INPUT_COST_PER_1M = 0.15
OUTPUT_COST_PER_1M = 0.60


class SQLiteSpanExporter(SpanExporter):
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
            """
        )
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )

        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self, timeout_millis=30000):
        return True


def setup_tracing(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("llm-zoomcamp")


def get_usage(response):
    usage = response.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cost = (
        input_tokens * INPUT_COST_PER_1M
        + output_tokens * OUTPUT_COST_PER_1M
    ) / 1_000_000
    return input_tokens, output_tokens, cost


def duration_ms(row):
    return (row["end_time"] - row["start_time"]) / 1_000_000


def closest_token_option(value):
    options = [700, 7000, 70000, 700000]
    return min(options, key=lambda option: abs(option - value))


def timing_bucket(ms):
    if ms < 100:
        return "Under 100ms"
    if ms < 500:
        return "100-500ms"
    if ms < 2000:
        return "500-2000ms"
    return "Over 2000ms"


def token_variation_answer(tokens):
    if len(set(tokens)) == 1:
        return "They're identical"

    min_tokens = min(tokens)
    max_tokens = max(tokens)
    variation = (max_tokens - min_tokens) / min_tokens

    if variation <= 0.10:
        return "Within 10% of each other"
    if variation <= 0.50:
        return "Within 50% of each other"
    return "They vary more than 50%"


def fetch_rows(db_path=DB_PATH):
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT name, start_time, end_time, input_tokens, output_tokens, cost
            FROM spans
            ORDER BY start_time
            """
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return rows


def print_summary(db_path=DB_PATH):
    rows = fetch_rows(db_path)
    if not rows:
        print(f"No spans found in {db_path}. Run without --summary-only first.")
        return

    names = sorted({row["name"] for row in rows})
    llm_rows = [row for row in rows if row["name"] == "llm"]
    completed_llm_rows = [
        row for row in llm_rows
        if row["input_tokens"] is not None and row["output_tokens"] is not None
    ]
    child_rows = [row for row in rows if row["name"] != "rag"]

    totals = {}
    for row in child_rows:
        totals[row["name"]] = totals.get(row["name"], 0.0) + duration_ms(row)

    print()
    print("Homework answers")
    print("Q1 span count per RAG call: 3")

    if completed_llm_rows:
        input_tokens = [row["input_tokens"] for row in completed_llm_rows]
        print(f"Q2 input tokens: {input_tokens[-1]}")
        print(f"Q2 closest answer: {closest_token_option(input_tokens[-1])}")
        print(f"Q3 LLM duration: {duration_ms(completed_llm_rows[-1]):.0f}ms")
        print(f"Q3 closest answer: {timing_bucket(duration_ms(completed_llm_rows[-1]))}")
        print(f"Q6 input tokens across runs: {input_tokens}")
        print(f"Q6 answer: {token_variation_answer(input_tokens)}")
    else:
        print("Q2/Q3/Q5/Q6 need at least one completed llm span in the database.")

    print(f"Q4 span names: {', '.join(names)}")

    if totals and completed_llm_rows:
        for name, total in sorted(totals.items()):
            print(f"Q5 total {name} duration: {total:.0f}ms")
        slowest = max(totals, key=totals.get)
        print(f"Q5 answer: {slowest}")


class RAGTraced:
    def __init__(self, base_rag, tracer):
        self.base_rag = base_rag
        self.tracer = tracer

    def search(self, query, num_results=5):
        with self.tracer.start_as_current_span("search"):
            return self.base_rag.search(query, num_results=num_results)

    def llm(self, prompt):
        with self.tracer.start_as_current_span("llm") as span:
            response = self.base_rag.llm(prompt)
            input_tokens, output_tokens, cost = get_usage(response)
            span.set_attribute("input_tokens", input_tokens)
            span.set_attribute("output_tokens", output_tokens)
            span.set_attribute("cost", cost)
            return response

    def rag(self, query):
        with self.tracer.start_as_current_span("rag"):
            search_results = self.search(query)
            prompt = self.base_rag.build_prompt(query, search_results)
            response = self.llm(prompt)
            return response.output_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exporter",
        choices=["sqlite", "console"],
        default="sqlite",
        help="Use console for Q1-Q3 inspection, sqlite for Q4-Q6.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=4,
        help="Number of RAG calls to run. Q6 needs 4 total calls.",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Delete traces.db before running.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Read traces.db and print the homework summary without new calls.",
    )
    args = parser.parse_args()

    if args.summary_only:
        print_summary()
        return

    if args.reset_db and DB_PATH.exists():
        DB_PATH.unlink()

    exporter = (
        ConsoleSpanExporter()
        if args.exporter == "console"
        else SQLiteSpanExporter(DB_PATH)
    )
    tracer = setup_tracing(exporter)

    from starter import rag

    model = os.getenv("OPENAI_MODEL")
    if model:
        rag.model = model

    traced_rag = RAGTraced(rag, tracer)

    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}")
        started = time.perf_counter()
        answer = traced_rag.rag(QUERY)
        elapsed = time.perf_counter() - started
        print(answer)
        print(f"Elapsed: {elapsed:.2f}s")
        print()

    trace.get_tracer_provider().force_flush()
    trace.get_tracer_provider().shutdown()

    if args.exporter == "sqlite":
        print_summary()


if __name__ == "__main__":
    main()

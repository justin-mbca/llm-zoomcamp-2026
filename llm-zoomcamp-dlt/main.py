import argparse
import os

from dotenv import load_dotenv

load_dotenv(override=True)

import logfire

from agent import SearchDeps, faq_agent
from ingest import build_index, load_faq_data


DEFAULT_QUESTION = "How do I run Ollama locally?"
SERVICE_NAME = "llm-zoomcamp-dlt-homework"


def configure_logfire():
    logfire.configure(
        service_name=SERVICE_NAME,
        send_to_logfire="if-token-present",
    )
    logfire.instrument_pydantic_ai()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    if not os.getenv("LOGFIRE_TOKEN"):
        print("LOGFIRE_TOKEN is not set; the agent will run without uploading traces.")

    configure_logfire()

    documents = load_faq_data()
    index = build_index(documents)
    deps = SearchDeps(index=index)

    result = faq_agent.run_sync(args.question, deps=deps)
    print(result.output)
    print()
    print(f"Local Pydantic AI usage: {result.usage}")


if __name__ == "__main__":
    main()

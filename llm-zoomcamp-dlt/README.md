# LLM Zoomcamp DLT Homework

This folder contains the DLT workshop homework solution.

Add these to the repo root `.env`:

```text
OPENAI_API_KEY=...
LOGFIRE_TOKEN=...
LOGFIRE_READ_TOKEN=...
```

If your Logfire project is in Europe, also add:

```text
LOGFIRE_BASE_URL=https://logfire-eu.pydantic.dev
```

Run the homework flow:

```bash
uv run python llm-zoomcamp-dlt/main.py
uv run python llm-zoomcamp-dlt/logfire_pipeline.py
uv run python llm-zoomcamp-dlt/analyze.py
```

Submit the three printed answer choices in the course form.

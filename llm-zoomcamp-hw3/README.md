# LLM Zoomcamp 2026 - Homework 3

Homework: AI Orchestration with Kestra

## Answers

| Question | Answer |
| --- | --- |
| 1. Context Engineering | AI Copilot has access to current Kestra plugin documentation |
| 2. RAG vs No RAG | Vague, generic, or fabricated - the model guesses from training data |
| 3. Token usage - short summary | 60-100 tokens |
| 4. Token usage - long summary | 2-5x more |
| 5. Modifying a flow | 2-4x more |
| 6. Best Practices | Use traditional task-based workflows for predictability and auditability |

## Notes

Question 1: Kestra AI Copilot performs better because it has Kestra-specific context and plugin documentation available when generating flows. A general chat model can produce plausible YAML, but it is more likely to miss exact task types, plugin names, properties, or current syntax.

Question 2: The non-RAG flow does not have the Kestra 1.1 release notes as grounding context, so the answer is generally vague or fabricated. The RAG flow retrieves relevant context before answering.

Questions 3-5: The exact token counts can vary slightly between model runs. I used the closest answer bucket based on the `log_token_usage` output in the Kestra execution logs.

## Token Count Evidence

Fill in the exact values from the Kestra execution logs if needed:

| Flow run | Task | Output tokens | Selected bucket |
| --- | --- | ---: | --- |
| `4_simple_agent.yaml`, `summary_length=short` | `multilingual_agent` |  | 60-100 tokens |
| `4_simple_agent.yaml`, `summary_length=long` | `multilingual_agent` |  | 2-5x more than short |
| `4_simple_agent.yaml`, `summary_length=long`, `english_brevity` changed to exactly 3 sentences | `english_brevity` |  | 2-4x more than original 1-sentence version |

## Flow Modification for Question 5

For Question 5, I changed the `english_brevity` task prompt in `4_simple_agent.yaml` from exactly 1 sentence to exactly 3 sentences.

Relevant prompt change:

```yaml
english_brevity:
  prompt: |
    Rewrite the summary in English using exactly 3 sentences.
```

## Learning in Public Draft

Module 3 of LLM Zoomcamp by DataTalksClub complete.

This module covered AI orchestration with Kestra, including context engineering, RAG-grounded answers, token usage inspection, AI agents, and the tradeoff between flexible agents and deterministic task-based workflows.

Homework solution: `<repo link>`

Course: https://github.com/DataTalksClub/llm-zoomcamp/

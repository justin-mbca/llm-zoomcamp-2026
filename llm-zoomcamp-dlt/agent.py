from dataclasses import dataclass

from minsearch import Index
from pydantic_ai import Agent, RunContext


INSTRUCTIONS = """
You're a course teaching assistant. You're given a question from a course
student and your task is to answer it. If you want to look up information, use
the search function. Use as many keywords from the user question as possible
when making first requests. Make multiple searches. First perform search,
analyze the results and then perform more searches. The question has to be
about the course or its logistics, offtopic questions shouldn't be answered.

If the search returns nothing, it's likely an off-topic question. If you can't
answer the question using FAQ, don't do it yourself. Only use the facts from
the FAQ database. At the end, ask if there are other areas that the user wants
to explore.
""".strip()


@dataclass
class SearchDeps:
    index: Index


faq_agent = Agent(
    "openai:gpt-5.4-mini",
    deps_type=SearchDeps,
    instructions=INSTRUCTIONS,
)


@faq_agent.tool
def search(ctx: RunContext[SearchDeps], query: str) -> str:
    """Search the FAQ database for entries matching the given query."""
    boost_dict = {"question": 3.0, "section": 0.5}
    filter_dict = {"course": "llm-zoomcamp"}

    results = ctx.deps.index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
        filter_dict=filter_dict,
    )

    return str(results)

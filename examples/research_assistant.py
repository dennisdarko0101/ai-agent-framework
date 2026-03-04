#!/usr/bin/env python3
"""Research assistant — multi-agent pipeline that plans, researches, and reviews a topic.

Usage:
    python -m examples.research_assistant "quantum computing applications"

Demonstrates:
    - AgentPipeline for sequential multi-agent orchestration
    - PlannerAgent → ResearchAgent → ReviewerAgent workflow
    - Tool use (web search, summarisation)
    - Memory integration for context persistence
"""

from __future__ import annotations

import asyncio
import sys

from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearchAgent
from src.agents.reviewer import ReviewerAgent
from src.llm.provider import ProviderFactory
from src.memory.conversation import ConversationMemory
from src.orchestration.pipeline import AgentPipeline
from src.tools.registry import ToolRegistry
from src.utils.logger import setup_logging


def _register_tools(registry: ToolRegistry) -> None:
    from src.tools.calculator import CalculatorTool
    from src.tools.web_search import WebSearchTool

    for cls in [CalculatorTool, WebSearchTool]:
        try:
            registry.register(cls())
        except ValueError:
            pass


async def main(topic: str) -> None:
    setup_logging(level="INFO")

    # ── Setup ────────────────────────────────────────────────────────
    llm = ProviderFactory.create()
    tools = ToolRegistry()
    _register_tools(tools)
    memory = ConversationMemory(max_messages=30)

    # ── Agents ───────────────────────────────────────────────────────
    planner = PlannerAgent(llm=llm, tools=tools, memory=memory, name="planner")
    researcher = ResearchAgent(llm=llm, tools=tools, memory=memory, name="researcher")
    reviewer = ReviewerAgent(llm=llm, memory=memory, name="reviewer")

    # ── Pipeline: Plan → Research → Review ───────────────────────────
    pipeline = AgentPipeline(agents=[planner, researcher, reviewer])

    print(f"\n{'='*60}")
    print(f"  Research Assistant — Topic: {topic}")
    print(f"{'='*60}\n")

    result = await pipeline.run(f"Research the following topic thoroughly: {topic}")

    # ── Output ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Status: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Steps:  {len(result.steps)}")
    print(f"  Tokens: {result.total_tokens:,}")
    print(f"  Time:   {result.execution_time:.1f}s")
    print(f"{'─'*60}\n")
    print(result.output)

    if not result.success:
        print(f"\nError: {result.error}")


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "applications of quantum computing"
    asyncio.run(main(topic))

#!/usr/bin/env python3
"""Code assistant — writes, tests, and reviews code using multiple agents.

Usage:
    python -m examples.code_assistant "binary search function with tests"

Demonstrates:
    - CoderAgent for code generation and debugging
    - ReviewerAgent for quality assessment
    - AgentSupervisor for iterative delegation
    - Code execution tool for verification
"""

from __future__ import annotations

import asyncio
import sys

from src.agents.coder import CoderAgent
from src.agents.reviewer import ReviewerAgent
from src.llm.provider import ProviderFactory
from src.memory.conversation import ConversationMemory
from src.orchestration.supervisor import AgentSupervisor
from src.tools.registry import ToolRegistry
from src.utils.logger import setup_logging


def _register_tools(registry: ToolRegistry) -> None:
    from src.tools.calculator import CalculatorTool
    from src.tools.code_executor import CodeExecutorTool

    for cls in [CalculatorTool, CodeExecutorTool]:
        try:
            registry.register(cls())
        except ValueError:
            pass


async def main(task: str) -> None:
    setup_logging(level="INFO")

    # ── Setup ────────────────────────────────────────────────────────
    llm = ProviderFactory.create()
    tools = ToolRegistry()
    _register_tools(tools)
    memory = ConversationMemory(max_messages=30)

    # ── Agents ───────────────────────────────────────────────────────
    coder = CoderAgent(llm=llm, tools=tools, memory=memory, name="coder")
    reviewer = ReviewerAgent(llm=llm, memory=memory, name="reviewer")

    # ── Supervisor: delegates between coder and reviewer ─────────────
    supervisor = AgentSupervisor(
        llm=llm,
        agents={"coder": coder, "reviewer": reviewer},
        max_rounds=5,
    )

    print(f"\n{'='*60}")
    print(f"  Code Assistant — Task: {task}")
    print(f"{'='*60}\n")

    result = await supervisor.run(
        f"Write code for: {task}. "
        "First have the coder write it, then have the reviewer check it. "
        "If the review finds issues, have the coder fix them."
    )

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
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "a binary search function with unit tests"
    asyncio.run(main(task))

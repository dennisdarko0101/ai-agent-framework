#!/usr/bin/env python3
"""Data analyst — reads data, writes analysis code, executes it, and interprets results.

Usage:
    python -m examples.data_analyst "What is the average?" data.csv

Demonstrates:
    - ReActAgent with file read and code execution tools
    - Multi-step reasoning: read → analyse → execute → interpret
    - Tool chaining within a single agent run
"""

from __future__ import annotations

import asyncio
import sys

from src.agents.react import ReActAgent
from src.llm.provider import ProviderFactory
from src.memory.conversation import ConversationMemory
from src.tools.registry import ToolRegistry
from src.utils.logger import setup_logging

_SYSTEM_PROMPT = (
    "You are a data analysis agent. When given a question and a data file:\n"
    "1. Read the file to understand its structure\n"
    "2. Write Python code to analyse the data and answer the question\n"
    "3. Execute the code to get results\n"
    "4. Interpret the results and provide a clear answer\n\n"
    "Use the file_read tool to read data and code_executor to run analysis code."
)


def _register_tools(registry: ToolRegistry) -> None:
    from src.tools.calculator import CalculatorTool
    from src.tools.code_executor import CodeExecutorTool
    from src.tools.file_ops import FileReadTool

    for cls in [CalculatorTool, FileReadTool, CodeExecutorTool]:
        try:
            registry.register(cls())
        except ValueError:
            pass


async def main(question: str, file_path: str | None = None) -> None:
    setup_logging(level="INFO")

    # ── Setup ────────────────────────────────────────────────────────
    llm = ProviderFactory.create()
    tools = ToolRegistry()
    _register_tools(tools)
    memory = ConversationMemory(max_messages=20)

    # ── Agent ────────────────────────────────────────────────────────
    agent = ReActAgent(
        llm=llm,
        tools=tools,
        memory=memory,
        system_prompt=_SYSTEM_PROMPT,
        name="data-analyst",
        max_steps=15,
    )

    task = question
    if file_path:
        task = f"Question: {question}\nData file: {file_path}"

    print(f"\n{'='*60}")
    print(f"  Data Analyst")
    print(f"  Question: {question}")
    if file_path:
        print(f"  File:     {file_path}")
    print(f"{'='*60}\n")

    result = await agent.run(task)

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
    q = sys.argv[1] if len(sys.argv) > 1 else "Calculate the sum of numbers 1 to 100"
    fp = sys.argv[2] if len(sys.argv) > 2 else None
    asyncio.run(main(q, fp))

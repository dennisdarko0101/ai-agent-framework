"""Planner agent — decomposes complex tasks into structured sub-task plans."""

from __future__ import annotations

from src.agents.base import AgentResponse
from src.agents.react import ReActAgent
from src.llm.provider import BaseLLMProvider
from src.memory.base import BaseMemory
from src.tools.registry import ToolRegistry

_DEFAULT_SYSTEM_PROMPT = (
    "You are a planning agent. Your job is to break down complex tasks into "
    "clear, actionable sub-tasks.\n\n"
    "For each sub-task provide:\n"
    "1. A concise description of what needs to be done\n"
    "2. Which tools or agents are needed\n"
    "3. Expected output\n"
    "4. Dependencies on other sub-tasks\n\n"
    "Return a numbered plan. Be specific and practical."
)


class PlannerAgent(ReActAgent):
    """Creates structured plans by decomposing complex tasks."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        tools: ToolRegistry | None = None,
        memory: BaseMemory | None = None,
        system_prompt: str | None = None,
        max_steps: int = 10,
        name: str = "planner",
        **kwargs: object,
    ) -> None:
        super().__init__(
            llm=llm,
            tools=tools,
            memory=memory,
            system_prompt=system_prompt if system_prompt is not None else _DEFAULT_SYSTEM_PROMPT,
            max_steps=max_steps,
            name=name,
            **kwargs,  # type: ignore[arg-type]
        )

    async def create_plan(self, task: str) -> AgentResponse:
        """Create a detailed step-by-step plan for *task*."""
        prompt = (
            f"Create a detailed step-by-step plan for the following task:\n\n"
            f"{task}\n\n"
            f"For each step provide: description, required tools, "
            f"expected output, and dependencies."
        )
        return await self.run(prompt)

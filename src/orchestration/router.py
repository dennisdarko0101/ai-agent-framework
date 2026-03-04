"""Task router — LLM-based classification routes tasks to the right agent."""

from __future__ import annotations

import time

from src.agents.base import AgentResponse, BaseAgent
from src.llm.provider import BaseLLMProvider
from src.utils.logger import get_logger

log = get_logger(__name__)


class TaskRouter:
    """Classify a task using an LLM and dispatch it to the matching agent.

    *routes* maps category names to agents.  The LLM picks the best category
    and the corresponding agent handles the task.  If classification fails and
    no *default_agent* is set, an error response is returned.
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        routes: dict[str, BaseAgent],
        default_agent: BaseAgent | None = None,
    ) -> None:
        if not routes:
            raise ValueError("TaskRouter requires at least one route")
        self.llm = llm
        self.routes = routes
        self.default_agent = default_agent

    async def route(self, task: str) -> AgentResponse:
        start = time.monotonic()
        route_names = list(self.routes.keys())

        classification_prompt = (
            f"Classify the following task into exactly one of these categories: "
            f"{route_names}\n\n"
            f"Task: {task}\n\n"
            f"Respond with just the category name, nothing else."
        )

        response = await self.llm.generate(
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0.0,
        )
        total_tokens = response.usage.total_tokens

        category = response.content.strip().lower()
        log.info("task_routed", category=category, task_preview=task[:80])

        # Find matching agent (case-insensitive)
        agent = None
        for name, candidate in self.routes.items():
            if name.lower() == category:
                agent = candidate
                category = name  # preserve original casing
                break

        if agent is None:
            agent = self.default_agent

        if agent is None:
            return AgentResponse(
                output=f"No agent found for category: {category!r}",
                steps=[],
                success=False,
                error=f"unroutable_task: {category}",
                total_tokens=total_tokens,
                execution_time=time.monotonic() - start,
                metadata={"classified_as": category},
            )

        result = await agent.run(task)
        result.total_tokens += total_tokens
        result.execution_time = time.monotonic() - start
        result.metadata["routed_to"] = category
        return result

    def __repr__(self) -> str:
        routes = list(self.routes.keys())
        return f"<TaskRouter routes={routes}>"

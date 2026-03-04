"""Reviewer agent — evaluates outputs for correctness, quality, and completeness."""

from __future__ import annotations

from src.agents.base import AgentResponse
from src.agents.react import ReActAgent
from src.llm.provider import BaseLLMProvider
from src.memory.base import BaseMemory
from src.tools.registry import ToolRegistry

_DEFAULT_SYSTEM_PROMPT = (
    "You are a review agent. Analyse outputs from other agents or processes "
    "for correctness, completeness, and quality.\n\n"
    "Guidelines:\n"
    "- Check for logical errors and inconsistencies\n"
    "- Verify claims against available information\n"
    "- Assess completeness — are all requirements addressed?\n"
    "- Provide specific, actionable feedback\n"
    "- Rate the overall quality (excellent / good / needs improvement)"
)


class ReviewerAgent(ReActAgent):
    """Reviews and critiques content or agent outputs."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        tools: ToolRegistry | None = None,
        memory: BaseMemory | None = None,
        system_prompt: str | None = None,
        max_steps: int = 10,
        name: str = "reviewer",
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

    async def review(self, content: str, criteria: str | None = None) -> AgentResponse:
        """Review *content* against optional *criteria*."""
        prompt = f"Review the following content:\n\n{content}"
        if criteria:
            prompt += f"\n\nEvaluation criteria:\n{criteria}"
        prompt += "\n\nProvide detailed feedback with specific suggestions for improvement."
        return await self.run(prompt)

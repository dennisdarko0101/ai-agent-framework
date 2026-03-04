"""Research agent — focused on information gathering and synthesis."""

from __future__ import annotations

from src.agents.base import AgentResponse
from src.agents.react import ReActAgent
from src.llm.provider import BaseLLMProvider
from src.memory.base import BaseMemory
from src.tools.registry import ToolRegistry

_DEFAULT_SYSTEM_PROMPT = (
    "You are a research agent. Gather information thoroughly using search "
    "and extraction tools. Synthesise findings into clear, factual summaries.\n\n"
    "Guidelines:\n"
    "- Search for multiple perspectives on a topic\n"
    "- Cross-reference information across sources\n"
    "- Clearly distinguish facts from opinions\n"
    "- Cite sources when possible\n"
    "- Summarise findings concisely at the end"
)


class ResearchAgent(ReActAgent):
    """Gathers and synthesises information using search and extraction tools."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        tools: ToolRegistry | None = None,
        memory: BaseMemory | None = None,
        system_prompt: str | None = None,
        max_steps: int = 15,
        name: str = "researcher",
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

    async def research(self, topic: str, depth: str = "standard") -> AgentResponse:
        """Research *topic* at the given depth (standard / deep)."""
        prompt = (
            f"Research the following topic thoroughly:\n\n"
            f"{topic}\n\n"
            f"Depth: {depth}\n"
            f"Provide a comprehensive summary with key findings."
        )
        return await self.run(prompt)

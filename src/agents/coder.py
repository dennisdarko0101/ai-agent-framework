"""Coder agent — writes, debugs, and tests code."""

from __future__ import annotations

from src.agents.base import AgentResponse
from src.agents.react import ReActAgent
from src.llm.provider import BaseLLMProvider
from src.memory.base import BaseMemory
from src.tools.registry import ToolRegistry

_DEFAULT_SYSTEM_PROMPT = (
    "You are a coding agent. Write clean, well-tested code.\n\n"
    "Guidelines:\n"
    "- Follow best practices for the target language\n"
    "- Include error handling\n"
    "- Use the code executor tool to verify your solutions\n"
    "- Keep code simple and readable\n"
    "- Explain your approach before writing code"
)


class CoderAgent(ReActAgent):
    """Writes, debugs, and verifies code using tool execution."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        tools: ToolRegistry | None = None,
        memory: BaseMemory | None = None,
        system_prompt: str | None = None,
        max_steps: int = 15,
        name: str = "coder",
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

    async def write_code(self, spec: str, language: str = "python") -> AgentResponse:
        """Generate code from a specification."""
        prompt = (
            f"Write {language} code for the following specification:\n\n"
            f"{spec}\n\n"
            f"Verify the code works using the code executor tool."
        )
        return await self.run(prompt)

    async def debug_code(self, code: str, error: str) -> AgentResponse:
        """Diagnose and fix a code error."""
        prompt = (
            f"Debug the following code:\n\n```\n{code}\n```\n\n"
            f"Error:\n{error}\n\n"
            f"Find the root cause, fix the code, and verify the fix works."
        )
        return await self.run(prompt)

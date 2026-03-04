"""ReAct agent — Reasoning + Acting loop with native tool calling."""

from __future__ import annotations

import time
from typing import Any

from src.agents.base import AgentAction, AgentResponse, AgentStep, BaseAgent
from src.llm.provider import BaseLLMProvider
from src.memory.base import BaseMemory
from src.tools.registry import ToolRegistry
from src.utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are an AI assistant that solves tasks step by step.\n\n"
    "Think carefully before acting. When you need information or need to "
    "perform an action, use the available tools. When you have enough "
    "information to answer, provide your final response directly without "
    "calling any tools.\n\n"
    "Always explain your reasoning before using a tool."
)


class ReActAgent(BaseAgent):
    """Agent that follows the Reason → Act → Observe loop.

    On each step the LLM either:
    * Requests one or more tool calls — the agent executes them and feeds
      the results back for the next iteration.
    * Returns plain text — treated as the final answer and the loop ends.

    The loop is capped at *max_steps* to prevent runaway execution.
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        tools: ToolRegistry | None = None,
        memory: BaseMemory | None = None,
        system_prompt: str | None = None,
        max_steps: int = 10,
        name: str = "react",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(
            llm=llm,
            tools=tools,
            memory=memory,
            system_prompt=system_prompt if system_prompt is not None else _DEFAULT_SYSTEM_PROMPT,
            max_steps=max_steps,
            name=name,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ── Main loop ────────────────────────────────────────────────────

    async def run(self, task: str) -> AgentResponse:
        start = time.monotonic()
        steps: list[AgentStep] = []
        messages: list[dict[str, Any]] = []
        total_tokens = 0

        # Load conversation context from memory
        if self.memory:
            context = await self.memory.get_context()
            messages.extend(m.to_dict() for m in context)

        # Append the user task
        messages.append({"role": "user", "content": task})
        if self.memory:
            await self.memory.add("user", task)

        # Collect tool schemas
        tool_schemas = self.tools.get_schemas() if self.tools else None

        for step_num in range(1, self.max_steps + 1):
            log.debug("react_step", agent=self.name, step=step_num)

            response = await self.llm.generate(
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                system=self._build_system_prompt(),
            )
            total_tokens += response.usage.total_tokens

            # ── No tool calls → final answer ──────────────────────
            if not response.has_tool_calls:
                steps.append(AgentStep(
                    step_number=step_num,
                    reasoning=response.content,
                ))
                if self.memory and response.content:
                    await self.memory.add("assistant", response.content)
                return AgentResponse(
                    output=response.content,
                    steps=steps,
                    success=True,
                    total_tokens=total_tokens,
                    execution_time=time.monotonic() - start,
                )

            # ── Execute tool calls ────────────────────────────────
            actions: list[AgentAction] = []
            result_strings: list[str] = []
            error_flags: list[bool] = []

            for tc in response.tool_calls:
                if self.tools:
                    result = await self.tools.execute(tc.tool_name, tc.arguments)
                    actions.append(AgentAction(
                        tool_name=tc.tool_name,
                        arguments=tc.arguments,
                        call_id=tc.call_id,
                        result_output=result.output,
                        result_error=result.error,
                        success=result.success,
                    ))
                    result_strings.append(result.output if result.success else (result.error or ""))
                    error_flags.append(not result.success)
                else:
                    actions.append(AgentAction(
                        tool_name=tc.tool_name,
                        arguments=tc.arguments,
                        call_id=tc.call_id,
                        result_error="No tool registry configured",
                        success=False,
                    ))
                    result_strings.append("No tool registry configured")
                    error_flags.append(True)

            # Build observation text
            obs_parts: list[str] = []
            for action in actions:
                if action.success:
                    obs_parts.append(f"[{action.tool_name}]: {action.result_output}")
                else:
                    obs_parts.append(f"[{action.tool_name}] ERROR: {action.result_error}")
            observation = "\n".join(obs_parts)

            steps.append(AgentStep(
                step_number=step_num,
                reasoning=response.content,
                actions=actions,
                observation=observation,
            ))

            # Append messages for multi-turn tool use
            messages.append(self.llm.build_assistant_message(response))
            messages.extend(
                self.llm.build_tool_result_messages(
                    response.tool_calls, result_strings, error_flags,
                )
            )

        # ── Max steps exhausted ───────────────────────────────────────
        return AgentResponse(
            output="Maximum steps reached without a final answer.",
            steps=steps,
            success=False,
            error="max_steps_exceeded",
            total_tokens=total_tokens,
            execution_time=time.monotonic() - start,
        )

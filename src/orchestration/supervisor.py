"""Agent supervisor — delegates sub-tasks to specialist agents and synthesises results."""

from __future__ import annotations

import time

from src.agents.base import AgentResponse, AgentStep, BaseAgent
from src.llm.provider import BaseLLMProvider
from src.utils.logger import get_logger

log = get_logger(__name__)


class AgentSupervisor:
    """Coordinates multiple agents by deciding which agent to invoke next.

    On each round the supervisor LLM either:
    * Delegates by outputting ``DELEGATE <agent_name>: <sub-task>``
    * Finishes by outputting ``DONE: <final answer>``

    The loop is capped at *max_rounds*.
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        agents: dict[str, BaseAgent],
        max_rounds: int = 5,
    ) -> None:
        if not agents:
            raise ValueError("Supervisor requires at least one agent")
        self.llm = llm
        self.agents = agents
        self.max_rounds = max_rounds

    async def run(self, task: str) -> AgentResponse:
        start = time.monotonic()
        all_steps: list[AgentStep] = []
        total_tokens = 0
        results_so_far: list[str] = []

        agent_descriptions = ", ".join(self.agents.keys())

        for round_num in range(1, self.max_rounds + 1):
            log.debug("supervisor_round", round=round_num)

            prompt = self._build_prompt(task, agent_descriptions, results_so_far)
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            total_tokens += response.usage.total_tokens
            content = response.content.strip()

            # ── Check for DONE signal ─────────────────────────────
            if content.upper().startswith("DONE:"):
                final_answer = content.split(":", 1)[1].strip()
                return AgentResponse(
                    output=final_answer,
                    steps=all_steps,
                    success=True,
                    total_tokens=total_tokens,
                    execution_time=time.monotonic() - start,
                    metadata={
                        "rounds": round_num,
                        "delegations": len(results_so_far),
                    },
                )

            # ── Parse delegation ──────────────────────────────────
            agent_name, sub_task = self._parse_delegation(content)

            if agent_name not in self.agents:
                results_so_far.append(
                    f"[supervisor] Unknown agent {agent_name!r}. "
                    f"Available: {agent_descriptions}"
                )
                continue

            log.info("supervisor_delegate", agent=agent_name, task_preview=sub_task[:80])
            agent = self.agents[agent_name]
            result = await agent.run(sub_task)
            all_steps.extend(result.steps)
            total_tokens += result.total_tokens

            summary = result.output if result.success else f"ERROR: {result.error}"
            results_so_far.append(f"[{agent_name}]: {summary}")

        # ── Max rounds exhausted ──────────────────────────────────
        combined = "\n\n".join(results_so_far) if results_so_far else "No results produced."
        return AgentResponse(
            output=combined,
            steps=all_steps,
            success=False,
            error="max_rounds_exceeded",
            total_tokens=total_tokens,
            execution_time=time.monotonic() - start,
            metadata={"rounds": self.max_rounds, "delegations": len(results_so_far)},
        )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(
        task: str,
        agent_names: str,
        results: list[str],
    ) -> str:
        parts = [
            f"You are a supervisor coordinating specialist agents: {agent_names}.",
            f"\nMain task: {task}",
        ]
        if results:
            parts.append("\nResults so far:")
            for r in results:
                parts.append(f"  {r}")
        parts.append(
            "\nDecide what to do next. Either:\n"
            "  DELEGATE <agent_name>: <sub-task description>\n"
            "  DONE: <final synthesised answer>"
        )
        return "\n".join(parts)

    @staticmethod
    def _parse_delegation(text: str) -> tuple[str, str]:
        """Extract ``(agent_name, sub_task)`` from a DELEGATE line."""
        cleaned = text.strip()
        if cleaned.upper().startswith("DELEGATE"):
            cleaned = cleaned.split(None, 1)[1] if " " in cleaned else ""
        if ":" in cleaned:
            name, sub = cleaned.split(":", 1)
            return name.strip(), sub.strip()
        return cleaned.strip(), cleaned.strip()

    def __repr__(self) -> str:
        agents = list(self.agents.keys())
        return f"<AgentSupervisor agents={agents}>"

"""Agent pipeline — sequential chain where each agent's output feeds the next."""

from __future__ import annotations

import time
from typing import Any

from src.agents.base import AgentResponse, AgentStep, BaseAgent
from src.utils.logger import get_logger

log = get_logger(__name__)


class AgentPipeline:
    """Run agents in sequence — the output of stage *n* becomes the input of stage *n+1*.

    If any stage fails the pipeline stops and returns the error.
    """

    def __init__(self, agents: list[BaseAgent]) -> None:
        if not agents:
            raise ValueError("Pipeline requires at least one agent")
        self.agents = agents

    async def run(self, task: str) -> AgentResponse:
        start = time.monotonic()
        current_input = task
        all_steps: list[AgentStep] = []
        total_tokens = 0
        stage_outputs: list[dict[str, Any]] = []

        for i, agent in enumerate(self.agents):
            log.debug("pipeline_stage", stage=i, agent=agent.name)
            response = await agent.run(current_input)
            all_steps.extend(response.steps)
            total_tokens += response.total_tokens
            stage_outputs.append({"agent": agent.name, "output": response.output})

            if not response.success:
                return AgentResponse(
                    output=response.output,
                    steps=all_steps,
                    success=False,
                    error=f"Pipeline failed at stage {i} ({agent.name}): {response.error}",
                    total_tokens=total_tokens,
                    execution_time=time.monotonic() - start,
                    metadata={
                        "failed_stage": i,
                        "failed_agent": agent.name,
                        "stage_outputs": stage_outputs,
                    },
                )

            current_input = response.output

        return AgentResponse(
            output=current_input,
            steps=all_steps,
            success=True,
            total_tokens=total_tokens,
            execution_time=time.monotonic() - start,
            metadata={
                "stages_completed": len(self.agents),
                "stage_outputs": stage_outputs,
            },
        )

    def __repr__(self) -> str:
        names = [a.name for a in self.agents]
        return f"<AgentPipeline stages={names}>"

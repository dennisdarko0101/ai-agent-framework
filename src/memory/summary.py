"""Summary memory — compresses older messages into a running LLM summary."""

from __future__ import annotations

from typing import Any

from src.llm.provider import BaseLLMProvider
from src.memory.base import BaseMemory, Message, estimate_tokens
from src.utils.logger import get_logger

log = get_logger(__name__)


class SummaryMemory(BaseMemory):
    """Keeps a running summary plus the most recent messages.

    When the buffer exceeds *summarize_threshold*, the older messages are
    compressed into ``current_summary`` via an LLM call and only the last
    *recent_count* messages are retained.

    ``get_context`` returns:
        1. A system message containing the summary (if any).
        2. The recent unconsumed messages.
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        summarize_threshold: int = 10,
        recent_count: int = 5,
    ) -> None:
        self._llm = llm
        self._summarize_threshold = summarize_threshold
        self._recent_count = recent_count
        self._messages: list[Message] = []
        self._summary: str = ""
        self._total_added: int = 0

    # ── Interface ───────────────────────────────────────────────────────

    async def add(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        msg = Message(role=role, content=content, metadata=metadata or {})
        self._messages.append(msg)
        self._total_added += 1

        if len(self._messages) > self._summarize_threshold:
            await self._summarize()

    async def get_context(self, max_tokens: int = 4000) -> list[Message]:
        context: list[Message] = []

        if self._summary:
            summary_msg = Message(
                role="system",
                content=f"Conversation summary so far:\n{self._summary}",
            )
            context.append(summary_msg)

        tokens_used = sum(estimate_tokens(m.content) for m in context)
        for msg in self._messages:
            msg_tokens = estimate_tokens(msg.content)
            if tokens_used + msg_tokens > max_tokens:
                break
            context.append(msg)
            tokens_used += msg_tokens

        return context

    async def search(self, query: str, k: int = 5) -> list[Message]:
        query_lower = query.lower()
        hits: list[Message] = []
        for msg in reversed(self._messages):
            if query_lower in msg.content.lower():
                hits.append(msg)
                if len(hits) >= k:
                    break
        hits.reverse()
        return hits

    def clear(self) -> None:
        self._messages.clear()
        self._summary = ""
        self._total_added = 0

    def get_stats(self) -> dict[str, Any]:
        return {
            "type": "summary",
            "message_count": len(self._messages),
            "total_added": self._total_added,
            "has_summary": bool(self._summary),
            "summary_tokens": estimate_tokens(self._summary) if self._summary else 0,
        }

    # ── Internal ────────────────────────────────────────────────────────

    async def _summarize(self) -> None:
        """Compress older messages into ``_summary``, keeping recent ones."""
        to_summarize = self._messages[: -self._recent_count]
        conversation = "\n".join(f"{m.role}: {m.content}" for m in to_summarize)

        if self._summary:
            prompt = (
                f"Previous summary:\n{self._summary}\n\n"
                f"New conversation:\n{conversation}\n\n"
                "Update the summary to include the new information. Be concise."
            )
        else:
            prompt = (
                f"Summarise this conversation concisely:\n{conversation}"
            )

        try:
            response = await self._llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )
            self._summary = response.content
            self._messages = list(self._messages[-self._recent_count :])
            log.info("summary_updated", summary_len=len(self._summary))
        except Exception as exc:
            log.warning("summary_failed", error=str(exc))

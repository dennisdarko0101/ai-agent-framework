"""Sliding-window conversation memory with token-aware context building."""

from __future__ import annotations

from collections import deque
from typing import Any

from src.memory.base import BaseMemory, Message, estimate_tokens
from src.utils.logger import get_logger

log = get_logger(__name__)


class ConversationMemory(BaseMemory):
    """Fixed-size buffer of the most recent messages (FIFO eviction).

    ``get_context`` walks backwards from the newest message and includes as
    many as fit within *max_tokens*.
    """

    def __init__(self, max_messages: int = 20) -> None:
        self.max_messages = max_messages
        self._messages: deque[Message] = deque(maxlen=max_messages)

    # ── Interface ───────────────────────────────────────────────────────

    async def add(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        msg = Message(role=role, content=content, metadata=metadata or {})
        self._messages.append(msg)
        log.debug("conversation_add", role=role, buffer=len(self._messages))

    async def get_context(self, max_tokens: int = 4000) -> list[Message]:
        """Return the most recent messages that fit within *max_tokens*."""
        selected: list[Message] = []
        tokens_used = 0
        for msg in reversed(self._messages):
            msg_tokens = estimate_tokens(msg.content)
            if tokens_used + msg_tokens > max_tokens:
                break
            selected.append(msg)
            tokens_used += msg_tokens
        selected.reverse()
        return selected

    async def search(self, query: str, k: int = 5) -> list[Message]:
        """Naive keyword search — returns the last *k* messages whose content
        contains *query* (case-insensitive).  For semantic search use
        ``VectorMemory``.
        """
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

    def get_stats(self) -> dict[str, Any]:
        total_tokens = sum(estimate_tokens(m.content) for m in self._messages)
        return {
            "type": "conversation",
            "message_count": len(self._messages),
            "max_messages": self.max_messages,
            "total_tokens": total_tokens,
        }

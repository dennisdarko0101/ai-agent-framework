"""Composite memory — merges results from multiple memory backends."""

from __future__ import annotations

from typing import Any

from src.memory.base import BaseMemory, Message, estimate_tokens
from src.utils.logger import get_logger

log = get_logger(__name__)


class CompositeMemory(BaseMemory):
    """Combines several memory backends and deduplicates their output.

    * ``add`` broadcasts to every backend.
    * ``get_context`` merges results from all backends, deduplicates by
      message id, and trims to *max_tokens*.
    * ``search`` unions results from all backends.
    """

    def __init__(self, memories: list[BaseMemory]) -> None:
        if not memories:
            raise ValueError("CompositeMemory requires at least one backend")
        self.memories = memories

    # ── Interface ───────────────────────────────────────────────────────

    async def add(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        for mem in self.memories:
            await mem.add(role, content, metadata)

    async def get_context(self, max_tokens: int = 4000) -> list[Message]:
        seen_ids: set[str] = set()
        seen_content: set[tuple[str, str]] = set()
        unique: list[Message] = []

        for mem in self.memories:
            for msg in await mem.get_context(max_tokens):
                content_key = (msg.role, msg.content)
                if msg.id not in seen_ids and content_key not in seen_content:
                    seen_ids.add(msg.id)
                    seen_content.add(content_key)
                    unique.append(msg)

        # Chronological order, trimmed to budget
        unique.sort(key=lambda m: m.timestamp)
        result: list[Message] = []
        tokens_used = 0
        for msg in unique:
            t = estimate_tokens(msg.content)
            if tokens_used + t > max_tokens:
                break
            result.append(msg)
            tokens_used += t
        return result

    async def search(self, query: str, k: int = 5) -> list[Message]:
        seen_ids: set[str] = set()
        seen_content: set[tuple[str, str]] = set()
        results: list[Message] = []
        for mem in self.memories:
            for msg in await mem.search(query, k):
                content_key = (msg.role, msg.content)
                if msg.id not in seen_ids and content_key not in seen_content:
                    seen_ids.add(msg.id)
                    seen_content.add(content_key)
                    results.append(msg)
        return results[:k]

    def clear(self) -> None:
        for mem in self.memories:
            mem.clear()

    def get_stats(self) -> dict[str, Any]:
        children = [mem.get_stats() for mem in self.memories]
        return {
            "type": "composite",
            "backends": len(self.memories),
            "children": children,
        }

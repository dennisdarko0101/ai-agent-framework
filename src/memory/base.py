"""Base memory abstractions — extend BaseMemory to create new memory backends."""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count.  Uses tiktoken when available, else ~4 chars/token."""
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    except Exception:  # ImportError, KeyError, etc.
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single message in the conversation history."""

    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, str]:
        """Convert to the ``{"role": ..., "content": ...}`` format expected by LLM providers."""
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseMemory(ABC):
    """Abstract memory backend.

    All memory types share this interface so agents can swap implementations
    without changing their reasoning loop.  Methods are async because some
    backends (summary, vector) perform I/O.
    """

    @abstractmethod
    async def add(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a message to memory."""
        ...

    @abstractmethod
    async def get_context(self, max_tokens: int = 4000) -> list[Message]:
        """Return messages suitable for the next LLM call, respecting *max_tokens*."""
        ...

    @abstractmethod
    async def search(self, query: str, k: int = 5) -> list[Message]:
        """Retrieve the *k* most relevant messages for *query*."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Erase all stored messages."""
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic counters (message count, token estimate, etc.)."""
        ...

"""Memory system — pluggable backends for conversation history and retrieval."""

from src.memory.base import BaseMemory, Message, estimate_tokens
from src.memory.composite import CompositeMemory
from src.memory.conversation import ConversationMemory
from src.memory.summary import SummaryMemory

__all__ = [
    "BaseMemory",
    "CompositeMemory",
    "ConversationMemory",
    "Message",
    "SummaryMemory",
    "VectorMemory",
    "estimate_tokens",
]

# VectorMemory depends on chromadb which may not be importable on all platforms.
import contextlib

with contextlib.suppress(Exception):
    from src.memory.vector_memory import VectorMemory

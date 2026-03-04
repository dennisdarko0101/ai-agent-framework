"""Tests for the memory system — 35+ tests covering all backends."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.base import Message, estimate_tokens
from src.memory.composite import CompositeMemory
from src.memory.conversation import ConversationMemory
from src.memory.summary import SummaryMemory

# ───────── Message dataclass ──────────────────────────────────────────

class TestMessage:
    def test_defaults(self) -> None:
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert isinstance(msg.timestamp, float)
        assert len(msg.id) == 12
        assert msg.metadata == {}

    def test_with_metadata(self) -> None:
        msg = Message(role="assistant", content="hi", metadata={"source": "test"})
        assert msg.metadata["source"] == "test"

    def test_to_dict(self) -> None:
        msg = Message(role="user", content="hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "hello"}

    def test_unique_ids(self) -> None:
        a = Message(role="user", content="a")
        b = Message(role="user", content="b")
        assert a.id != b.id


# ───────── estimate_tokens ────────────────────────────────────────────

class TestEstimateTokens:
    def test_nonempty(self) -> None:
        tokens = estimate_tokens("Hello world, this is a test.")
        assert tokens > 0

    def test_empty(self) -> None:
        assert estimate_tokens("") >= 0

    def test_proportional(self) -> None:
        short = estimate_tokens("hello")
        long = estimate_tokens("hello " * 100)
        assert long > short


# ───────── ConversationMemory ─────────────────────────────────────────

class TestConversationMemory:
    @pytest.fixture
    def mem(self) -> ConversationMemory:
        return ConversationMemory(max_messages=5)

    @pytest.mark.asyncio
    async def test_add_message(self, mem: ConversationMemory) -> None:
        await mem.add("user", "hello")
        stats = mem.get_stats()
        assert stats["message_count"] == 1

    @pytest.mark.asyncio
    async def test_get_context_returns_messages(self, mem: ConversationMemory) -> None:
        await mem.add("user", "hi")
        await mem.add("assistant", "hello!")
        ctx = await mem.get_context()
        assert len(ctx) == 2
        assert ctx[0].role == "user"
        assert ctx[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_fifo_eviction(self, mem: ConversationMemory) -> None:
        for i in range(8):
            await mem.add("user", f"msg-{i}")
        stats = mem.get_stats()
        assert stats["message_count"] == 5
        ctx = await mem.get_context()
        assert ctx[0].content == "msg-3"

    @pytest.mark.asyncio
    async def test_get_context_token_limit(self) -> None:
        mem = ConversationMemory(max_messages=100)
        for i in range(20):
            await mem.add("user", f"Message number {i} with some extra text padding")
        # Very small token limit should return fewer messages
        ctx = await mem.get_context(max_tokens=20)
        assert len(ctx) < 20

    @pytest.mark.asyncio
    async def test_empty_context(self, mem: ConversationMemory) -> None:
        ctx = await mem.get_context()
        assert ctx == []

    @pytest.mark.asyncio
    async def test_search_keyword(self, mem: ConversationMemory) -> None:
        await mem.add("user", "I like Python programming")
        await mem.add("user", "The weather is nice")
        await mem.add("user", "Python is great")
        results = await mem.search("Python")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_no_match(self, mem: ConversationMemory) -> None:
        await mem.add("user", "hello")
        results = await mem.search("zzzzz")
        assert results == []

    def test_clear(self) -> None:
        mem = ConversationMemory()
        # Sync check — clear is not async
        mem.clear()
        assert mem.get_stats()["message_count"] == 0

    def test_get_stats(self) -> None:
        mem = ConversationMemory(max_messages=10)
        stats = mem.get_stats()
        assert stats["type"] == "conversation"
        assert stats["max_messages"] == 10
        assert stats["message_count"] == 0
        assert stats["total_tokens"] == 0


# ───────── SummaryMemory ──────────────────────────────────────────────

class TestSummaryMemory:
    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        llm = AsyncMock()
        llm.generate.return_value = MagicMock(content="This is the summary.")
        return llm

    @pytest.fixture
    def mem(self, mock_llm: AsyncMock) -> SummaryMemory:
        return SummaryMemory(
            llm=mock_llm,
            summarize_threshold=5,
            recent_count=2,
        )

    @pytest.mark.asyncio
    async def test_add_below_threshold(self, mem: SummaryMemory, mock_llm: AsyncMock) -> None:
        for i in range(3):
            await mem.add("user", f"msg-{i}")
        mock_llm.generate.assert_not_awaited()
        assert mem.get_stats()["message_count"] == 3

    @pytest.mark.asyncio
    async def test_auto_summarize_trigger(self, mem: SummaryMemory, mock_llm: AsyncMock) -> None:
        for i in range(6):
            await mem.add("user", f"msg-{i}")
        # Should have triggered summarization
        mock_llm.generate.assert_awaited()
        assert mem.get_stats()["has_summary"] is True
        # Only recent_count messages should remain
        assert mem.get_stats()["message_count"] == 2

    @pytest.mark.asyncio
    async def test_get_context_without_summary(
        self, mem: SummaryMemory, mock_llm: AsyncMock,
    ) -> None:
        await mem.add("user", "hello")
        ctx = await mem.get_context()
        # No summary yet — just the message
        assert len(ctx) == 1
        assert ctx[0].role == "user"

    @pytest.mark.asyncio
    async def test_get_context_with_summary(
        self, mem: SummaryMemory, mock_llm: AsyncMock,
    ) -> None:
        for i in range(6):
            await mem.add("user", f"msg-{i}")
        ctx = await mem.get_context()
        # First message should be the system summary
        assert ctx[0].role == "system"
        assert "summary" in ctx[0].content.lower()
        # Followed by recent messages
        assert len(ctx) > 1

    @pytest.mark.asyncio
    async def test_clear_resets_summary(
        self, mem: SummaryMemory, mock_llm: AsyncMock,
    ) -> None:
        for i in range(6):
            await mem.add("user", f"msg-{i}")
        mem.clear()
        assert mem.get_stats()["has_summary"] is False
        assert mem.get_stats()["message_count"] == 0

    def test_get_stats(self, mem: SummaryMemory) -> None:
        stats = mem.get_stats()
        assert stats["type"] == "summary"
        assert stats["total_added"] == 0

    @pytest.mark.asyncio
    async def test_summarize_failure_keeps_messages(
        self, mock_llm: AsyncMock,
    ) -> None:
        mock_llm.generate.side_effect = RuntimeError("LLM down")
        mem = SummaryMemory(llm=mock_llm, summarize_threshold=3, recent_count=1)
        for i in range(5):
            await mem.add("user", f"msg-{i}")
        # Summarization failed — all messages should still be present
        assert mem.get_stats()["message_count"] == 5


# ───────── VectorMemory (mocked ChromaDB) ─────────────────────────────

def _make_mock_chromadb(mock_collection: MagicMock) -> MagicMock:
    """Create a fake ``chromadb`` module suitable for ``sys.modules``."""
    mock_client_cls = MagicMock()
    mock_client_instance = MagicMock()
    mock_client_instance.get_or_create_collection.return_value = mock_collection
    mock_client_cls.return_value = mock_client_instance

    mock_persistent_cls = MagicMock()
    mock_persistent_cls.return_value = mock_client_instance

    fake_chromadb = MagicMock()
    fake_chromadb.Client = mock_client_cls
    fake_chromadb.PersistentClient = mock_persistent_cls
    return fake_chromadb


class TestVectorMemory:
    """All ChromaDB calls are mocked so tests don't need embeddings."""

    @pytest.fixture
    def mock_collection(self) -> MagicMock:
        coll = MagicMock()
        coll.count.return_value = 0
        coll.name = "test_memory"
        return coll

    @pytest.fixture
    def mem(self, mock_collection: MagicMock) -> Any:
        import sys

        fake_chromadb = _make_mock_chromadb(mock_collection)
        original = sys.modules.get("chromadb")
        sys.modules["chromadb"] = fake_chromadb
        try:
            # Force re-import so the lazy import picks up our mock
            import importlib

            import src.memory.vector_memory as vm_mod

            importlib.reload(vm_mod)
            memory = vm_mod.VectorMemory(collection_name="test_memory")
        finally:
            if original is not None:
                sys.modules["chromadb"] = original
            else:
                sys.modules.pop("chromadb", None)
        return memory

    @pytest.mark.asyncio
    async def test_add_stores_in_collection(
        self, mem: Any, mock_collection: MagicMock,
    ) -> None:
        await mem.add("user", "hello world")
        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args
        assert call_kwargs[1]["documents"] == ["hello world"]

    @pytest.mark.asyncio
    async def test_search_returns_results(
        self, mem: Any, mock_collection: MagicMock,
    ) -> None:
        mock_collection.count.return_value = 3
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc one", "doc two"]],
            "metadatas": [[
                {"role": "user", "timestamp": 1.0},
                {"role": "assistant", "timestamp": 2.0},
            ]],
        }
        results = await mem.search("test query", k=5)
        assert len(results) == 2
        assert results[0].content == "doc one"
        assert results[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(
        self, mem: Any, mock_collection: MagicMock,
    ) -> None:
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["only user msg"]],
            "metadatas": [[{"role": "user", "timestamp": 1.0}]],
        }
        results = await mem.search("test", k=5, where={"role": "user"})
        assert len(results) == 1
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["where"] == {"role": "user"}

    @pytest.mark.asyncio
    async def test_get_context_includes_recent(
        self, mem: Any, mock_collection: MagicMock,
    ) -> None:
        mock_collection.count.return_value = 0
        mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        await mem.add("user", "first")
        await mem.add("assistant", "second")
        ctx = await mem.get_context()
        assert len(ctx) == 2

    @pytest.mark.asyncio
    async def test_deduplication(
        self, mem: Any, mock_collection: MagicMock,
    ) -> None:
        """Messages already in recent context shouldn't appear twice."""
        await mem.add("user", "hello")
        msg_id = mem._messages[0].id

        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "ids": [[msg_id]],
            "documents": [["hello"]],
            "metadatas": [[{"role": "user", "timestamp": 1.0}]],
        }
        ctx = await mem.get_context()
        ids = [m.id for m in ctx]
        assert len(ids) == len(set(ids))

    def test_clear(self, mem: Any, mock_collection: MagicMock) -> None:
        mem._messages.append(Message(role="user", content="x"))
        mem.clear()
        assert mem.get_stats()["message_count"] == 0

    def test_get_stats(self, mem: Any, mock_collection: MagicMock) -> None:
        mock_collection.count.return_value = 42
        stats = mem.get_stats()
        assert stats["type"] == "vector"
        assert stats["collection_count"] == 42


# ───────── CompositeMemory ────────────────────────────────────────────

class TestCompositeMemory:
    @pytest.fixture
    def conv_mem(self) -> ConversationMemory:
        return ConversationMemory(max_messages=10)

    @pytest.fixture
    def conv_mem2(self) -> ConversationMemory:
        return ConversationMemory(max_messages=10)

    @pytest.fixture
    def composite(
        self, conv_mem: ConversationMemory, conv_mem2: ConversationMemory,
    ) -> CompositeMemory:
        return CompositeMemory(memories=[conv_mem, conv_mem2])

    @pytest.mark.asyncio
    async def test_add_propagates(
        self,
        composite: CompositeMemory,
        conv_mem: ConversationMemory,
        conv_mem2: ConversationMemory,
    ) -> None:
        await composite.add("user", "broadcast")
        assert conv_mem.get_stats()["message_count"] == 1
        assert conv_mem2.get_stats()["message_count"] == 1

    @pytest.mark.asyncio
    async def test_get_context_merges(self, composite: CompositeMemory) -> None:
        await composite.add("user", "hello")
        ctx = await composite.get_context()
        # Both backends received the message but dedup should yield 1
        assert len(ctx) == 1

    @pytest.mark.asyncio
    async def test_get_context_deduplicates(
        self,
        conv_mem: ConversationMemory,
        conv_mem2: ConversationMemory,
    ) -> None:
        # Add different messages to each backend directly
        await conv_mem.add("user", "unique-A")
        await conv_mem2.add("user", "unique-B")
        composite = CompositeMemory(memories=[conv_mem, conv_mem2])
        ctx = await composite.get_context()
        assert len(ctx) == 2

    @pytest.mark.asyncio
    async def test_search_combines(self, composite: CompositeMemory) -> None:
        await composite.add("user", "Python rocks")
        await composite.add("user", "I love JavaScript")
        results = await composite.search("Python")
        assert any("Python" in m.content for m in results)

    def test_clear_all(
        self,
        composite: CompositeMemory,
        conv_mem: ConversationMemory,
        conv_mem2: ConversationMemory,
    ) -> None:
        composite.clear()
        assert conv_mem.get_stats()["message_count"] == 0
        assert conv_mem2.get_stats()["message_count"] == 0

    def test_get_stats(self, composite: CompositeMemory) -> None:
        stats = composite.get_stats()
        assert stats["type"] == "composite"
        assert stats["backends"] == 2
        assert len(stats["children"]) == 2

    def test_empty_memories_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CompositeMemory(memories=[])

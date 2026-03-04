"""Vector memory — stores messages as embeddings in ChromaDB for semantic retrieval."""

from __future__ import annotations

from typing import Any

from src.memory.base import BaseMemory, Message, estimate_tokens
from src.utils.logger import get_logger

log = get_logger(__name__)


class VectorMemory(BaseMemory):
    """Embeds every message into ChromaDB and retrieves by semantic similarity.

    ``get_context`` returns recent messages combined with the most relevant
    retrieved messages, with deduplication.
    """

    def __init__(
        self,
        collection_name: str = "agent_memory",
        persist_dir: str | None = None,
        recent_count: int = 5,
    ) -> None:
        import chromadb  # lazy — chromadb has heavy deps that may not be installed

        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
        )
        self._messages: list[Message] = []
        self._recent_count = recent_count

    # ── Interface ───────────────────────────────────────────────────────

    async def add(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        meta = dict(metadata) if metadata else {}
        msg = Message(role=role, content=content, metadata=meta)
        self._messages.append(msg)

        # Store in ChromaDB
        doc_meta: dict[str, Any] = {"role": role, "timestamp": msg.timestamp}
        doc_meta.update(meta)
        try:
            self._collection.add(
                documents=[content],
                metadatas=[doc_meta],
                ids=[msg.id],
            )
        except Exception as exc:
            log.warning("vector_add_failed", error=str(exc))

    async def get_context(self, max_tokens: int = 4000) -> list[Message]:
        """Return recent messages plus relevant retrieved messages (deduplicated)."""
        recent = self._messages[-self._recent_count :]
        recent_ids = {m.id for m in recent}

        # Retrieve relevant older messages via the most recent user message
        retrieved: list[Message] = []
        user_msgs = [m for m in recent if m.role == "user"]
        if user_msgs and len(self._messages) > self._recent_count:
            query = user_msgs[-1].content
            search_results = await self.search(query, k=self._recent_count)
            for msg in search_results:
                if msg.id not in recent_ids:
                    retrieved.append(msg)
                    recent_ids.add(msg.id)

        # Merge: retrieved context first (older), then recent
        merged = sorted(retrieved + list(recent), key=lambda m: m.timestamp)

        # Trim to fit token budget
        result: list[Message] = []
        tokens_used = 0
        for msg in merged:
            t = estimate_tokens(msg.content)
            if tokens_used + t > max_tokens:
                break
            result.append(msg)
            tokens_used += t
        return result

    async def search(
        self,
        query: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[Message]:
        """Semantic search over stored messages.

        Args:
            query: Natural-language query.
            k: Maximum results.
            where: Optional ChromaDB metadata filter
                   (e.g. ``{"role": "user"}``).
        """
        try:
            kwargs: dict[str, Any] = {
                "query_texts": [query],
                "n_results": min(k, self._collection.count() or 1),
            }
            if where:
                kwargs["where"] = where
            results = self._collection.query(**kwargs)
        except Exception as exc:
            log.warning("vector_search_failed", error=str(exc))
            return []

        messages: list[Message] = []
        if results and results.get("ids"):
            ids = results["ids"][0]
            docs = results["documents"][0] if results.get("documents") else [""] * len(ids)
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
            for doc_id, doc, meta in zip(ids, docs, metas, strict=False):
                messages.append(
                    Message(
                        id=doc_id,
                        role=meta.get("role", "unknown"),
                        content=doc,
                        metadata={k: v for k, v in meta.items() if k not in ("role", "timestamp")},
                        timestamp=meta.get("timestamp", 0.0),
                    )
                )
        return messages

    def clear(self) -> None:
        self._messages.clear()
        # Re-create collection to drop all embeddings
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(name=name)

    def get_stats(self) -> dict[str, Any]:
        return {
            "type": "vector",
            "message_count": len(self._messages),
            "collection_count": self._collection.count(),
            "total_tokens": sum(estimate_tokens(m.content) for m in self._messages),
        }

# Agents Guide

## Memory System Overview

The framework provides pluggable memory backends that control how much conversation history an agent retains and how it retrieves relevant past context.

All backends implement `BaseMemory` (async interface):

```python
await memory.add(role, content, metadata)  # store a message
await memory.get_context(max_tokens)       # get messages for the next LLM call
await memory.search(query, k)             # retrieve relevant past messages
memory.clear()                            # erase everything
memory.get_stats()                        # diagnostic counters
```

### Memory Types

| Type | Best for | Requires |
|------|----------|----------|
| `ConversationMemory` | Short tasks, chatbots | Nothing extra |
| `SummaryMemory` | Long conversations | LLM provider |
| `VectorMemory` | Knowledge-heavy agents | ChromaDB |
| `CompositeMemory` | Combining approaches | 2+ backends |

### ConversationMemory

Fixed-size sliding window (default 20 messages). When the buffer is full, the oldest message is evicted (FIFO). `get_context` walks backward from the newest message and includes as many as fit within the token budget.

```python
from src.memory.conversation import ConversationMemory

memory = ConversationMemory(max_messages=20)
```

**When to use:** Quick tasks, single-turn Q&A, chatbots where older context is disposable.

### SummaryMemory

Keeps a running LLM-generated summary plus the N most recent messages. When the message buffer exceeds a threshold, older messages are compressed into the summary.

```python
from src.memory.summary import SummaryMemory

memory = SummaryMemory(
    llm=provider,
    summarize_threshold=10,
    recent_count=5,
)
```

`get_context` returns:
1. A system message containing the summary (if any)
2. The recent unconsumed messages

**When to use:** Long conversations where you want to retain the big picture without blowing up the token budget.

### VectorMemory

Stores every message as an embedding in ChromaDB and retrieves semantically similar past messages when building context.

```python
from src.memory.vector_memory import VectorMemory

memory = VectorMemory(
    collection_name="agent_memory",
    persist_dir="./chroma_data",  # None for ephemeral
    recent_count=5,
)
```

`get_context` combines:
- The N most recent messages
- Semantically relevant older messages (queried using the latest user message)

Supports metadata filtering in `search(query, k, where={"role": "user"})`.

**When to use:** Research agents, knowledge workers, or any task where relevant older context matters more than recency.

### CompositeMemory

Merges multiple backends. `add` broadcasts to all; `get_context` unions and deduplicates.

```python
from src.memory.composite import CompositeMemory

memory = CompositeMemory(memories=[conv_memory, vector_memory])
```

**When to use:** When you want both a sliding window *and* semantic retrieval.

## How Memory Affects Agent Behaviour

- **Too little memory** — the agent forgets earlier instructions and repeats questions.
- **Too much memory** — the context window fills up, costs increase, and irrelevant old messages dilute focus.
- **Summary memory** strikes a balance by preserving the gist while keeping token usage stable.
- **Vector memory** is ideal when the agent needs to recall specific facts from hundreds of messages ago.

## Configuration

Memory settings in `.env` or environment variables:

```
MEMORY_TYPE=conversation          # conversation | summary | vector
MEMORY_MAX_MESSAGES=20            # ConversationMemory buffer size
SUMMARY_THRESHOLD=10              # messages before summary triggers
SUMMARY_RECENT_COUNT=5            # recent messages kept after summary
CHROMA_PERSIST_DIR=./chroma_data  # VectorMemory persistence (optional)
```

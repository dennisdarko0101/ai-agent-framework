# Agents Guide

## Agent System

The framework provides a ReAct-based agent architecture with specialised variants and multi-agent orchestration — all built from primitives, no LangChain dependency.

### Core Concepts

| Concept | Description |
|---------|-------------|
| `BaseAgent` | Abstract class — implement `run(task) -> AgentResponse` |
| `ReActAgent` | Reason → Act → Observe loop with native tool calling |
| `AgentResponse` | Final output with steps, tokens, timing, metadata |
| `AgentStep` | One think → act → observe cycle |
| `AgentAction` | A single tool invocation within a step |

### ReActAgent

The core agent follows the **ReAct** pattern (Reasoning + Acting):

1. Receive a task
2. Load memory context and tool schemas
3. Send to LLM → LLM either returns text (final answer) or tool calls
4. Execute tool calls, feed results back to LLM
5. Repeat until final answer or max_steps

```python
from src.agents.react import ReActAgent
from src.llm.provider import ProviderFactory
from src.tools.registry import ToolRegistry

llm = ProviderFactory.create()
tools = ToolRegistry.get_instance()
agent = ReActAgent(llm=llm, tools=tools, max_steps=10)

result = await agent.run("What is 2 + 2?")
print(result.output)       # "4"
print(result.total_tokens)  # token count
print(len(result.steps))    # reasoning steps taken
```

#### Constructor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm` | required | `BaseLLMProvider` instance |
| `tools` | `None` | `ToolRegistry` with registered tools |
| `memory` | `None` | `BaseMemory` for conversation history |
| `system_prompt` | ReAct default | Custom system prompt |
| `max_steps` | 10 | Safety cap on reasoning iterations |
| `name` | `"react"` | Agent name for logging |
| `temperature` | 0.7 | LLM temperature |
| `max_tokens` | 4096 | Max tokens per LLM response |

### Specialised Agents

All specialised agents extend `ReActAgent` with domain-specific system prompts and convenience methods.

#### PlannerAgent

Decomposes complex tasks into structured sub-task plans.

```python
from src.agents.planner import PlannerAgent

planner = PlannerAgent(llm=llm, tools=tools)
plan = await planner.create_plan("Build a REST API with authentication")
```

#### ResearchAgent

Focused on information gathering using search and extraction tools.

```python
from src.agents.researcher import ResearchAgent

researcher = ResearchAgent(llm=llm, tools=tools)
findings = await researcher.research("quantum computing", depth="deep")
```

#### CoderAgent

Writes, debugs, and verifies code.

```python
from src.agents.coder import CoderAgent

coder = CoderAgent(llm=llm, tools=tools)
code = await coder.write_code("A binary search function", language="python")
fix = await coder.debug_code("def f(): retrun 1", "SyntaxError: invalid syntax")
```

#### ReviewerAgent

Reviews and critiques content or agent outputs.

```python
from src.agents.reviewer import ReviewerAgent

reviewer = ReviewerAgent(llm=llm)
feedback = await reviewer.review(code.output, criteria="correctness, readability")
```

### Custom Agents

Extend `ReActAgent` for your own specialisation:

```python
from src.agents.react import ReActAgent
from src.agents.base import AgentResponse

class MyAgent(ReActAgent):
    def __init__(self, llm, **kwargs):
        super().__init__(
            llm=llm,
            system_prompt="You are a helpful assistant specialised in X.",
            name="my-agent",
            **kwargs,
        )

    async def do_task(self, task: str) -> AgentResponse:
        return await self.run(f"Perform X on: {task}")
```

---

## Orchestration Patterns

Three composable patterns for multi-agent coordination.

### AgentPipeline (Sequential)

Chain agents — the output of stage N feeds stage N+1.

```python
from src.orchestration.pipeline import AgentPipeline

pipe = AgentPipeline(agents=[planner, coder, reviewer])
result = await pipe.run("Build a calculator")
# planner output → coder input → reviewer input → final output
```

If any stage fails, the pipeline stops and returns the error with metadata about which stage failed.

### TaskRouter (Classification)

LLM classifies the task and routes it to the right agent.

```python
from src.orchestration.router import TaskRouter

router = TaskRouter(
    llm=llm,
    routes={
        "coding": coder,
        "research": researcher,
        "planning": planner,
    },
    default_agent=general_agent,  # fallback
)
result = await router.route("Write a sorting algorithm")
print(result.metadata["routed_to"])  # "coding"
```

### AgentSupervisor (Delegation)

A supervisor LLM decides which agent to invoke next, collects results, and synthesises a final answer.

```python
from src.orchestration.supervisor import AgentSupervisor

supervisor = AgentSupervisor(
    llm=llm,
    agents={"coder": coder, "reviewer": reviewer},
    max_rounds=5,
)
result = await supervisor.run("Build and review a function")
```

The supervisor uses two signals:
- `DELEGATE <agent>: <sub-task>` — invoke an agent
- `DONE: <answer>` — stop and return the final answer

---

## Memory System

Pluggable memory backends that control how much conversation history an agent retains.

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

---

## API Integration

All agents are accessible via the REST API and WebSocket.

### REST — Execute Any Agent

```bash
curl -X POST http://localhost:8000/api/v1/run \
  -H "Content-Type: application/json" \
  -d '{"task": "What is 2+2?", "agent_type": "react", "max_steps": 10}'
```

Specialised endpoints:

| Endpoint | Agent | Request Fields |
|----------|-------|---------------|
| `POST /api/v1/plan` | PlannerAgent | `task` |
| `POST /api/v1/research` | ResearchAgent | `topic` |
| `POST /api/v1/code` | CoderAgent | `task`, `language` |
| `POST /api/v1/pipeline` | AgentPipeline | `task`, `stages` |

### WebSocket — Real-Time Streaming

Connect to `/ws/run` for live thought/action/observation events:

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://localhost:8000/ws/run") as ws:
        await ws.send(json.dumps({"task": "Explain AI", "agent_type": "react"}))
        async for message in ws:
            event = json.loads(message)
            print(f"[{event['event_type']}] {event['data']}")
            if event["event_type"] in ("result", "error"):
                break

asyncio.run(main())
```

Event types: `thought`, `action`, `observation`, `result`, `error`

---

## Configuration

Memory and agent settings in `.env` or environment variables:

```
# Agent
MAX_AGENT_STEPS=10               # Max reasoning steps per run

# Memory
MEMORY_TYPE=conversation          # conversation | summary | vector
MEMORY_MAX_MESSAGES=20            # ConversationMemory buffer size
SUMMARY_THRESHOLD=10              # messages before summary triggers
SUMMARY_RECENT_COUNT=5            # recent messages kept after summary
CHROMA_PERSIST_DIR=./chroma_data  # VectorMemory persistence (optional)
```

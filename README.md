# Memory Persistence & Recall Agent

Small Python CLI conversational agent with SQLite-backed selective semantic memory.

The main assistant calls DeepSeek directly through its OpenAI-compatible API. After each answer, a smaller memory-manager call proposes `ADD`, `SUPERSEDE`, `DELETE`, or `NO_OP`; application code validates those operations before mutating SQLite.

## How To Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env`, then verify the API:

```bash
python scripts/test_deepseek_api.py
```

Start the CLI:

```bash
python -m src.app --user default --session s1
```

Start the browser UI:

```bash
python scripts/ui_server.py
```

Then open:

```text
http://127.0.0.1:8765
```

The UI supports chat, memory inspection, semantic memory search, manual memory seeding, forgetting memories, reset, session list, and session history. Memory tools work without an API key; chat requires `DEEPSEEK_API_KEY`.

CLI commands:

```text
/help
/memories
/search <query>
/show <memory_id>
/forget <memory_id>
/stats
/sessions
/history [limit]
/export <path>
/reset
/exit
```

Inspection commands work even without `DEEPSEEK_API_KEY`; chat calls still require it.

## Demo

Deterministic local demo:

```bash
python scripts/demo.py
```

Live DeepSeek demo:

```bash
python scripts/demo.py --live
```

The demo walks through:

```text
remember preference -> restart -> recall preference -> supersede preference -> restart -> forget preference
```

More rigorous deterministic multi-session scenario:

```bash
python scripts/multi_session_test.py
```

This script verifies:

```text
multiple turns across four sessions
cross-process persistence through fresh agent instances
preference recall
durable project-decision recall
Python preference superseded by Java preference
deleted concise-style memory no longer active
secret-like API key rejected by policy
toolbox stats/search/session history
```

## Architecture

Runtime flow:

```text
User message
  -> save raw turn as provenance
  -> retrieve active relevant memories
  -> append top memories to the user-message payload
  -> stream DeepSeek response to the user
  -> save assistant turn
  -> schedule memory-manager work in the background
  -> validate and apply memory operations asynchronously
```

Components:

```text
ChatAgent: orchestrates turn handling.
AgentToolbox: exposes safe local memory/session tools for CLI inspection.
MemoryStore: owns SQLite persistence and status transitions.
MemoryRetriever: filters active memories and ranks by embedding similarity, keywords, importance, and staleness.
MemoryManager: asks the LLM for memory operations as JSON.
MemoryPolicy: rejects invalid operations and likely secrets.
```

The main system prompt is a concise environment contract: it tells the model it is running inside a local Python memory-agent harness, has no direct tool access, receives memory context appended to the user message, and that a separate background memory manager may add, supersede, delete, or ignore memories after the response.

## Memory Philosophy

This is intentionally not a general RAG system. It stores curated semantic memories:

```text
preference
working_style
project_decision
fact
temporary
```

It does not inject raw transcript history by default. Conversation turns are stored as provenance/debugging data, while durable memory is separately selected and typed. This avoids the common failure mode where a memory store fills with stale one-off task text and retrieves noise later.

## Write Path

The memory manager can propose:

```text
ADD: store a new active memory.
SUPERSEDE: mark an older active memory superseded and insert a newer active memory.
DELETE: soft-delete a memory.
NO_OP: do nothing.
```

The LLM never writes directly to the database. Code owns IDs, timestamps, embeddings, status transitions, and validation.

Memory management is deliberately off the response critical path. Once the assistant response is generated and saved, `ChatAgent` schedules memory analysis/application on a background worker. CLI and UI responses can return immediately; tests and deterministic demos call `wait_for_memory_updates()` when they need final memory state.

## Streaming

The DeepSeek client supports streaming chat completions. The CLI prints chunks as they arrive, and the browser UI uses `/api/chat_stream` with newline-delimited JSON events:

```text
{"type": "chunk", "content": "..."}
{"type": "done", "state": {...}, "memory_update_pending": true}
```

## Read Path

Retrieval loads active memories for the user, excludes deleted/superseded/expired rows, embeds the current query, and performs a bounded cosine scan. At 1,000 memories, brute-force scan is simple and fast enough for this assignment scale.

Retrieved memories are not placed in the system prompt. They are added to the user-message payload as:

```text
Relevant memory context retrieved for this turn:
- User prefers Java examples for coding questions.

Latest user message:
Show me a queue implementation.
```

The default embedding client is a deterministic local hashing vectorizer so tests, demos, and benchmarks are stable without a second hosted provider. The code also supports an OpenAI-compatible embedding endpoint through:

```text
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=...
EMBEDDING_MODEL=...
```

## Data Model

SQLite tables:

```text
turns: raw conversation provenance, not retrieved by default.
memories: typed semantic memories with status, confidence, importance, timestamps, source turn, supersession link, and embedding JSON.
```

Memory statuses:

```text
active
superseded
deleted
rejected
```

## Conflict Handling

Conflicting changes use `SUPERSEDE` rather than in-place update. This keeps an audit trail:

```text
old memory: status=superseded, valid_until=<time of change>
new memory: status=active, supersedes_id=<old id>
```

Current user messages still win over retrieved memory in the assistant prompt.

## Safety

`MemoryPolicy` rejects likely secrets before storage, including:

```text
password=
api_key=
secret=
token=
Bearer ...
sk-...
-----BEGIN PRIVATE KEY-----
```

This is a lightweight safety layer, not a production PII classifier. Production hard-delete should also remove derived indexes and backups; this v1 uses soft delete so behavior is easy to inspect.

## Latency

Run:

```bash
python scripts/benchmark_latency.py
```

The benchmark isolates retrieval overhead rather than provider network latency. It reports p50 retrieval time with 1 memory and 1,000 memories, plus the delta. Target: p50 delta under 200 ms.

Latest local result:

```text
p50_retrieval_ms_1_memory=0.135
p50_retrieval_ms_1000_memories=51.916
p50_delta_ms=51.780
target_delta_ms<=200
```

## Tests

```bash
pytest
```

Covered behavior:

```text
SQLite persistence after reopen
deleted memories hidden from active retrieval
superseded memories hidden from active retrieval
top-k retrieval
secret rejection
ADD / SUPERSEDE / DELETE validation and application
cross-instance recall through persisted memory
streaming chat response
background memory update scheduling
toolbox search / inspect / stats / forget / export / sessions / history
rigorous four-session scenario with recall, supersession, deletion, durable project decisions, and secret rejection
```

## Tradeoffs

Brute-force cosine scan is used instead of a vector database because the target scale is around 1,000 memories. 


No reranker is included.

## What I Would Build Next

```text
Async background memory consolidation.
PII classifier beyond regexes.
Memory browser/editor.
```

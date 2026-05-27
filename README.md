# Memory Agent

Conversational agent with persistent semantic memory. Uses DeepSeek for chat/memory analysis, sentence-transformers for local embeddings, SQLite for storage.

## Quick Start

```bash
cp .env.example .env
# Set DEEPSEEK_API_KEY in .env
./run.sh
```

Opens at `http://127.0.0.1:8765`. Override with `PORT=9000 HOST=0.0.0.0 ./run.sh`.

`run.sh` creates the venv, installs deps, and starts the UI server. Memory tools work without an API key; chat requires `DEEPSEEK_API_KEY`.

## How It Works

```
User message
  -> save raw turn as provenance
  -> retrieve active relevant memories (sentence-transformer cosine similarity)
  -> append top memories to the user-message payload
  -> stream DeepSeek response to the user
  -> save assistant turn
  -> schedule memory-manager work in background thread
  -> LLM proposes memory ops (ADD / SUPERSEDE / DELETE / NO_OP)
  -> MemoryPolicy validates, application code applies to SQLite
```

The LLM never writes directly to the database. Code owns IDs, timestamps, embeddings, status transitions, and validation. Memory management is off the response critical path — the assistant response returns immediately, memory analysis runs async.

**Components:**
- `ChatAgent` — orchestrates turn handling and streaming
- `MemoryStore` — SQLite persistence and status transitions
- `MemoryRetriever` — filters active memories, ranks by embedding similarity + keywords + importance - staleness
- `MemoryManager` — asks the LLM for memory operations as JSON
- `MemoryPolicy` — rejects invalid operations and likely secrets
- `AgentToolbox` — safe local memory/session tools for CLI and UI

**Memory types:** `preference`, `working_style`, `project_decision`, `fact`, `temporary`

**Memory statuses:** `active`, `superseded`, `deleted`, `rejected`

**Conflict handling:** Uses `SUPERSEDE` instead of in-place update, keeping an audit trail (old memory gets `status=superseded, valid_until=<timestamp>`, new memory links back via `supersedes_id`).

**Memory structure (SQLite):**

| Field | Description |
|---|---|
| `type` | `preference` / `working_style` / `project_decision` / `fact` / `temporary` |
| `content` | The actual memory text |
| `status` | `active` / `superseded` / `deleted` / `rejected` |
| `importance` | 1-5, influences retrieval ranking |
| `confidence` | 0.0-1.0, LLM's confidence in the memory |
| `supersedes_id` | Points to the older memory this one replaced |
| `embedding_json` | Vector embedding for similarity search |

## UI Features

- Streaming chat with DeepSeek
- Memory context annotations on user messages (click to expand)
- Memory change annotations on assistant messages after background update
- Memory list, search, manual add, forget
- Session management and history
- DB reset

## CLI

```bash
.venv/bin/python -m src.app --user default
```

Commands: `/memories`, `/search <q>`, `/show <id>`, `/forget <id>`, `/stats`, `/sessions`, `/history`, `/export <path>`, `/reset`, `/exit`

## Embeddings

Default: `all-MiniLM-L6-v2` via sentence-transformers (local, no API). Retrieval filters by cosine similarity threshold (0.1) before ranking by similarity + importance + keyword overlap - staleness.

Set `EMBEDDING_PROVIDER=openai_compatible` in `.env` for a hosted embedding API.

## Safety

`MemoryPolicy` rejects likely secrets (`password=`, `api_key=`, `sk-...`, `Bearer ...`, private keys) before storage.

## Tests

```bash
.venv/bin/python -m pytest
```

## Tradeoffs

- Brute-force cosine scan instead of vector DB — target scale is ~1,000 memories
- No reranker
- Secret rejection is regex-based, not a production PII classifier
- Soft delete only — production hard-delete should also remove derived indexes and backups

## What I Would Build Next

- Async background memory consolidation
- PII classifier beyond regexes
- Memory browser/editor

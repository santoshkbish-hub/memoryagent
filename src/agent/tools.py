from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.memory.models import Memory
from src.memory.retriever import MemoryRetriever
from src.memory.store import MemoryStore


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    message: str
    data: Any = None


class AgentToolbox:
    """Safe local tools for memory and session inspection.

    These tools do not execute shell commands or access arbitrary application
    data. They expose only the agent's own SQLite-backed state.
    """

    def __init__(self, store: MemoryStore, retriever: MemoryRetriever):
        self.store = store
        self.retriever = retriever

    def list_memories(self, user_id: str, *, include_inactive: bool = True) -> ToolResult:
        memories = self.store.list_memories(user_id, include_inactive=include_inactive)
        return ToolResult(ok=True, message=f"{len(memories)} memories found.", data=memories)

    def inspect_memory(self, memory_id: str) -> ToolResult:
        memory = self.store.get_memory(memory_id)
        if not memory:
            return ToolResult(ok=False, message=f"No memory found for id {memory_id}.")
        return ToolResult(ok=True, message="Memory found.", data=memory)

    def search_memories(self, user_id: str, query: str, *, limit: int = 5) -> ToolResult:
        memories = self.retriever.retrieve(user_id, query, k=limit)
        return ToolResult(ok=True, message=f"{len(memories)} relevant memories found.", data=memories)

    def forget_memory(self, memory_id: str) -> ToolResult:
        deleted = self.store.mark_deleted(memory_id)
        if not deleted:
            return ToolResult(ok=False, message="No active matching memory found.")
        return ToolResult(ok=True, message=f"Deleted {memory_id}.")

    def memory_stats(self, user_id: str) -> ToolResult:
        memories = self.store.list_memories(user_id, include_inactive=True)
        stats: dict[str, Any] = {
            "total": len(memories),
            "by_status": {},
            "by_type": {},
        }
        for memory in memories:
            _increment(stats["by_status"], memory.status)
            _increment(stats["by_type"], memory.type)
        return ToolResult(ok=True, message="Memory stats ready.", data=stats)

    def list_sessions(self, *, limit: int = 20) -> ToolResult:
        sessions = self.store.list_sessions(limit=limit)
        return ToolResult(ok=True, message=f"{len(sessions)} sessions found.", data=sessions)

    def session_history(self, session_id: str | None = None, *, limit: int = 20) -> ToolResult:
        turns = self.store.list_turns(session_id, limit=limit)
        return ToolResult(ok=True, message=f"{len(turns)} turns found.", data=turns)

    def export_memories(self, user_id: str, path: str) -> ToolResult:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        memories = self.store.list_memories(user_id, include_inactive=True)
        payload = [memory_to_dict(memory) for memory in memories]
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return ToolResult(ok=True, message=f"Exported {len(payload)} memories to {destination}.", data=str(destination))


def memory_to_dict(memory: Memory) -> dict[str, Any]:
    return asdict(memory)


def _increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1

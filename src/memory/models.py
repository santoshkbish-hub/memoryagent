from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ALLOWED_MEMORY_TYPES = {"preference", "working_style", "project_decision", "fact", "temporary"}
ALLOWED_STATUSES = {"active", "superseded", "deleted", "rejected"}
ALLOWED_OPS = {"ADD", "SUPERSEDE", "DELETE", "NO_OP"}


@dataclass(frozen=True)
class Memory:
    id: str
    user_id: str
    type: str
    content: str
    status: str
    importance: int
    confidence: float
    source_turn_id: str | None
    supersedes_id: str | None
    embedding: list[float]
    created_at: str
    updated_at: str
    valid_until: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MemoryOperation:
    op: str
    reason: str = ""
    target_memory_id: str | None = None
    content: str | None = None
    type: str | None = None
    importance: int = 3
    confidence: float = 0.8

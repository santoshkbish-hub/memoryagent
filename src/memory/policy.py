from __future__ import annotations

import re
from dataclasses import dataclass

from src.memory.models import ALLOWED_MEMORY_TYPES, ALLOWED_OPS, Memory, MemoryOperation


SENSITIVE_PATTERNS = [
    re.compile(r"password\s*[:=]", re.IGNORECASE),
    re.compile(r"api[_-]?key\s*[:=]", re.IGNORECASE),
    re.compile(r"\bsecret\s*[:=]", re.IGNORECASE),
    re.compile(r"\btoken\s*[:=]", re.IGNORECASE),
    re.compile(r"bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{12,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


@dataclass
class MemoryPolicy:
    def is_sensitive(self, text: str | None) -> bool:
        if not text:
            return False
        return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)

    def normalize_type(self, value: str | None) -> str | None:
        if not value:
            return None
        normalized = value.strip().lower()
        return normalized if normalized in ALLOWED_MEMORY_TYPES else None

    def validate_operation(self, op: MemoryOperation) -> bool:
        if op.op not in ALLOWED_OPS:
            return False
        if op.op == "NO_OP":
            return True
        if op.op in {"DELETE", "SUPERSEDE"} and not op.target_memory_id:
            return False
        if op.op in {"ADD", "SUPERSEDE"}:
            if not op.content or not op.content.strip():
                return False
            if self.is_sensitive(op.content):
                return False
            if self.normalize_type(op.type) is None:
                return False
        return True

    def is_duplicate(self, content: str, memories: list[Memory]) -> bool:
        normalized = self._normalize_content(content)
        return any(self._normalize_content(memory.content) == normalized for memory in memories)

    @staticmethod
    def _normalize_content(content: str) -> str:
        return re.sub(r"\s+", " ", content.strip().lower()).rstrip(".")

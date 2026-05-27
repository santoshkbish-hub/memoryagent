from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.llm_client import LLMClient
from src.memory.models import ALLOWED_OPS, Memory, MemoryOperation
from src.memory.prompts import MEMORY_MANAGER_PROMPT


@dataclass
class MemoryManager:
    llm_client: LLMClient
    model: str = "deepseek-chat"

    def analyze_turn(
        self,
        *,
        user_message: str,
        assistant_message: str,
        related_memories: list[Memory],
        current_time: str | None = None,
    ) -> list[MemoryOperation]:
        prompt = MEMORY_MANAGER_PROMPT.format(
            related_memories=self._format_related_memories(related_memories),
            user_message=user_message,
            assistant_message=assistant_message,
            now=current_time or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        raw = self.llm_client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000,
            model=self.model,
        )
        return parse_memory_operations(raw)

    @staticmethod
    def _format_related_memories(memories: list[Memory]) -> str:
        if not memories:
            return "None."
        lines = []
        for memory in memories:
            lines.append(
                f"- id={memory.id}; type={memory.type}; importance={memory.importance}; "
                f"content={memory.content}"
            )
        return "\n".join(lines)


def parse_memory_operations(raw: str) -> list[MemoryOperation]:
    try:
        data = json.loads(_extract_json_object(raw))
    except (json.JSONDecodeError, ValueError, TypeError):
        return [MemoryOperation(op="NO_OP", reason="Memory manager did not return valid JSON.")]

    operations = data.get("operations", [])
    if not isinstance(operations, list):
        return [MemoryOperation(op="NO_OP", reason="Memory manager JSON had no operations list.")]

    parsed: list[MemoryOperation] = []
    for item in operations:
        if not isinstance(item, dict):
            continue
        op_name = str(item.get("op", "NO_OP")).strip().upper()
        if op_name not in ALLOWED_OPS:
            continue
        parsed.append(
            MemoryOperation(
                op=op_name,
                reason=_optional_str(item.get("reason")) or "",
                target_memory_id=_optional_str(item.get("target_memory_id")),
                content=_optional_str(item.get("content")),
                type=_optional_str(item.get("type")),
                importance=_bounded_int(item.get("importance"), default=3, minimum=1, maximum=5),
                confidence=_bounded_float(item.get("confidence"), default=0.8, minimum=0.0, maximum=1.0),
            )
        )
    return parsed or [MemoryOperation(op="NO_OP", reason="No valid operations.")]


def _extract_json_object(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found.")
    return stripped[start : end + 1]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _bounded_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))

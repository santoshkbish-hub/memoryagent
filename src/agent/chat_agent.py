from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterator

from src.embedding_client import EmbeddingClient
from src.llm_client import LLMClient
from src.memory.manager import MemoryManager
from src.memory.models import Memory, MemoryOperation
from src.memory.policy import MemoryPolicy
from src.memory.prompts import MAIN_SYSTEM_PROMPT
from src.memory.retriever import MemoryRetriever
from src.memory.store import MemoryStore


@dataclass
class ChatAgent:
    store: MemoryStore
    retriever: MemoryRetriever
    memory_manager: MemoryManager
    llm_client: LLMClient
    embedding_client: EmbeddingClient
    policy: MemoryPolicy
    chat_model: str = "deepseek-chat"
    _memory_executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-manager"),
        init=False,
        repr=False,
    )
    _pending_memory_tasks: list[Future[list[str]]] = field(default_factory=list, init=False, repr=False)
    _pending_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def chat(self, user_message: str, *, session_id: str, user_id: str) -> str:
        user_turn_id = self.store.save_turn(session_id, "user", user_message)
        related_memories = self.retriever.retrieve(user_id, user_message, k=5)
        messages = [
            {"role": "system", "content": MAIN_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_message(user_message, related_memories)},
        ]
        assistant_response = self.llm_client.chat(
            messages,
            temperature=0.2,
            max_tokens=900,
            model=self.chat_model,
        )
        self.store.save_turn(session_id, "assistant", assistant_response)
        self.schedule_memory_update(
            user_message=user_message,
            assistant_message=assistant_response,
            related_memories=related_memories,
            user_id=user_id,
            source_turn_id=user_turn_id,
        )
        return assistant_response

    def chat_stream(self, user_message: str, *, session_id: str, user_id: str) -> Iterator[str]:
        user_turn_id = self.store.save_turn(session_id, "user", user_message)
        related_memories = self.retriever.retrieve(user_id, user_message, k=5)
        messages = [
            {"role": "system", "content": MAIN_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_message(user_message, related_memories)},
        ]
        chunks: list[str] = []

        for chunk in self.llm_client.stream_chat(
            messages,
            temperature=0.2,
            max_tokens=900,
            model=self.chat_model,
        ):
            chunks.append(chunk)
            yield chunk

        assistant_response = "".join(chunks).strip()
        self.store.save_turn(session_id, "assistant", assistant_response)
        self.schedule_memory_update(
            user_message=user_message,
            assistant_message=assistant_response,
            related_memories=related_memories,
            user_id=user_id,
            source_turn_id=user_turn_id,
        )

    def schedule_memory_update(
        self,
        *,
        user_message: str,
        assistant_message: str,
        related_memories: list[Memory],
        user_id: str,
        source_turn_id: str | None,
    ) -> Future[list[str]]:
        future = self._memory_executor.submit(
            self._analyze_and_apply_memory_update,
            user_message,
            assistant_message,
            related_memories,
            user_id,
            source_turn_id,
        )
        with self._pending_lock:
            self._pending_memory_tasks = [task for task in self._pending_memory_tasks if not task.done()]
            self._pending_memory_tasks.append(future)
        return future

    def wait_for_memory_updates(self, timeout: float | None = None) -> list[list[str]]:
        with self._pending_lock:
            futures = list(self._pending_memory_tasks)
        results = [future.result(timeout=timeout) for future in futures]
        with self._pending_lock:
            self._pending_memory_tasks = [task for task in self._pending_memory_tasks if not task.done()]
        return results

    def _analyze_and_apply_memory_update(
        self,
        user_message: str,
        assistant_message: str,
        related_memories: list[Memory],
        user_id: str,
        source_turn_id: str | None,
    ) -> list[str]:
        operations = self.memory_manager.analyze_turn(
            user_message=user_message,
            assistant_message=assistant_message,
            related_memories=related_memories,
        )
        return self.apply_memory_operations(operations, user_id=user_id, source_turn_id=source_turn_id)

    def apply_memory_operations(
        self,
        operations: list[MemoryOperation],
        *,
        user_id: str,
        source_turn_id: str | None = None,
    ) -> list[str]:
        active_memories = self.store.list_active_memories(user_id)
        changed_ids: list[str] = []

        for operation in operations:
            if not self.policy.validate_operation(operation):
                continue
            if operation.op == "NO_OP":
                continue
            if operation.op == "DELETE":
                if operation.target_memory_id and self.store.mark_deleted(operation.target_memory_id):
                    changed_ids.append(operation.target_memory_id)
                continue

            content = (operation.content or "").strip()
            memory_type = self.policy.normalize_type(operation.type)
            if not content or not memory_type:
                continue

            if operation.op == "ADD":
                if self.policy.is_duplicate(content, active_memories):
                    continue
                memory_id = self.store.create_memory(
                    user_id=user_id,
                    type=memory_type,
                    content=content,
                    embedding=self.embedding_client.embed_text(content),
                    importance=operation.importance,
                    confidence=operation.confidence,
                    source_turn_id=source_turn_id,
                    metadata={"reason": operation.reason},
                )
                changed_ids.append(memory_id)
                active_memories = self.store.list_active_memories(user_id)
                continue

            if operation.op == "SUPERSEDE" and operation.target_memory_id:
                memory_id = self.store.supersede_memory(
                    target_memory_id=operation.target_memory_id,
                    user_id=user_id,
                    type=memory_type,
                    content=content,
                    embedding=self.embedding_client.embed_text(content),
                    importance=operation.importance,
                    confidence=operation.confidence,
                    source_turn_id=source_turn_id,
                    metadata={"reason": operation.reason},
                )
                if memory_id:
                    changed_ids.append(memory_id)
                    active_memories = self.store.list_active_memories(user_id)

        return changed_ids

    @staticmethod
    def _build_user_message(user_message: str, memories: list[Memory]) -> str:
        if memories:
            memory_block = "\n".join(f"- {memory.content}" for memory in memories)
        else:
            memory_block = "None."
        return (
            "Relevant memory context retrieved for this turn:\n"
            f"{memory_block}\n\n"
            "Latest user message:\n"
            f"{user_message}"
        )

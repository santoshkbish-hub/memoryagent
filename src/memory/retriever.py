from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from src.embedding_client import EmbeddingClient, cosine_similarity
from src.memory.models import Memory
from src.memory.store import MemoryStore

log = logging.getLogger(__name__)


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]*")


@dataclass
class ScoredMemory:
    memory: Memory
    score: float


@dataclass
class MemoryRetriever:
    store: MemoryStore
    embedding_client: EmbeddingClient

    min_similarity: float = 0.1

    def retrieve(self, user_id: str, query: str, k: int = 5) -> list[Memory]:
        memories = self.store.list_active_memories(user_id)
        if not memories:
            return []

        query_embedding = self.embedding_client.embed_text(query)
        query_tokens = self._tokens(query)
        scored = []
        for memory in memories:
            similarity = cosine_similarity(query_embedding, memory.embedding)
            if similarity < self.min_similarity:
                log.info("retrieve q=%r mem=%r sim=%.4f SKIP (below %.2f)",
                         query[:40], memory.content[:40], similarity, self.min_similarity)
                continue
            keyword_boost = self._keyword_overlap_boost(query_tokens, self._tokens(memory.content))
            stale_penalty = self._stale_penalty(memory.updated_at)
            score = similarity + (0.05 * memory.importance) + keyword_boost - stale_penalty
            scored.append(ScoredMemory(memory=memory, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        for item in scored[:k]:
            log.info("retrieve q=%r mem=%r score=%.4f PASS", query[:40], item.memory.content[:40], item.score)
        return [item.memory for item in scored[:k]]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token.lower() for token in TOKEN_RE.findall(text)}

    @staticmethod
    def _keyword_overlap_boost(query_tokens: set[str], memory_tokens: set[str]) -> float:
        if not query_tokens or not memory_tokens:
            return 0.0
        overlap = len(query_tokens & memory_tokens)
        return min(0.15, 0.03 * overlap)

    @staticmethod
    def _stale_penalty(updated_at: str) -> float:
        try:
            updated = datetime.fromisoformat(updated_at)
        except ValueError:
            return 0.0
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (datetime.now(timezone.utc) - updated).total_seconds() / 86400)
        return min(0.05, age_days / 3650)

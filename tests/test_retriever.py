from __future__ import annotations

from src.memory.retriever import MemoryRetriever
from src.memory.store import MemoryStore


class FakeEmbeddingClient:
    def embed_text(self, text: str) -> list[float]:
        lower = text.lower()
        if "python" in lower or "stack" in lower:
            return [1.0, 0.0, 0.0]
        if "java" in lower:
            return [0.9, 0.1, 0.0]
        if "food" in lower:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_retrieves_relevant_memory(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.sqlite3"))
    embeddings = FakeEmbeddingClient()
    python_id = store.create_memory(
        user_id="u1",
        type="preference",
        content="User prefers Python examples.",
        embedding=embeddings.embed_text("User prefers Python examples."),
    )
    store.create_memory(
        user_id="u1",
        type="preference",
        content="User likes food recommendations.",
        embedding=embeddings.embed_text("User likes food recommendations."),
    )

    results = MemoryRetriever(store, embeddings).retrieve("u1", "Show me a stack implementation", k=1)

    assert [memory.id for memory in results] == [python_id]


def test_does_not_return_deleted_memory(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.sqlite3"))
    embeddings = FakeEmbeddingClient()
    memory_id = store.create_memory(
        user_id="u1",
        type="preference",
        content="User prefers Python examples.",
        embedding=embeddings.embed_text("User prefers Python examples."),
    )
    store.mark_deleted(memory_id)

    results = MemoryRetriever(store, embeddings).retrieve("u1", "python stack", k=5)

    assert results == []


def test_does_not_return_superseded_memory(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.sqlite3"))
    embeddings = FakeEmbeddingClient()
    memory_id = store.create_memory(
        user_id="u1",
        type="preference",
        content="User prefers Python examples.",
        embedding=embeddings.embed_text("User prefers Python examples."),
    )
    store.mark_superseded(memory_id)

    results = MemoryRetriever(store, embeddings).retrieve("u1", "python stack", k=5)

    assert results == []


def test_top_k_limit(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.sqlite3"))
    embeddings = FakeEmbeddingClient()
    for index in range(5):
        content = f"User prefers Python examples #{index}."
        store.create_memory(
            user_id="u1",
                type="preference",
            content=content,
            embedding=embeddings.embed_text(content),
        )

    results = MemoryRetriever(store, embeddings).retrieve("u1", "python stack", k=3)

    assert len(results) == 3

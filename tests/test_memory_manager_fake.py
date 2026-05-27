from __future__ import annotations

from src.agent.chat_agent import ChatAgent
from src.embedding_client import HashEmbeddingClient
from src.memory.models import MemoryOperation
from src.memory.policy import MemoryPolicy
from src.memory.retriever import MemoryRetriever
from src.memory.store import MemoryStore


class NoOpLLM:
    def chat(self, messages, *, temperature=0.2, max_tokens=700, model=None):
        return "ok"


class NoOpManager:
    def analyze_turn(self, **kwargs):
        return [MemoryOperation(op="NO_OP")]


def make_agent(tmp_path) -> ChatAgent:
    store = MemoryStore(str(tmp_path / "memory.sqlite3"))
    embedding = HashEmbeddingClient()
    return ChatAgent(
        store=store,
        retriever=MemoryRetriever(store, embedding),
        memory_manager=NoOpManager(),
        llm_client=NoOpLLM(),
        embedding_client=embedding,
        policy=MemoryPolicy(),
    )


def test_add_operation_creates_memory(tmp_path):
    agent = make_agent(tmp_path)

    changed = agent.apply_memory_operations(
        [
            MemoryOperation(
                op="ADD",
                content="User prefers Python examples.",
                type="preference",
            )
        ],
        user_id="u1",
    )

    assert len(changed) == 1
    assert agent.store.list_active_memories("u1")[0].content == "User prefers Python examples."


def test_supersede_operation_marks_old_and_creates_new(tmp_path):
    agent = make_agent(tmp_path)
    old_id = agent.store.create_memory(
        user_id="u1",
        type="preference",
        content="User prefers Python examples.",
        embedding=agent.embedding_client.embed_text("User prefers Python examples."),
    )

    changed = agent.apply_memory_operations(
        [
            MemoryOperation(
                op="SUPERSEDE",
                target_memory_id=old_id,
                content="User prefers Java examples.",
                type="preference",
            )
        ],
        user_id="u1",
    )

    assert len(changed) == 1
    assert agent.store.get_memory(old_id).status == "superseded"
    active = agent.store.list_active_memories("u1")
    assert len(active) == 1
    assert active[0].content == "User prefers Java examples."
    assert active[0].supersedes_id == old_id


def test_delete_operation_marks_deleted(tmp_path):
    agent = make_agent(tmp_path)
    memory_id = agent.store.create_memory(
        user_id="u1",
        type="preference",
        content="User prefers Python examples.",
        embedding=agent.embedding_client.embed_text("User prefers Python examples."),
    )

    changed = agent.apply_memory_operations(
        [MemoryOperation(op="DELETE", target_memory_id=memory_id)],
        user_id="u1",
    )

    assert changed == [memory_id]
    assert agent.store.get_memory(memory_id).status == "deleted"


def test_invalid_operation_ignored(tmp_path):
    agent = make_agent(tmp_path)

    changed = agent.apply_memory_operations(
        [
            MemoryOperation(
                op="ADD",
                content="api_key=abc123",
                type="preference",
            )
        ],
        user_id="u1",
    )

    assert changed == []
    assert agent.store.list_active_memories("u1") == []

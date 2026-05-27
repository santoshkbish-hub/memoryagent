from __future__ import annotations

from threading import Event

from src.agent.chat_agent import ChatAgent
from src.embedding_client import HashEmbeddingClient
from src.memory.models import MemoryOperation
from src.memory.policy import MemoryPolicy
from src.memory.retriever import MemoryRetriever
from src.memory.store import MemoryStore


class BasicLLM:
    def chat(self, messages, *, temperature=0.2, max_tokens=700, model=None):
        return "Assistant response."

    def stream_chat(self, messages, *, temperature=0.2, max_tokens=700, model=None):
        yield "Assistant "
        yield "response."


class BlockingMemoryManager:
    def __init__(self):
        self.started = Event()
        self.release = Event()

    def analyze_turn(self, **kwargs):
        self.started.set()
        self.release.wait(timeout=2)
        return [
            MemoryOperation(
                op="ADD",
                reason="Explicit preference.",
                content="User prefers Java examples.",
                type="preference",
            )
        ]


class ImmediateMemoryManager:
    def analyze_turn(self, **kwargs):
        return [
            MemoryOperation(
                op="ADD",
                reason="Explicit preference.",
                content="User prefers streamed answers.",
                type="preference",
            )
        ]


def make_agent(tmp_path, memory_manager) -> ChatAgent:
    store = MemoryStore(str(tmp_path / "memory.sqlite3"))
    embedding = HashEmbeddingClient()
    return ChatAgent(
        store=store,
        retriever=MemoryRetriever(store, embedding),
        memory_manager=memory_manager,
        llm_client=BasicLLM(),
        embedding_client=embedding,
        policy=MemoryPolicy(),
    )


def test_chat_returns_before_memory_update_finishes(tmp_path):
    manager = BlockingMemoryManager()
    agent = make_agent(tmp_path, manager)

    response = agent.chat("Remember that I prefer Java examples.", session_id="s1", user_id="u1")

    assert response == "Assistant response."
    assert manager.started.wait(timeout=1)
    assert agent.store.list_active_memories("u1") == []

    manager.release.set()
    agent.wait_for_memory_updates(timeout=2)

    memories = agent.store.list_active_memories("u1")
    assert len(memories) == 1
    assert memories[0].content == "User prefers Java examples."


def test_chat_stream_yields_chunks_and_schedules_memory_update(tmp_path):
    agent = make_agent(tmp_path, ImmediateMemoryManager())

    chunks = list(agent.chat_stream("Remember that I prefer streamed answers.", session_id="s1", user_id="u1"))

    assert chunks == ["Assistant ", "response."]
    agent.wait_for_memory_updates(timeout=2)
    turns = agent.store.list_turns("s1", limit=10)
    assert [turn["role"] for turn in turns] == ["assistant", "user"]
    memories = agent.store.list_active_memories("u1")
    assert len(memories) == 1
    assert memories[0].content == "User prefers streamed answers."

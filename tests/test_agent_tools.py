from __future__ import annotations

import json

from src.agent.tools import AgentToolbox
from src.embedding_client import HashEmbeddingClient
from src.memory.retriever import MemoryRetriever
from src.memory.store import MemoryStore


def make_toolbox(tmp_path) -> tuple[AgentToolbox, MemoryStore, HashEmbeddingClient]:
    store = MemoryStore(str(tmp_path / "memory.sqlite3"))
    embedding = HashEmbeddingClient()
    retriever = MemoryRetriever(store, embedding)
    return AgentToolbox(store, retriever), store, embedding


def test_toolbox_search_inspect_stats_forget_and_export(tmp_path):
    toolbox, store, embedding = make_toolbox(tmp_path)
    python_id = store.create_memory(
        user_id="u1",
        type="preference",
        content="User prefers Python examples.",
        embedding=embedding.embed_text("User prefers Python examples."),
    )
    store.create_memory(
        user_id="u1",
        type="project_decision",
        content="Ledger project uses pytest.",
        embedding=embedding.embed_text("Ledger project uses pytest."),
    )

    search = toolbox.search_memories("u1", "Python stack")
    assert search.ok
    assert search.data[0].id == python_id

    inspect = toolbox.inspect_memory(python_id)
    assert inspect.ok
    assert inspect.data.content == "User prefers Python examples."

    stats = toolbox.memory_stats("u1")
    assert stats.data["total"] == 2
    assert stats.data["by_status"] == {"active": 2}
    assert stats.data["by_type"]["preference"] == 1

    export_path = tmp_path / "memories.json"
    export = toolbox.export_memories("u1", str(export_path))
    assert export.ok
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    exported_ids = {item["id"] for item in payload}
    assert len(exported_ids) == 2
    assert python_id in exported_ids

    deleted = toolbox.forget_memory(python_id)
    assert deleted.ok
    assert store.get_memory(python_id).status == "deleted"


def test_toolbox_sessions_and_history(tmp_path):
    toolbox, store, _ = make_toolbox(tmp_path)
    store.save_turn("s1", "user", "hello")
    store.save_turn("s1", "assistant", "hi")
    store.save_turn("s2", "user", "different session")

    sessions = toolbox.list_sessions()
    assert sessions.ok
    assert {session["session_id"] for session in sessions.data} == {"s1", "s2"}

    history = toolbox.session_history("s1", limit=10)
    assert history.ok
    assert len(history.data) == 2
    assert {turn["session_id"] for turn in history.data} == {"s1"}

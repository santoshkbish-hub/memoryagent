from __future__ import annotations

from src.memory.store import MemoryStore


def test_memory_persists_after_reopen(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    store = MemoryStore(str(db_path))
    memory_id = store.create_memory(
        user_id="u1",
        type="preference",
        content="User prefers Python examples.",
        embedding=[1.0, 0.0, 0.0],
    )

    reopened = MemoryStore(str(db_path))
    memory = reopened.get_memory(memory_id)

    assert memory is not None
    assert memory.content == "User prefers Python examples."


def test_mark_deleted_hides_memory_from_active_list(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.sqlite3"))
    memory_id = store.create_memory(
        user_id="u1",
        type="preference",
        content="User prefers Python examples.",
        embedding=[1.0],
    )

    assert store.mark_deleted(memory_id)

    assert store.list_active_memories("u1") == []
    assert store.get_memory(memory_id).status == "deleted"


def test_mark_superseded_hides_old_memory(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.sqlite3"))
    memory_id = store.create_memory(
        user_id="u1",
        type="preference",
        content="User prefers Python examples.",
        embedding=[1.0],
    )

    assert store.mark_superseded(memory_id)

    assert store.list_active_memories("u1") == []
    assert store.get_memory(memory_id).status == "superseded"

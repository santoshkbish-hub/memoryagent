from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.memory.models import Memory


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                  id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  role TEXT NOT NULL,
                  content TEXT NOT NULL,
                  created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                  id TEXT PRIMARY KEY,
                  user_id TEXT NOT NULL,
                  type TEXT NOT NULL,
                  content TEXT NOT NULL,
                  status TEXT NOT NULL,
                  importance INTEGER NOT NULL DEFAULT 3,
                  confidence REAL NOT NULL DEFAULT 0.8,
                  source_turn_id TEXT,
                  supersedes_id TEXT,
                  embedding_json TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  valid_until TEXT,
                  metadata_json TEXT
                )
                """
            )
            self._migrate_memories_schema(conn)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_status ON memories(user_id, status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, created_at)")

    def _migrate_memories_schema(self, conn: sqlite3.Connection) -> None:
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(memories)").fetchall()]
        if "scope" not in columns:
            return

        conn.execute("ALTER TABLE memories RENAME TO memories_legacy_scope")
        conn.execute(
            """
            CREATE TABLE memories (
              id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              type TEXT NOT NULL,
              content TEXT NOT NULL,
              status TEXT NOT NULL,
              importance INTEGER NOT NULL DEFAULT 3,
              confidence REAL NOT NULL DEFAULT 0.8,
              source_turn_id TEXT,
              supersedes_id TEXT,
              embedding_json TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              valid_until TEXT,
              metadata_json TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO memories (
              id, user_id, type, content, status, importance, confidence,
              source_turn_id, supersedes_id, embedding_json, created_at, updated_at,
              valid_until, metadata_json
            )
            SELECT
              id, user_id, type, content, status, importance, confidence,
              source_turn_id, supersedes_id, embedding_json, created_at, updated_at,
              valid_until, metadata_json
            FROM memories_legacy_scope
            """
        )
        conn.execute("DROP TABLE memories_legacy_scope")

    def save_turn(self, session_id: str, role: str, content: str) -> str:
        turn_id = f"turn_{uuid.uuid4().hex}"
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO turns (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (turn_id, session_id, role, content, now),
            )
        return turn_id

    def create_memory(
        self,
        *,
        user_id: str,
        type: str,
        content: str,
        embedding: list[float],
        importance: int = 3,
        confidence: float = 0.8,
        source_turn_id: str | None = None,
        supersedes_id: str | None = None,
        valid_until: str | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "active",
    ) -> str:
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                  id, user_id, type, content, status, importance, confidence,
                  source_turn_id, supersedes_id, embedding_json, created_at, updated_at,
                  valid_until, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    user_id,
                    type,
                    content,
                    status,
                    importance,
                    confidence,
                    source_turn_id,
                    supersedes_id,
                    json.dumps(embedding),
                    now,
                    now,
                    valid_until,
                    json.dumps(metadata or {}),
                ),
            )
        return memory_id

    def supersede_memory(
        self,
        *,
        target_memory_id: str,
        user_id: str,
        type: str,
        content: str,
        embedding: list[float],
        importance: int = 3,
        confidence: float = 0.8,
        source_turn_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        now = utc_now()
        new_id = f"mem_{uuid.uuid4().hex[:12]}"
        with self._connect() as conn:
            target = conn.execute(
                "SELECT * FROM memories WHERE id = ? AND user_id = ? AND status = 'active'",
                (target_memory_id, user_id),
            ).fetchone()
            if not target:
                return None
            conn.execute(
                "UPDATE memories SET status = 'superseded', valid_until = ?, updated_at = ? WHERE id = ?",
                (now, now, target_memory_id),
            )
            conn.execute(
                """
                INSERT INTO memories (
                  id, user_id, type, content, status, importance, confidence,
                  source_turn_id, supersedes_id, embedding_json, created_at, updated_at,
                  valid_until, metadata_json
                )
                VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    new_id,
                    user_id,
                    type,
                    content,
                    importance,
                    confidence,
                    source_turn_id,
                    target_memory_id,
                    json.dumps(embedding),
                    now,
                    now,
                    json.dumps(metadata or {}),
                ),
            )
        return new_id

    def list_active_memories(self, user_id: str) -> list[Memory]:
        now = utc_now()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ?
                  AND status = 'active'
                  AND (valid_until IS NULL OR valid_until > ?)
                ORDER BY updated_at DESC
                """,
                (user_id, now),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def list_memories(self, user_id: str, *, include_inactive: bool = False) -> list[Memory]:
        status_clause = "" if include_inactive else "AND status = 'active'"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE user_id = ?
                {status_clause}
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def get_memory(self, memory_id: str) -> Memory | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def mark_deleted(self, memory_id: str) -> bool:
        now = utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE memories SET status = 'deleted', updated_at = ? WHERE id = ? AND status != 'deleted'",
                (now, memory_id),
            )
            return cursor.rowcount > 0

    def mark_superseded(self, memory_id: str, valid_until: str | None = None) -> bool:
        now = utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE memories SET status = 'superseded', valid_until = ?, updated_at = ? WHERE id = ?",
                (valid_until or now, now, memory_id),
            )
            return cursor.rowcount > 0

    def search_by_keyword(self, user_id: str, query: str) -> list[Memory]:
        terms = [term.strip() for term in query.split() if term.strip()]
        if not terms:
            return []
        clauses = " OR ".join(["LOWER(content) LIKE ?" for _ in terms])
        params = [user_id, *[f"%{term.lower()}%" for term in terms]]
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE user_id = ?
                  AND status = 'active'
                  AND ({clauses})
                ORDER BY updated_at DESC
                """,
                params,
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def list_turns(self, session_id: str | None = None, *, limit: int = 20) -> list[dict[str, str]]:
        limit = max(1, min(200, limit))
        params: list[Any] = []
        session_clause = ""
        if session_id:
            session_clause = "WHERE session_id = ?"
            params.append(session_id)
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, session_id, role, content, created_at
                FROM turns
                {session_clause}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def list_sessions(self, *, limit: int = 20) -> list[dict[str, str | int]]:
        limit = max(1, min(200, limit))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, COUNT(*) AS turn_count, MAX(created_at) AS last_turn_at
                FROM turns
                GROUP BY session_id
                ORDER BY last_turn_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def reembed_all(self, embed_fn: Any) -> int:
        """Re-embed every memory using the given callable (text -> list[float])."""
        count = 0
        with self._connect() as conn:
            rows = conn.execute("SELECT id, content FROM memories").fetchall()
            for row in rows:
                embedding = embed_fn(row["content"])
                conn.execute(
                    "UPDATE memories SET embedding_json = ? WHERE id = ?",
                    (json.dumps(embedding), row["id"]),
                )
                count += 1
        return count

    def reset_all(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM turns")
            conn.execute("DELETE FROM memories")

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            user_id=row["user_id"],
            type=row["type"],
            content=row["content"],
            status=row["status"],
            importance=int(row["importance"]),
            confidence=float(row["confidence"]),
            source_turn_id=row["source_turn_id"],
            supersedes_id=row["supersedes_id"],
            embedding=json.loads(row["embedding_json"] or "[]"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            valid_until=row["valid_until"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

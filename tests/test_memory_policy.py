from __future__ import annotations

from src.memory.policy import MemoryPolicy


def test_rejects_api_key():
    assert MemoryPolicy().is_sensitive("api_key=abc123")


def test_rejects_password():
    assert MemoryPolicy().is_sensitive("password = hunter2")


def test_allows_normal_preference():
    assert not MemoryPolicy().is_sensitive("User prefers concise Python examples.")

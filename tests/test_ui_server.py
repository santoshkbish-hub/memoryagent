from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from src.agent.chat_agent import ChatAgent
from src.embedding_client import HashEmbeddingClient
from src.memory.models import MemoryOperation
from src.memory.policy import MemoryPolicy
from src.memory.retriever import MemoryRetriever
from src.memory.store import MemoryStore
from src.ui_server import MemoryAgentUiServer


class FakeLLM:
    def chat(self, messages, *, temperature=0.2, max_tokens=700, model=None):
        return "Got it."

    def stream_chat(self, messages, *, temperature=0.2, max_tokens=700, model=None):
        yield "Got "
        yield "it."


class FakeMemoryManager:
    def analyze_turn(self, **kwargs):
        return [MemoryOperation(op="NO_OP")]


def test_ui_state_add_search_and_reset(tmp_path):
    server, thread = start_server(tmp_path)
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"

        state = get_json(f"{base_url}/api/state?user_id=u1&session_id=s1")
        assert state["ok"] is True
        assert state["stats"]["total"] == 0

        added = post_json(
            f"{base_url}/api/memories",
            {
                "user_id": "u1",
                "session_id": "s1",
                "content": "User prefers Java examples for coding questions.",
                "type": "preference",
            },
        )
        assert added["ok"] is True
        assert added["state"]["stats"]["total"] == 1

        found = post_json(
            f"{base_url}/api/search",
            {"user_id": "u1", "query": "Java queue"},
        )
        assert len(found["memories"]) == 1
        assert found["memories"][0]["content"] == "User prefers Java examples for coding questions."

        reset = post_json(f"{base_url}/api/reset", {"user_id": "u1", "session_id": "s1"})
        assert reset["ok"] is True
        assert reset["state"]["stats"]["total"] == 0
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_ui_chat_stream_endpoint(tmp_path):
    server, thread = start_server(tmp_path)
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        request = urllib.request.Request(
            f"{base_url}/api/chat_stream",
            data=json.dumps({"user_id": "u1", "session_id": "s1", "message": "hello"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            events = [json.loads(line) for line in response.read().decode("utf-8").splitlines() if line]

        assert events[0] == {"type": "chunk", "content": "Got "}
        assert events[1] == {"type": "chunk", "content": "it."}
        assert events[-1]["type"] == "done"
        assert events[-1]["memory_update_pending"] is True
    finally:
        server.shutdown()
        thread.join(timeout=2)


def start_server(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.sqlite3"))
    embedding = HashEmbeddingClient()
    agent = ChatAgent(
        store=store,
        retriever=MemoryRetriever(store, embedding),
        memory_manager=FakeMemoryManager(),
        llm_client=FakeLLM(),
        embedding_client=embedding,
        policy=MemoryPolicy(),
    )
    server = MemoryAgentUiServer(("127.0.0.1", 0), agent)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        raise AssertionError(body) from exc

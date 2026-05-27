from __future__ import annotations

from src.agent.chat_agent import ChatAgent
from src.embedding_client import HashEmbeddingClient
from src.memory.models import MemoryOperation
from src.memory.policy import MemoryPolicy
from src.memory.retriever import MemoryRetriever
from src.memory.store import MemoryStore


class CapturingLLM:
    def __init__(self):
        self.last_messages = None

    def chat(self, messages, *, temperature=0.2, max_tokens=700, model=None):
        self.last_messages = messages
        return "Got it."


class RememberPythonManager:
    def analyze_turn(self, **kwargs):
        user_message = kwargs["user_message"].lower()
        if "remember" in user_message:
            return [
                MemoryOperation(
                    op="ADD",
                    content="User prefers Python examples for coding questions.",
                    type="preference",
                    importance=4,
                    confidence=0.95,
                )
            ]
        return [MemoryOperation(op="NO_OP")]


def build_agent(db_path, llm):
    store = MemoryStore(str(db_path))
    embedding = HashEmbeddingClient()
    return ChatAgent(
        store=store,
        retriever=MemoryRetriever(store, embedding),
        memory_manager=RememberPythonManager(),
        llm_client=llm,
        embedding_client=embedding,
        policy=MemoryPolicy(),
    )


def test_remember_then_retrieve_across_new_agent_instance(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    first_llm = CapturingLLM()
    first_agent = build_agent(db_path, first_llm)

    first_agent.chat(
        "Remember that I prefer Python examples for coding questions.",
        session_id="s1",
        user_id="u1",
    )
    first_agent.wait_for_memory_updates()

    second_llm = CapturingLLM()
    second_agent = build_agent(db_path, second_llm)
    second_agent.chat("Show me a stack implementation.", session_id="s2", user_id="u1")
    second_agent.wait_for_memory_updates()

    system_prompt = second_llm.last_messages[0]["content"]
    user_payload = second_llm.last_messages[1]["content"]
    assert "User prefers Python examples for coding questions." not in system_prompt
    assert "User prefers Python examples for coding questions." in user_payload
    assert "Relevant memory context retrieved for this turn:" in user_payload

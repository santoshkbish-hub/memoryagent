from __future__ import annotations

from src.agent.chat_agent import ChatAgent
from src.agent.tools import AgentToolbox
from src.embedding_client import HashEmbeddingClient
from src.memory.models import Memory, MemoryOperation
from src.memory.policy import MemoryPolicy
from src.memory.retriever import MemoryRetriever
from src.memory.store import MemoryStore


class ScenarioLLM:
    def __init__(self):
        self.system_prompts: list[str] = []
        self.user_payloads: list[str] = []

    def chat(self, messages, *, temperature=0.2, max_tokens=700, model=None):
        system_prompt = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        self.system_prompts.append(system_prompt)
        user_payload = messages[-1]["content"]
        self.user_payloads.append(user_payload)
        user_message = user_payload.lower()
        if "stack" in user_message:
            if "User prefers Python examples" in user_payload:
                return "Concise Python stack example."
            return "Generic stack example."
        if "queue" in user_message:
            if "User prefers Java examples" in user_payload and "User prefers Python examples" not in user_payload:
                return "Java queue example."
            return "Generic queue example."
        if "ledger project" in user_message and "pytest" in user_payload:
            return "Use pytest for the Ledger project tests."
        return "Got it."


class ScenarioMemoryManager:
    def analyze_turn(
        self,
        *,
        user_message: str,
        assistant_message: str,
        related_memories: list[Memory],
        current_time: str | None = None,
    ) -> list[MemoryOperation]:
        text = user_message.lower()
        if "prefer python examples" in text:
            return [
                MemoryOperation(
                    op="ADD",
                    content="User prefers Python examples for coding questions.",
                    type="preference",
                    importance=4,
                    confidence=0.95,
                ),
                MemoryOperation(
                    op="ADD",
                    content="User prefers concise explanations.",
                    type="working_style",
                    importance=4,
                    confidence=0.95,
                ),
            ]
        if "ledger project uses pytest" in text:
            return [
                MemoryOperation(
                    op="ADD",
                    content="Ledger project uses pytest for tests.",
                    type="project_decision",
                    importance=5,
                    confidence=0.95,
                )
            ]
        if "use java examples" in text:
            target = _find_memory(related_memories, "Python examples")
            return [
                MemoryOperation(
                    op="SUPERSEDE",
                    target_memory_id=target.id if target else None,
                    content="User prefers Java examples for coding questions.",
                    type="preference",
                    importance=4,
                    confidence=0.95,
                )
            ]
        if "api key" in text:
            return [
                MemoryOperation(
                    op="ADD",
                    content="User API key is sk-test1234567890.",
                    type="fact",
                    importance=5,
                    confidence=1.0,
                )
            ]
        if "forget" in text and "concise" in text:
            target = _find_memory(related_memories, "concise explanations")
            return [MemoryOperation(op="DELETE", target_memory_id=target.id if target else None)]
        return [MemoryOperation(op="NO_OP")]


def build_agent(db_path, llm: ScenarioLLM) -> ChatAgent:
    store = MemoryStore(str(db_path))
    embedding = HashEmbeddingClient()
    return ChatAgent(
        store=store,
        retriever=MemoryRetriever(store, embedding),
        memory_manager=ScenarioMemoryManager(),
        llm_client=llm,
        embedding_client=embedding,
        policy=MemoryPolicy(),
    )


def test_rigorous_multi_turn_multi_session_memory_flow(tmp_path):
    db_path = tmp_path / "memory.sqlite3"

    llm1 = ScenarioLLM()
    agent1 = build_agent(db_path, llm1)
    agent1.chat("Remember that I prefer Python examples and concise explanations.", session_id="s1", user_id="u1")
    agent1.wait_for_memory_updates()
    agent1.chat("Remember that the Ledger project uses pytest for tests.", session_id="s1", user_id="u1")
    agent1.wait_for_memory_updates()

    llm2 = ScenarioLLM()
    agent2 = build_agent(db_path, llm2)
    stack_response = agent2.chat("Show me a stack implementation.", session_id="s2", user_id="u1")
    assert stack_response == "Concise Python stack example."
    agent2.chat("Actually, use Java examples going forward, not Python.", session_id="s2", user_id="u1")
    agent2.wait_for_memory_updates()
    agent2.chat("Remember this API key: sk-test1234567890.", session_id="s2", user_id="u1")
    agent2.wait_for_memory_updates()

    llm3 = ScenarioLLM()
    agent3 = build_agent(db_path, llm3)
    queue_response = agent3.chat("Show me a queue implementation.", session_id="s3", user_id="u1")
    assert queue_response == "Java queue example."
    assert "User prefers Python examples" not in llm3.system_prompts[-1]
    assert "User prefers Python examples" not in llm3.user_payloads[-1]
    agent3.chat("Forget that I prefer concise explanations.", session_id="s3", user_id="u1")
    agent3.wait_for_memory_updates()

    llm4 = ScenarioLLM()
    agent4 = build_agent(db_path, llm4)
    project_response = agent4.chat("How should we test the Ledger project?", session_id="s4", user_id="u1")
    assert project_response == "Use pytest for the Ledger project tests."
    agent4.wait_for_memory_updates()

    toolbox = AgentToolbox(agent4.store, agent4.retriever)
    stats = toolbox.memory_stats("u1").data
    assert stats["by_status"] == {"active": 2, "deleted": 1, "superseded": 1}
    assert not any("sk-test" in memory.content for memory in agent4.store.list_memories("u1", include_inactive=True))
    assert {session["session_id"] for session in toolbox.list_sessions().data} == {"s1", "s2", "s3", "s4"}


def _find_memory(memories: list[Memory], text: str) -> Memory | None:
    needle = text.lower()
    for memory in memories:
        if needle in memory.content.lower():
            return memory
    return None

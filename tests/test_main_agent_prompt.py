from __future__ import annotations

from src.agent.chat_agent import ChatAgent
from src.memory.models import Memory
from src.memory.prompts import MAIN_SYSTEM_PROMPT


def test_main_system_prompt_describes_harness_and_capabilities():
    assert "local Python memory-agent harness" in MAIN_SYSTEM_PROMPT
    assert "You do not have direct tool access" in MAIN_SYSTEM_PROMPT
    assert "no web search, browser, code execution, image generation, or file access" in MAIN_SYSTEM_PROMPT
    assert "background memory manager may add, supersede, delete, or ignore memories" in MAIN_SYSTEM_PROMPT
    assert "Relevant memories:" not in MAIN_SYSTEM_PROMPT


def test_memory_context_is_injected_into_user_message_not_system_prompt():
    memory = Memory(
        id="mem_1",
        user_id="u1",
        type="preference",
        content="User prefers Java examples.",
        status="active",
        importance=4,
        confidence=0.9,
        source_turn_id=None,
        supersedes_id=None,
        embedding=[],
        created_at="2026-05-27T00:00:00+00:00",
        updated_at="2026-05-27T00:00:00+00:00",
        valid_until=None,
        metadata={},
    )

    user_payload = ChatAgent._build_user_message("Show me a queue.", [memory])

    assert "Relevant memory context retrieved for this turn:" in user_payload
    assert "- User prefers Java examples." in user_payload
    assert "Latest user message:\nShow me a queue." in user_payload

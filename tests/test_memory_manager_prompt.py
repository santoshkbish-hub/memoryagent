from __future__ import annotations

from src.memory.manager import parse_memory_operations
from src.memory.prompts import MEMORY_MANAGER_PROMPT


def test_memory_manager_prompt_documents_edit_delete_and_target_id():
    assert "Use SUPERSEDE when a current memory is wrong, stale, contradicted, or should be edited." in MEMORY_MANAGER_PROMPT
    assert "Use DELETE when the user asks to forget an existing memory" in MEMORY_MANAGER_PROMPT
    assert "one-off task details for only this session" in MEMORY_MANAGER_PROMPT
    assert "target_memory_id is the id of an existing related memory" in MEMORY_MANAGER_PROMPT
    assert "For ADD and NO_OP, target_memory_id must be null." in MEMORY_MANAGER_PROMPT


def test_memory_manager_prompt_has_examples_and_no_scope():
    assert "Good memories:" in MEMORY_MANAGER_PROMPT
    assert "Bad memories:" in MEMORY_MANAGER_PROMPT
    assert "User prefers Java examples for coding questions." in MEMORY_MANAGER_PROMPT
    assert '"scope"' not in MEMORY_MANAGER_PROMPT
    assert "scope=" not in MEMORY_MANAGER_PROMPT


def test_memory_manager_prompt_orders_reason_after_op():
    op_index = MEMORY_MANAGER_PROMPT.index('"op": "ADD | SUPERSEDE | DELETE | NO_OP"')
    reason_index = MEMORY_MANAGER_PROMPT.index('"reason": "short reason"')
    target_index = MEMORY_MANAGER_PROMPT.index('"target_memory_id": "string or null"')

    assert op_index < reason_index < target_index


def test_parse_memory_operations_ignores_legacy_scope_field():
    operations = parse_memory_operations(
        """
        {
          "operations": [
            {
              "op": "ADD",
              "reason": "User explicitly stated a preference.",
              "target_memory_id": null,
              "content": "User prefers Java examples.",
              "type": "preference",
              "scope": "project",
              "importance": 4,
              "confidence": 0.9
            }
          ]
        }
        """
    )

    assert len(operations) == 1
    assert operations[0].op == "ADD"
    assert operations[0].reason == "User explicitly stated a preference."
    assert not hasattr(operations[0], "scope")

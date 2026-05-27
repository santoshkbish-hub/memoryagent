from __future__ import annotations

import argparse
from pathlib import Path
import sys
import uuid

from src.agent.tools import AgentToolbox, ToolResult
from src.agent.chat_agent import ChatAgent
from src.config import load_settings
from src.embedding_client import get_embedding_client
from src.llm_client import DeepSeekClient, UnavailableLLMClient
from src.memory.manager import MemoryManager
from src.memory.policy import MemoryPolicy
from src.memory.retriever import MemoryRetriever
from src.memory.store import MemoryStore


def build_agent(db_path: str | None = None) -> ChatAgent:
    settings = load_settings()
    store = MemoryStore(db_path or settings.db_path)
    embedding_client = get_embedding_client(settings)
    count = store.reembed_all(embedding_client.embed_text)
    if count:
        print(f"Re-embedded {count} memories with {settings.embedding_provider} provider.")
    retriever = MemoryRetriever(store=store, embedding_client=embedding_client)
    if settings.deepseek_api_key:
        llm_client = DeepSeekClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_chat_model,
        )
    else:
        llm_client = UnavailableLLMClient("DEEPSEEK_API_KEY is required for chat calls.")
    memory_manager = MemoryManager(llm_client=llm_client, model=settings.deepseek_memory_model)
    return ChatAgent(
        store=store,
        retriever=retriever,
        memory_manager=memory_manager,
        llm_client=llm_client,
        embedding_client=embedding_client,
        policy=MemoryPolicy(),
        chat_model=settings.deepseek_chat_model,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="SQLite-backed conversational memory agent.")
    parser.add_argument("--user", default="default", help="User id for memory partitioning.")
    parser.add_argument("--session", default=None, help="Session id. Defaults to a fresh session.")
    parser.add_argument("--db", default=None, help="SQLite database path.")
    args = parser.parse_args()

    session_id = args.session or f"sess_{uuid.uuid4().hex[:8]}"
    try:
        agent = build_agent(args.db)
    except Exception as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 1

    print(f"Memory agent ready. user={args.user} session={session_id}")
    print("Commands: /help, /memories, /search <query>, /show <memory_id>, /forget <memory_id>, /stats, /sessions, /history [limit], /export <path>, /reset, /exit")
    toolbox = AgentToolbox(store=agent.store, retriever=agent.retriever)

    while True:
        try:
            user_message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_message:
            continue
        if user_message == "/exit":
            return 0
        if user_message == "/help":
            _print_help()
            continue
        if user_message == "/memories":
            _print_memories(toolbox, args.user)
            continue
        if user_message.startswith("/search "):
            query = user_message.split(maxsplit=1)[1].strip()
            _print_memories_result(toolbox.search_memories(args.user, query))
            continue
        if user_message.startswith("/show "):
            memory_id = user_message.split(maxsplit=1)[1].strip()
            _print_memory_detail(toolbox.inspect_memory(memory_id))
            continue
        if user_message.startswith("/forget "):
            memory_id = user_message.split(maxsplit=1)[1].strip()
            print(toolbox.forget_memory(memory_id).message)
            continue
        if user_message == "/stats":
            _print_stats(toolbox.memory_stats(args.user))
            continue
        if user_message == "/sessions":
            _print_sessions(toolbox.list_sessions())
            continue
        if user_message.startswith("/history"):
            parts = user_message.split()
            limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 20
            _print_history(toolbox.session_history(session_id, limit=limit))
            continue
        if user_message.startswith("/export "):
            path = user_message.split(maxsplit=1)[1].strip()
            print(toolbox.export_memories(args.user, str(Path(path))).message)
            continue
        if user_message == "/reset":
            agent.store.reset_all()
            print("All turns and memories removed from this database.")
            continue

        try:
            print("\nAssistant: ", end="", flush=True)
            for chunk in agent.chat_stream(user_message, session_id=session_id, user_id=args.user):
                print(chunk, end="", flush=True)
            print()
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue


def _print_help() -> None:
    print(
        """Available commands:
  /memories                 List all memories including inactive ones.
  /search <query>           Semantic search over active memories.
  /show <memory_id>         Inspect one memory with metadata.
  /forget <memory_id>       Soft-delete one memory.
  /stats                    Count memories by status and type.
  /sessions                 List known conversation sessions.
  /history [limit]          Show recent turns for this session.
  /export <path>            Export this user's memories as JSON.
  /reset                    Delete all turns and memories in this DB.
  /exit                     Quit."""
    )


def _print_memories(toolbox: AgentToolbox, user_id: str) -> None:
    result = toolbox.list_memories(user_id, include_inactive=True)
    memories = result.data
    if not memories:
        print("No memories stored.")
        return
    for memory in memories:
        print(f"{memory.id} [{memory.status}/{memory.type}] {memory.content}")


def _print_memories_result(result: ToolResult) -> None:
    if not result.ok:
        print(result.message)
        return
    memories = result.data or []
    if not memories:
        print("No matching active memories.")
        return
    for memory in memories:
        print(f"{memory.id} [{memory.status}/{memory.type}] {memory.content}")


def _print_memory_detail(result: ToolResult) -> None:
    if not result.ok:
        print(result.message)
        return
    memory = result.data
    print(f"id: {memory.id}")
    print(f"status: {memory.status}")
    print(f"type: {memory.type}")
    print(f"importance/confidence: {memory.importance}/{memory.confidence}")
    print(f"source_turn_id: {memory.source_turn_id}")
    print(f"supersedes_id: {memory.supersedes_id}")
    print(f"created_at: {memory.created_at}")
    print(f"updated_at: {memory.updated_at}")
    print(f"valid_until: {memory.valid_until}")
    print(f"content: {memory.content}")
    print(f"metadata: {memory.metadata}")


def _print_stats(result: ToolResult) -> None:
    stats = result.data
    print(f"total: {stats['total']}")
    print(f"by_status: {stats['by_status']}")
    print(f"by_type: {stats['by_type']}")


def _print_sessions(result: ToolResult) -> None:
    sessions = result.data or []
    if not sessions:
        print("No sessions stored.")
        return
    for session in sessions:
        print(f"{session['session_id']} turns={session['turn_count']} last_turn_at={session['last_turn_at']}")


def _print_history(result: ToolResult) -> None:
    turns = list(reversed(result.data or []))
    if not turns:
        print("No turns stored for this session.")
        return
    for turn in turns:
        content = turn["content"].replace("\n", " ")
        if len(content) > 140:
            content = content[:137] + "..."
        print(f"{turn['created_at']} {turn['role']}: {content}")


if __name__ == "__main__":
    raise SystemExit(main())

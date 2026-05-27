#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

VENV_DIR="$ROOT/.venv"

# --- Load .env if present ---
if [ -f "$ROOT/.env" ]; then
  set -a
  . "$ROOT/.env"
  set +a
fi

PORT="${PORT:-8765}"
HOST="${HOST:-127.0.0.1}"
DB="${DB:-${MEMORY_DB_PATH:-./memory_agent.sqlite3}}"

# --- Create virtual environment if missing ---
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

# --- Install dependencies ---
echo "Installing dependencies..."
"$VENV_DIR/bin/pip" install -q -r requirements.txt

# --- Check for API key ---
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo ""
  echo "WARNING: DEEPSEEK_API_KEY is not set."
  echo "Chat will be disabled. Memory tools still work."
  echo "Usage:  DEEPSEEK_API_KEY='sk-...' ./run.sh"
  echo ""
fi

# --- Start the UI server ---
echo "Starting Memory Agent UI at http://$HOST:$PORT"
echo "Database: $DB"
echo "Press Ctrl+C to stop."
echo ""

exec "$VENV_DIR/bin/python" scripts/ui_server.py \
  --host "$HOST" \
  --port "$PORT" \
  --db "$DB"

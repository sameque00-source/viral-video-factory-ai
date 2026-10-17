#!/usr/bin/env bash
# PostToolUse (Edit|Write|MultiEdit): formata arquivos .py com ruff. Nunca falha a edição.
set -uo pipefail
path="$(jq -r '.tool_input.file_path // empty' 2>/dev/null)"
case "$path" in
  *.py) cd "${CLAUDE_PROJECT_DIR:-.}" && uv run --no-sync --quiet ruff format "$path" >/dev/null 2>&1 ;;
esac
exit 0

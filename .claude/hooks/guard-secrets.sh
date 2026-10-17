#!/usr/bin/env bash
# PreToolUse (Edit|Write|MultiEdit): bloqueia escrita em arquivos .env reais.
set -uo pipefail
path="$(jq -r '.tool_input.file_path // empty' 2>/dev/null)"
[ -n "$path" ] || exit 0
base="$(basename "$path")"
case "$base" in
  .env.example) exit 0 ;;
  .env | .env.*)
    echo "Bloqueado: não edite $base pelo agente. Chaves ficam só no ambiente/.env local; use .env.example." >&2
    exit 2 ;;
esac
exit 0

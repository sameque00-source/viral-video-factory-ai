#!/usr/bin/env bash
# SessionStart: prepara o ambiente Python do projeto em sessões Cloud. Rápido e idempotente.
set -uo pipefail
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
command -v uv >/dev/null 2>&1 || { echo "uv não encontrado; rode scripts/bootstrap.sh" >&2; exit 0; }
uv sync --locked --quiet 2>&1 | tail -3 >&2 || true
command -v ffmpeg >/dev/null 2>&1 || echo "AVISO: ffmpeg ausente (apt-get install ffmpeg)" >&2
exit 0

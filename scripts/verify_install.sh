#!/usr/bin/env bash
# Verifica a instalação sem custo: lint, formatação, testes e versões das ferramentas.
set -uo pipefail
cd "$(dirname "$0")/.."
NODE_DIR="/opt/node-v22.23.3-linux-x64"
fail=0
run() { echo "== $*"; "$@" || { echo "FALHOU: $*"; fail=1; }; }

run uv run ruff check .
run uv run ruff format --check .
run uv run pytest -q

echo "== versões (informativo)"
for c in "ffmpeg -version" "uv --version" "bun --version" "headroom --version"; do
  printf '%-20s' "${c%% *}"; ($c 2>&1 | head -1) || echo "não instalado"
done
printf '%-20s' omniroute; (PATH="${NODE_DIR}/bin:$PATH" omniroute --version 2>&1 | tail -1) || echo "não instalado"
exit $fail

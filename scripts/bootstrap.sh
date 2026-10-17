#!/usr/bin/env bash
# Instala o ambiente do projeto.
#   scripts/bootstrap.sh          -> dependências Python (uv sync --locked)
#   scripts/bootstrap.sh tools    -> + Headroom, OmniRoute e CLI do Cartographer (opcionais)
# Idempotente. Não grava chaves. Não executa chamadas pagas.
set -euo pipefail
cd "$(dirname "$0")/.."

NODE_VERSION="v22.23.3"          # OmniRoute exige Node >=22.22.2
NODE_DIR="/opt/node-${NODE_VERSION}-linux-x64"
CARTO_DIR="${HOME}/.local/share/cartographer"

uv sync --locked

[ "${1:-}" = "tools" ] || { echo "ok: ambiente Python pronto (use '$0 tools' para as ferramentas extras)"; exit 0; }

# Headroom (compressão de contexto) — sem torch; beacon desligado por padrão em .env.example
if ! command -v headroom >/dev/null; then
  uv tool install --python 3.13 "headroom-ai[proxy,code]==0.41.0"
fi

# Node compatível com o OmniRoute, ao lado do Node do sistema (com checksum)
if [ ! -x "${NODE_DIR}/bin/node" ]; then
  tmp="$(mktemp -d)"
  base="https://nodejs.org/dist/${NODE_VERSION}"
  curl -fsSL -o "${tmp}/node.tar.xz" "${base}/node-${NODE_VERSION}-linux-x64.tar.xz"
  curl -fsSL -o "${tmp}/SHASUMS256.txt" "${base}/SHASUMS256.txt"
  (cd "$tmp" && grep "node-${NODE_VERSION}-linux-x64.tar.xz" SHASUMS256.txt | sed 's#  .*#  node.tar.xz#' | sha256sum -c -)
  tar -xJf "${tmp}/node.tar.xz" -C "$(dirname "$NODE_DIR")"
  rm -rf "$tmp"
fi
PATH="${NODE_DIR}/bin:${PATH}" npm install -g omniroute@3.8.48

# CLI do Cartographer (não está no npm; clone oficial + bun)
if [ ! -d "${CARTO_DIR}/.git" ]; then
  git clone --depth 1 https://github.com/kingbootoshi/cartographer "${CARTO_DIR}"
fi
command -v bun >/dev/null && (cd "${CARTO_DIR}" && bun install --frozen-lockfile)
echo "ok: ferramentas extras instaladas"

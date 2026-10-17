# viral-video-factory-ai

Fábrica automatizada de vídeos curtos com IA para TikTok, Instagram Reels e YouTube Shorts.

> Estado: **ambiente de desenvolvimento pronto; nenhuma funcionalidade implementada ainda.**

## Início rápido
```bash
scripts/bootstrap.sh          # uv sync --locked
scripts/verify_install.sh     # ruff + pytest + versões (sem custo, sem rede)
cp .env.example .env          # preencha só o que for usar
```
Requisitos: Python 3.12+, [uv](https://docs.astral.sh/uv/), FFmpeg (com libx264/aac/libass).

## Documentação
- [`CLAUDE.md`](CLAUDE.md) — instruções para o Claude Code (comandos, mapa, regras)
- [`docs/SETUP.md`](docs/SETUP.md) — ferramentas instaladas, como usar e pendências
- [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md) — origem dos vídeos, consentimento e rótulo de IA

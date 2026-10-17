# Setup das ferramentas

Última verificação: 2026-10-17 (Claude Code Cloud, Ubuntu 24.04, Python 3.13, Node 22, sem GPU).

## Núcleo (versionado, testado por `scripts/verify_install.sh` e pelo CI)
`uv.lock` fixa tudo. Principais: typer, pydantic, pydantic-settings, httpx, tenacity, diskcache,
anthropic, fal-client, replicate, faster-whisper, scenedetect, opencv-python-headless, pysubs2,
ffmpeg-normalize; dev: pytest, pytest-cov, ruff. FFmpeg vem do sistema (`apt install ffmpeg`).

## Plugins do Claude Code (`.claude/settings.json`; instalados no início de cada sessão Cloud)
| Plugin | Estado | Uso |
|---|---|---|
| claude-code-setup | ativo | `"recomende automações para este projeto"` (somente leitura) |
| plugin-dev | ativo | criar skills, agentes, hooks e MCPs deste projeto |
| context7 (MCP) | ativo | docs atuais de fal, Replicate, Remotion... (pede autenticação na 1ª vez) |
| cartographer | ativo | `"map this codebase"` → `docs/CODEBASE_MAP.md` (usar quando houver código) |
| remotion | ativo | skills de vídeo programático (React). Confira a licença do Remotion antes de uso comercial |
| claude-mem | **desativado** | veja abaixo |

Hooks (`.claude/hooks/`): `session-start.sh` (uv sync em sessões Cloud), `guard-secrets.sh`
(bloqueia edição de `.env`), `ruff-format.sh` (formata .py após editar).

## Ferramentas externas (opcionais; `scripts/bootstrap.sh tools`)
- **Headroom 0.41.0** (compressão de contexto). `headroom proxy --port 8787` com
  `HEADROOM_BEACON=off DO_NOT_TRACK=1` (o beacon vem LIGADO por padrão). Em Python:
  `from headroom import compress`. Na sessão Cloud o endpoint do Claude é gerenciado pelo host, então
  `headroom wrap claude` só vale no seu computador/CLI próprio; aqui ele serve ao código do projeto.
- **OmniRoute 3.8.48** (roteador/fallback de modelos). Exige Node >=22.22.2 (o script baixa o 22.23.3
  ao lado do Node do sistema). `PATH=/opt/node-v22.23.3-linux-x64/bin:$PATH omniroute serve --no-open`
  → API `http://localhost:20128/v1` (exige autenticação). Falta conectar provedores com SUAS chaves no painel.
- **Cartographer CLI** (clone em `~/.local/share/cartographer`, requer bun):
  `bun run --cwd ~/.local/share/cartographer cartographer -- index --root . --out .cartographer`.

## Claude-Mem (memória entre sessões) — decisão pendente
Instala e roda no Cloud (testado em HOME isolado), mas: o disco Cloud é efêmero (memória local some),
o provedor padrão usa tokens do seu plano, e a telemetria é opt-out. Para ativar:
`"claude-mem@thedotmack": true` em `.claude/settings.json` + login em cmem.ai (sincronização) +
`npx claude-mem telemetry disable`. Sem isso, a memória persistente do projeto é este repositório
(`CLAUDE.md` + `docs/`).

## Não instalado (de propósito)
- **ComfyUI / Wan local**: sem GPU no Cloud e sua RX 580 não comporta. Usar fal.ai/Replicate.
- **agent-browser / Playwright MCP**: sem necessidade agora; Chromium já existe no ambiente.
- **security-guidance, code-review, code-simplifier**: redundantes com `/security-review`, `/code-review`, `/simplify`.
- **Headroom `[ml]` (torch)**: pesado; a compressão por regras já funciona.

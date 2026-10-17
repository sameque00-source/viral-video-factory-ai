# viral-video-factory-ai

Fábrica de vídeos curtos (TikTok, Reels, Shorts) com IA: recebe vídeos originais, analisa cenas,
gera ideias/roteiros, aplica transformações via APIs na nuvem, legenda, mixa áudio e exporta.
Python 3.12+ gerenciado com **uv**. Máquina do dono (RX 580, 16 GB) NÃO roda modelos pesados:
geração/transformação de vídeo é sempre via provedor na nuvem; edição/legenda/cortes rodam local (FFmpeg).

## Comandos
- Ambiente: `scripts/bootstrap.sh` (extras: `scripts/bootstrap.sh tools`)
- Validar tudo (grátis): `scripts/verify_install.sh`  |  testes: `uv run pytest -q`  |  lint: `uv run ruff check . && uv run ruff format --check .`
- Dependências: edite `pyproject.toml` e rode `uv lock`; nunca edite `uv.lock` à mão.

## Mapa (leia só o que precisar)
- `src/vvf/config.py` — settings via env/.env (sem segredos no código)
- `src/vvf/{pipeline,providers,models}/` — vazios; arquitetura-alvo em `docs/SETUP.md` e plano no histórico
- `presets/` formatos de saída · `projects/` dados de execução (ignorado pelo Git) · `tests/`
- `docs/SETUP.md` ferramentas instaladas/pendências · `docs/COMPLIANCE.md` origem, consentimento, rótulo de IA
- Mapa de código: `docs/CODEBASE_MAP.md` (gerado pelo Cartographer quando houver código suficiente)

## Regras
1. **Custo**: nenhuma chamada paga (fal, Replicate, ElevenLabs, LLM) sem `VVF_ALLOW_PAID_CALLS=true` e teto `VVF_MAX_SPEND_USD`. Testes nunca chamam APIs pagas (marcador `cloud`, desligado por padrão).
2. **Segredos**: nunca em Git/logs/relatórios. `.env` é bloqueado por hook; use `.env.example` para documentar.
3. **Mídia** (vídeo/áudio/imagem) nunca vai ao Git (`.gitignore`). Dados vão em `projects/<nome>/`.
4. **Conteúdo sintético realista**: seguir `docs/COMPLIANCE.md` (registro de origem + consentimento + rótulo de IA). Sem rosto/voz de pessoa real sem consentimento.
5. Após editar Python: ruff formata automaticamente (hook); rode `uv run pytest -q` antes de commitar.

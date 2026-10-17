# Origem, consentimento e rótulo de IA

Todo projeto em `projects/<nome>/` deve ter um `manifest.json` com, no mínimo:

| Campo | Conteúdo |
|---|---|
| `sources[]` | arquivo original, hash SHA-256, quem filmou, data |
| `rights` | `own` / `licensed` / `third_party_with_consent` + prova (arquivo/link) |
| `people[]` | pessoas que aparecem/falam e `consent: true/false` |
| `transformations[]` | provedor, modelo, parâmetros, custo estimado e real |
| `synthetic` | `true` se o vídeo final tem conteúdo realista gerado/alterado por IA |
| `disclosure` | rótulo aplicado por plataforma (TikTok, Reels, Shorts) |

Regras: sem consentimento registrado, não se gera rosto/voz de pessoa real; vídeos com `synthetic: true`
só são exportados com o aviso de IA exigido pela plataforma. (Implementação do manifesto: próxima etapa.)

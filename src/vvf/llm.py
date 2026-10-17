"""Cliente de LLM para roteiros e planos. Interface mínima + implementação Anthropic."""

from __future__ import annotations

import json
from typing import Any, Protocol

from vvf.costs import Estimate
from vvf.errors import MissingCredential, ProviderError

# US$ por 1M de tokens (entrada, saída). Fonte: tabela de modelos da skill claude-api
# (cache de 2026-10-06). Confirme em https://platform.claude.com/docs/en/about-claude/pricing
# Modelo ausente da tabela => preço desconhecido => execução paga recusada.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
PRICE_SOURCE = "tabela claude-api em cache de 2026-10-06 (limite superior)"


class LLMClient(Protocol):
    model: str

    def estimate(self, system: str, user: str, max_tokens: int) -> Estimate: ...

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int) -> dict: ...


def estimate_tokens(text: str) -> int:
    """Estimativa conservadora (português tokeniza pior): ~2.8 caracteres por token."""
    return int(len(text) / 2.8) + 1


def estimate_cost(model: str, system: str, user: str, max_tokens: int) -> Estimate:
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return Estimate("anthropic", model, "llm", None, "modelo fora da tabela de preços")
    tin, tout = prices
    usd = estimate_tokens(system + user) * tin / 1e6 + max_tokens * tout / 1e6
    return Estimate("anthropic", model, "llm", round(usd, 6), PRICE_SOURCE)


class AnthropicLLM:
    """Usa o SDK oficial. Saída JSON garantida por `output_config.format` (json_schema)."""

    def __init__(self, api_key: str | None, model: str = "claude-opus-5-5", client=None):
        if client is None:
            if not api_key:
                raise MissingCredential("ANTHROPIC_API_KEY não configurada")
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
        self.client = client
        self.model = model

    def estimate(self, system: str, user: str, max_tokens: int) -> Estimate:
        return estimate_cost(self.model, system, user, max_tokens)

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int) -> dict:
        import anthropic

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "effort": "medium",
                "format": {"type": "json_schema", "schema": schema},
            },
        }
        try:
            response = self.client.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"API da Anthropic recusou a chamada: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"falha de rede ao chamar a Anthropic: {exc}") from exc
        if response.stop_reason == "refusal":
            raise ProviderError("o modelo recusou a solicitação (stop_reason=refusal)")
        if response.stop_reason == "max_tokens":
            raise ProviderError("resposta truncada (max_tokens); reduza o vídeo ou aumente o limite")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"o modelo não devolveu JSON válido: {exc}") from exc

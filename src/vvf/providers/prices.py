"""Tabela de preços mantida pelo usuário (config/provider_prices.json).

Preços de geração de vídeo mudam com frequência e nem todo provedor os expõe por API.
Por isso nada é assumido: sem preço aqui (ou na API do provedor), a execução paga é recusada.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from vvf.presets import config_dir


@dataclass
class UserPrice:
    usd_per_second: float | None
    usd_per_video: float | None
    usd_per_1k_chars: float | None
    verified_at: str
    source: str


def load_user_price(provider: str, model: str) -> UserPrice | None:
    path = config_dir() / "provider_prices.json"
    if not path.exists():
        return None
    entry = json.loads(path.read_text(encoding="utf-8")).get("prices", {}).get(f"{provider}:{model}")
    if not entry:
        return None
    return UserPrice(
        usd_per_second=entry.get("usd_per_second"),
        usd_per_video=entry.get("usd_per_video"),
        usd_per_1k_chars=entry.get("usd_per_1k_chars"),
        verified_at=entry.get("verified_at", "?"),
        source=entry.get("source", "?"),
    )


def price_from_table(provider: str, model: str, duration_s: float) -> tuple[float | None, str]:
    p = load_user_price(provider, model)
    if p is None:
        return None, "sem preço em config/provider_prices.json"
    note = f"tabela do usuário (verificado em {p.verified_at}; fonte: {p.source})"
    if p.usd_per_video is not None:
        return round(p.usd_per_video, 6), note
    if p.usd_per_second is not None:
        return round(p.usd_per_second * duration_s, 6), note
    return None, "entrada de preço sem valor"


def price_for_chars(provider: str, model: str, chars: int) -> tuple[float | None, str]:
    p = load_user_price(provider, model)
    if p is None or p.usd_per_1k_chars is None:
        return None, "sem usd_per_1k_chars em config/provider_prices.json"
    note = f"tabela do usuário (verificado em {p.verified_at}; fonte: {p.source})"
    return round(p.usd_per_1k_chars * chars / 1000, 6), note

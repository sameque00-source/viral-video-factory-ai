"""Fábrica de provedores. Adicionar um provedor = implementar `VideoProvider` e registrar aqui."""

from __future__ import annotations

from pathlib import Path

from vvf.config import Settings
from vvf.errors import UserInputError
from vvf.providers.base import VideoProvider

PROVIDER_NAMES = ("fal", "replicate")


def make_provider(name: str, settings: Settings) -> VideoProvider:
    cache = settings.vvf_cache_dir
    if name == "fal":
        from vvf.providers.fal import FalProvider

        key = settings.fal_key.get_secret_value() if settings.fal_key else None
        return FalProvider(key, cache_dir=Path(cache))
    if name == "replicate":
        from vvf.providers.replicate import ReplicateProvider

        token = settings.replicate_api_token.get_secret_value() if settings.replicate_api_token else None
        return ReplicateProvider(token)
    raise UserInputError(f"provedor desconhecido: {name} (disponíveis: {list(PROVIDER_NAMES)})")

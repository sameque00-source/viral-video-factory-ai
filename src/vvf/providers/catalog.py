"""Modelos de vídeo-para-vídeo sugeridos. Conferidos em 2026-10-17 (ver `verified`)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model: str
    summary: str
    verified: str  # o que foi conferido de fato e quando
    deprecated_on: str | None = None


CATALOG: list[ModelInfo] = [
    ModelInfo("fal", "fal-ai/luma-dream-machine/ray-2-flash/modify", "Luma Ray-2 Flash: edição por prompt",
              "esquema OpenAPI público conferido em 2026-10-17"),
    ModelInfo("fal", "fal-ai/luma-dream-machine/ray-2/modify", "Luma Ray-2: edição por prompt",
              "esquema OpenAPI público conferido em 2026-10-17", deprecated_on="2026-10-25"),
    ModelInfo("fal", "fal-ai/wan/v2.2-a14b/video-to-video", "Wan 2.2 A14B: vídeo para vídeo (strength)",
              "esquema OpenAPI público conferido em 2026-10-17"),
    ModelInfo("fal", "fal-ai/kling-video/o1/video-to-video/edit", "Kling O1: edição de vídeo",
              "esquema OpenAPI público conferido em 2026-10-17"),
    ModelInfo("fal", "fal-ai/hunyuan-video/video-to-video", "Hunyuan: vídeo para vídeo",
              "esquema OpenAPI público conferido em 2026-10-17"),
    ModelInfo("fal", "fal-ai/wan-vace-14b", "Wan VACE 14B: edição guiada",
              "esquema OpenAPI público conferido em 2026-10-17"),
    ModelInfo("replicate", "luma/modify-video", "Luma Modify Video: estilo e edição por prompt",
              "página existe (2026-10-17); esquema só com REPLICATE_API_TOKEN"),
    ModelInfo("replicate", "runwayml/gen4-aleph", "Runway Aleph: editar/transformar vídeo",
              "página existe (2026-10-17); esquema só com REPLICATE_API_TOKEN"),
]  # fmt: skip


def find(provider: str, model: str) -> ModelInfo | None:
    return next((m for m in CATALOG if m.provider == provider and m.model == model), None)

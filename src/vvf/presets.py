"""Presets de exportação (presets/*.json). Valores validados com pydantic."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from vvf.errors import UserInputError


class Preset(BaseModel):
    name: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0, le=120)
    video_codec: str = "libx264"
    profile: str = "high"
    x264_preset: str = "medium"
    crf: int = Field(ge=0, le=51)
    maxrate: str
    bufsize: str
    pix_fmt: str = "yuv420p"
    audio_codec: str = "aac"
    audio_bitrate: str = "160k"
    audio_rate: int = 48000
    loudness_lufs: float = -14.0
    true_peak_db: float = -1.5
    recommended_max_seconds: float
    caption_font: str = "Inter"
    caption_size: int = 84
    caption_safe_bottom: int = 380
    notes: str = ""


def presets_dir() -> Path:
    override = os.environ.get("VVF_PRESETS_DIR")
    return Path(override) if override else Path(__file__).resolve().parents[2] / "presets"


def available() -> list[str]:
    return sorted(p.stem for p in presets_dir().glob("*.json"))


def load_preset(name: str) -> Preset:
    path = presets_dir() / f"{name}.json"
    if not path.exists():
        raise UserInputError(f"preset desconhecido: {name} (disponíveis: {available()})")
    return Preset.model_validate_json(path.read_text(encoding="utf-8"))

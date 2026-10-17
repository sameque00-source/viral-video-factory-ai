"""Manifesto por vídeo: entradas, etapas, custos, saídas e conformidade (fonte da verdade)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Rights(StrEnum):
    own = "own"
    licensed = "licensed"
    third_party_with_consent = "third_party_with_consent"


class StepStatus(StrEnum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class Provenance(BaseModel):
    description: str = ""
    rights: Rights
    rights_evidence: str | None = None
    people_present: bool = False
    consent_confirmed: bool = False
    recorded_at: str = Field(default_factory=utcnow)


class MediaProbe(BaseModel):
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None = None
    has_audio: bool = False
    size_bytes: int = 0
    rotation: int = 0


class SourceInfo(BaseModel):
    original_name: str
    stored_path: str
    sha256: str
    probe: MediaProbe


class StepRecord(BaseModel):
    name: str
    status: StepStatus = StepStatus.pending
    attempts: int = 0
    input_hash: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    pid: int | None = None
    outputs: dict[str, str] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class CostEntry(BaseModel):
    at: str = Field(default_factory=utcnow)
    provider: str
    model: str
    kind: str
    estimated_usd: float | None
    actual_usd: float | None = None
    dry_run: bool = False
    request_id: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None


class RenderRecord(BaseModel):
    at: str = Field(default_factory=utcnow)
    preset: str
    path: str
    sha256: str
    duration: float
    width: int
    height: int
    checks: dict[str, bool] = Field(default_factory=dict)
    synthetic: bool = False
    clips: list[str] = Field(default_factory=list)


class TransformRecord(BaseModel):
    id: str
    clip_id: str
    provider: str
    model: str
    prompt: str
    params: dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"
    request_id: str | None = None
    output: str | None = None
    sha256: str | None = None
    cost_usd: float | None = None
    cached: bool = False
    error: str | None = None
    created_at: str = Field(default_factory=utcnow)


class Manifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    slug: str
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    provenance: Provenance | None = None
    source: SourceInfo | None = None
    steps: dict[str, StepRecord] = Field(default_factory=dict)
    costs: list[CostEntry] = Field(default_factory=list)
    transforms: list[TransformRecord] = Field(default_factory=list)
    renders: list[RenderRecord] = Field(default_factory=list)
    synthetic: bool = False
    synthetic_reasons: list[str] = Field(default_factory=list)

    def spent_usd(self) -> float:
        """Soma do que foi realmente gasto (ignora simulações)."""
        return round(
            sum((c.actual_usd if c.actual_usd is not None else (c.estimated_usd or 0.0))
                for c in self.costs if not c.dry_run),
            6,
        )  # fmt: skip

    def mark_synthetic(self, reason: str) -> None:
        self.synthetic = True
        if reason not in self.synthetic_reasons:
            self.synthetic_reasons.append(reason)

    def step(self, name: str) -> StepRecord:
        return self.steps.setdefault(name, StepRecord(name=name))


def save_manifest(manifest: Manifest, path: Path) -> None:
    """Escrita atômica: nunca deixa um manifesto pela metade."""
    manifest.updated_at = utcnow()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))

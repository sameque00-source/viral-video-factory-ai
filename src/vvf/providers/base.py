"""Contrato comum dos provedores de transformação de vídeo (fal.ai, Replicate, ...)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from vvf.costs import Estimate


@dataclass
class TransformRequest:
    input_video: Path
    prompt: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


@dataclass
class JobHandle:
    provider: str
    model: str
    request_id: str


@dataclass
class JobStatus:
    state: Literal["queued", "running", "succeeded", "failed"]
    message: str | None = None
    output_url: str | None = None


class VideoProvider(Protocol):
    name: str

    def has_credentials(self) -> bool: ...

    def validate(self, req: TransformRequest) -> tuple[list[str], list[str]]:
        """Devolve (problemas bloqueantes, avisos). Sem custo."""

    def estimate(self, req: TransformRequest) -> Estimate: ...

    def submit(self, req: TransformRequest) -> JobHandle:
        """Envia o trabalho (PAGO). Nunca deve ser chamado sem passar pelo CostGuard."""

    def poll(self, handle: JobHandle) -> JobStatus: ...

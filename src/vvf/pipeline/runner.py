"""Executor de etapas: cache por hash de entrada, verificação de saídas, retentativas e logs."""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vvf.errors import StepFailed, VVFError
from vvf.models.manifest import StepRecord, StepStatus, utcnow
from vvf.project import Project


@dataclass
class StepResult:
    outputs: dict[str, str]  # nome lógico -> caminho relativo ao projeto
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepOutcome:
    record: StepRecord
    cached: bool


def input_hash(inputs: dict[str, Any]) -> str:
    blob = json.dumps(inputs, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def upstream(project: Project, *names: str) -> dict[str, str | None]:
    """Hashes das etapas anteriores: mudar uma etapa invalida as que dependem dela."""
    return {n: (project.manifest.steps[n].input_hash if n in project.manifest.steps else None)
            for n in names}  # fmt: skip


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def outputs_valid(project: Project, outputs: dict[str, str]) -> bool:
    """Saída só vale se o arquivo existe e não está vazio."""
    if not outputs:
        return False
    for rel in outputs.values():
        p = project.path(rel)
        if not p.exists() or p.stat().st_size == 0:
            return False
    return True


def is_interrupted(rec: StepRecord) -> bool:
    return rec.status == StepStatus.running and not _pid_alive(rec.pid)


def _log(project: Project, name: str, message: str) -> None:
    log = project.root / "logs" / f"{name}.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"[{utcnow()}] {message}\n")


def run_step(
    project: Project,
    name: str,
    inputs: dict[str, Any],
    fn: Callable[[], StepResult],
    *,
    force: bool = False,
    retries: int = 2,
    sleep: Callable[[float], None] = time.sleep,
    validate: Callable[[StepResult], None] | None = None,
) -> StepOutcome:
    """Roda `fn` uma vez por combinação de entradas. Reaproveita resultado válido.

    - cache: etapa `done` com o mesmo hash e saídas íntegras não é refeita
    - retomada: etapa `running` sem processo vivo (interrompida) é refeita
    - retentativas: só para erros `retryable`; backoff exponencial limitado
    """
    h = input_hash(inputs)
    rec = project.manifest.step(name)
    if (
        not force
        and rec.status == StepStatus.done
        and rec.input_hash == h
        and outputs_valid(project, rec.outputs)
    ):
        _log(project, name, f"cache hit ({h})")
        return StepOutcome(rec, cached=True)

    last_error = ""
    for attempt in range(1, retries + 2):
        rec.status = StepStatus.running
        rec.attempts += 1
        rec.input_hash = h
        rec.started_at = utcnow()
        rec.finished_at = None
        rec.pid = os.getpid()
        rec.error = None
        project.save()
        _log(project, name, f"início tentativa {attempt} ({h})")
        try:
            result = fn()
            if validate:
                validate(result)
            if not outputs_valid(project, result.outputs):
                raise VVFError(f"etapa {name}: saídas ausentes ou vazias: {result.outputs}")
        except Exception as exc:  # noqa: BLE001 - registramos tudo e decidimos abaixo
            last_error = f"{type(exc).__name__}: {exc}"
            _log(project, name, f"falhou: {last_error}\n{traceback.format_exc()}")
            retryable = getattr(exc, "retryable", True)
            rec.status = StepStatus.failed
            rec.error = last_error
            rec.finished_at = utcnow()
            rec.pid = None
            project.save()
            if not retryable or attempt > retries:
                raise StepFailed(f"etapa '{name}' falhou: {last_error}") from exc
            sleep(min(2 ** (attempt - 1), 30))
            continue
        rec.status = StepStatus.done
        rec.outputs = result.outputs
        rec.meta = result.meta
        rec.finished_at = utcnow()
        rec.pid = None
        project.save()
        _log(project, name, f"concluída: {result.outputs}")
        return StepOutcome(rec, cached=False)
    raise StepFailed(f"etapa '{name}' falhou: {last_error}")  # pragma: no cover

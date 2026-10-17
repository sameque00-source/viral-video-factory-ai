"""Etapa D: transformações de vídeo por IA via provedores intercambiáveis.

Garantias:
- nada pago roda sem consentimento registrado, estimativa conhecida e dentro do teto
- o request_id é gravado ANTES de esperar o resultado: retomar nunca paga duas vezes
- resultados idênticos são reaproveitados do cache global (mesmo vídeo+modelo+prompt+parâmetros)
- o arquivo devolvido é verificado com ffprobe antes de ser marcado como concluído
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from vvf import media
from vvf.config import Settings
from vvf.costs import CostGuard, Estimate
from vvf.errors import ConsentRequired, MissingCredential, PlanError, ProviderError, UserInputError
from vvf.models.manifest import TransformRecord
from vvf.pipeline.plan import Clip, load_plan, save_plan
from vvf.pipeline.runner import StepResult, run_step
from vvf.project import Project
from vvf.providers.base import JobHandle, TransformRequest, VideoProvider
from vvf.providers.catalog import find

MAX_INPUT_SIDE = 1280
MAX_POLL_ERRORS = 5
SCALE_FILTER = (
    f"scale='min({MAX_INPUT_SIDE},iw)':'min({MAX_INPUT_SIDE},ih)'"
    ":force_original_aspect_ratio=decrease:force_divisible_by=2"
)


def default_download(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as resp:
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_bytes(1 << 20):
                f.write(chunk)


def prepare_input(project: Project, clip: Clip, settings: Settings) -> Path:
    """Recorta o trecho do original (sem áudio, lado maior limitado) para enviar ao provedor."""
    src = project.manifest.source
    assert src is not None
    out = project.root / "work" / "transforms" / f"{clip.id}_input.mp4"

    def work() -> StepResult:
        out.parent.mkdir(parents=True, exist_ok=True)
        media.run_ffmpeg(
            ["-ss", f"{clip.start:.3f}", "-t", f"{clip.end - clip.start:.3f}",
             "-i", str(project.path(src.stored_path)), "-an",
             "-vf", SCALE_FILTER,
             "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out)],
            what="recorte do trecho para transformação",
        )  # fmt: skip
        return StepResult({"input": project.rel(out)})

    run_step(
        project, f"transform_input:{clip.id}",
        {"sha": src.sha256, "start": clip.start, "end": clip.end}, work,
        retries=settings.vvf_step_retries,
    )  # fmt: skip
    return out


def cache_key(input_sha: str, req: TransformRequest, provider: str) -> str:
    blob = json.dumps(
        [input_sha, provider, req.model, req.prompt, req.params], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _next_id(project: Project) -> str:
    n = max([int(t.id[1:]) for t in project.manifest.transforms if t.id[1:].isdigit()] or [0])
    return f"t{n + 1}"


def _finish_download(
    project: Project, settings: Settings, rec: TransformRecord, url: str, clip: Clip,
    download: Callable[[str, Path], None], cache_file: Path | None,
) -> None:  # fmt: skip
    dest = project.root / "work" / "transforms" / f"{rec.id}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    download(url, dest)
    probe = media.probe(dest)  # levanta se não for vídeo legível
    wanted = clip.end - clip.start
    if probe.duration < max(0.3, wanted * 0.5):
        raise ProviderError(
            f"saída do provedor curta demais ({probe.duration:.1f}s para {wanted:.1f}s pedidos)"
        )
    rec.output, rec.sha256, rec.status = project.rel(dest), media.sha256_file(dest), "done"
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, cache_file)
    project.manifest.mark_synthetic(f"transformação por IA ({rec.provider}/{rec.model})")


def _poll_until_done(
    project: Project, provider: VideoProvider, rec: TransformRecord, handle: JobHandle,
    poll_interval: float, timeout_s: float, sleep: Callable[[float], None],
) -> str:  # fmt: skip
    deadline = time.monotonic() + timeout_s
    errors = 0
    while True:
        try:
            status = provider.poll(handle)
            errors = 0
        except Exception as exc:  # noqa: BLE001 - rede instável não pode perder o request_id
            errors += 1
            if errors >= MAX_POLL_ERRORS:
                rec.error = f"falhas seguidas ao consultar o status: {exc}"
                project.save()
                raise ProviderError(
                    rec.error + f" (retome com: vvf transform resume {project.slug} {rec.id})"
                ) from exc
            sleep(poll_interval)
            continue
        rec.status = "running" if status.state in ("queued", "running") else rec.status
        if status.state == "succeeded" and status.output_url:
            return status.output_url
        if status.state == "failed":
            rec.status, rec.error = "failed", status.message or "falha no provedor"
            project.save()
            raise ProviderError(f"o provedor reportou falha: {rec.error}")
        if time.monotonic() > deadline:
            project.save()
            raise ProviderError(
                "tempo esgotado aguardando o provedor; "
                f"retome com: vvf transform resume {project.slug} {rec.id}"
            )
        sleep(poll_interval)


def run_transform(
    project: Project,
    settings: Settings,
    provider: VideoProvider,
    *,
    clip_id: str,
    model: str,
    prompt: str,
    params: dict | None = None,
    execute: bool = False,
    price_usd: float | None = None,
    guard: CostGuard | None = None,
    poll_interval: float = 5.0,
    timeout_s: float = 1800.0,
    sleep: Callable[[float], None] = time.sleep,
    download: Callable[[str, Path], None] = default_download,
) -> TransformRecord:
    manifest = project.manifest
    plan = load_plan(project)
    clip = plan.get(clip_id)
    prov = manifest.provenance
    if execute and prov and prov.people_present and not prov.consent_confirmed:
        raise ConsentRequired(
            "há pessoas no vídeo sem consentimento confirmado: registre o consentimento antes de "
            "aplicar transformações por IA (vvf import --people --consent)"
        )
    info = find(provider.name, model)
    if info and info.deprecated_on:
        notice = f"modelo marcado como descontinuado em {info.deprecated_on}"
    else:
        notice = None

    input_path = prepare_input(project, clip, settings)
    req = TransformRequest(input_path, prompt, model, params or {}, clip.end - clip.start)
    problems, warnings = provider.validate(req)
    if notice:
        warnings.append(notice)
    if problems:
        raise UserInputError("entrada inválida para o modelo: " + "; ".join(problems))

    estimate = provider.estimate(req)
    if price_usd is not None:  # teto informado pelo usuário substitui um preço desconhecido
        estimate = Estimate(
            provider.name, model, estimate.kind, price_usd, "informado pelo usuário (--price-usd)"
        )
    guard = guard or CostGuard(settings, manifest)
    if guard.manifest is None:
        guard.manifest = manifest

    rec = TransformRecord(
        id=_next_id(project), clip_id=clip_id, provider=provider.name, model=model, prompt=prompt,
        params=params or {}, status="dry-run" if not execute else "pending",
    )  # fmt: skip
    input_sha = media.sha256_file(input_path)
    key = cache_key(input_sha, req, provider.name)
    cache_file = Path(settings.vvf_cache_dir) / "transforms" / f"{key}.mp4"

    if execute and cache_file.exists():  # resultado idêntico já pago antes: custo zero
        manifest.transforms.append(rec)
        dest = project.root / "work" / "transforms" / f"{rec.id}.mp4"
        shutil.copy2(cache_file, dest)
        rec.output, rec.sha256, rec.status, rec.cached = (
            project.rel(dest),
            media.sha256_file(dest),
            "done",
            True,
        )
        manifest.mark_synthetic(f"transformação por IA ({rec.provider}/{rec.model})")
        project.save()
        return rec

    entry = guard.authorize(
        estimate, dry_run=not execute, params={"clip": clip_id, "prompt": prompt, **(params or {})}
    )  # fmt: skip
    rec.cost_usd = estimate.usd
    manifest.transforms.append(rec)
    if not execute:
        project.save()
        rec.error = "; ".join(warnings) or None
        return rec
    if not provider.has_credentials():
        manifest.transforms.remove(rec)
        raise MissingCredential(f"credencial do provedor '{provider.name}' não configurada")

    handle = provider.submit(req)  # <- única chamada que cobra
    rec.request_id, rec.status = handle.request_id, "submitted"
    entry.request_id = handle.request_id
    project.save()  # request_id persistido antes de esperar: permite retomar sem pagar de novo

    url = _poll_until_done(project, provider, rec, handle, poll_interval, timeout_s, sleep)
    try:
        _finish_download(project, settings, rec, url, clip, download, cache_file)
    except Exception as exc:
        rec.status, rec.error = "failed", f"resultado inválido: {exc}"
        project.save()
        raise
    project.save()
    return rec


def resume_transform(
    project: Project, settings: Settings, provider: VideoProvider, transform_id: str, *,
    poll_interval: float = 5.0, timeout_s: float = 1800.0,
    sleep: Callable[[float], None] = time.sleep,
    download: Callable[[str, Path], None] = default_download,
) -> TransformRecord:  # fmt: skip
    """Continua uma transformação já enviada (não reenvia nem cobra de novo)."""
    rec = next((t for t in project.manifest.transforms if t.id == transform_id), None)
    if rec is None:
        raise UserInputError(f"transformação inexistente: {transform_id}")
    if rec.status == "done":
        return rec
    if not rec.request_id:
        raise UserInputError("esta transformação nunca foi enviada ao provedor (não há o que retomar)")
    clip = load_plan(project).get(rec.clip_id)
    handle = JobHandle(rec.provider, rec.model, rec.request_id)
    url = _poll_until_done(project, provider, rec, handle, poll_interval, timeout_s, sleep)
    try:
        _finish_download(project, settings, rec, url, clip, download, None)
    except Exception as exc:
        rec.status, rec.error = "failed", f"resultado inválido: {exc}"
        project.save()
        raise
    project.save()
    return rec


def use_transform(project: Project, clip_id: str, transform_id: str) -> None:
    """Faz o trecho passar a usar o vídeo transformado no render."""
    rec = next((t for t in project.manifest.transforms if t.id == transform_id), None)
    if rec is None or rec.status != "done":
        raise PlanError(f"transformação '{transform_id}' não está concluída")
    plan = load_plan(project)
    clip = plan.get(clip_id)
    clip.source = f"transform:{transform_id}"
    plan.edited, plan.status = True, "draft"
    save_plan(project, plan)

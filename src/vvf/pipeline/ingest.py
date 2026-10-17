"""Etapa: importar e validar o vídeo original; registrar origem e consentimento."""

from __future__ import annotations

import shutil
from pathlib import Path

from vvf import media
from vvf.config import Settings
from vvf.errors import IngestError
from vvf.models.manifest import MediaProbe, Provenance, Rights, SourceInfo
from vvf.pipeline.runner import StepResult, run_step
from vvf.project import Project

ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def validate_probe(probe: MediaProbe, settings: Settings) -> list[str]:
    """Levanta IngestError para problemas bloqueantes; devolve avisos não bloqueantes."""
    if probe.duration < settings.vvf_min_duration_s:
        raise IngestError(f"vídeo curto demais ({probe.duration:.2f}s)")
    if probe.duration > settings.vvf_max_duration_s:
        raise IngestError(f"vídeo longo demais ({probe.duration:.0f}s)")
    if min(probe.width, probe.height) < settings.vvf_min_height:
        raise IngestError(f"resolução baixa demais ({probe.width}x{probe.height})")
    warnings = []
    if not probe.has_audio:
        warnings.append("sem trilha de áudio: não haverá transcrição nem legendas")
    if probe.height < 720 and probe.width < 720:
        warnings.append("resolução abaixo de 720p: o 9:16 final ficará suavizado")
    return warnings


def make_provenance(
    *,
    description: str,
    rights: Rights,
    rights_evidence: str | None,
    people_present: bool,
    consent_confirmed: bool,
) -> Provenance:
    if rights == Rights.third_party_with_consent and not (rights_evidence or consent_confirmed):
        raise IngestError("direitos de terceiros exigem --evidence ou --consent confirmado")
    return Provenance(
        description=description,
        rights=rights,
        rights_evidence=rights_evidence,
        people_present=people_present,
        consent_confirmed=consent_confirmed,
    )


def import_video(project: Project, video: Path, provenance: Provenance, settings: Settings) -> list[str]:
    """Copia o original para o projeto (nunca altera o arquivo de origem) e extrai o áudio."""
    if not video.is_file():
        raise IngestError(f"arquivo não encontrado: {video}")
    if video.suffix.lower() not in ALLOWED_EXT:
        raise IngestError(f"formato não suportado: {video.suffix} (aceitos: {sorted(ALLOWED_EXT)})")
    probe = media.probe(video)
    warnings = validate_probe(probe, settings)

    def work() -> StepResult:
        dest = project.root / "original" / f"source{video.suffix.lower()}"
        shutil.copy2(video, dest)
        digest = media.sha256_file(dest)
        outputs = {"original": project.rel(dest)}
        if probe.has_audio:
            wav = project.root / "work" / "audio.wav"
            media.extract_audio(dest, wav)
            outputs["audio"] = project.rel(wav)
        project.manifest.provenance = provenance
        project.manifest.source = SourceInfo(
            original_name=video.name,
            stored_path=project.rel(dest),
            sha256=digest,
            probe=probe,
        )
        return StepResult(outputs, {"warnings": warnings, "sha256": digest})

    sha = media.sha256_file(video)
    run_step(project, "ingest", {"sha256": sha, "name": video.name}, work, retries=0)
    return warnings

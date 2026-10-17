"""Etapa: transcrição com timestamps por palavra. O backend é intercambiável (Transcriber)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from vvf.config import Settings
from vvf.errors import UserInputError
from vvf.pipeline.runner import StepOutcome, StepResult, run_step, upstream
from vvf.project import Project


class Word(BaseModel):
    text: str
    start: float
    end: float
    probability: float | None = None


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = []


class Transcript(BaseModel):
    language: str
    duration: float
    model: str
    segments: list[Segment]
    edits: list[str] = []  # correções manuais ("povo→polvo (2x)")

    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: Path, language: str | None) -> Transcript: ...


class FasterWhisperTranscriber:
    """Transcrição local (CPU, int8). O modelo é baixado do Hugging Face no primeiro uso."""

    def __init__(self, model: str = "small", device: str = "cpu", compute_type: str = "int8"):
        self.model_name, self.device, self.compute_type = model, device, compute_type
        self.name = f"faster-whisper:{model}"
        self._model = None

    def transcribe(self, audio: Path, language: str | None) -> Transcript:
        from faster_whisper import WhisperModel

        if self._model is None:
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        segments, info = self._model.transcribe(
            str(audio), language=language, word_timestamps=True, vad_filter=True
        )
        out = []
        for i, seg in enumerate(segments):
            words = [
                Word(
                    text=w.word.strip(),
                    start=round(float(w.start), 3),
                    end=round(float(w.end), 3),
                    probability=round(float(w.probability), 3),
                )
                for w in (seg.words or [])
            ]
            out.append(
                Segment(
                    id=i,
                    start=round(float(seg.start), 3),
                    end=round(float(seg.end), 3),
                    text=seg.text.strip(),
                    words=words,
                )
            )
        return Transcript(
            language=info.language,
            duration=round(float(info.duration), 3),
            model=self.name,
            segments=out,
        )


def _case_like(original: str, replacement: str) -> str:
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def replace_word(project: Project, old: str, new: str) -> int:
    """Corrige uma palavra mal transcrita (palavra inteira, preserva maiúsculas e pontuação).

    Atualiza transcrição, legendas (palavras com timestamp) e o plano (gancho/título/citações
    que vieram da transcrição). Devolve quantas ocorrências foram trocadas.
    """
    if not old.strip() or not new.strip():
        raise UserInputError("informe a palavra errada e a correta")
    pattern = re.compile(rf"(?<!\w){re.escape(old)}(?!\w)", re.IGNORECASE)
    count = 0

    def sub(text: str) -> str:
        nonlocal count
        out, n = pattern.subn(lambda m: _case_like(m.group(0), new), text)
        count += n
        return out

    transcript = load_transcript(project)
    word_hits = 0
    for seg in transcript.segments:
        seg.text = sub(seg.text)
        for w in seg.words:
            before = w.text
            w.text = pattern.sub(lambda m: _case_like(m.group(0), new), w.text)
            word_hits += before != w.text
    if count == 0:
        raise UserInputError(f"'{old}' não aparece na transcrição")
    transcript.edits.append(f"{old}→{new} ({count}x)")
    rec = project.manifest.steps["transcribe"]
    project.path(rec.outputs["transcript"]).write_text(transcript.model_dump_json(indent=2), encoding="utf-8")

    from vvf.pipeline import plan as planmod  # evita import circular

    if planmod.plan_path(project).exists():
        plan = planmod.load_plan(project)
        for clip in plan.clips:
            if clip.hook_source == "transcript":
                clip.hook = pattern.sub(lambda m: _case_like(m.group(0), new), clip.hook)
                clip.title = pattern.sub(lambda m: _case_like(m.group(0), new), clip.title)
                clip.description = pattern.sub(lambda m: _case_like(m.group(0), new), clip.description)
            clip.quotes = [pattern.sub(lambda m: _case_like(m.group(0), new), q) for q in clip.quotes]
        plan.transcript_sha = planmod.transcript_sha(transcript)
        plan.status = "draft"  # conteúdo mudou: revisar e aprovar de novo
        planmod.save_plan(project, plan)
    project.save()
    return count


def transcribe(
    project: Project,
    settings: Settings,
    transcriber: Transcriber | None = None,
    *,
    force: bool = False,
    discard_edits: bool = False,
) -> StepOutcome:
    ingest = project.manifest.steps.get("ingest")
    assert ingest is not None, "importe o vídeo antes de transcrever"
    if force and not discard_edits:
        try:
            if load_transcript(project).edits:
                raise UserInputError(
                    "há correções manuais na transcrição: use --discard-edits para refazer do zero"
                )
        except FileNotFoundError:
            pass
    transcriber = transcriber or FasterWhisperTranscriber(
        settings.vvf_whisper_model, settings.vvf_whisper_device, settings.vvf_whisper_compute_type
    )

    def work() -> StepResult:
        out = project.root / "work" / "transcript.json"
        if "audio" not in ingest.outputs:  # sem áudio: registra vazio, nunca inventa fala
            transcript = Transcript(
                language=settings.vvf_language,
                duration=project.manifest.source.probe.duration,
                model="none",
                segments=[],
            )
        else:
            transcript = transcriber.transcribe(project.path(ingest.outputs["audio"]), settings.vvf_language)
        out.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
        return StepResult(
            {"transcript": project.rel(out)},
            {"segments": len(transcript.segments), "model": transcript.model},
        )

    return run_step(
        project,
        "transcribe",
        {"model": transcriber.name, "lang": settings.vvf_language, **upstream(project, "ingest")},
        work,
        force=force,
        retries=settings.vvf_step_retries,
    )


def load_transcript(project: Project) -> Transcript:
    rec = project.manifest.steps.get("transcribe")
    if not rec or "transcript" not in rec.outputs:
        raise FileNotFoundError("transcrição ainda não gerada: rode 'vvf analyze'")
    return Transcript.model_validate_json(project.path(rec.outputs["transcript"]).read_text())

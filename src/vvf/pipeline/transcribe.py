"""Etapa: transcrição com timestamps por palavra. O backend é intercambiável (Transcriber)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from vvf.config import Settings
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


def transcribe(
    project: Project,
    settings: Settings,
    transcriber: Transcriber | None = None,
    *,
    force: bool = False,
) -> StepOutcome:
    ingest = project.manifest.steps.get("ingest")
    assert ingest is not None, "importe o vídeo antes de transcrever"
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

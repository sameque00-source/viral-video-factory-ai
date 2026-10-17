"""Etapa: detecção de cenas (PySceneDetect) e energia de áudio por janela."""

from __future__ import annotations

import json
import math
import wave

import numpy as np
from scenedetect import ContentDetector, detect

from vvf.config import Settings
from vvf.pipeline.runner import StepOutcome, StepResult, run_step, upstream
from vvf.project import Project


def detect_scenes(project: Project, settings: Settings, *, force: bool = False) -> StepOutcome:
    source = project.manifest.source
    assert source is not None, "importe o vídeo antes de analisar cenas"
    threshold = settings.vvf_scene_threshold

    def work() -> StepResult:
        video = project.path(source.stored_path)
        found = detect(str(video), ContentDetector(threshold=threshold, min_scene_len=8))
        duration = source.probe.duration
        if found:
            scenes = [(s.seconds, e.seconds) for s, e in found]
            scenes[-1] = (scenes[-1][0], max(scenes[-1][1], duration))
        else:
            scenes = [(0.0, duration)]  # sem cortes: o vídeo todo é uma cena
        payload = {
            "threshold": threshold,
            "scenes": [
                {"index": i, "start": round(s, 3), "end": round(e, 3), "duration": round(e - s, 3)}
                for i, (s, e) in enumerate(scenes)
            ],
        }
        out = project.root / "work" / "scenes.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return StepResult({"scenes": project.rel(out)}, {"count": len(scenes)})

    return run_step(
        project,
        "scenes",
        {"sha": source.sha256, "threshold": threshold, **upstream(project, "ingest")},
        work,
        force=force,
        retries=settings.vvf_step_retries,
    )


def audio_energy(wav_path, window_s: float = 0.5) -> list[dict]:
    """RMS em dBFS por janela. Piso em -90 dB para silêncio digital."""
    with wave.open(str(wav_path), "rb") as w:
        rate, n = w.getframerate(), w.getnframes()
        samples = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    step = max(1, int(rate * window_s))
    out = []
    for i in range(0, len(samples), step):
        chunk = samples[i : i + step]
        rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) else 0.0
        db = 20 * math.log10(rms) if rms > 1e-5 else -90.0
        out.append({"t": round(i / rate, 3), "db": round(db, 1)})
    return out


def compute_audio_energy(project: Project, settings: Settings, *, force: bool = False):
    ingest = project.manifest.steps.get("ingest")
    if not ingest or "audio" not in ingest.outputs:
        return None
    wav = project.path(ingest.outputs["audio"])

    def work() -> StepResult:
        out = project.root / "work" / "energy.json"
        out.write_text(json.dumps(audio_energy(wav)), encoding="utf-8")
        return StepResult({"energy": project.rel(out)})

    return run_step(
        project, "energy", upstream(project, "ingest"), work, force=force,
        retries=settings.vvf_step_retries,
    )  # fmt: skip

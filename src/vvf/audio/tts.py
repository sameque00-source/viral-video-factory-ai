"""Narração por IA: Piper (local, grátis) e ElevenLabs (pago). Sincroniza com a duração do trecho."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

import httpx

from vvf import media
from vvf.config import Settings
from vvf.costs import CostGuard, Estimate
from vvf.errors import MissingCredential, ProviderError, UserInputError
from vvf.pipeline.plan import load_plan, save_plan
from vvf.pipeline.runner import StepResult, run_step
from vvf.project import Project
from vvf.providers.prices import price_for_chars

MAX_CHARS = 2500
ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


class TTSProvider(Protocol):
    name: str
    voice: str

    def has_credentials(self) -> bool: ...

    def estimate(self, text: str) -> Estimate: ...

    def synthesize(self, text: str, dest: Path) -> None: ...


class PiperTTS:
    """Piper roda como processo separado (licença GPL-3.0): não é importado nem distribuído aqui."""

    name = "piper"

    def __init__(self, voice_model: str | None, binary: str = "piper"):
        self.voice = voice_model or ""
        self.binary = binary

    def has_credentials(self) -> bool:  # local: precisa de binário e voz, não de chave
        return bool(self.voice) and Path(self.voice).exists() and shutil.which(self.binary) is not None

    def estimate(self, text: str) -> Estimate:
        return Estimate("piper", self.voice or "-", "tts", 0.0, "execução local, sem custo", local=True)

    def synthesize(self, text: str, dest: Path) -> None:
        if not self.has_credentials():
            raise MissingCredential(
                "Piper indisponível: instale o binário 'piper' e aponte VVF_PIPER_VOICE para o .onnx da voz"
            )
        proc = subprocess.run(
            [self.binary, "-m", self.voice, "-f", str(dest), "--", text], capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise ProviderError(f"piper falhou: {proc.stderr.strip()[-300:]}")


class ElevenLabsTTS:
    name = "elevenlabs"

    def __init__(self, api_key: str | None, voice_id: str | None, model_id: str = "eleven_multilingual_v2",
                 *, http: httpx.Client | None = None):  # fmt: skip
        self.api_key, self.voice, self.model_id = api_key, voice_id or "", model_id
        self.http = http or httpx.Client(timeout=120.0)

    def has_credentials(self) -> bool:
        return bool(self.api_key and self.voice)

    def estimate(self, text: str) -> Estimate:
        usd, note = price_for_chars("elevenlabs", self.model_id, len(text))
        return Estimate("elevenlabs", self.model_id, "tts", usd, note)

    def synthesize(self, text: str, dest: Path) -> None:
        if not self.has_credentials():
            raise MissingCredential("ELEVENLABS_API_KEY e VVF_ELEVENLABS_VOICE_ID são necessários")
        resp = self.http.post(
            ELEVEN_URL.format(voice_id=self.voice),
            params={"output_format": "mp3_44100_128"},
            headers={"xi-api-key": self.api_key or "", "Content-Type": "application/json"},
            json={"text": text, "model_id": self.model_id},
        )
        if resp.status_code != 200:
            raise ProviderError(f"ElevenLabs respondeu {resp.status_code}: {resp.text[:200]}")
        mp3 = dest.with_suffix(".mp3")
        mp3.write_bytes(resp.content)
        media.run_ffmpeg(
            ["-i", str(mp3), "-ar", "48000", "-ac", "2", str(dest)], what="conversão do MP3 para WAV"
        )
        mp3.unlink(missing_ok=True)


def make_tts(name: str, settings: Settings) -> TTSProvider:
    if name == "piper":
        return PiperTTS(settings.vvf_piper_voice, settings.vvf_piper_bin)
    if name == "elevenlabs":
        key = settings.elevenlabs_api_key.get_secret_value() if settings.elevenlabs_api_key else None
        return ElevenLabsTTS(key, settings.vvf_elevenlabs_voice_id, settings.vvf_elevenlabs_model)
    raise UserInputError(f"provedor de voz desconhecido: {name} (piper, elevenlabs)")


def narrate(
    project: Project,
    settings: Settings,
    tts: TTSProvider,
    *,
    clip_id: str,
    text: str,
    mode: str = "duck",
    execute: bool = True,
    price_usd: float | None = None,
    guard: CostGuard | None = None,
) -> Path | None:
    """Gera a narração do trecho, valida a duração e a associa ao plano. Dry-run só estima."""
    text = text.strip()
    if not text:
        raise UserInputError("texto da narração vazio")
    if len(text) > MAX_CHARS:
        raise UserInputError(f"texto longo demais ({len(text)} > {MAX_CHARS} caracteres)")
    plan = load_plan(project)
    clip = plan.get(clip_id)
    est = tts.estimate(text)
    if price_usd is not None:
        est = Estimate(
            est.provider, est.model, est.kind, price_usd, "informado pelo usuário (--price-usd)", est.local
        )
    guard = guard or CostGuard(settings, project.manifest)
    if guard.manifest is None:
        guard.manifest = project.manifest
    guard.authorize(est, dry_run=not execute, params={"clip": clip_id, "chars": len(text)})
    if not execute:
        project.save()
        return None

    if not tts.has_credentials():
        hint = {
            "piper": "instale o binário 'piper' e aponte VVF_PIPER_VOICE para o .onnx da voz",
            "elevenlabs": "defina ELEVENLABS_API_KEY e VVF_ELEVENLABS_VOICE_ID",
        }.get(tts.name, "configure o provedor")
        raise MissingCredential(f"provedor de voz '{tts.name}' indisponível: {hint}")
    digest = hashlib.sha256(f"{tts.name}|{tts.voice}|{text}".encode()).hexdigest()[:12]
    out = project.root / "work" / "narration" / f"{clip_id}_{digest}.wav"

    def work() -> StepResult:
        out.parent.mkdir(parents=True, exist_ok=True)
        tts.synthesize(text, out)
        return StepResult({"narration": project.rel(out)}, {"seconds": media.probe_audio(out)})

    run_step(project, f"narrate:{clip_id}", {"d": digest}, work, retries=settings.vvf_step_retries)
    seconds = media.probe_audio(out)
    wanted = clip.end - clip.start
    if seconds > wanted + 0.3:
        raise UserInputError(
            f"a narração dura {seconds:.1f}s e o trecho {wanted:.1f}s: encurte o texto ou amplie o trecho "
            f"(vvf plan edit {project.slug} {clip_id} --end ...)"
        )
    clip.narration, clip.narration_mode = project.rel(out), mode  # type: ignore[assignment]
    plan.edited, plan.status = True, "draft"
    save_plan(project, plan)
    project.manifest.mark_synthetic("narração com voz sintética")
    project.save()
    return out

"""Wrappers finos de FFmpeg/ffprobe. Toda chamada falha com mensagem útil, nunca silenciosamente."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from vvf.errors import ProviderError, UserInputError
from vvf.models.manifest import MediaProbe


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _fraction(value: str | None) -> float:
    if not value or "/" not in value:
        return float(value) if value else 0.0
    num, den = value.split("/")
    return float(num) / float(den) if float(den) else 0.0


def probe(path: Path) -> MediaProbe:
    """Lê metadados via ffprobe. Levanta UserInputError se o arquivo não for mídia legível."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
         str(path)],
        capture_output=True, text=True,
    )  # fmt: skip
    if proc.returncode != 0:
        raise UserInputError(f"ffprobe não conseguiu ler {path.name}: {proc.stderr.strip()[:300]}")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise UserInputError(f"{path.name} não tem trilha de vídeo")
    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"]) % 360
    width, height = int(video["width"]), int(video["height"])
    if rotation in (90, 270):
        width, height = height, width
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    return MediaProbe(
        duration=duration,
        width=width,
        height=height,
        fps=round(fps, 3),
        video_codec=video.get("codec_name", "?"),
        audio_codec=audio.get("codec_name") if audio else None,
        has_audio=audio is not None,
        size_bytes=int(data.get("format", {}).get("size") or path.stat().st_size),
        rotation=rotation,
    )


def run_ffmpeg(args: list[str], *, what: str, cwd: Path | None = None) -> None:
    """Executa ffmpeg; em falha inclui o final do stderr na exceção."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise ProviderError(f"ffmpeg falhou ({what}): {tail}")


def extract_audio(video: Path, dest: Path, sample_rate: int = 16000) -> None:
    """Áudio mono 16 kHz PCM (formato ideal para transcrição)."""
    run_ffmpeg(
        ["-i", str(video), "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
         str(dest)],
        what="extração de áudio",
    )  # fmt: skip


def probe_audio(path: Path) -> float:
    """Duração de um arquivo de áudio (s). Recusa arquivos sem trilha de áudio."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )  # fmt: skip
    out = proc.stdout.strip()
    if proc.returncode != 0 or not out or out == "N/A":
        raise UserInputError(f"{path.name} não é um áudio válido")
    return round(float(out), 3)

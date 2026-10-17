import subprocess
from pathlib import Path

import pytest

from vvf.config import Settings
from vvf.models.manifest import Provenance, Rights
from vvf.pipeline.transcribe import Segment, Transcript, Word
from vvf.project import Project


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    """Nenhum teste lê .env real nem credenciais do ambiente."""
    for var in (
        "ANTHROPIC_API_KEY", "FAL_KEY", "REPLICATE_API_TOKEN", "ELEVENLABS_API_KEY",
        "LLM_BASE_URL", "LLM_API_KEY", "VVF_ALLOW_PAID_CALLS", "VVF_MAX_SPEND_USD",
    ):  # fmt: skip
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Vídeo sintético 640x360, 8 s: 3 cenas coloridas + tom de áudio (sem direitos autorais)."""
    out = tmp_path_factory.mktemp("media") / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=red:s=640x360:r=30:d=3",
            "-f", "lavfi", "-i", "color=c=green:s=640x360:r=30:d=3",
            "-f", "lavfi", "-i", "color=c=blue:s=640x360:r=30:d=2",
            "-f", "lavfi", "-i", "sine=frequency=330:duration=8",
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(out),
        ],
        check=True,
    )  # fmt: skip
    return out


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def provenance() -> Provenance:
    return Provenance(description="vídeo sintético de teste", rights=Rights.own, people_present=False)


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "projects", "demo")


class FakeTranscriber:
    """Dublê de teste: devolve uma transcrição fixa. NÃO é usado fora dos testes."""

    name = "fake"

    def __init__(self, transcript: Transcript):
        self.transcript = transcript
        self.calls = 0

    def transcribe(self, audio: Path, language: str | None) -> Transcript:
        self.calls += 1
        return self.transcript


def make_transcript(duration: float = 8.0) -> Transcript:
    sentences = [
        (0.0, 2.8, "Você sabia que o polvo tem três corações?"),
        (3.0, 5.8, "Dois bombeiam sangue para as brânquias."),
        (6.0, 7.9, "Incrível, ninguém imagina isso!"),
    ]
    segs = []
    for i, (s, e, text) in enumerate(sentences):
        words = text.split()
        step = (e - s) / len(words)
        segs.append(
            Segment(
                id=i, start=s, end=e, text=text,
                words=[
                    Word(text=w, start=round(s + k * step, 3), end=round(s + (k + 1) * step, 3))
                    for k, w in enumerate(words)
                ],
            )
        )  # fmt: skip
    return Transcript(language="pt", duration=duration, model="fake", segments=segs)


@pytest.fixture
def transcript() -> Transcript:
    return make_transcript()

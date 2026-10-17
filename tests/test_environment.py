"""Valida que o ambiente tem as ferramentas e bibliotecas do projeto. Sem rede, sem custo."""

import importlib
import shutil
import subprocess

import pytest

MODULES = [
    "typer",
    "pydantic",
    "pydantic_settings",
    "httpx",
    "tenacity",
    "diskcache",
    "anthropic",
    "fal_client",
    "replicate",
    "faster_whisper",
    "scenedetect",
    "cv2",
    "pysubs2",
    "ffmpeg_normalize",
    "vvf",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


@pytest.mark.parametrize("binary", ["ffmpeg", "ffprobe"])
def test_binary_available(binary: str) -> None:
    assert shutil.which(binary), f"{binary} não encontrado no PATH"


def test_ffmpeg_has_required_codecs() -> None:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True
    ).stdout
    assert "libx264" in out
    assert "aac" in out

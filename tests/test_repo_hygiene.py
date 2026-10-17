"""Garante que segredos e mídia não entrem no Git."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [f for f in out.splitlines() if (ROOT / f).is_file()]


def test_env_example_has_no_values_for_secrets() -> None:
    pattern = re.compile(r"^(\w*(KEY|TOKEN|SECRET)\w*)=(\S+)", re.MULTILINE)
    text = (ROOT / ".env.example").read_text()
    assert not pattern.findall(text), "o .env.example não pode conter valores de chaves"


def test_no_env_or_media_is_tracked() -> None:
    bad = [
        f for f in _tracked_files()
        if re.fullmatch(r"(.*/)?\.env(\..+)?", f) and not f.endswith(".env.example")
        or f.lower().endswith((".mp4", ".mov", ".mkv", ".wav", ".mp3", ".safetensors"))
    ]  # fmt: skip
    assert bad == []


def test_no_secret_looking_strings_in_tracked_files() -> None:
    patterns = [
        re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
        re.compile(r"sk-[A-Za-z0-9]{32,}"),
        re.compile(r"ghp_[A-Za-z0-9]{30,}"),
        re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
        re.compile(r"r8_[A-Za-z0-9]{30,}"),
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ]
    this_file = Path(__file__).resolve()
    hits = []
    for rel in _tracked_files():
        path = ROOT / rel
        if path == this_file or path.name == "uv.lock":
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        hits += [rel for p in patterns if p.search(text)]
    assert hits == []

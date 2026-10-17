"""Exercita FFmpeg, PySceneDetect, pysubs2 e ffmpeg-normalize com mídia sintética local."""

import json
import subprocess
from pathlib import Path

import pysubs2
import pytest
from scenedetect import ContentDetector, detect


@pytest.fixture(scope="module")
def two_scene_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """2 s vermelhos + 2 s azuis, com tom de áudio, 320x568 (9:16)."""
    out = tmp_path_factory.mktemp("media") / "scenes.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=red:s=320x568:r=24:d=2",
            "-f", "lavfi", "-i", "color=c=blue:s=320x568:r=24:d=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out),
        ],
        check=True,
    )  # fmt: skip
    return out


def test_ffprobe_reads_generated_video(two_scene_video: Path) -> None:
    info = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
             str(two_scene_video)],
            capture_output=True, text=True, check=True,
        ).stdout
    )  # fmt: skip
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (320, 568)


def test_scenedetect_finds_the_cut(two_scene_video: Path) -> None:
    scenes = detect(str(two_scene_video), ContentDetector())
    assert len(scenes) == 2


def test_subtitles_burn_in(two_scene_video: Path, tmp_path: Path) -> None:
    subs = pysubs2.SSAFile()
    subs.append(pysubs2.SSAEvent(start=0, end=1500, text="teste de legenda"))
    ass = tmp_path / "caption.ass"
    subs.save(str(ass))
    out = tmp_path / "burned.mp4"
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(two_scene_video),
         "-vf", f"ass={ass}", "-c:a", "copy", str(out)],
        capture_output=True, text=True,
    )  # fmt: skip
    if result.returncode != 0 and "No such filter" in result.stderr:
        pytest.skip("FFmpeg sem libass: legendas queimadas indisponíveis neste build")
    assert result.returncode == 0, result.stderr
    assert out.stat().st_size > 0


def test_loudness_normalization_runs(two_scene_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "normalized.mp4"
    result = subprocess.run(
        ["uv", "run", "--no-sync", "ffmpeg-normalize", str(two_scene_video), "-o", str(out),
         "-c:a", "aac", "-f"],
        capture_output=True, text=True,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert out.exists()

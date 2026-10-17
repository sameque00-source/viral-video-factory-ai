import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import FakeTranscriber
from vvf.audio.library import Library
from vvf.errors import PlanError, RenderVerificationError, UserInputError
from vvf.models.manifest import TransformRecord
from vvf.pipeline import plan as planmod
from vvf.pipeline.ingest import import_video
from vvf.pipeline.render import (
    AudioItem,
    RenderOptions,
    parse_loudnorm_json,
    pick_mode,
    render_video,
    verify_output,
)
from vvf.pipeline.scenes import compute_audio_energy, detect_scenes
from vvf.pipeline.transcribe import transcribe
from vvf.presets import load_preset


@pytest.fixture
def ready(project, sample_video, provenance, settings, transcript):
    """Projeto analisado, com plano gerado e aprovado (1 trecho cobrindo o vídeo curto)."""
    import_video(project, sample_video, provenance, settings)
    detect_scenes(project, settings)
    compute_audio_energy(project, settings)
    transcribe(project, settings, FakeTranscriber(transcript))
    planmod.generate_plan(project, settings, n_clips=1)
    planmod.approve_plan(project)
    return project


def _opts(**kw) -> RenderOptions:
    # ultrafast só acelera os testes; os presets reais usam "medium"
    preset = load_preset("tiktok").model_copy(update={"x264_preset": "ultrafast"})
    return RenderOptions(preset=preset, **kw)


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return json.loads(out)


def _frame(video: Path, t: float):
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, check=True,
    )  # fmt: skip
    return cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)


def test_renders_vertical_video_that_passes_all_checks(ready, settings):
    rec = render_video(ready, settings, [planmod.load_plan(ready).get("c1")], _opts(), name="demo_c1")
    assert all(rec.checks.values()), rec.checks
    info = _probe(ready.path(rec.path))
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert (v["width"], v["height"], v["codec_name"], v["pix_fmt"]) == (1080, 1920, "h264", "yuv420p")
    assert a["codec_name"] == "aac" and int(a["sample_rate"]) == 48000
    assert rec.sha256 and ready.manifest.renders[0].path == rec.path


def test_loudness_is_normalized_to_target(ready, settings):
    rec = render_video(ready, settings, [planmod.load_plan(ready).get("c1")], _opts(), name="loud")
    from vvf.pipeline.render import measure_output_loudness

    assert abs(measure_output_loudness(ready.path(rec.path)) - (-14.0)) <= 1.5


def test_captions_are_actually_burned_into_the_frame(ready, settings):
    clip = planmod.load_plan(ready).get("c1")
    with_caps = render_video(ready, settings, [clip], _opts(), name="caps")
    no_caps = render_video(ready, settings, [clip], _opts(captions=False), name="nocaps")
    t = 0.5  # "VOCÊ" é a palavra ativa (amarela)

    def yellow(video):
        img = _frame(ready.path(video.path), t)
        low = img[int(img.shape[0] * 0.55) :]
        b, g, r = low[..., 0].astype(int), low[..., 1].astype(int), low[..., 2].astype(int)
        return int(((r > 200) & (g > 180) & (b < 90)).sum())

    assert yellow(with_caps) > 500
    assert yellow(no_caps) == 0


def test_second_render_hits_cache(ready, settings):
    clip = planmod.load_plan(ready).get("c1")
    render_video(ready, settings, [clip], _opts(), name="cache")
    step = ready.manifest.steps["render:tiktok:cache"]
    first = (step.attempts, step.finished_at)
    render_video(ready, settings, [clip], _opts(), name="cache")
    assert (step.attempts, step.finished_at) == first
    render_video(ready, settings, [clip], _opts(), name="cache", force=True)
    assert step.attempts == first[0] + 1


def test_changing_options_invalidates_cache(ready, settings):
    clip = planmod.load_plan(ready).get("c1")
    render_video(ready, settings, [clip], _opts(), name="opt")
    attempts = ready.manifest.steps["render:tiktok:opt"].attempts
    render_video(ready, settings, [clip], _opts(highlight=False), name="opt")
    assert ready.manifest.steps["render:tiktok:opt"].attempts == attempts + 1


def test_dry_run_returns_commands_and_creates_nothing(ready, settings):
    clip = planmod.load_plan(ready).get("c1")
    cmds = render_video(ready, settings, [clip], _opts(), name="dry", dry_run=True)
    assert cmds[0][0] == "ffmpeg" and "loudnorm" in " ".join(cmds[0])
    assert "ass=" in " ".join(cmds[1])
    assert not (ready.root / "output" / "tiktok" / "dry.mp4").exists()
    assert "render:tiktok:dry" not in ready.manifest.steps


def test_combine_two_clips(ready, settings):
    planmod.add_clip(ready, 0.0, 3.0, title="a")
    planmod.add_clip(ready, 4.0, 7.0, title="b")
    clips = [c for c in planmod.load_plan(ready).clips if c.id in ("c2", "c3")]
    rec = render_video(ready, settings, clips, _opts(), name="combo")
    assert abs(rec.duration - 6.0) < 0.3 and all(rec.checks.values())


def test_music_with_ducking_and_sfx_render(ready, settings, tmp_path):
    lib = Library(tmp_path / "lib")
    lib.synth_builtin()
    music = lib.get("pad-calm")
    pop = lib.get("pop")
    opts = _opts(
        music=AudioItem(music.id, lib.path(music), music.sha256, music.license),
        sfx=[(AudioItem(pop.id, lib.path(pop), pop.sha256, pop.license), 1.0, -6.0)],
    )
    clip = planmod.load_plan(ready).get("c1")
    cmds = render_video(ready, settings, [clip], opts, name="mix", dry_run=True)
    joined = " ".join(cmds[1])
    assert "sidechaincompress" in joined and "adelay=1000|1000" in joined
    rec = render_video(ready, settings, [clip], opts, name="mix")
    assert all(rec.checks.values()), rec.checks


def test_too_long_for_preset_is_refused(ready, settings):
    clip = planmod.load_plan(ready).get("c1").model_copy(update={"end": 700.0})
    with pytest.raises(UserInputError, match="excede"):
        render_video(ready, settings, [clip], _opts(), name="long")


def test_silent_source_renders_with_silence(project, tmp_path, provenance, settings, transcript):
    silent = tmp_path / "s.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=640x360:r=30:d=6",
         "-pix_fmt", "yuv420p", str(silent)],
        check=True,
    )  # fmt: skip
    import_video(project, silent, provenance, settings)
    transcribe(project, settings, FakeTranscriber(transcript))
    plan = planmod.generate_plan(project, settings, n_clips=1)
    assert plan.clips == []  # sem fala não há sugestões automáticas
    planmod.add_clip(project, 0.0, 4.0, title="manual")
    planmod.approve_plan(project)
    rec = render_video(
        project, settings, [planmod.load_plan(project).get("c1")], _opts(captions=False), name="silent"
    )
    assert all(rec.checks.values())


def test_transform_source_marks_content_as_synthetic(ready, settings):
    # "Transformação" fabricada só para o teste (recolore o vídeo); registra como se viesse de um provedor.
    tdir = ready.root / "work" / "transforms"
    tdir.mkdir(parents=True)
    out = tdir / "t1.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(ready.path(ready.manifest.source.stored_path)),
         "-vf", "hue=h=90", "-an", str(out)],
        check=True,
    )  # fmt: skip
    ready.manifest.transforms.append(
        TransformRecord(id="t1", clip_id="c1", provider="teste", model="teste", prompt="x",
                        status="done", output=ready.rel(out), sha256="abc")
    )  # fmt: skip
    planmod.edit_clip(ready, "c1", source="transform:t1")
    clip = planmod.load_plan(ready).get("c1")
    rec = render_video(ready, settings, [clip], _opts(), name="synthetic")
    assert rec.synthetic and ready.manifest.synthetic
    disclosure = json.loads(ready.path(rec.path).with_suffix(".disclosure.json").read_text())
    assert disclosure["ai_generated_or_altered"] is True and "TikTok" in disclosure["action"]
    meta = _probe(ready.path(rec.path))["format"]["tags"]
    assert "sim" in meta["comment"]


def test_unfinished_transform_is_refused(ready, settings):
    planmod.edit_clip(ready, "c1", source="transform:nao-existe")
    with pytest.raises(PlanError, match="não está concluída"):
        render_video(ready, settings, [planmod.load_plan(ready).get("c1")], _opts(), name="x")


def test_non_synthetic_render_disclosure_and_publish_sidecars(ready, settings):
    rec = render_video(ready, settings, [planmod.load_plan(ready).get("c1")], _opts(), name="side")
    base = ready.path(rec.path)
    disclosure = json.loads(base.with_suffix(".disclosure.json").read_text())
    publish = json.loads(base.with_suffix(".publish.json").read_text())
    assert disclosure["ai_generated_or_altered"] is False
    assert disclosure["provenance"]["rights"] == "own" and publish["hashtags"]


def test_verify_output_flags_wrong_size(sample_video):
    checks = verify_output(sample_video, load_preset("tiktok"), 8.0)
    assert checks["size"] is False and checks["video_h264"] is True


def test_verification_error_type_is_not_retryable():
    assert RenderVerificationError.retryable is False


def test_pick_mode_rules():
    assert pick_mode("auto", 1920, 1080) == "blurpad"
    assert pick_mode("auto", 1080, 1920) == "crop"
    assert pick_mode("crop", 1920, 1080) == "crop"


def test_parse_loudnorm_json_handles_silence_and_noise():
    ok = 'noise\n{\n "input_i" : "-20.1",\n "input_tp" : "-3.0",\n "input_lra" : "1.0"\n}'
    assert parse_loudnorm_json(ok)["input_i"] == "-20.1"
    silent = '{\n "input_i" : "-inf",\n "input_tp" : "-inf"\n}'
    assert parse_loudnorm_json(silent) is None

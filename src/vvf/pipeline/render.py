"""Etapa C: renderização com FFmpeg (corte, 9:16, legendas, áudio, loudness) + verificação."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from vvf import media
from vvf.captions import CapWord, build_ass, collect_words
from vvf.config import Settings
from vvf.errors import PlanError, ProviderError, RenderVerificationError, UserInputError
from vvf.models.manifest import RenderRecord
from vvf.pipeline.plan import Clip, Plan, load_plan
from vvf.pipeline.runner import StepResult, run_step, upstream
from vvf.pipeline.transcribe import Transcript, load_transcript
from vvf.presets import Preset
from vvf.project import Project

HOOK_SECONDS = 3.0
TARGET_ASPECT = 9 / 16


@dataclass
class AudioItem:
    id: str
    path: Path
    sha256: str
    license: str = ""
    attribution: str = ""


@dataclass
class RenderOptions:
    preset: Preset
    mode: str = "auto"  # auto | blurpad | crop
    focus: float = 0.5  # 0..1: posição horizontal do recorte (modo crop)
    captions: bool = True
    highlight: bool = True
    hook_overlay: bool = False
    music: AudioItem | None = None
    music_db: float = -20.0
    sfx: list[tuple[AudioItem, float, float]] = field(default_factory=list)  # (item, em_s, dB)
    allow_long: bool = False


@dataclass
class ClipSource:
    """De onde vêm vídeo e áudio de um trecho."""

    clip: Clip
    video_path: Path
    video_is_transform: bool
    audio_path: Path | None  # None => sem áudio original (usa silêncio)
    width: int
    height: int
    narration: Path | None = None
    narration_mode: str = "duck"


# ---------- geometria 9:16 ----------


def pick_mode(mode: str, width: int, height: int) -> str:
    if mode in ("blurpad", "crop"):
        return mode
    ratio = width / height
    return "crop" if abs(ratio - TARGET_ASPECT) / TARGET_ASPECT < 0.05 else "blurpad"


def video_chain(i: int, src_label: str, mode: str, w: int, h: int, preset: Preset, focus: float) -> str:
    W, H, fps = preset.width, preset.height, preset.fps
    resolved = pick_mode(mode, w, h)
    if resolved == "crop":
        return (
            f"[{src_label}]setpts=PTS-STARTPTS,scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H}:x='(iw-{W})*{focus}':y='(ih-{H})/2',setsar=1,fps={fps},format={preset.pix_fmt}[v{i}]"
        )
    return (
        f"[{src_label}]setpts=PTS-STARTPTS,split=2[bg{i}][fg{i}];"
        f"[bg{i}]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30:3[bgb{i}];"
        f"[fg{i}]scale={W}:{H}:force_original_aspect_ratio=decrease[fgs{i}];"
        f"[bgb{i}][fgs{i}]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={fps},format={preset.pix_fmt}[v{i}]"
    )


# ---------- grafo ----------

_FMT = "aresample={rate},aformat=sample_fmts=fltp:channel_layouts=stereo"


@dataclass
class Graph:
    inputs: list[str]
    video: list[str]
    audio: list[str]
    total: float
    ass_clips: list[list[CapWord]]
    hooks: list[tuple[float, float, str]]


def build_graph(sources: list[ClipSource], opts: RenderOptions, transcript: Transcript | None) -> Graph:
    preset = opts.preset
    inputs: list[str] = []
    video: list[str] = []
    audio: list[str] = []
    n_in = 0

    def add_input(args: list[str], target: str) -> int:
        nonlocal n_in
        inputs.extend([*args, "-i", target])
        n_in += 1
        return n_in - 1

    offset = 0.0
    ass_clips: list[list[CapWord]] = []
    hooks: list[tuple[float, float, str]] = []
    for i, s in enumerate(sources):
        dur = s.clip.end - s.clip.start
        seek = ["-ss", f"{s.clip.start:.3f}", "-t", f"{dur:.3f}"]
        if s.video_is_transform:
            vi = add_input(["-t", f"{dur:.3f}"], str(s.video_path))
        else:
            vi = add_input(seek, str(s.video_path))
        video.append(video_chain(i, f"{vi}:v", opts.mode, s.width, s.height, preset, opts.focus))

        if s.audio_path is not None:
            ai = vi if not s.video_is_transform else add_input(seek, str(s.audio_path))
            src_a = f"{ai}:a"
        else:
            ai = add_input(["-f", "lavfi", "-t", f"{dur:.3f}"], f"anullsrc=r={preset.audio_rate}:cl=stereo")
            src_a = f"{ai}:a"
        base = (
            f"[{src_a}]{_FMT.format(rate=preset.audio_rate)},apad=whole_dur={dur:.3f},"
            f"atrim=end={dur:.3f},asetpts=PTS-STARTPTS"
        )
        if s.narration is not None:
            ni = add_input([], str(s.narration))
            narr = (
                f"[{ni}:a]{_FMT.format(rate=preset.audio_rate)},apad=whole_dur={dur:.3f},"
                f"atrim=end={dur:.3f},asetpts=PTS-STARTPTS"
            )
            if s.narration_mode == "replace":
                audio.append(f"{narr}[ac{i}]")
            elif s.narration_mode == "mix":
                audio.append(f"{base}[o{i}]")
                audio.append(f"{narr}[n{i}]")
                audio.append(f"[o{i}][n{i}]amix=inputs=2:duration=first:normalize=0[ac{i}]")
            else:  # duck: a fala original abaixa enquanto a narração toca
                audio.append(f"{base}[o{i}]")
                audio.append(f"{narr},asplit=2[n{i}][nk{i}]")
                audio.append(
                    f"[o{i}][nk{i}]sidechaincompress=threshold=0.02:ratio=10:attack=15:release=250[d{i}]"
                )
                audio.append(f"[d{i}][n{i}]amix=inputs=2:duration=first:normalize=0[ac{i}]")
        else:
            audio.append(f"{base}[ac{i}]")

        if opts.captions and transcript is not None:
            ass_clips.append(collect_words(transcript, s.clip, offset))
        if opts.hook_overlay and s.clip.hook:
            hooks.append((offset, offset + min(HOOK_SECONDS, dur), s.clip.hook))
        offset += dur

    n = len(sources)
    video.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vcat]")
    audio.append("".join(f"[ac{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[acat]")

    last = "acat"
    if opts.music is not None:
        mi = add_input(["-stream_loop", "-1", "-t", f"{offset:.3f}"], str(opts.music.path))
        fade_out = max(0.0, offset - 1.5)
        audio.append(
            f"[{mi}:a]{_FMT.format(rate=preset.audio_rate)},volume={opts.music_db}dB,"
            f"afade=t=in:d=1,afade=t=out:st={fade_out:.3f}:d=1.5[mus]"
        )
        audio.append(f"[{last}]asplit=2[vk][vm]")
        audio.append("[mus][vk]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=300[musd]")
        audio.append("[vm][musd]amix=inputs=2:duration=first:normalize=0[amus]")
        last = "amus"
    if opts.sfx:
        labels = [f"[{last}]"]
        for k, (item, at_s, db) in enumerate(opts.sfx):
            si = add_input([], str(item.path))
            ms = int(at_s * 1000)
            audio.append(
                f"[{si}:a]{_FMT.format(rate=preset.audio_rate)},volume={db}dB,adelay={ms}|{ms}[sfx{k}]"
            )
            labels.append(f"[sfx{k}]")
        audio.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0[asfx]")
        last = "asfx"
    audio.append(f"[{last}]anull[apre]")
    return Graph(inputs, video, audio, offset, ass_clips, hooks)


def loudnorm_filter(preset: Preset, measured: dict | None) -> str:
    base = f"loudnorm=I={preset.loudness_lufs}:TP={preset.true_peak_db}:LRA=11"
    if not measured:
        return base + ":print_format=json"
    return (
        f"{base}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )


def parse_loudnorm_json(stderr: str) -> dict | None:
    """Extrai o bloco JSON do loudnorm; None se o áudio é silêncio digital (-inf)."""
    blocks = list(re.finditer(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, re.S))
    if not blocks:
        raise ProviderError("não foi possível medir o loudness (saída do loudnorm ausente)")
    data = json.loads(blocks[-1].group(0))
    if any(str(data.get(k, "")).strip() in ("-inf", "inf", "nan") for k in ("input_i", "input_tp")):
        return None
    return data


def encode_args(preset: Preset, total: float, comment: str, title: str) -> list[str]:
    return [
        "-c:v", preset.video_codec, "-profile:v", preset.profile, "-preset", preset.x264_preset,
        "-crf", str(preset.crf), "-maxrate", preset.maxrate, "-bufsize", preset.bufsize,
        "-pix_fmt", preset.pix_fmt, "-color_primaries", "bt709", "-color_trc", "bt709",
        "-colorspace", "bt709", "-r", str(preset.fps),
        "-c:a", preset.audio_codec, "-b:a", preset.audio_bitrate, "-ar", str(preset.audio_rate), "-ac", "2",
        "-movflags", "+faststart", "-t", f"{total:.3f}",
        "-metadata", f"title={title}", "-metadata", f"comment={comment}",
    ]  # fmt: skip


def build_commands(
    graph: Graph, opts: RenderOptions, ass_rel: str | None, out_rel: str, measured: dict | None,
    comment: str, title: str,
) -> tuple[list[str], list[str]]:  # fmt: skip
    """Devolve (comando de medição de loudness, comando final de codificação)."""
    head = ["ffmpeg", "-y", "-hide_banner", "-nostdin"]
    audio_graph = ";".join(graph.audio)
    measure = [
        *head, "-loglevel", "info", *graph.inputs,
        "-filter_complex", f"{audio_graph};[apre]{loudnorm_filter(opts.preset, None)}[aout]",
        "-map", "[aout]", "-f", "null", "-",
    ]  # fmt: skip
    video = list(graph.video)
    video.append(f"[vcat]ass={ass_rel}[vout]" if ass_rel else "[vcat]null[vout]")
    final_audio = f"[apre]{loudnorm_filter(opts.preset, measured)}[aout]" if measured else "[apre]anull[aout]"
    final = [
        *head, "-loglevel", "error", *graph.inputs,
        "-filter_complex", ";".join([*video, *graph.audio, final_audio]),
        "-map", "[vout]", "-map", "[aout]", *encode_args(opts.preset, graph.total, comment, title),
        out_rel,
    ]  # fmt: skip
    return measure, final


# ---------- verificação ----------


def measure_output_loudness(path: Path) -> float | None:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info", "-i", str(path), "-vn",
         "-af", "loudnorm=I=-14:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    )  # fmt: skip
    data = parse_loudnorm_json(proc.stderr)
    return None if data is None else float(data["input_i"])


def verify_output(path: Path, preset: Preset, expected_duration: float) -> dict[str, bool]:
    checks = {"exists": path.exists() and path.stat().st_size > 0}
    if not checks["exists"]:
        return checks
    raw = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
    )  # fmt: skip
    video = next((s for s in raw["streams"] if s["codec_type"] == "video"), {})
    audio = next((s for s in raw["streams"] if s["codec_type"] == "audio"), {})
    num, den = (video.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    duration = float(raw["format"].get("duration", 0))
    checks.update(
        video_h264=video.get("codec_name") == "h264",
        pix_fmt=video.get("pix_fmt") == preset.pix_fmt,
        size=(video.get("width"), video.get("height")) == (preset.width, preset.height),
        fps=abs(fps - preset.fps) < 0.5,
        audio_aac=audio.get("codec_name") == "aac",
        duration=abs(duration - expected_duration) <= 0.5,
    )  # fmt: skip
    loud = measure_output_loudness(path) if checks["audio_aac"] else None
    checks["loudness"] = True if loud is None else abs(loud - preset.loudness_lufs) <= 1.5
    return checks


# ---------- orquestração ----------


def _audio_sha(item: AudioItem | None) -> str | None:
    return item.sha256 if item else None


def resolve_sources(project: Project, plan: Plan, clips: list[Clip]) -> list[ClipSource]:
    src = project.manifest.source
    if src is None:
        raise UserInputError("projeto sem vídeo importado")
    original = project.path(src.stored_path)
    out = []
    for clip in clips:
        video_path, is_transform, w, h = original, False, src.probe.width, src.probe.height
        if clip.source.startswith("transform:"):
            tid = clip.source.split(":", 1)[1]
            rec = next((t for t in project.manifest.transforms if t.id == tid), None)
            if rec is None or rec.status != "done" or not rec.output:
                raise PlanError(f"transformação '{tid}' não está concluída")
            video_path = project.path(rec.output)
            tp = media.probe(video_path)
            w, h, is_transform = tp.width, tp.height, True
        narration = None
        if clip.narration:
            narration = project.path(clip.narration)
        out.append(
            ClipSource(
                clip=clip, video_path=video_path, video_is_transform=is_transform,
                audio_path=original if src.probe.has_audio else None, width=w, height=h,
                narration=narration, narration_mode=clip.narration_mode,
            )
        )  # fmt: skip
    return out


def render_video(
    project: Project,
    settings: Settings,
    clips: list[Clip],
    opts: RenderOptions,
    *,
    name: str,
    force: bool = False,
    dry_run: bool = False,
) -> list[list[str]] | RenderRecord:
    """Renderiza um vídeo a partir de `clips` (concatenados). Em dry_run só devolve os comandos."""
    plan = load_plan(project)
    total = sum(c.end - c.start for c in clips)
    if total > opts.preset.recommended_max_seconds and not opts.allow_long:
        raise UserInputError(
            f"duração {total:.0f}s excede a meta do preset {opts.preset.name} "
            f"({opts.preset.recommended_max_seconds:.0f}s); use --allow-long para ignorar"
        )
    transcript = load_transcript(project) if opts.captions or opts.hook_overlay else None
    sources = resolve_sources(project, plan, clips)
    graph = build_graph(sources, opts, transcript)
    manifest = project.manifest
    synthetic = manifest.synthetic or any(s.video_is_transform or s.narration for s in sources)
    comment = (
        "Conteúdo gerado ou alterado por IA: sim" if synthetic else "Conteúdo gerado ou alterado por IA: não"
    )
    title = clips[0].title or name
    out_dir = project.root / "output" / opts.preset.name
    out_path = out_dir / f"{name}.mp4"
    ass_path = project.root / "work" / "render" / f"{opts.preset.name}_{name}.ass"
    ass_rel = project.rel(ass_path) if (opts.captions or graph.hooks) else None

    if dry_run:
        measure, final = build_commands(graph, opts, ass_rel, project.rel(out_path), None, comment, title)
        return [measure, final]

    step = f"render:{opts.preset.name}:{name}"
    inputs = {
        "preset": opts.preset.model_dump(), "mode": opts.mode, "focus": opts.focus,
        "captions": opts.captions, "highlight": opts.highlight, "hook": opts.hook_overlay,
        "clips": [c.model_dump() for c in clips],
        "src": [(s.video_path.name, s.narration and str(s.narration)) for s in sources],
        "src_sha": manifest.source.sha256 if manifest.source else None,
        "tsha": [t.sha256 for t in manifest.transforms if t.sha256],
        "music": [_audio_sha(opts.music), opts.music_db],
        "sfx": [(_audio_sha(i), a, d) for i, a, d in opts.sfx],
        "synthetic": synthetic,
        **upstream(project, "transcribe"),
    }  # fmt: skip

    def work() -> StepResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        ass_path.parent.mkdir(parents=True, exist_ok=True)
        if ass_rel:
            subs = build_ass(graph.ass_clips if opts.captions else [], opts.preset,
                             highlight=opts.highlight, hooks=graph.hooks)  # fmt: skip
            subs.save(str(ass_path))
        measure_cmd, _ = build_commands(graph, opts, ass_rel, project.rel(out_path), None, comment, title)
        proc = subprocess.run(measure_cmd, capture_output=True, text=True, cwd=project.root)
        if proc.returncode != 0:
            raise ProviderError(
                "ffmpeg falhou (medição de loudness): " + "\n".join(proc.stderr.splitlines()[-6:])
            )
        measured = parse_loudnorm_json(proc.stderr)
        _, final_cmd = build_commands(graph, opts, ass_rel, project.rel(out_path), measured, comment, title)
        proc = subprocess.run(final_cmd, capture_output=True, text=True, cwd=project.root)
        if proc.returncode != 0:
            raise ProviderError("ffmpeg falhou (codificação): " + "\n".join(proc.stderr.splitlines()[-8:]))
        checks = verify_output(out_path, opts.preset, graph.total)
        failed = [k for k, v in checks.items() if not v]
        if failed:
            raise RenderVerificationError(f"verificação do arquivo falhou: {failed}")
        return StepResult({"video": project.rel(out_path)}, {"checks": checks, "duration": graph.total})

    outcome = run_step(project, step, inputs, work, force=force, retries=settings.vvf_step_retries)
    probe = media.probe(out_path)
    record = RenderRecord(
        preset=opts.preset.name, path=project.rel(out_path), sha256=media.sha256_file(out_path),
        duration=probe.duration, width=probe.width, height=probe.height,
        checks=outcome.record.meta.get("checks", {}), synthetic=synthetic, clips=[c.id for c in clips],
    )  # fmt: skip
    manifest.renders = [r for r in manifest.renders if r.path != record.path] + [record]
    if synthetic:
        for reason in _synthetic_reasons(sources):
            manifest.mark_synthetic(reason)
    project.save()
    write_sidecars(project, out_path, clips, opts, synthetic)
    return record


def _synthetic_reasons(sources: list[ClipSource]) -> list[str]:
    reasons = []
    if any(s.video_is_transform for s in sources):
        reasons.append("vídeo transformado por IA")
    if any(s.narration for s in sources):
        reasons.append("narração com voz sintética")
    return reasons


PLATFORM_LABELS = {
    "tiktok": "Ative o rótulo de conteúdo gerado por IA do TikTok ao publicar.",
    "reels": "Use o rótulo de conteúdo gerado/alterado por IA do Instagram ao publicar.",
    "shorts": "Marque 'conteúdo alterado ou sintético' no YouTube Studio ao publicar.",
}


def write_sidecars(project: Project, video: Path, clips: list[Clip], opts: RenderOptions, synthetic: bool):
    tags: list[str] = []
    for c in clips:
        tags += [t for t in c.hashtags if t not in tags]
    attributions = [
        {"id": a.id, "license": a.license, "attribution": a.attribution}
        for a in ([opts.music] if opts.music else []) + [i for i, _, _ in opts.sfx]
        if a.license
    ]  # fmt: skip
    publish = {
        "title": clips[0].title, "description": clips[0].description, "hashtags": tags,
        "clips": [{"id": c.id, "title": c.title, "hook": c.hook} for c in clips],
        "audio_credits": attributions,
    }  # fmt: skip
    prov = project.manifest.provenance
    action = (
        PLATFORM_LABELS.get(opts.preset.name, "")
        if synthetic
        else "Nenhum rótulo de IA exigido pelo registro."
    )
    disclosure = {
        "ai_generated_or_altered": synthetic,
        "reasons": project.manifest.synthetic_reasons if synthetic else [],
        "action": action,
        "rule": "Os nomes exatos dos rótulos variam: confira no app de cada plataforma antes de publicar.",
        "provenance": prov.model_dump() if prov else None,
        "source_sha256": project.manifest.source.sha256 if project.manifest.source else None,
    }  # fmt: skip
    (video.with_suffix(".publish.json")).write_text(json.dumps(publish, indent=2, ensure_ascii=False))
    (video.with_suffix(".disclosure.json")).write_text(json.dumps(disclosure, indent=2, ensure_ascii=False))

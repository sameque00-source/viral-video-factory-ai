"""Implementação das tarefas da fila (analyze, render, export). Usadas pela CLI e por `queue run`."""

from __future__ import annotations

from pathlib import Path

from vvf.audio.library import Library
from vvf.config import Settings
from vvf.errors import PlanError, UserInputError
from vvf.jobs import handler
from vvf.pipeline.plan import load_plan
from vvf.pipeline.render import AudioItem, RenderOptions, render_video
from vvf.pipeline.scenes import compute_audio_energy, detect_scenes
from vvf.pipeline.transcribe import FasterWhisperTranscriber, transcribe
from vvf.presets import available, load_preset
from vvf.project import Project


@handler("analyze")
def analyze(settings: Settings, project: Project, args: dict) -> None:
    force = bool(args.get("force"))
    detect_scenes(project, settings, force=force)
    compute_audio_energy(project, settings, force=force)
    model = args.get("whisper_model") or settings.vvf_whisper_model
    backend = FasterWhisperTranscriber(model, settings.vvf_whisper_device, settings.vvf_whisper_compute_type)
    transcribe(project, settings, backend, force=force, discard_edits=bool(args.get("discard_edits")))


def _audio_item(lib: Library, item_id: str) -> AudioItem:
    item = lib.get(item_id)
    return AudioItem(item.id, lib.path(item), item.sha256, item.license, item.attribution)


def build_options(settings: Settings, preset_name: str, args: dict) -> RenderOptions:
    lib = Library(Path(settings.vvf_library_dir))
    music = _audio_item(lib, args["music"]) if args.get("music") else None
    sfx = [(_audio_item(lib, s["id"]), float(s["at"]), float(s.get("db", -8.0))) for s in args.get("sfx", [])]
    return RenderOptions(
        preset=load_preset(preset_name),
        mode=args.get("mode", "auto"),
        focus=float(args.get("focus", 0.5)),
        captions=bool(args.get("captions", True)),
        highlight=bool(args.get("highlight", True)),
        hook_overlay=bool(args.get("hook_overlay", False)),
        music=music,
        music_db=float(args.get("music_db", -20.0)),
        sfx=sfx,
        allow_long=bool(args.get("allow_long", False)),
    )


def render_preset(
    settings: Settings, project: Project, preset_name: str, args: dict, *, dry_run: bool = False
):
    plan = load_plan(project)
    if plan.status != "approved":
        raise PlanError("plano não aprovado: revise com 'vvf plan show' e rode 'vvf plan approve'")
    wanted = args.get("clips") or [c.id for c in plan.enabled_clips()]
    clips = [plan.get(cid) for cid in wanted]
    if not clips:
        raise UserInputError("nenhum trecho ativo para renderizar")
    opts = build_options(settings, preset_name, args)
    force = bool(args.get("force"))
    groups = [clips] if args.get("combine") else [[c] for c in clips]
    records = []
    for group in groups:
        name = f"{project.slug}_{'-'.join(c.id for c in group)}"
        records.append(render_video(project, settings, group, opts, name=name, force=force, dry_run=dry_run))
    return records


@handler("render")
def render(settings: Settings, project: Project, args: dict) -> None:
    render_preset(settings, project, args["preset"], args)


@handler("export")
def export(settings: Settings, project: Project, args: dict) -> None:
    presets = args.get("presets") or available()
    for name in presets:
        render_preset(settings, project, name, args)

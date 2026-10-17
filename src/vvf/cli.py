"""CLI `vvf`: importar, analisar, planejar, renderizar, exportar, transformar, narrar, retomar."""

from __future__ import annotations

import functools
import json
import shutil
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from vvf import __version__, handlers  # noqa: F401  (handlers registra as tarefas da fila)
from vvf.audio.library import ALLOWED_LICENSES, Library
from vvf.audio.tts import make_tts, narrate
from vvf.config import Settings, get_settings
from vvf.costs import CostGuard
from vvf.errors import UserInputError, VVFError
from vvf.jobs import JobQueue, run_queue, run_recorded
from vvf.llm import AnthropicLLM
from vvf.models.manifest import Rights
from vvf.pipeline import plan as planmod
from vvf.pipeline.ingest import import_video, make_provenance
from vvf.pipeline.runner import is_interrupted
from vvf.pipeline.transform import resume_transform, run_transform, use_transform
from vvf.presets import available as available_presets
from vvf.project import Project, project_lock, slugify
from vvf.providers import PROVIDER_NAMES, make_provider
from vvf.providers.catalog import CATALOG

console = Console()
err = Console(stderr=True)
app = typer.Typer(help="Fábrica de vídeos virais com IA.", no_args_is_help=True, add_completion=False)
plan_app = typer.Typer(help="Plano de cortes: gerar, revisar, editar, aprovar.", no_args_is_help=True)
transform_app = typer.Typer(help="Transformações de vídeo por IA (pagas).", no_args_is_help=True)
queue_app = typer.Typer(help="Fila de tarefas.", no_args_is_help=True)
library_app = typer.Typer(help="Biblioteca de músicas e efeitos.", no_args_is_help=True)
app.add_typer(plan_app, name="plan")
app.add_typer(transform_app, name="transform")
app.add_typer(queue_app, name="queue")
app.add_typer(library_app, name="library")


def _settings() -> Settings:
    return get_settings()


def _projects_dir(s: Settings) -> Path:
    return Path(s.vvf_projects_dir)


def _open(s: Settings, slug: str) -> Project:
    return Project.open(_projects_dir(s), slug)


def _mutating(fn):
    """Trava o projeto durante comandos que alteram o manifesto (evita dois processos competindo)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        s = _settings()
        slug = kwargs["slug"]
        if not (_projects_dir(s) / slug / "manifest.json").exists():
            raise UserInputError(f"projeto não encontrado: {slug}")
        with project_lock(_projects_dir(s) / slug, slug):
            return fn(*args, **kwargs)

    return wrapper


def _queue(s: Settings) -> JobQueue:
    return JobQueue(_projects_dir(s) / ".queue.sqlite")


def _money(v: float | None) -> str:
    return "desconhecido" if v is None else f"US$ {v:.4f}"


def _parse_params(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise UserInputError(f"--param deve ser chave=valor (recebido: {item!r})")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"vvf {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool, typer.Option("--version", help="Mostra a versão.", callback=_version_callback, is_eager=True)
    ] = False,
):
    """Fábrica de vídeos virais com IA."""


# ---------- importar / analisar ----------


@app.command("import")
def import_cmd(
    video: Annotated[Path, typer.Argument(help="Vídeo original (mp4, mov, mkv, webm, avi, m4v).")],
    rights: Annotated[
        Rights, typer.Option(help="Direitos de uso: own | licensed | third_party_with_consent.")
    ],
    slug: Annotated[str | None, typer.Option(help="Nome do projeto (padrão: nome do arquivo).")] = None,
    description: Annotated[str, typer.Option(help="De onde veio o vídeo / quem filmou.")] = "",
    evidence: Annotated[
        str | None, typer.Option(help="Prova de licença/autorização (link, arquivo).")
    ] = None,
    people: Annotated[bool, typer.Option("--people/--no-people", help="Aparecem pessoas?")] = False,
    consent: Annotated[
        bool, typer.Option("--consent/--no-consent", help="Consentimento das pessoas confirmado.")
    ] = False,
):
    """Importa um vídeo, valida com ffprobe e registra origem, direitos e consentimento."""
    s = _settings()
    provenance = make_provenance(
        description=description, rights=rights, rights_evidence=evidence,
        people_present=people, consent_confirmed=consent,
    )  # fmt: skip
    project = Project.create(_projects_dir(s), slug or slugify(video.stem))
    try:
        warnings = import_video(project, video, provenance, s)
    except Exception:
        shutil.rmtree(project.root, ignore_errors=True)  # não deixa projeto pela metade
        raise
    p = project.manifest.source.probe  # type: ignore[union-attr]
    console.print(f"[green]✔[/] projeto [bold]{project.slug}[/] criado")
    console.print(
        f"  {p.width}x{p.height} · {p.fps} fps · {p.duration:.1f}s · áudio: {p.audio_codec or 'não'}"
    )
    for w in warnings:
        console.print(f"  [yellow]aviso:[/] {w}")
    if people and not consent:
        console.print(
            "  [yellow]aviso:[/] pessoas sem consentimento confirmado: transformações por IA serão bloqueadas"
        )


@app.command()
def analyze(
    slug: str,
    whisper_model: Annotated[
        str | None, typer.Option(help="tiny|base|small|medium (padrão: VVF_WHISPER_MODEL).")
    ] = None,
    force: Annotated[bool, typer.Option(help="Refaz mesmo que já esteja em cache.")] = False,
    discard_edits: Annotated[
        bool, typer.Option(help="Com --force, descarta correções manuais da transcrição.")
    ] = False,
):
    """Detecta cenas, mede energia do áudio e transcreve (local, sem custo)."""
    s = _settings()
    _open(s, slug)
    run_recorded(
        s,
        _queue(s),
        slug,
        "analyze",
        {"whisper_model": whisper_model, "force": force, "discard_edits": discard_edits},
    )
    project = _open(s, slug)
    scenes = project.manifest.steps["scenes"].meta.get("count")
    segs = project.manifest.steps["transcribe"].meta.get("segments")
    console.print(f"[green]✔[/] {scenes} cena(s) · {segs} segmento(s) de fala")


# ---------- transcrição ----------

transcript_app = typer.Typer(help="Revisar e corrigir a transcrição.", no_args_is_help=True)
app.add_typer(transcript_app, name="transcript")


@transcript_app.command("show")
def transcript_show(slug: str):
    from vvf.pipeline.transcribe import load_transcript

    t = load_transcript(_open(_settings(), slug))
    console.print(
        f"{t.model} · {t.language} · {t.duration:.1f}s" + (f" · correções: {t.edits}" if t.edits else "")
    )
    for seg in t.segments:
        console.print(f"[{seg.id}] {seg.start:6.1f}–{seg.end:6.1f}  {seg.text}", markup=False)


@transcript_app.command("replace")
@_mutating
def transcript_replace(slug: str, old: str, new: str):
    """Corrige uma palavra mal transcrita (ex.: vvf transcript replace polvo povo polvo)."""
    from vvf.pipeline.transcribe import replace_word

    n = replace_word(_open(_settings(), slug), old, new)
    console.print(
        f"[green]✔[/] {n} ocorrência(s) corrigida(s) (transcrição, legendas, plano). Aprove o plano de novo."
    )


# ---------- plano ----------


@plan_app.command("generate")
@_mutating
def plan_generate(
    slug: str,
    engine: Annotated[
        str, typer.Option(help="heuristic (grátis, offline) | llm (Claude, pago).")
    ] = "heuristic",
    clips: Annotated[int, typer.Option(min=1, max=10, help="Quantidade de trechos.")] = 3,
    min_len: Annotated[float, typer.Option(help="Duração mínima (s).")] = 15.0,
    max_len: Annotated[float, typer.Option(help="Duração máxima (s).")] = 45.0,
    execute: Annotated[
        bool, typer.Option("--execute", help="Necessário para a engine llm (chamada paga).")
    ] = False,
    price_usd: Annotated[
        float | None, typer.Option(help="Teto de preço aceito se o modelo não tiver preço conhecido.")
    ] = None,
    force: Annotated[bool, typer.Option(help="Sobrescreve plano editado/aprovado.")] = False,
):
    s = _settings()
    project = _open(s, slug)
    llm = None
    if engine == "llm":
        key = s.anthropic_api_key.get_secret_value() if s.anthropic_api_key else None
        llm = AnthropicLLM(key, s.vvf_llm_model) if (execute or key) else None
        probe_llm = llm or AnthropicLLM(None, s.vvf_llm_model, client=object())
        est = planmod.llm_plan_estimate(project, probe_llm, clips, min_len, max_len)
        if price_usd is not None:
            from vvf.costs import Estimate

            est = Estimate(
                est.provider, est.model, est.kind, price_usd, "informado pelo usuário (--price-usd)"
            )
        CostGuard(s, project.manifest).authorize(est, dry_run=not execute, params={"engine": "llm"})
        project.save()
        console.print(f"estimativa (limite superior): {_money(est.usd)} · {est.basis}")
        if not execute:
            console.print(
                "[yellow]simulação:[/] nada foi enviado. Use --execute (e VVF_ALLOW_PAID_CALLS=true)."
            )
            return
    elif engine != "heuristic":
        raise UserInputError("engine deve ser 'heuristic' ou 'llm'")
    plan = planmod.generate_plan(
        project,
        s,
        engine=engine,
        llm=llm,
        n_clips=clips,
        min_len=min_len,
        max_len=max_len,
        force=force,  # type: ignore[arg-type]
    )
    _print_plan(plan)


def _print_plan(plan: planmod.Plan) -> None:
    table = Table(title=f"Plano ({plan.engine}) · {plan.status}{' · editado' if plan.edited else ''}")
    for col in ("id", "ativo", "início–fim", "dur", "nota", "gancho", "título"):
        table.add_column(col)
    for c in plan.clips:
        span = f"{c.start:.1f}–{c.end:.1f}s"
        table.add_row(c.id, "sim" if c.enabled else "não", span, f"{c.duration:.1f}s",
                      f"{c.score:.2f}", c.hook[:40], c.title[:40])  # fmt: skip
    console.print(table)
    for idea in plan.ideas:
        console.print(f"  ideia: {idea}")
    for w in plan.warnings:
        console.print(f"  [yellow]aviso:[/] {w}")


@plan_app.command("show")
def plan_show(
    slug: str, details: Annotated[bool, typer.Option("--details", help="Mostra justificativas.")] = False
):
    s = _settings()
    plan = planmod.load_plan(_open(s, slug))
    _print_plan(plan)
    if details:
        for c in plan.clips:
            tags = " ".join(c.hashtags)
            console.print(f"[bold]{c.id}[/] {c.reason}\n  descrição: {c.description}\n  hashtags: {tags}")
            console.print(f"  citações: {c.quotes}\n  notas: {c.script_notes}")


@plan_app.command("edit")
@_mutating
def plan_edit(
    slug: str,
    clip: str,
    start: Annotated[float | None, typer.Option()] = None,
    end: Annotated[float | None, typer.Option()] = None,
    title: Annotated[str | None, typer.Option()] = None,
    description: Annotated[str | None, typer.Option()] = None,
    hook: Annotated[str | None, typer.Option()] = None,
    enabled: Annotated[bool | None, typer.Option("--enable/--disable")] = None,
):
    """Ajusta um trecho. Editar desfaz a aprovação (aprove novamente antes de renderizar)."""
    s = _settings()
    project = _open(s, slug)
    changes = {
        "start": start,
        "end": end,
        "title": title,
        "description": description,
        "hook": hook,
        "enabled": enabled,
    }
    if hook is not None:
        changes["hook_source"] = "user"
    _print_plan(planmod.edit_clip(project, clip, **changes))


@plan_app.command("add")
@_mutating
def plan_add(slug: str, start: float, end: float, title: str = ""):
    """Adiciona um trecho manual."""
    s = _settings()
    _print_plan(planmod.add_clip(_open(s, slug), start, end, title))


@plan_app.command("approve")
@_mutating
def plan_approve(slug: str):
    s = _settings()
    plan = planmod.approve_plan(_open(s, slug))
    console.print(f"[green]✔[/] plano aprovado com {len(plan.enabled_clips())} trecho(s)")


# ---------- render / export ----------


def _sfx_args(items: list[str]) -> list[dict]:
    out = []
    for item in items:  # formato: id@segundos[:dB]
        try:
            ident, rest = item.split("@", 1)
            at, _, db = rest.partition(":")
            out.append({"id": ident, "at": float(at), "db": float(db) if db else -8.0})
        except ValueError as exc:
            raise UserInputError(f"--sfx deve ser id@segundos[:dB] (recebido: {item!r})") from exc
    return out


def _render_args(**kw: Any) -> dict:
    kw["sfx"] = _sfx_args(kw.pop("sfx") or [])
    kw["clips"] = kw.pop("clip") or []
    return kw


@app.command()
def render(
    slug: str,
    preset: Annotated[str, typer.Option(help=f"Preset ({', '.join(available_presets())}).")] = "tiktok",
    clip: Annotated[
        list[str] | None, typer.Option(help="Renderiza só estes trechos (repita a opção).")
    ] = None,
    combine: Annotated[bool, typer.Option(help="Junta os trechos em um único vídeo.")] = False,
    mode: Annotated[str, typer.Option(help="auto | blurpad | crop.")] = "auto",
    focus: Annotated[
        float, typer.Option(min=0, max=1, help="Posição horizontal do recorte (modo crop).")
    ] = 0.5,
    captions: Annotated[bool, typer.Option("--captions/--no-captions")] = True,
    highlight: Annotated[
        bool, typer.Option("--highlight/--no-highlight", help="Destaca a palavra falada.")
    ] = True,
    hook_overlay: Annotated[bool, typer.Option(help="Mostra o gancho nos 3 primeiros segundos.")] = False,
    music: Annotated[str | None, typer.Option(help="ID da música na biblioteca.")] = None,
    music_db: Annotated[float, typer.Option(help="Volume da música (dB).")] = -20.0,
    sfx: Annotated[list[str] | None, typer.Option(help="Efeito: id@segundos[:dB] (repita).")] = None,
    allow_long: Annotated[bool, typer.Option(help="Ignora a meta de duração do preset.")] = False,
    dry_run: Annotated[bool, typer.Option(help="Mostra os comandos FFmpeg sem renderizar.")] = False,
    force: Annotated[bool, typer.Option(help="Refaz mesmo que já esteja pronto.")] = False,
):
    """Corta, converte para 9:16, legenda, mixa áudio e exporta com verificação."""
    s = _settings()
    project = _open(s, slug)
    args = _render_args(
        preset=preset, clip=clip, combine=combine, mode=mode, focus=focus, captions=captions,
        highlight=highlight, hook_overlay=hook_overlay, music=music, music_db=music_db, sfx=sfx,
        allow_long=allow_long, force=force,
    )  # fmt: skip
    if dry_run:
        for cmds in handlers.render_preset(s, project, preset, args, dry_run=True):
            for c in cmds:
                console.print(" ".join(c), soft_wrap=True, markup=False)
        return
    run_recorded(s, _queue(s), slug, "render", args)
    _print_renders(_open(s, slug), preset)


def _print_renders(project: Project, preset: str | None = None) -> None:
    for r in project.manifest.renders:
        if preset and r.preset != preset:
            continue
        ok = all(r.checks.values())
        mark = "[green]✔[/]" if ok else "[red]✘[/]"
        flag = " · [magenta]IA[/]" if r.synthetic else ""
        console.print(f"{mark} {project.root / r.path}  ({r.width}x{r.height}, {r.duration:.1f}s){flag}")


@app.command()
def export(
    slug: str,
    preset: Annotated[list[str] | None, typer.Option(help="Presets (padrão: todos).")] = None,
    combine: Annotated[bool, typer.Option()] = False,
    music: Annotated[str | None, typer.Option()] = None,
    force: Annotated[bool, typer.Option()] = False,
):
    """Exporta para TikTok, Reels e Shorts de uma vez (cada preset em output/<preset>/)."""
    s = _settings()
    _open(s, slug)
    args = _render_args(presets=preset, clip=None, combine=combine, music=music, force=force, sfx=None)
    run_recorded(s, _queue(s), slug, "export", args)
    _print_renders(_open(s, slug))


# ---------- transformações ----------


@transform_app.command("models")
def transform_models():
    """Modelos sugeridos (conferidos em fontes oficiais) e o que ainda não foi verificado."""
    table = Table(title="Modelos de vídeo-para-vídeo")
    for col in ("provedor", "modelo", "descrição", "verificação"):
        table.add_column(col)
    for m in CATALOG:
        note = m.verified + (f" · [red]descontinuado em {m.deprecated_on}[/]" if m.deprecated_on else "")
        table.add_row(m.provider, m.model, m.summary, note)
    console.print(table)


def _transform_report(rec, show_cost: bool = True) -> None:
    console.print(
        f"  {rec.id}: {rec.provider}/{rec.model} → [bold]{rec.status}[/]" + (" (cache)" if rec.cached else "")
    )
    if show_cost:
        console.print(f"  custo estimado: {_money(rec.cost_usd)}")
    if rec.error:
        console.print(f"  [yellow]{rec.error}[/]")


@transform_app.command("run")
@_mutating
def transform_run(
    slug: str,
    clip: Annotated[str, typer.Option(help="Trecho do plano (ex.: c1).")],
    provider: Annotated[str, typer.Option(help=f"{' | '.join(PROVIDER_NAMES)}")],
    model: Annotated[str, typer.Option(help="ID do modelo (veja 'vvf transform models').")],
    prompt: Annotated[str, typer.Option(help="Instrução de transformação.")] = "",
    param: Annotated[
        list[str] | None, typer.Option(help="Parâmetro do modelo: chave=valor (repita).")
    ] = None,
    execute: Annotated[
        bool, typer.Option("--execute", help="Executa DE VERDADE (pago). Sem isso, só simula.")
    ] = False,
    price_usd: Annotated[
        float | None, typer.Option(help="Teto de preço aceito quando o preço é desconhecido.")
    ] = None,
    timeout: Annotated[float, typer.Option(help="Espera máxima pelo provedor (s).")] = 1800.0,
    poll_interval: Annotated[float, typer.Option()] = 5.0,
):
    """Transforma um trecho com IA. Padrão = simulação (estimativa + validação, sem custo)."""
    s = _settings()
    project = _open(s, slug)
    rec = run_transform(
        project, s, make_provider(provider, s), clip_id=clip, model=model, prompt=prompt,
        params=_parse_params(param or []), execute=execute, price_usd=price_usd,
        poll_interval=poll_interval, timeout_s=timeout,
    )  # fmt: skip
    console.print("[green]✔[/] simulação" if not execute else "[green]✔[/] transformação")
    _transform_report(rec)
    if not execute:
        console.print(
            "  nada foi enviado. Para executar: --execute (com VVF_ALLOW_PAID_CALLS=true e teto definido)."
        )
    elif rec.status == "done":
        console.print(f"  use no render: vvf transform use {slug} {clip} {rec.id}")


@transform_app.command("compare")
@_mutating
def transform_compare(
    slug: str,
    clip: Annotated[str, typer.Option()],
    prompt: Annotated[str, typer.Option()] = "",
    target: Annotated[
        list[str] | None, typer.Option(help="provedor:modelo (repita). Padrão: catálogo.")
    ] = None,
    param: Annotated[list[str] | None, typer.Option()] = None,
    execute: Annotated[bool, typer.Option("--execute", help="Roda todos (pago, sob o mesmo teto).")] = False,
):
    """Compara provedores/modelos lado a lado (estimativa e validação; --execute roda de verdade)."""
    s = _settings()
    project = _open(s, slug)
    targets = [t.split(":", 1) for t in target] if target else [[m.provider, m.model] for m in CATALOG]
    guard = CostGuard(s, project.manifest)
    table = Table(title=f"Comparação para {slug}/{clip}")
    for col in ("provedor", "modelo", "resultado", "custo estimado", "observação"):
        table.add_column(col)
    for prov_name, model in targets:
        try:
            rec = run_transform(
                project, s, make_provider(prov_name, s), clip_id=clip, model=model, prompt=prompt,
                params=_parse_params(param or []), execute=execute, guard=guard,
            )  # fmt: skip
            table.add_row(prov_name, model, rec.status, _money(rec.cost_usd), rec.error or "")
        except VVFError as exc:  # um alvo inválido não interrompe a comparação
            table.add_row(prov_name, model, "[red]recusado[/]", "-", str(exc)[:90])
    project.save()
    console.print(table)


@transform_app.command("resume")
@_mutating
def transform_resume(slug: str, transform_id: str, provider: Annotated[str | None, typer.Option()] = None):
    """Continua uma transformação já enviada, sem reenviar nem cobrar de novo."""
    s = _settings()
    project = _open(s, slug)
    rec = next((t for t in project.manifest.transforms if t.id == transform_id), None)
    if rec is None:
        raise UserInputError(f"transformação inexistente: {transform_id}")
    out = resume_transform(project, s, make_provider(provider or rec.provider, s), transform_id)
    _transform_report(out, show_cost=False)


@transform_app.command("use")
@_mutating
def transform_use(slug: str, clip: str, transform_id: str):
    """Faz o trecho usar o vídeo transformado no próximo render (exige aprovar o plano de novo)."""
    s = _settings()
    use_transform(_open(s, slug), clip, transform_id)
    console.print(
        f"[green]✔[/] {clip} agora usa {transform_id}. Rode 'vvf plan approve {slug}' e depois 'vvf render'."
    )


@transform_app.command("list")
def transform_list(slug: str):
    s = _settings()
    project = _open(s, slug)
    for t in project.manifest.transforms:
        _transform_report(t)


# ---------- narração / biblioteca ----------


@app.command("narrate")
@_mutating
def narrate_cmd(
    slug: str,
    clip: Annotated[str, typer.Option()],
    text: Annotated[str | None, typer.Option(help="Texto da narração.")] = None,
    use_hook: Annotated[bool, typer.Option("--hook", help="Narra o gancho do trecho.")] = False,
    provider: Annotated[str, typer.Option(help="piper (local, grátis) | elevenlabs (pago).")] = "piper",
    mode: Annotated[str, typer.Option(help="duck (abaixa a fala original) | mix | replace.")] = "duck",
    execute: Annotated[
        bool | None, typer.Option("--execute/--dry-run", help="Padrão: executa se local, simula se pago.")
    ] = None,
    price_usd: Annotated[float | None, typer.Option()] = None,
):
    s = _settings()
    project = _open(s, slug)
    plan_clip = planmod.load_plan(project).get(clip)
    body = text or (plan_clip.hook if use_hook else "")
    if not body:
        raise UserInputError("informe --text ou --hook")
    tts = make_tts(provider, s)
    run = execute if execute is not None else (provider == "piper")
    out = narrate(project, s, tts, clip_id=clip, text=body, mode=mode, execute=run, price_usd=price_usd)
    if out:
        console.print(f"[green]✔[/] narração: {out} (voz sintética: o vídeo será marcado como IA)")
    else:
        est = project.manifest.costs[-1]
        console.print(f"simulação: {_money(est.estimated_usd)} · {est.note}")


@library_app.command("list")
def library_list():
    lib = Library(Path(_settings().vvf_library_dir))
    table = Table(title="Biblioteca de áudio")
    for col in ("id", "tipo", "título", "licença", "duração"):
        table.add_column(col)
    for i in lib.list():
        table.add_row(i.id, i.kind, i.title, i.license, f"{i.duration:.1f}s")
    console.print(table)


@library_app.command("add")
def library_add(
    file: Path,
    item_id: Annotated[str, typer.Option("--id")],
    kind: Annotated[str, typer.Option(help="music | sfx")],
    license: Annotated[str, typer.Option(help=f"Uma de: {', '.join(sorted(ALLOWED_LICENSES))}")],
    source: Annotated[str, typer.Option(help="Onde obteve (URL/loja/autor).")],
    title: Annotated[str, typer.Option()] = "",
    attribution: Annotated[str, typer.Option(help="Crédito exigido pela licença.")] = "",
):
    """Só entra áudio com licença declarada."""
    lib = Library(Path(_settings().vvf_library_dir))
    item = lib.add(
        file,
        item_id=item_id,
        kind=kind,
        title=title or item_id,
        license=license,
        source=source,
        attribution=attribution,
    )
    console.print(f"[green]✔[/] {item.id} adicionado ({item.license})")


@library_app.command("synth")
def library_synth():
    """Gera efeitos/trilha simples com FFmpeg (licença generated-by-vvf)."""
    lib = Library(Path(_settings().vvf_library_dir))
    made = lib.synth_builtin()
    console.print(f"[green]✔[/] {len(made)} item(ns) gerado(s): {[i.id for i in made]}")


# ---------- estado / retomada / fila ----------


def _status_data(project: Project) -> dict:
    m = project.manifest
    steps = {
        n: {"status": "interrompida" if is_interrupted(r) else r.status.value, "tentativas": r.attempts,
            "erro": r.error}
        for n, r in m.steps.items()
    }  # fmt: skip
    return {
        "slug": m.slug,
        "origem": m.provenance.model_dump() if m.provenance else None,
        "video": m.source.probe.model_dump() if m.source else None,
        "etapas": steps,
        "sintetico": m.synthetic,
        "motivos_sinteticos": m.synthetic_reasons,
        "gasto_usd": m.spent_usd(),
        "simulacoes": sum(1 for c in m.costs if c.dry_run),
        "transformacoes": [t.model_dump() for t in m.transforms],
        "renders": [r.model_dump() for r in m.renders],
    }


@app.command()
def status(
    slug: Annotated[str | None, typer.Argument()] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
):
    """Estado do projeto (ou lista de projetos)."""
    s = _settings()
    if slug is None:
        slugs = Project.list_slugs(_projects_dir(s))
        console.print("\n".join(slugs) if slugs else "nenhum projeto ainda: use 'vvf import'")
        return
    project = _open(s, slug)
    data = _status_data(project)
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        return
    table = Table(title=f"Projeto {slug}")
    for col in ("etapa", "estado", "tentativas", "erro"):
        table.add_column(col)
    for n, st in data["etapas"].items():
        color = {"done": "green", "failed": "red", "interrompida": "yellow"}.get(st["status"], "white")
        table.add_row(n, f"[{color}]{st['status']}[/]", str(st["tentativas"]), (st["erro"] or "")[:60])
    console.print(table)
    ai = "sim — " + ", ".join(data["motivos_sinteticos"]) if data["sintetico"] else "não"
    console.print(
        f"gasto real: US$ {data['gasto_usd']:.4f} · simulações: {data['simulacoes']} · conteúdo IA: {ai}"
    )
    pending = [t for t in project.manifest.transforms if t.status in ("submitted", "running")]
    for t in pending:
        console.print(f"[yellow]transformação {t.id} em andamento:[/] vvf transform resume {slug} {t.id}")


@app.command()
def resume(slug: str):
    """Repete tarefas que falharam ou foram interrompidas (etapas prontas vêm do cache)."""
    s = _settings()
    _open(s, slug)
    summary = run_queue(s, _queue(s), slug=slug, include_failed=True, resume=True)
    console.print(f"[green]✔[/] retomada: {summary['done']} concluída(s), {summary['failed']} falha(s), "
                  f"{summary['recovered']} recuperada(s) de interrupção")  # fmt: skip
    for t in _open(s, slug).manifest.transforms:
        if t.status in ("submitted", "running"):
            console.print(
                f"[yellow]transformação paga {t.id} aguardando:[/] vvf transform resume {slug} {t.id}"
            )


@queue_app.command("add")
def queue_add(slug: str, kind: str, args_json: Annotated[str, typer.Option("--args")] = "{}"):
    s = _settings()
    _open(s, slug)
    job_id = _queue(s).add(slug, kind, json.loads(args_json))
    console.print(f"[green]✔[/] tarefa {job_id} enfileirada ({kind} · {slug})")


@queue_app.command("list")
def queue_list(slug: Annotated[str | None, typer.Option()] = None):
    table = Table(title="Fila")
    for col in ("id", "projeto", "tipo", "estado", "tentativas", "erro"):
        table.add_column(col)
    for j in _queue(_settings()).list(slug):
        table.add_row(
            str(j.id), j.slug, j.kind, j.status, f"{j.attempts}/{j.max_attempts}", (j.error or "")[:50]
        )
    console.print(table)


@queue_app.command("run")
def queue_run(
    max_jobs: Annotated[int | None, typer.Option()] = None,
    retry_failed: Annotated[bool, typer.Option()] = False,
):
    s = _settings()
    summary = run_queue(s, _queue(s), include_failed=retry_failed, max_jobs=max_jobs)
    console.print(
        f"concluídas: {summary['done']} · falhas: {summary['failed']} · recuperadas: {summary['recovered']}"
    )


# ---------- diagnóstico ----------


@app.command()
def doctor():
    """Confere ambiente, ferramentas e quais integrações têm credenciais (sem chamar APIs)."""
    s = _settings()
    rows = []
    for tool in ("ffmpeg", "ffprobe"):
        rows.append((tool, bool(shutil.which(tool)), shutil.which(tool) or "não encontrado"))
    for name, ok in s.configured_providers().items():
        rows.append((f"credencial {name}", ok, "configurada" if ok else "ausente"))
    rows.append(
        (
            "chamadas pagas",
            s.vvf_allow_paid_calls,
            f"{'LIGADAS' if s.vvf_allow_paid_calls else 'desligadas'} · teto US$ {s.vvf_max_spend_usd:.2f}",
        )
    )
    rows.append(
        (
            "piper (voz local)",
            bool(s.vvf_piper_voice and shutil.which(s.vvf_piper_bin)),
            s.vvf_piper_voice or "VVF_PIPER_VOICE não definida",
        )
    )
    table = Table(title=f"vvf {__version__} · diagnóstico")
    for col in ("item", "ok", "detalhe"):
        table.add_column(col)
    for item, ok, detail in rows:
        table.add_row(item, "[green]sim[/]" if ok else "[yellow]não[/]", detail)
    console.print(table)


def main() -> None:
    """Ponto de entrada. O Typer trata erros de uso; aqui tratamos os erros do domínio."""
    try:
        app()
    except UserInputError as exc:
        err.print(f"[red]erro:[/] {exc}")
        sys.exit(2)
    except ValidationError as exc:  # arquivo de configuração/preset/manifesto malformado
        err.print(f"[red]erro de configuração:[/] {exc.error_count()} problema(s): {exc.errors()[0]['loc']}")
        sys.exit(2)
    except VVFError as exc:
        err.print(f"[red]falha:[/] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

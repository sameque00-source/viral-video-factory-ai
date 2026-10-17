"""Etapa B: escolhe os melhores trechos e gera ganchos, títulos, descrições e hashtags.

Duas engines produzem o mesmo `plan.json` editável:
- heurística: offline, grátis, determinística; só reutiliza texto que existe na transcrição
- llm: usa Claude com saída JSON; cada trecho é validado contra a transcrição (sem invenção)
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from vvf.config import Settings
from vvf.errors import PlanError
from vvf.llm import LLMClient
from vvf.pipeline.runner import StepResult, run_step, upstream
from vvf.pipeline.transcribe import Segment, Transcript, load_transcript
from vvf.project import Project

MIN_CLIP_S, MAX_CLIP_S = 1.0, 180.0

HOOK_WORDS = {
    "segredo", "incrível", "incrivel", "nunca", "ninguém", "ninguem", "atenção", "atencao",
    "cuidado", "erro", "verdade", "mentira", "descubra", "surpreendente", "chocante",
    "impressionante", "jamais", "truque", "dica", "errado", "pare", "veja", "sabia",
    "imagine", "milhões", "milhoes", "bilhões", "bilhoes", "grátis", "gratis", "urgente",
}  # fmt: skip
STOPWORDS = set(
    "para como mais muito você voce isso esse essa esses essas aquele aquela quando onde "
    "porque pois então entao ainda sobre entre depois antes também tambem apenas porém porem "
    "dois duas tudo todos todas cada outro outra mesmo seja ficar fazer pode podem tem têm "
    "foram será sera estão estao era são sao dele dela deles delas minha meu seus suas nosso "
    "nossa aqui ali lá".split()
)  # fmt: skip


class Clip(BaseModel):
    id: str
    start: float
    end: float
    enabled: bool = True
    score: float = 0.0
    reason: str = ""
    hook: str = ""
    hook_source: Literal["transcript", "llm", "user"] = "transcript"
    quotes: list[str] = Field(default_factory=list)
    segment_ids: list[int] = Field(default_factory=list)
    title: str = ""
    description: str = ""
    hashtags: list[str] = Field(default_factory=list)
    script_notes: str = ""
    source: str = "original"  # "original" ou "transform:<id>"
    narration: str | None = None  # caminho relativo ao projeto (wav)
    narration_mode: Literal["duck", "mix", "replace"] = "duck"
    generated_by: str = "manual"

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


class Plan(BaseModel):
    version: int = 1
    engine: str
    status: Literal["draft", "approved"] = "draft"
    edited: bool = False
    transcript_sha: str
    ideas: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    clips: list[Clip]

    def enabled_clips(self) -> list[Clip]:
        return [c for c in self.clips if c.enabled]

    def get(self, clip_id: str) -> Clip:
        for c in self.clips:
            if c.id == clip_id:
                return c
        raise PlanError(f"trecho inexistente: {clip_id}")


# ---------- utilidades de texto ----------


def _norm(text: str) -> str:
    """Minúsculas, sem acentos nem pontuação: compara citações com a transcrição."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text).split())


def transcript_sha(t: Transcript) -> str:
    return hashlib.sha256(t.model_dump_json().encode()).hexdigest()[:16]


def _slug_tag(word: str) -> str:
    base = unicodedata.normalize("NFKD", word.lower())
    base = "".join(c for c in base if not unicodedata.combining(c))
    return "#" + re.sub(r"[^a-z0-9]", "", base)


def _keywords(text: str, n: int = 3) -> list[str]:
    words = [w for w in re.findall(r"[^\W\d_]{5,}", text.lower()) if w not in STOPWORDS]
    return [w for w, _ in Counter(words).most_common(n)]


# ---------- engine heurística ----------


def _hook_score(first: Segment) -> tuple[float, str]:
    text = first.text.strip()
    low = text.lower()
    score, parts = 0.0, []
    if "?" in text:
        score += 0.5
        parts.append("abre com pergunta")
    if any(w in HOOK_WORDS for w in re.findall(r"[^\W\d_]+", low)):
        score += 0.3
        parts.append("palavra de gancho")
    if re.search(r"\d", text):
        score += 0.2
        parts.append("traz número")
    if "!" in text:
        score += 0.1
        parts.append("exclamação")
    return min(score, 1.0), ", ".join(parts)


def _gap_penalty(segs: list[Segment]) -> float:
    gaps = sum(max(0.0, b.start - a.end - 2.0) for a, b in zip(segs, segs[1:], strict=False))
    total = segs[-1].end - segs[0].start
    return min(1.0, gaps / total) if total > 0 else 0.0


def _mean_db(energy: list[dict] | None, start: float, end: float) -> float | None:
    if not energy:
        return None
    vals = [e["db"] for e in energy if start <= e["t"] < end]
    return sum(vals) / len(vals) if vals else None


def _cuts_in(scenes: list[dict], start: float, end: float) -> int:
    return sum(1 for s in scenes if start < s["start"] < end)


def build_heuristic_plan(
    transcript: Transcript,
    scenes: list[dict],
    energy: list[dict] | None,
    *,
    n_clips: int = 3,
    min_len: float = 15.0,
    max_len: float = 45.0,
) -> Plan:
    segs = [s for s in transcript.segments if s.text.strip()]
    sha = transcript_sha(transcript)
    if not segs:
        return Plan(
            engine="heuristic", transcript_sha=sha, clips=[],
            warnings=["sem fala detectada: nada para sugerir. Edite o plano manualmente."],
        )  # fmt: skip
    warnings: list[str] = []
    speech_span = segs[-1].end - segs[0].start
    if speech_span < min_len:
        min_len = max(1.0, speech_span * 0.9)
        warnings.append("vídeo curto: o trecho único cobre quase toda a fala")

    video_db = _mean_db(energy, 0, transcript.duration)
    candidates = []
    for i in range(len(segs)):
        for j in range(i, len(segs)):
            dur = segs[j].end - segs[i].start
            if dur > max_len:
                break
            if dur < min_len and j != len(segs) - 1:
                continue
            window = segs[i : j + 1]
            words = sum(len(s.text.split()) for s in window)
            density = min(1.0, (words / dur) / 3.0) if dur > 0 else 0.0
            hook, hook_why = _hook_score(window[0])
            db = _mean_db(energy, window[0].start, window[-1].end)
            loud = 0.5 if db is None or video_db is None else min(1, max(0, (db - video_db + 6) / 12))
            cuts = _cuts_in(scenes, window[0].start, window[-1].end)
            dyn = min(1.0, (cuts / (dur / 10)) / 2) if dur > 0 else 0.0
            complete = 1.0 if window[-1].text.strip()[-1:] in ".!?…" else 0.5
            score = (
                0.30 * hook + 0.25 * density + 0.15 * loud + 0.10 * dyn + 0.10 * complete
                + 0.10 * (1 - _gap_penalty(window))
            )  # fmt: skip
            why = [
                f"{words / dur:.1f} palavras/s",
                *([hook_why] if hook_why else []),
                f"{cuts} corte(s) de cena",
                "termina em frase completa" if complete == 1.0 else "termina no meio da frase",
            ]
            candidates.append((score, i, j, dur, "; ".join(why)))

    candidates.sort(key=lambda c: (-c[0], c[1]))
    chosen, used = [], []
    for score, i, j, dur, why in candidates:
        if len(chosen) >= n_clips:
            break
        if any(not (j < ui or i > uj) for ui, uj in used):
            continue
        used.append((i, j))
        chosen.append((score, i, j, dur, why))
    chosen.sort(key=lambda c: c[1])

    clips = []
    for k, (score, i, j, _dur, why) in enumerate(chosen, start=1):
        window = segs[i : j + 1]
        start = max(0.0, window[0].start - 0.15)
        end = min(transcript.duration, window[-1].end + 0.25)
        text = " ".join(s.text.strip() for s in window)
        hook_text = window[0].text.strip()
        tags = [_slug_tag(w) for w in _keywords(text)]
        clips.append(
            Clip(
                id=f"c{k}", start=round(start, 3), end=round(end, 3), score=round(score, 3),
                reason=why, hook=hook_text, hook_source="transcript",
                quotes=[hook_text], segment_ids=[s.id for s in window],
                title=hook_text[:60].rstrip(" ,;:"), description=hook_text,
                hashtags=[*tags, "#shorts"], generated_by="heuristic",
            )
        )  # fmt: skip
    return Plan(engine="heuristic", transcript_sha=sha, clips=clips, warnings=warnings)


# ---------- engine LLM ----------

PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ideas", "clips"],
    "properties": {
        "ideas": {"type": "array", "items": {"type": "string"}},
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "start_segment",
                    "end_segment",
                    "reason",
                    "hook",
                    "title",
                    "description",
                    "hashtags",
                    "script_notes",
                    "quotes",
                ],  # fmt: skip
                "properties": {
                    "start_segment": {"type": "integer"},
                    "end_segment": {"type": "integer"},
                    "reason": {"type": "string"},
                    "hook": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                    "script_notes": {"type": "string"},
                    "quotes": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}

SYSTEM_PROMPT = """Você é editor de vídeos curtos virais (TikTok, Reels, Shorts) do Brasil.
Recebe a transcrição numerada de um vídeo e escolhe os trechos com maior potencial de retenção.

Regras obrigatórias:
- Escolha trechos SOMENTE por intervalo de segmentos (start_segment..end_segment, inclusive).
- Use apenas o que está na transcrição. Nunca invente fatos, números, nomes ou falas.
- Cada trecho deve ter 1 ou 2 citações literais (campo quotes) copiadas da transcrição.
- hook: texto curto (até 12 palavras) para os 3 primeiros segundos; pode reformular, mas sem
  afirmar nada que o vídeo não diga.
- Português do Brasil. Título até 70 caracteres. Descrição de 1 a 2 frases. 3 a 6 hashtags.
- reason: justifique em uma frase por que o trecho funciona (retenção, emoção, curiosidade).
- script_notes: sugestão de edição (corte, zoom, texto na tela) sem alterar o conteúdo falado.
"""


def format_transcript(t: Transcript) -> str:
    return "\n".join(f"[{s.id}] ({s.start:.1f}s-{s.end:.1f}s) {s.text.strip()}" for s in t.segments)


def build_user_prompt(t: Transcript, n_clips: int, min_len: float, max_len: float) -> str:
    return (
        f"Duração do vídeo: {t.duration:.1f}s. Idioma: {t.language}.\n"
        f"Escolha até {n_clips} trechos de {min_len:.0f} a {max_len:.0f} segundos "
        "(ou o vídeo todo se for mais curto) e proponha de 3 a 5 ideias de ângulo/roteiro.\n\n"
        f"TRANSCRIÇÃO:\n{format_transcript(t)}"
    )


def validate_llm_clips(
    raw: dict, transcript: Transcript, min_len: float, max_len: float
) -> tuple[list[Clip], list[str]]:
    """Descarta (com aviso) qualquer trecho que não se sustente na transcrição."""
    by_id = {s.id: s for s in transcript.segments}
    clips: list[Clip] = []
    warnings: list[str] = []
    for n, item in enumerate(raw.get("clips", []), start=1):
        a, b = item.get("start_segment"), item.get("end_segment")
        if a not in by_id or b not in by_id or a > b:
            warnings.append(f"trecho {n} descartado: segmentos inválidos ({a}..{b})")
            continue
        window = [by_id[i] for i in range(a, b + 1) if i in by_id]
        start, end = window[0].start, window[-1].end
        dur = end - start
        if dur < min(min_len, transcript.duration * 0.5) or dur > max_len:
            warnings.append(f"trecho {n} descartado: duração {dur:.1f}s fora do limite")
            continue
        window_norm = _norm(" ".join(s.text for s in window))
        quotes = [q for q in item.get("quotes", []) if q.strip()]
        bad = [q for q in quotes if _norm(q) not in window_norm]
        if not quotes or bad:
            warnings.append(f"trecho {n} descartado: citação não encontrada na transcrição")
            continue
        clips.append(
            Clip(
                id=f"c{len(clips) + 1}", start=round(max(0.0, start - 0.15), 3),
                end=round(min(transcript.duration, end + 0.25), 3),
                reason=item.get("reason", ""), hook=item.get("hook", ""), hook_source="llm",
                quotes=quotes, segment_ids=[s.id for s in window], title=item.get("title", ""),
                description=item.get("description", ""),
                hashtags=[h if h.startswith("#") else f"#{h}" for h in item.get("hashtags", [])],
                script_notes=item.get("script_notes", ""), generated_by="llm",
            )
        )  # fmt: skip
    return clips, warnings


def build_llm_plan(
    transcript: Transcript, llm: LLMClient, *, n_clips: int, min_len: float, max_len: float,
    max_tokens: int = 8000,
) -> Plan:  # fmt: skip
    raw = llm.complete_json(
        SYSTEM_PROMPT, build_user_prompt(transcript, n_clips, min_len, max_len), PLAN_SCHEMA,
        max_tokens,
    )  # fmt: skip
    clips, warnings = validate_llm_clips(raw, transcript, min_len, max_len)
    if not clips:
        raise PlanError("nenhum trecho do modelo passou na validação: " + "; ".join(warnings))
    return Plan(
        engine=f"llm:{llm.model}", transcript_sha=transcript_sha(transcript),
        ideas=[i for i in raw.get("ideas", []) if i.strip()], warnings=warnings, clips=clips,
    )  # fmt: skip


# ---------- persistência e edição ----------


def plan_path(project: Project):
    return project.root / "work" / "plan.json"


def load_plan(project: Project) -> Plan:
    p = plan_path(project)
    if not p.exists():
        raise PlanError("plano ainda não gerado: rode 'vvf plan generate'")
    return Plan.model_validate_json(p.read_text(encoding="utf-8"))


def save_plan(project: Project, plan: Plan) -> None:
    plan_path(project).write_text(plan.model_dump_json(indent=2), encoding="utf-8")


def _load_aux(project: Project) -> tuple[list[dict], list[dict] | None]:
    scenes_rec = project.manifest.steps.get("scenes")
    scenes = []
    if scenes_rec and "scenes" in scenes_rec.outputs:
        scenes = json.loads(project.path(scenes_rec.outputs["scenes"]).read_text())["scenes"]
    energy_rec = project.manifest.steps.get("energy")
    energy = None
    if energy_rec and "energy" in energy_rec.outputs:
        energy = json.loads(project.path(energy_rec.outputs["energy"]).read_text())
    return scenes, energy


def generate_plan(
    project: Project,
    settings: Settings,
    *,
    engine: Literal["heuristic", "llm"] = "heuristic",
    llm: LLMClient | None = None,
    n_clips: int = 3,
    min_len: float = 15.0,
    max_len: float = 45.0,
    force: bool = False,
) -> Plan:
    """Gera plan.json. Um plano editado ou aprovado só é sobrescrito com force=True."""
    transcript = load_transcript(project)
    if plan_path(project).exists() and not force:
        existing = load_plan(project)
        if existing.edited or existing.status == "approved":
            raise PlanError("o plano foi editado/aprovado; use --force para regenerar do zero")
    scenes, energy = _load_aux(project)
    if engine == "llm" and llm is None:
        raise PlanError("engine llm exige um cliente configurado")

    def work() -> StepResult:
        if engine == "llm":
            plan = build_llm_plan(
                transcript,
                llm,
                n_clips=n_clips,
                min_len=min_len,
                max_len=max_len,  # type: ignore[arg-type]
            )
        else:
            plan = build_heuristic_plan(
                transcript, scenes, energy, n_clips=n_clips, min_len=min_len, max_len=max_len
            )
        save_plan(project, plan)
        return StepResult({"plan": project.rel(plan_path(project))}, {"clips": len(plan.clips)})

    run_step(
        project,
        "plan",
        {
            "engine": engine, "model": getattr(llm, "model", None), "n": n_clips,
            "min": min_len, "max": max_len, "sha": transcript_sha(transcript),
            **upstream(project, "transcribe", "scenes"),
        },
        work,
        force=force,
        retries=0 if engine == "llm" else settings.vvf_step_retries,
    )  # fmt: skip
    return load_plan(project)


def edit_clip(project: Project, clip_id: str, **changes) -> Plan:
    plan = load_plan(project)
    clip = plan.get(clip_id)
    duration = project.manifest.source.probe.duration if project.manifest.source else None
    new = clip.model_copy(update={k: v for k, v in changes.items() if v is not None})
    if new.start < 0 or (duration is not None and new.end > duration + 0.05):
        raise PlanError(f"tempos fora do vídeo (0..{duration:.1f}s)")
    if new.end - new.start < MIN_CLIP_S:
        raise PlanError(f"trecho curto demais (mín. {MIN_CLIP_S:.0f}s)")
    if new.end - new.start > MAX_CLIP_S:
        raise PlanError(f"trecho longo demais (máx. {MAX_CLIP_S:.0f}s)")
    plan.clips[plan.clips.index(clip)] = new
    plan.edited, plan.status = True, "draft"
    save_plan(project, plan)
    return plan


def add_clip(project: Project, start: float, end: float, title: str = "") -> Plan:
    plan = load_plan(project)
    clip_id = f"c{max([int(c.id[1:]) for c in plan.clips if c.id[1:].isdigit()] or [0]) + 1}"
    plan.clips.append(
        Clip(id=clip_id, start=start, end=end, title=title, reason="manual", generated_by="manual")
    )
    plan.edited, plan.status = True, "draft"
    save_plan(project, plan)
    return edit_clip(project, clip_id)  # reaproveita a validação


def approve_plan(project: Project) -> Plan:
    plan = load_plan(project)
    if not plan.enabled_clips():
        raise PlanError("nenhum trecho ativo para aprovar")
    plan.status = "approved"
    save_plan(project, plan)
    return plan


def llm_plan_estimate(project: Project, llm: LLMClient, n_clips: int, min_len: float, max_len: float):
    """Estimativa de custo da geração do plano por LLM (sem chamar a API)."""
    transcript = load_transcript(project)
    return llm.estimate(SYSTEM_PROMPT, build_user_prompt(transcript, n_clips, min_len, max_len), 8000)

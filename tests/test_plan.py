import json

import pytest

from conftest import FakeTranscriber, make_transcript
from vvf.errors import PlanError
from vvf.llm import AnthropicLLM, estimate_cost
from vvf.pipeline import plan as planmod
from vvf.pipeline.ingest import import_video
from vvf.pipeline.scenes import compute_audio_energy, detect_scenes
from vvf.pipeline.transcribe import Segment, Transcript, Word, transcribe


def _long_transcript() -> Transcript:
    """90 s de fala: um trecho forte (pergunta + número) entre trechos fracos."""
    lines = [
        (0, 6, "Bem vindos ao canal de hoje, vamos conversar um pouco."),
        (6, 12, "Primeiro um aviso sobre os assuntos que veremos em seguida."),
        (14, 20, "Você sabia que o polvo tem três corações?"),
        (20, 27, "Dois bombeiam sangue para as brânquias e um para o corpo todo."),
        (27, 34, "Incrível, ninguém imagina isso sobre esse animal!"),
        (50, 58, "Agora voltando ao assunto comum que falávamos antes de tudo."),
        (60, 70, "Era só isso por hoje, obrigado por assistir até aqui no vídeo."),
    ]
    segs = []
    for i, (s, e, t) in enumerate(lines):
        ws = t.split()
        step = (e - s) / len(ws)
        segs.append(
            Segment(id=i, start=s, end=e, text=t,
                    words=[Word(text=w, start=s + k * step, end=s + (k + 1) * step)
                           for k, w in enumerate(ws)])
        )  # fmt: skip
    return Transcript(language="pt", duration=72.0, model="fake", segments=segs)


def test_heuristic_prefers_hook_window_and_never_invents_text():
    t = _long_transcript()
    plan = planmod.build_heuristic_plan(t, [], None, n_clips=1, min_len=15, max_len=40)
    clip = plan.clips[0]
    assert "polvo" in clip.hook  # gancho = frase real da transcrição
    assert clip.hook_source == "transcript"
    full = planmod._norm(t.full_text())
    assert all(planmod._norm(q) in full for q in clip.quotes)
    assert clip.start >= 13.5 and clip.end <= 35


def test_heuristic_clips_do_not_overlap_and_respect_bounds():
    t = _long_transcript()
    plan = planmod.build_heuristic_plan(t, [], None, n_clips=3, min_len=10, max_len=30)
    spans = sorted((c.start, c.end) for c in plan.clips)
    assert all(a_end <= b_start + 0.5 for (_, a_end), (b_start, _) in zip(spans, spans[1:], strict=False))
    assert all(0 <= c.start < c.end <= t.duration for c in plan.clips)


def test_heuristic_without_speech_returns_empty_plan_with_warning():
    t = Transcript(language="pt", duration=10, model="none", segments=[])
    plan = planmod.build_heuristic_plan(t, [], None)
    assert plan.clips == [] and plan.warnings


def test_short_video_gets_single_clip():
    plan = planmod.build_heuristic_plan(make_transcript(), [], None, n_clips=3)
    assert len(plan.clips) == 1 and plan.warnings


class FakeLLM:
    """Dublê de teste do LLM: devolve JSON fixo. Não é usado fora dos testes."""

    model = "claude-opus-5-5"

    def __init__(self, payload):
        self.payload, self.calls = payload, 0

    def estimate(self, system, user, max_tokens):
        return estimate_cost(self.model, system, user, max_tokens)

    def complete_json(self, system, user, schema, max_tokens):
        self.calls += 1
        return self.payload


def _clip_payload(**over):
    base = {
        "start_segment": 2, "end_segment": 4, "reason": "pergunta curiosa",
        "hook": "Polvo tem 3 corações?",
        "title": "O segredo do polvo", "description": "Curiosidade rápida.",
        "hashtags": ["polvo", "#curiosidades"], "script_notes": "zoom no gancho",
        "quotes": ["Você sabia que o polvo tem três corações"],
    }  # fmt: skip
    return {**base, **over}


def test_llm_plan_accepts_grounded_clip():
    t = _long_transcript()
    plan = planmod.build_llm_plan(
        t, FakeLLM({"ideas": ["a", "b"], "clips": [_clip_payload()]}),
        n_clips=1, min_len=15, max_len=40,
    )  # fmt: skip
    clip = plan.clips[0]
    assert plan.engine == "llm:claude-opus-5-5" and clip.hook_source == "llm"
    assert clip.hashtags == ["#polvo", "#curiosidades"]
    assert clip.segment_ids == [2, 3, 4]


def test_llm_plan_drops_clip_with_invented_quote():
    t = _long_transcript()
    bad = _clip_payload(quotes=["O polvo tem nove cérebros e mora no espaço"])
    with pytest.raises(PlanError, match="citação"):
        planmod.build_llm_plan(t, FakeLLM({"ideas": [], "clips": [bad]}), n_clips=1, min_len=15, max_len=40)


def test_llm_plan_keeps_valid_and_warns_about_invalid():
    t = _long_transcript()
    payload = {
        "ideas": [],
        "clips": [_clip_payload(), _clip_payload(start_segment=99, end_segment=100)],
    }
    plan = planmod.build_llm_plan(t, FakeLLM(payload), n_clips=2, min_len=15, max_len=40)
    assert len(plan.clips) == 1 and any("inválidos" in w for w in plan.warnings)


def test_llm_plan_rejects_quote_from_outside_the_window():
    t = _long_transcript()
    wrong_window = _clip_payload(quotes=["obrigado por assistir até aqui"])
    with pytest.raises(PlanError):
        planmod.build_llm_plan(
            t, FakeLLM({"ideas": [], "clips": [wrong_window]}), n_clips=1, min_len=15, max_len=40
        )


def _analyzed(project, sample_video, provenance, settings, transcript):
    import_video(project, sample_video, provenance, settings)
    detect_scenes(project, settings)
    compute_audio_energy(project, settings)
    transcribe(project, settings, FakeTranscriber(transcript))


def test_generate_plan_persists_caches_and_protects_edits(
    project, sample_video, provenance, settings, transcript
):
    _analyzed(project, sample_video, provenance, settings, transcript)
    plan = planmod.generate_plan(project, settings, n_clips=1)
    assert plan.clips and planmod.plan_path(project).exists()
    # edição manual valida limites e marca o plano como editado
    edited = planmod.edit_clip(project, "c1", title="Meu título", start=0.5, end=6.0)
    assert edited.edited and edited.status == "draft" and edited.get("c1").title == "Meu título"
    with pytest.raises(PlanError, match="editado"):
        planmod.generate_plan(project, settings, n_clips=1)  # não sobrescreve sem --force
    assert planmod.generate_plan(project, settings, n_clips=1, force=True).edited is False


def test_edit_validation_and_approval(project, sample_video, provenance, settings, transcript):
    _analyzed(project, sample_video, provenance, settings, transcript)
    planmod.generate_plan(project, settings, n_clips=1)
    with pytest.raises(PlanError, match="fora do vídeo"):
        planmod.edit_clip(project, "c1", end=999.0)
    with pytest.raises(PlanError, match="curto"):
        planmod.edit_clip(project, "c1", start=1.0, end=1.2)
    with pytest.raises(PlanError, match="inexistente"):
        planmod.edit_clip(project, "c9", title="x")
    assert planmod.approve_plan(project).status == "approved"
    planmod.edit_clip(project, "c1", title="mudou")  # editar desfaz a aprovação
    assert planmod.load_plan(project).status == "draft"


def test_add_manual_clip_and_disable_all_blocks_approval(
    project, sample_video, provenance, settings, transcript
):
    _analyzed(project, sample_video, provenance, settings, transcript)
    planmod.generate_plan(project, settings, n_clips=1)
    plan = planmod.add_clip(project, 1.0, 4.0, title="manual")
    assert plan.get("c2").generated_by == "manual"
    planmod.edit_clip(project, "c1", enabled=False)
    planmod.edit_clip(project, "c2", enabled=False)
    with pytest.raises(PlanError, match="nenhum trecho"):
        planmod.approve_plan(project)


def test_llm_cost_estimate_known_and_unknown_models():
    known = estimate_cost("claude-opus-5-5", "s" * 1000, "u" * 10000, 8000)
    assert known.usd and 0 < known.usd < 1
    assert estimate_cost("modelo-desconhecido", "a", "b", 100).usd is None


class _FakeMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason, self.kwargs = text, stop_reason, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        block = type("B", (), {"type": "text", "text": self.text})()
        return type("R", (), {"content": [block], "stop_reason": self.stop_reason})()


def _client(text, stop="end_turn"):
    return type("C", (), {"messages": _FakeMessages(text, stop)})()


def test_anthropic_llm_sends_structured_output_request_and_parses():
    client = _client(json.dumps({"ideas": [], "clips": []}))
    llm = AnthropicLLM(None, client=client)
    out = llm.complete_json("sys", "user", {"type": "object"}, 100)
    sent = client.messages.kwargs
    assert out == {"ideas": [], "clips": []}
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in sent and "tool_choice" not in sent and "temperature" not in sent


def test_anthropic_llm_refusal_truncation_and_bad_json():
    from vvf.errors import ProviderError

    for text, stop in [("{}", "refusal"), ("{}", "max_tokens"), ("não é json", "end_turn")]:
        with pytest.raises(ProviderError):
            AnthropicLLM(None, client=_client(text, stop)).complete_json("s", "u", {}, 10)


def test_anthropic_llm_requires_key_without_injected_client():
    from vvf.errors import MissingCredential

    with pytest.raises(MissingCredential):
        AnthropicLLM(None)

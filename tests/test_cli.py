"""A CLI de ponta a ponta com entradas válidas e inválidas (sem rede, sem custo)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import FakeTranscriber
from vvf import cli
from vvf.config import Settings
from vvf.models.manifest import Provenance, Rights
from vvf.pipeline.ingest import import_video
from vvf.pipeline.transcribe import transcribe
from vvf.project import Project


def vvf(monkeypatch, capsys, *args: str, expect: int = 0):
    """Executa `vvf <args>` no processo e devolve (saída, erro). Falha se o código de saída divergir."""
    monkeypatch.setenv("COLUMNS", "220")  # evita quebra de linha das tabelas Rich
    monkeypatch.setattr(sys, "argv", ["vvf", *args])
    with pytest.raises(SystemExit) as exc:
        cli.main()  # o Typer sempre encerra com SystemExit (0 em caso de sucesso)
    out = capsys.readouterr()
    assert exc.value.code == expect, f"código {exc.value.code} != {expect}\n{out.out}\n{out.err}"
    return out.out, out.err


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def analyzed(cwd, sample_video, transcript, monkeypatch, capsys):
    """Projeto 'demo' importado pela CLI e transcrito com o dublê (o analyze real usa Whisper)."""
    vvf(monkeypatch, capsys, "import", str(sample_video), "--rights", "own", "--slug", "demo")
    project = Project.open(Path("projects"), "demo")
    s = Settings(_env_file=None)
    from vvf.pipeline.scenes import compute_audio_energy, detect_scenes

    detect_scenes(project, s)
    compute_audio_energy(project, s)
    transcribe(project, s, FakeTranscriber(transcript))
    return cwd


def test_import_happy_path_and_status(cwd, sample_video, monkeypatch, capsys):
    out, _ = vvf(monkeypatch, capsys, "import", str(sample_video), "--rights", "own", "--slug", "demo")
    assert "projeto demo criado" in out and "640x360" in out
    out, _ = vvf(monkeypatch, capsys, "status", "demo", "--json")
    data = json.loads(out)
    assert data["video"]["width"] == 640 and data["origem"]["rights"] == "own"
    assert data["etapas"]["ingest"]["status"] == "done" and data["gasto_usd"] == 0
    out, _ = vvf(monkeypatch, capsys, "status")
    assert "demo" in out


@pytest.mark.parametrize(
    "args,needle",
    [
        (["import", "nao-existe.mp4", "--rights", "own"], "não encontrado"),
        (["status", "fantasma"], "projeto não encontrado"),
        (["plan", "show", "fantasma"], "projeto não encontrado"),
    ],
)
def test_user_errors_exit_2_with_clear_message(cwd, monkeypatch, capsys, args, needle):
    _, err = vvf(monkeypatch, capsys, *args, expect=2)
    assert needle in err


def test_invalid_rights_value_is_rejected_by_cli_parser(cwd, sample_video, monkeypatch, capsys):
    _, err = vvf(monkeypatch, capsys, "import", str(sample_video), "--rights", "roubado", expect=2)
    assert "roubado" in err or "invalid" in err.lower()


def test_failed_import_leaves_no_half_created_project(cwd, tmp_path, monkeypatch, capsys):
    bad = tmp_path / "ruim.mp4"
    bad.write_bytes(b"lixo" * 100)
    vvf(monkeypatch, capsys, "import", str(bad), "--rights", "own", "--slug", "ruim", expect=2)
    assert not Path("projects/ruim").exists()


def test_duplicate_slug_is_refused(analyzed, sample_video, monkeypatch, capsys):
    _, err = vvf(
        monkeypatch, capsys, "import", str(sample_video), "--rights", "own", "--slug", "demo", expect=2
    )
    assert "já existe" in err


def test_people_without_consent_warns(cwd, sample_video, monkeypatch, capsys):
    out, _ = vvf(
        monkeypatch, capsys, "import", str(sample_video), "--rights", "own", "--people", "--slug", "p"
    )
    assert "consentimento" in out


def test_third_party_rights_without_evidence_is_refused(cwd, sample_video, monkeypatch, capsys):
    _, err = vvf(
        monkeypatch, capsys, "import", str(sample_video), "--rights", "third_party_with_consent", expect=2
    )
    assert "evidence" in err or "consent" in err


def test_plan_flow_edit_approve_and_render_gate(analyzed, monkeypatch, capsys):
    vvf(monkeypatch, capsys, "plan", "generate", "demo", "--clips", "1")
    _, err = vvf(monkeypatch, capsys, "render", "demo", expect=2)  # plano ainda não aprovado
    assert "não aprovado" in err
    _, err = vvf(monkeypatch, capsys, "plan", "edit", "demo", "c1", "--end", "999", expect=2)
    assert "fora do vídeo" in err
    out, _ = vvf(monkeypatch, capsys, "plan", "edit", "demo", "c1", "--title", "Meu título")
    assert "Meu título" in out
    out, _ = vvf(monkeypatch, capsys, "plan", "approve", "demo")
    assert "aprovado" in out
    _, err = vvf(monkeypatch, capsys, "plan", "generate", "demo", expect=2)  # protege a edição
    assert "editado" in err


def test_render_dry_run_prints_ffmpeg_and_writes_nothing(analyzed, monkeypatch, capsys):
    vvf(monkeypatch, capsys, "plan", "generate", "demo", "--clips", "1")
    vvf(monkeypatch, capsys, "plan", "approve", "demo")
    out, _ = vvf(monkeypatch, capsys, "render", "demo", "--preset", "reels", "--dry-run")
    assert "ffmpeg" in out and "loudnorm" in out and "ass=" in out
    assert not list(Path("projects/demo/output").rglob("*.mp4"))
    _, err = vvf(monkeypatch, capsys, "render", "demo", "--preset", "inexistente", "--dry-run", expect=2)
    assert "preset desconhecido" in err


def test_render_real_then_cache_then_status_and_resume(analyzed, monkeypatch, capsys):
    vvf(monkeypatch, capsys, "plan", "generate", "demo", "--clips", "1")
    vvf(monkeypatch, capsys, "plan", "approve", "demo")
    out, _ = vvf(monkeypatch, capsys, "render", "demo", "--preset", "shorts", "--no-captions")
    assert "1080x1920" in out and Path("projects/demo/output/shorts/demo_c1.mp4").exists()
    out, _ = vvf(monkeypatch, capsys, "status", "demo", "--json")
    assert json.loads(out)["etapas"]["render:shorts:demo_c1"]["status"] == "done"
    out, _ = vvf(monkeypatch, capsys, "resume", "demo")
    assert "retomada" in out and "0 falha" in out


def test_transform_is_a_dry_run_by_default_and_never_calls_the_network(analyzed, monkeypatch, capsys):
    vvf(monkeypatch, capsys, "plan", "generate", "demo", "--clips", "1")
    # fal sem chave e sem rede: o esquema não pôde ser verificado (aviso), preço desconhecido
    monkeypatch.setattr("vvf.providers.fal.FalProvider.input_schema", lambda self, model: None)
    out, _ = vvf(
        monkeypatch, capsys, "transform", "run", "demo", "--clip", "c1", "--provider", "fal",
        "--model", "fal-ai/x/y", "--prompt", "estilo anime",
    )  # fmt: skip
    assert "simulação" in out and "desconhecido" in out and "nada foi enviado" in out
    _, err = vvf(
        monkeypatch, capsys, "transform", "run", "demo", "--clip", "c1", "--provider", "fal",
        "--model", "fal-ai/x/y", "--execute", expect=2,
    )  # fmt: skip
    assert "VVF_ALLOW_PAID_CALLS" in err
    _, err = vvf(
        monkeypatch, capsys, "transform", "run", "demo", "--clip", "c1", "--provider", "inexistente",
        "--model", "m", expect=2,
    )  # fmt: skip
    assert "provedor desconhecido" in err


def test_llm_plan_without_execute_only_estimates(analyzed, monkeypatch, capsys):
    out, _ = vvf(monkeypatch, capsys, "plan", "generate", "demo", "--engine", "llm")
    assert "estimativa" in out and "nada foi enviado" in out
    _, err = vvf(monkeypatch, capsys, "plan", "generate", "demo", "--engine", "llm", "--execute", expect=2)
    assert "ANTHROPIC_API_KEY" in err  # sem chave: recusa antes de qualquer chamada
    monkeypatch.setenv("ANTHROPIC_API_KEY", "chave-falsa-de-teste")  # nunca usada: o guarda barra antes
    _, err = vvf(monkeypatch, capsys, "plan", "generate", "demo", "--engine", "llm", "--execute", expect=2)
    assert "VVF_ALLOW_PAID_CALLS" in err
    _, err = vvf(monkeypatch, capsys, "plan", "generate", "demo", "--engine", "xyz", expect=2)
    assert "heuristic" in err


def test_transcript_correction_updates_captions_and_plan(analyzed, monkeypatch, capsys):
    vvf(monkeypatch, capsys, "plan", "generate", "demo", "--clips", "1")
    out, _ = vvf(monkeypatch, capsys, "transcript", "replace", "demo", "polvo", "pulpo")
    assert "1 ocorrência" in out
    out, _ = vvf(monkeypatch, capsys, "transcript", "show", "demo")
    assert "pulpo" in out and "polvo→pulpo" in out
    t = json.loads(Path("projects/demo/work/transcript.json").read_text())
    assert any(w["text"] == "pulpo" for s in t["segments"] for w in s["words"])
    plan = json.loads(Path("projects/demo/work/plan.json").read_text())
    assert "pulpo" in plan["clips"][0]["hook"] and plan["status"] == "draft"
    _, err = vvf(monkeypatch, capsys, "transcript", "replace", "demo", "inexistente", "x", expect=2)
    assert "não aparece" in err


def test_narrate_requires_text_and_valid_provider(analyzed, monkeypatch, capsys):
    vvf(monkeypatch, capsys, "plan", "generate", "demo", "--clips", "1")
    _, err = vvf(monkeypatch, capsys, "narrate", "demo", "--clip", "c1", expect=2)
    assert "--text" in err
    _, err = vvf(
        monkeypatch, capsys, "narrate", "demo", "--clip", "c1", "--text", "oi", "--provider", "x", expect=2
    )
    assert "desconhecido" in err
    out, _ = vvf(
        monkeypatch, capsys, "narrate", "demo", "--clip", "c1", "--text", "Olá",
        "--provider", "elevenlabs",
    )  # fmt: skip
    assert "simulação" in out  # pago => padrão é simular


def test_library_requires_license_and_attribution(cwd, tmp_path, monkeypatch, capsys):
    wav = tmp_path / "a.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=d=1", str(wav)], check=True
    )
    _, err = vvf(
        monkeypatch, capsys, "library", "add", str(wav), "--id", "x", "--kind", "music",
        "--license", "pirata", "--source", "web", expect=2,
    )  # fmt: skip
    assert "licença" in err
    _, err = vvf(
        monkeypatch, capsys, "library", "add", str(wav), "--id", "x", "--kind", "music",
        "--license", "CC-BY-4.0", "--source", "web", expect=2,
    )  # fmt: skip
    assert "attribution" in err
    out, _ = vvf(
        monkeypatch, capsys, "library", "add", str(wav), "--id", "x", "--kind", "music",
        "--license", "CC-BY-4.0", "--source", "web", "--attribution", "Autor X",
    )  # fmt: skip
    assert "adicionado" in out
    vvf(monkeypatch, capsys, "library", "synth")
    out, _ = vvf(monkeypatch, capsys, "library", "list")
    assert "pad-calm" in out and "CC-BY-4.0" in out


def test_doctor_and_models_run_without_credentials(cwd, monkeypatch, capsys):
    out, _ = vvf(monkeypatch, capsys, "doctor")
    assert "desligadas" in out and "ffmpeg" in out
    out, _ = vvf(monkeypatch, capsys, "transform", "models")
    assert "luma-dream-machine" in out and "2026-10-25" in out


def test_queue_commands(analyzed, monkeypatch, capsys):
    vvf(monkeypatch, capsys, "queue", "add", "demo", "analyze", "--args", '{"whisper_model": "tiny"}')
    out, _ = vvf(monkeypatch, capsys, "queue", "list")
    assert "analyze" in out and "queued" in out
    _, err = vvf(monkeypatch, capsys, "queue", "add", "fantasma", "analyze", expect=2)
    assert "projeto não encontrado" in err


def test_version_flag(cwd, monkeypatch, capsys):
    out, _ = vvf(monkeypatch, capsys, "--version")
    assert out.startswith("vvf ")


def test_provenance_object_reused_by_import(analyzed):
    m = Project.open(Path("projects"), "demo").manifest
    assert isinstance(m.provenance, Provenance) and m.provenance.rights == Rights.own
    assert import_video  # noqa: B015 - mantém o import usado nos testes de integração

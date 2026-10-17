import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import FakeTranscriber
from vvf.costs import CostGuard, Estimate
from vvf.errors import (
    BudgetExceeded,
    ConsentRequired,
    PaidCallsDisabled,
    ProviderError,
    UnknownPrice,
    UserInputError,
)
from vvf.models.manifest import Provenance, Rights
from vvf.pipeline import plan as planmod
from vvf.pipeline.ingest import import_video
from vvf.pipeline.transcribe import transcribe
from vvf.pipeline.transform import resume_transform, run_transform, use_transform
from vvf.providers.base import JobHandle, JobStatus


class FakeProvider:
    """Dublê de teste de um provedor. Conta chamadas; não gera vídeo de verdade."""

    name = "fake"

    def __init__(self, price=1.0, states=("queued", "running", "succeeded"), problems=(), creds=True):
        self.price, self.states, self.problems, self.creds = price, list(states), list(problems), creds
        self.submits = 0
        self.polls = 0

    def has_credentials(self):
        return self.creds

    def validate(self, req):
        return list(self.problems), []

    def estimate(self, req):
        return Estimate("fake", req.model, "video-transform", self.price, "preço de teste")

    def submit(self, req):
        self.submits += 1
        return JobHandle("fake", req.model, f"req-{self.submits}")

    def poll(self, handle):
        state = self.states[min(self.polls, len(self.states) - 1)]
        self.polls += 1
        if state == "succeeded":
            return JobStatus("succeeded", output_url="http://exemplo.invalido/saida.mp4")
        if state == "failed":
            return JobStatus("failed", "modelo recusou")
        return JobStatus(state)


@pytest.fixture
def ready(project, sample_video, provenance, settings, transcript):
    import_video(project, sample_video, provenance, settings)
    transcribe(project, settings, FakeTranscriber(transcript))
    planmod.generate_plan(project, settings, n_clips=1)
    return project


@pytest.fixture
def paid_settings(settings, tmp_path):
    return settings.model_copy(
        update={"vvf_allow_paid_calls": True, "vvf_max_spend_usd": 5.0, "vvf_cache_dir": tmp_path / "cache"}
    )


@pytest.fixture
def good_download(sample_video):
    def fake(url: str, dest: Path):  # entrega um vídeo válido (fixture), como se fosse a saída do provedor
        shutil.copy2(sample_video, dest)

    return fake


def _run(ready, settings, provider, download, **kw):
    kw.setdefault("clip_id", "c1")
    prompt = kw.pop("prompt", "estilo anime")
    return run_transform(
        ready, settings, provider, model="m1", prompt=prompt, download=download,
        sleep=lambda _: None, **kw,
    )  # fmt: skip


def test_dry_run_calls_nothing_and_spends_nothing(ready, paid_settings, good_download):
    p = FakeProvider()
    rec = _run(ready, paid_settings, p, good_download, execute=False)
    assert rec.status == "dry-run" and p.submits == 0 and p.polls == 0
    cost = ready.manifest.costs[-1]
    assert cost.dry_run and cost.estimated_usd == 1.0 and ready.manifest.spent_usd() == 0.0
    assert not ready.manifest.synthetic


def test_paid_calls_are_blocked_by_default(ready, settings, good_download):
    p = FakeProvider()
    with pytest.raises(PaidCallsDisabled):
        _run(ready, settings, p, good_download, execute=True)
    assert p.submits == 0


def test_unknown_price_is_refused_unless_user_sets_a_ceiling(ready, paid_settings, good_download):
    p = FakeProvider(price=None)
    with pytest.raises(UnknownPrice):
        _run(ready, paid_settings, p, good_download, execute=True)
    assert p.submits == 0
    rec = _run(ready, paid_settings, p, good_download, execute=True, price_usd=2.0)
    assert rec.status == "done" and p.submits == 1


def test_budget_ceiling_blocks_before_submitting(ready, paid_settings, good_download):
    p = FakeProvider(price=9.99)
    with pytest.raises(BudgetExceeded):
        _run(ready, paid_settings, p, good_download, execute=True)
    assert p.submits == 0


def test_budget_accumulates_across_calls_in_one_run(ready, paid_settings, good_download):
    guard = CostGuard(paid_settings, ready.manifest)
    first = FakeProvider(price=3.0)
    _run(ready, paid_settings, first, good_download, execute=True, guard=guard)
    second = FakeProvider(price=3.0)
    with pytest.raises(BudgetExceeded):  # 3 + 3 > teto de 5
        _run(ready, paid_settings, second, good_download, execute=True, guard=guard, prompt="outro estilo")
    assert second.submits == 0


def test_consent_required_when_people_present(
    project, sample_video, settings, transcript, paid_settings, good_download
):
    prov = Provenance(description="x", rights=Rights.own, people_present=True, consent_confirmed=False)
    import_video(project, sample_video, prov, settings)
    transcribe(project, settings, FakeTranscriber(transcript))
    planmod.generate_plan(project, settings, n_clips=1)
    p = FakeProvider()
    with pytest.raises(ConsentRequired):
        _run(project, paid_settings, p, good_download, execute=True)
    assert p.submits == 0
    _run(project, paid_settings, p, good_download, execute=False)  # simular é permitido


def test_success_records_everything_and_marks_synthetic(ready, paid_settings, good_download):
    p = FakeProvider()
    rec = _run(ready, paid_settings, p, good_download, execute=True)
    assert rec.status == "done" and rec.request_id == "req-1" and rec.sha256 and not rec.cached
    assert ready.path(rec.output).exists()
    assert ready.manifest.synthetic and "fake/m1" in ready.manifest.synthetic_reasons[0]
    assert ready.manifest.spent_usd() == 1.0
    use_transform(ready, "c1", rec.id)
    assert planmod.load_plan(ready).get("c1").source == f"transform:{rec.id}"


def test_identical_request_is_served_from_cache_without_paying(ready, paid_settings, good_download):
    p = FakeProvider()
    first = _run(ready, paid_settings, p, good_download, execute=True)
    second = _run(ready, paid_settings, p, good_download, execute=True)
    assert p.submits == 1 and second.cached and second.status == "done" and second.id != first.id
    assert ready.manifest.spent_usd() == 1.0  # só o primeiro foi cobrado


def test_timeout_then_resume_does_not_pay_twice(ready, paid_settings, good_download):
    p = FakeProvider(states=("running", "running", "succeeded"))
    with pytest.raises(ProviderError, match="vvf transform resume"):
        _run(ready, paid_settings, p, good_download, execute=True, timeout_s=-1)
    rec = ready.manifest.transforms[-1]
    assert rec.request_id == "req-1" and rec.status in ("submitted", "running")
    assert p.submits == 1
    done = resume_transform(ready, paid_settings, p, rec.id, sleep=lambda _: None, download=good_download)
    assert done.status == "done" and p.submits == 1  # nenhum novo envio


def test_provider_failure_is_recorded_and_cost_is_kept(ready, paid_settings, good_download):
    p = FakeProvider(states=("running", "failed"))
    with pytest.raises(ProviderError, match="recusou"):
        _run(ready, paid_settings, p, good_download, execute=True)
    rec = ready.manifest.transforms[-1]
    assert rec.status == "failed" and "recusou" in rec.error
    assert ready.manifest.costs[-1].estimated_usd == 1.0 and not ready.manifest.synthetic


def test_invalid_output_file_is_rejected(ready, paid_settings):
    def garbage(url: str, dest: Path):
        dest.write_bytes(b"isto nao e um video" * 100)

    with pytest.raises(UserInputError):
        _run(ready, paid_settings, FakeProvider(), garbage, execute=True)
    assert ready.manifest.transforms[-1].status == "failed"
    assert not ready.manifest.synthetic


def test_too_short_output_is_rejected(ready, paid_settings, tmp_path):
    tiny = tmp_path / "tiny.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=320x240:d=1:r=10",
         "-pix_fmt", "yuv420p", str(tiny)],
        check=True,
    )  # fmt: skip
    planmod.edit_clip(ready, "c1", start=0.0, end=7.0)

    def short(url: str, dest: Path):
        shutil.copy2(tiny, dest)

    with pytest.raises(ProviderError, match="curta"):
        _run(ready, paid_settings, FakeProvider(), short, execute=True)


def test_validation_problems_block_even_dry_run(ready, paid_settings, good_download):
    p = FakeProvider(problems=["parâmetro desconhecido: 'foo'"])
    with pytest.raises(UserInputError, match="foo"):
        _run(ready, paid_settings, p, good_download, execute=False)


def test_missing_credentials_do_not_submit(ready, paid_settings, good_download):
    from vvf.errors import MissingCredential

    p = FakeProvider(creds=False)
    with pytest.raises(MissingCredential):
        _run(ready, paid_settings, p, good_download, execute=True)
    assert p.submits == 0 and all(t.status != "pending" for t in ready.manifest.transforms)

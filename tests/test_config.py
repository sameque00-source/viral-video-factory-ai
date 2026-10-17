from pathlib import Path

import pytest

from vvf.config import Settings

SECRET_VARS = [
    "ANTHROPIC_API_KEY", "FAL_KEY", "REPLICATE_API_TOKEN", "ELEVENLABS_API_KEY",
    "LLM_BASE_URL", "LLM_API_KEY", "VVF_ALLOW_PAID_CALLS", "VVF_MAX_SPEND_USD",
]  # fmt: skip


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in SECRET_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # evita ler um .env real do desenvolvedor


def test_defaults_block_paid_calls() -> None:
    s = Settings()
    assert s.vvf_allow_paid_calls is False
    assert s.vvf_max_spend_usd == 5.0
    assert not any(s.configured_providers().values())


def test_secrets_are_not_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAL_KEY", "valor-secreto-de-teste")
    s = Settings()
    assert s.configured_providers()["fal"] is True
    assert "valor-secreto-de-teste" not in repr(s)
    assert "valor-secreto-de-teste" not in str(s.model_dump())


def test_negative_budget_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VVF_MAX_SPEND_USD", "-1")
    with pytest.raises(ValueError):
        Settings()

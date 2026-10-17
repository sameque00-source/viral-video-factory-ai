import json
import os
import stat
import subprocess
from pathlib import Path

import httpx
import pytest

from conftest import FakeTranscriber
from vvf.audio.tts import ElevenLabsTTS, PiperTTS, narrate
from vvf.errors import MissingCredential, PaidCallsDisabled, ProviderError, UnknownPrice, UserInputError
from vvf.pipeline import plan as planmod
from vvf.pipeline.ingest import import_video
from vvf.pipeline.transcribe import transcribe


@pytest.fixture
def ready(project, sample_video, provenance, settings, transcript):
    import_video(project, sample_video, provenance, settings)
    transcribe(project, settings, FakeTranscriber(transcript))
    planmod.generate_plan(project, settings, n_clips=1)
    return project


@pytest.fixture
def fake_piper(tmp_path) -> tuple[str, str]:
    """Binário 'piper' de teste: gera um WAV senoidal de duração proporcional ao texto."""
    voice = tmp_path / "voz.onnx"
    voice.write_bytes(b"x")
    script = tmp_path / "piper"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'while [ "$1" != "--" ]; do case "$1" in -f) out="$2";; esac; shift; done; shift\n'
        "secs=$(python3 -c \"print(max(0.5, len('$1')/15))\")\n"
        'ffmpeg -y -loglevel error -f lavfi -i "sine=f=300:d=$secs" -ar 22050 -ac 1 "$out"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(voice), str(script)


def test_piper_narration_is_local_free_synced_and_attached(ready, settings, fake_piper):
    voice, binary = fake_piper
    tts = PiperTTS(voice, binary)
    assert settings.vvf_allow_paid_calls is False  # local não precisa de chamadas pagas
    out = narrate(ready, settings, tts, clip_id="c1", text="Texto curto de narração.")
    assert out and out.exists()
    clip = planmod.load_plan(ready).get("c1")
    assert clip.narration == ready.rel(out) and clip.narration_mode == "duck"
    assert ready.manifest.synthetic and "narração" in ready.manifest.synthetic_reasons[0]
    assert ready.manifest.spent_usd() == 0.0


def test_narration_longer_than_clip_is_rejected(ready, settings, fake_piper):
    voice, binary = fake_piper
    with pytest.raises(UserInputError, match="encurte"):
        narrate(ready, settings, PiperTTS(voice, binary), clip_id="c1", text="palavra " * 120)


def test_empty_and_huge_texts_are_rejected(ready, settings, fake_piper):
    voice, binary = fake_piper
    tts = PiperTTS(voice, binary)
    with pytest.raises(UserInputError):
        narrate(ready, settings, tts, clip_id="c1", text="   ")
    with pytest.raises(UserInputError):
        narrate(ready, settings, tts, clip_id="c1", text="a" * 3000)


def test_piper_missing_binary_or_voice_gives_clear_error(ready, settings, tmp_path):
    tts = PiperTTS(str(tmp_path / "nao-existe.onnx"), "piper-inexistente")
    assert tts.has_credentials() is False
    with pytest.raises(MissingCredential, match="piper"):
        narrate(ready, settings, tts, clip_id="c1", text="oi")


def test_dry_run_estimates_without_synthesizing(ready, settings, fake_piper):
    voice, binary = fake_piper
    assert narrate(ready, settings, PiperTTS(voice, binary), clip_id="c1", text="oi", execute=False) is None
    assert ready.manifest.costs[-1].dry_run
    assert planmod.load_plan(ready).get("c1").narration is None


def _mp3_bytes(tmp_path: Path) -> bytes:
    mp3 = tmp_path / "x.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=400:d=1.5", str(mp3)], check=True
    )
    return mp3.read_bytes()


def _eleven(tmp_path, status=200, key="k", voice="v1"):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["key"] = str(request.url), request.headers.get("xi-api-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(status, content=_mp3_bytes(tmp_path) if status == 200 else b"erro")

    return ElevenLabsTTS(key, voice, http=httpx.Client(transport=httpx.MockTransport(handler))), seen


def test_elevenlabs_request_shape_and_wav_conversion(ready, settings, tmp_path):
    tts, seen = _eleven(tmp_path)
    paid = settings.model_copy(update={"vvf_allow_paid_calls": True})
    out = narrate(ready, paid, tts, clip_id="c1", text="Olá, mundo.", price_usd=0.01)
    assert seen["url"].startswith("https://api.elevenlabs.io/v1/text-to-speech/v1")
    assert seen["key"] == "k" and seen["body"] == {
        "text": "Olá, mundo.",
        "model_id": "eleven_multilingual_v2",
    }
    assert out.suffix == ".wav" and out.stat().st_size > 0
    assert ready.manifest.spent_usd() == 0.01


def test_elevenlabs_is_blocked_without_paid_flag_and_without_price(ready, settings, tmp_path):
    tts, seen = _eleven(tmp_path)
    with pytest.raises(PaidCallsDisabled):
        narrate(ready, settings, tts, clip_id="c1", text="oi")
    paid = settings.model_copy(update={"vvf_allow_paid_calls": True})
    with pytest.raises(UnknownPrice):
        narrate(ready, paid, tts, clip_id="c1", text="oi")
    assert "url" not in seen  # nenhuma requisição saiu


def test_elevenlabs_http_error_is_reported(ready, settings, tmp_path):
    tts, _ = _eleven(tmp_path, status=401)
    paid = settings.model_copy(update={"vvf_allow_paid_calls": True, "vvf_step_retries": 0})
    from vvf.errors import StepFailed

    with pytest.raises(StepFailed, match="401"):
        narrate(ready, paid, tts, clip_id="c1", text="oi", price_usd=0.01)


def test_elevenlabs_requires_key_and_voice():
    assert ElevenLabsTTS(None, None).has_credentials() is False
    with pytest.raises(MissingCredential):
        ElevenLabsTTS("k", None).synthesize("oi", Path("x.wav"))
    assert isinstance(ProviderError("x"), Exception)


@pytest.mark.models
def test_real_piper_voice_end_to_end(ready, settings):
    voice, binary = os.environ["VVF_PIPER_VOICE"], os.environ.get("VVF_PIPER_BIN", "piper")
    out = narrate(
        ready,
        settings,
        PiperTTS(voice, binary),
        clip_id="c1",
        text="Você sabia que o polvo tem três corações?",
    )
    assert out and out.stat().st_size > 1000

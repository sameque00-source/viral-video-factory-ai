import json
import subprocess

import pytest

from conftest import FakeTranscriber
from vvf.errors import IngestError
from vvf.models.manifest import Rights
from vvf.pipeline.ingest import import_video, make_provenance
from vvf.pipeline.scenes import audio_energy, compute_audio_energy, detect_scenes
from vvf.pipeline.transcribe import load_transcript, transcribe
from vvf.project import Project


def _ingest(project, sample_video, provenance, settings):
    return import_video(project, sample_video, provenance, settings)


def test_ingest_registers_source_provenance_and_audio(project, sample_video, provenance, settings):
    _ingest(project, sample_video, provenance, settings)
    m = project.manifest
    assert m.source.probe.width == 640 and m.source.probe.has_audio
    assert abs(m.source.probe.duration - 8.0) < 0.3
    assert m.provenance.rights == Rights.own
    assert (project.root / m.source.stored_path).exists()
    assert project.path(m.steps["ingest"].outputs["audio"]).stat().st_size > 0


def test_ingest_never_modifies_original(project, sample_video, provenance, settings):
    before = sample_video.read_bytes()
    _ingest(project, sample_video, provenance, settings)
    assert sample_video.read_bytes() == before


def test_ingest_rejects_unsupported_extension(project, tmp_path, provenance, settings):
    bogus = tmp_path / "a.txt"
    bogus.write_text("não é vídeo")
    with pytest.raises(IngestError, match="formato"):
        import_video(project, bogus, provenance, settings)


def test_ingest_rejects_corrupt_video(project, tmp_path, provenance, settings):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"isto nao e um mp4 valido" * 50)
    with pytest.raises(Exception, match="ffprobe"):
        import_video(project, bad, provenance, settings)


def test_ingest_rejects_missing_file(project, tmp_path, provenance, settings):
    with pytest.raises(IngestError, match="não encontrado"):
        import_video(project, tmp_path / "nada.mp4", provenance, settings)


def test_ingest_rejects_too_short(project, tmp_path, provenance, settings):
    short = tmp_path / "short.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=320x240:d=0.3",
         "-pix_fmt", "yuv420p", str(short)],
        check=True,
    )  # fmt: skip
    with pytest.raises(IngestError, match="curto"):
        import_video(project, short, provenance, settings)


def test_silent_video_warns(project, tmp_path, provenance, settings):
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=640x360:d=3",
         "-pix_fmt", "yuv420p", str(silent)],
        check=True,
    )  # fmt: skip
    warnings = import_video(project, silent, provenance, settings)
    assert any("sem trilha de áudio" in w for w in warnings)


def test_third_party_rights_need_evidence():
    with pytest.raises(IngestError):
        make_provenance(
            description="x", rights=Rights.third_party_with_consent, rights_evidence=None,
            people_present=True, consent_confirmed=False,
        )  # fmt: skip


def test_scene_detection_finds_three_scenes(project, sample_video, provenance, settings):
    _ingest(project, sample_video, provenance, settings)
    detect_scenes(project, settings)
    data = json.loads(project.path(project.manifest.steps["scenes"].outputs["scenes"]).read_text())
    assert len(data["scenes"]) == 3
    assert abs(data["scenes"][1]["start"] - 3.0) < 0.2


def test_scenes_is_cached_on_second_run(project, sample_video, provenance, settings):
    _ingest(project, sample_video, provenance, settings)
    assert detect_scenes(project, settings).cached is False
    assert detect_scenes(project, settings).cached is True


def test_audio_energy_distinguishes_silence_from_tone(project, sample_video, provenance, settings):
    _ingest(project, sample_video, provenance, settings)
    levels = audio_energy(project.path(project.manifest.steps["ingest"].outputs["audio"]))
    assert max(v["db"] for v in levels) > -30  # o tom de 330 Hz
    compute_audio_energy(project, settings)
    assert "energy" in project.manifest.steps


def test_transcribe_uses_backend_and_caches(project, sample_video, provenance, settings, transcript):
    _ingest(project, sample_video, provenance, settings)
    fake = FakeTranscriber(transcript)
    assert transcribe(project, settings, fake).cached is False
    assert transcribe(project, settings, fake).cached is True
    assert fake.calls == 1
    loaded = load_transcript(project)
    assert loaded.segments[0].text.startswith("Você sabia")
    assert loaded.segments[0].words[0].start == 0.0


def test_transcribe_without_audio_does_not_invent_speech(project, tmp_path, provenance, settings):
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=640x360:d=3",
         "-pix_fmt", "yuv420p", str(silent)],
        check=True,
    )  # fmt: skip
    import_video(project, silent, provenance, settings)
    fake = FakeTranscriber(None)  # se fosse chamado, quebraria
    transcribe(project, settings, fake)
    assert fake.calls == 0
    assert load_transcript(project).segments == []


def test_project_slug_validation(tmp_path):
    from vvf.errors import UserInputError

    with pytest.raises(UserInputError):
        Project.create(tmp_path, "../fuga")

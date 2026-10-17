import pytest

from vvf.errors import StepFailed, UserInputError
from vvf.models.manifest import StepStatus
from vvf.pipeline.runner import StepResult, is_interrupted, run_step


def _writer(project, counter, content="ok"):
    def fn():
        counter.append(1)
        out = project.root / "work" / "out.txt"
        out.write_text(content)
        return StepResult({"out": project.rel(out)})

    return fn


def test_step_runs_then_hits_cache(project):
    calls: list[int] = []
    first = run_step(project, "s", {"a": 1}, _writer(project, calls))
    second = run_step(project, "s", {"a": 1}, _writer(project, calls))
    assert (first.cached, second.cached) == (False, True)
    assert len(calls) == 1
    assert project.manifest.steps["s"].status == StepStatus.done


def test_changed_inputs_invalidate_cache(project):
    calls: list[int] = []
    run_step(project, "s", {"a": 1}, _writer(project, calls))
    run_step(project, "s", {"a": 2}, _writer(project, calls))
    assert len(calls) == 2


def test_missing_output_forces_rerun(project):
    calls: list[int] = []
    run_step(project, "s", {"a": 1}, _writer(project, calls))
    (project.root / "work" / "out.txt").unlink()
    run_step(project, "s", {"a": 1}, _writer(project, calls))
    assert len(calls) == 2


def test_retries_then_succeeds(project):
    attempts = []
    sleeps = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("rede caiu")
        out = project.root / "work" / "o.txt"
        out.write_text("x")
        return StepResult({"o": project.rel(out)})

    out = run_step(project, "s", {}, flaky, retries=3, sleep=sleeps.append)
    assert not out.cached and len(attempts) == 3
    assert sleeps == [1, 2]  # backoff exponencial


def test_gives_up_after_retry_limit(project):
    def always_fails():
        raise RuntimeError("sempre")

    with pytest.raises(StepFailed):
        run_step(project, "s", {}, always_fails, retries=1, sleep=lambda _: None)
    rec = project.manifest.steps["s"]
    assert rec.status == StepStatus.failed and rec.attempts == 2 and "sempre" in rec.error


def test_non_retryable_error_is_not_retried(project):
    attempts = []

    def bad_input():
        attempts.append(1)
        raise UserInputError("entrada inválida")

    with pytest.raises(StepFailed):
        run_step(project, "s", {}, bad_input, retries=5, sleep=lambda _: None)
    assert len(attempts) == 1


def test_empty_output_is_rejected(project):
    def writes_empty():
        out = project.root / "work" / "empty.txt"
        out.write_text("")
        return StepResult({"o": project.rel(out)})

    with pytest.raises(StepFailed):
        run_step(project, "s", {}, writes_empty, retries=0)


def test_interrupted_step_is_detected_and_rerun(project):
    rec = project.manifest.step("s")
    rec.status, rec.pid = StepStatus.running, 999_999_999  # processo que não existe
    assert is_interrupted(rec)
    calls: list[int] = []
    run_step(project, "s", {"a": 1}, _writer(project, calls))
    assert len(calls) == 1 and rec.status == StepStatus.done


def test_manifest_persisted_on_disk(project):
    run_step(project, "s", {"a": 1}, _writer(project, []))
    from vvf.project import Project

    reopened = Project.open(project.root.parent, "demo")
    assert reopened.manifest.steps["s"].status == StepStatus.done


def test_every_file_in_presets_dir_is_a_valid_loadable_preset():
    """Regressão: um JSON de outra natureza em presets/ quebrava o `export` de todos os presets."""
    from vvf import presets

    names = presets.available()
    assert {"tiktok", "reels", "shorts"} <= set(names)
    for name in names:
        assert presets.load_preset(name).name == name

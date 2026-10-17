import os

import pytest

from vvf.errors import UserInputError
from vvf.jobs import JobQueue, run_queue, run_recorded


@pytest.fixture
def queue(tmp_path):
    return JobQueue(tmp_path / "q.sqlite")


def _ok(settings, job):
    pass


def test_jobs_run_in_order_and_are_marked_done(queue, settings):
    ids = [queue.add("a", "analyze", {"n": i}) for i in range(3)]
    done = []
    summary = run_queue(settings, queue, executor=lambda s, j: done.append(j.id))
    assert done == ids and summary["done"] == 3 and summary["failed"] == 0
    assert {j.status for j in queue.list()} == {"done"}


def test_failure_does_not_stop_other_jobs_and_is_retried_by_resume(queue, settings):
    a = queue.add("x", "analyze", {})
    b = queue.add("x", "analyze", {})
    calls = {a: 0, b: 0}

    def flaky(s, job):
        calls[job.id] += 1
        if job.id == a and calls[a] == 1:
            raise RuntimeError("falha de rede")

    first = run_queue(settings, queue, executor=flaky, sleep=lambda _: None)
    assert first == {"done": 1, "failed": 1, "recovered": 0}
    assert queue.get(a).status == "failed" and "falha de rede" in queue.get(a).error
    assert queue.get(b).status == "done"
    second = run_queue(settings, queue, include_failed=True, executor=flaky, sleep=lambda _: None)
    assert second["done"] == 1 and queue.get(a).status == "done" and calls == {a: 2, b: 1}


def test_attempts_are_limited(queue, settings):
    job = queue.add("x", "analyze", {}, max_attempts=2)

    def always(s, j):
        raise RuntimeError("sempre")

    for _ in range(4):
        run_queue(settings, queue, include_failed=True, executor=always, sleep=lambda _: None)
    assert queue.get(job).attempts == 2 and queue.get(job).status == "failed"


def test_interrupted_job_is_recovered(queue, settings):
    job = queue.add("x", "analyze", {}, status="running")
    queue._set(job, pid=999_999_999, attempts=1)  # processo que não existe mais
    summary = run_queue(settings, queue, executor=_ok)
    assert summary["recovered"] == 1 and summary["done"] == 1


def test_live_running_job_is_not_stolen(queue, settings):
    job = queue.add("x", "analyze", {}, status="running")
    queue._set(job, pid=os.getpid())
    assert queue.recover_stale() == 0


def test_filter_by_slug_and_max_jobs(queue, settings):
    queue.add("a", "analyze", {})
    queue.add("b", "analyze", {})
    queue.add("b", "analyze", {})
    assert run_queue(settings, queue, slug="b", max_jobs=1, executor=_ok)["done"] == 1
    assert [j.status for j in queue.list("b")] == ["done", "queued"]
    assert queue.list("a")[0].status == "queued"


def test_unknown_job_kind_and_missing_job(queue, settings):
    with pytest.raises(UserInputError):
        queue.get(99)
    job = queue.add("x", "inexistente", {})
    summary = run_queue(settings, queue, sleep=lambda _: None)
    assert summary["failed"] == 1 and "tipo de tarefa desconhecido" in queue.get(job).error
    missing = queue.add("fantasma", "analyze", {})
    run_queue(settings, queue, sleep=lambda _: None)
    assert "projeto não encontrado" in queue.get(missing).error


def test_run_recorded_marks_failure_and_reraises(queue, settings, monkeypatch):
    from vvf import jobs

    monkeypatch.setitem(jobs.HANDLERS, "boom", lambda s, p, a: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(jobs, "execute_job", lambda s, j: jobs.HANDLERS[j.kind](s, None, j.args))
    with pytest.raises(RuntimeError):
        run_recorded(settings, queue, "x", "boom", {})
    assert queue.list()[0].status == "failed"


def test_project_lock_blocks_a_second_process_and_is_released_on_death(project, tmp_path):
    """Dois processos no mesmo projeto corromperiam o manifesto: o segundo deve falhar com mensagem clara."""
    import subprocess
    import sys
    import textwrap

    from vvf.errors import UserInputError

    code = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, "src")
        from pathlib import Path
        from vvf.project import project_lock
        with project_lock(Path({str(project.root)!r}), "demo"):
            print("locked", flush=True)
            time.sleep(30)
        """
    )
    holder = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(UserInputError, match="em uso por outro processo"):
            with project.lock():
                pass
        holder.kill()  # morte abrupta (kill -9): o lock precisa sumir sozinho
        holder.wait()
        with project.lock():
            pass
    finally:
        holder.kill()


def test_lock_is_reentrant_across_sequential_use(project):
    with project.lock():
        pass
    with project.lock():
        pass


def test_resume_drops_force_so_finished_work_is_reused(queue, settings):
    queue.add("x", "render", {"force": True, "preset": "tiktok"})
    seen = []
    run_queue(settings, queue, resume=True, executor=lambda s, j: seen.append(j.args))
    assert seen == [{"force": False, "preset": "tiktok"}]
    queue.add("x", "render", {"force": True})
    seen.clear()
    run_queue(settings, queue, executor=lambda s, j: seen.append(j.args))  # execução normal respeita o force
    assert seen == [{"force": True}]

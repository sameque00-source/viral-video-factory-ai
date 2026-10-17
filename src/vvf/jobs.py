"""Fila de tarefas persistente (SQLite) + execução registrada para retomar trabalho interrompido.

Todo comando pesado (analyze, render, export) vira uma linha em `projects/.queue.sqlite`:
- `queue run` processa a fila; `resume` repete o que falhou ou foi interrompido (sem refazer
  etapas já concluídas, graças ao cache por hash do executor de etapas);
- tentativas são limitadas por `max_attempts`.
Transformações pagas NÃO passam por aqui: reenviar cobraria duas vezes (use `transform resume`).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vvf.config import Settings
from vvf.errors import UserInputError
from vvf.models.manifest import utcnow
from vvf.project import Project, project_lock

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  slug TEXT NOT NULL, kind TEXT NOT NULL, args TEXT NOT NULL,
  status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 3,
  pid INTEGER, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
)"""


@dataclass
class Job:
    id: int
    slug: str
    kind: str
    args: dict[str, Any]
    status: str
    attempts: int
    max_attempts: int
    error: str | None


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class JobQueue:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(self._conn()) as c:
            c.execute(SCHEMA)
            c.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        return Job(row["id"], row["slug"], row["kind"], json.loads(row["args"]), row["status"],
                   row["attempts"], row["max_attempts"], row["error"])  # fmt: skip

    def add(self, slug: str, kind: str, args: dict, *, max_attempts: int = 3, status: str = "queued") -> int:
        now = utcnow()
        with closing(self._conn()) as c:
            cur = c.execute(
                "INSERT INTO jobs(slug,kind,args,status,max_attempts,created_at,updated_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (slug, kind, json.dumps(args, sort_keys=True), status, max_attempts, now, now),
            )
            c.commit()
            return int(cur.lastrowid)

    def _set(self, job_id: int, **fields: Any) -> None:
        fields["updated_at"] = utcnow()
        sets = ", ".join(f"{k}=?" for k in fields)
        with closing(self._conn()) as c:
            c.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*fields.values(), job_id))
            c.commit()

    def get(self, job_id: int) -> Job:
        with closing(self._conn()) as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise UserInputError(f"tarefa inexistente: {job_id}")
        return self._job(row)

    def list(self, slug: str | None = None, status: str | None = None) -> list[Job]:
        q, params = "SELECT * FROM jobs WHERE 1=1", []
        if slug:
            q, params = q + " AND slug=?", [*params, slug]
        if status:
            q, params = q + " AND status=?", [*params, status]
        with closing(self._conn()) as c:
            return [self._job(r) for r in c.execute(q + " ORDER BY id", params)]

    def recover_stale(self, slug: str | None = None) -> int:
        """`running` sem processo vivo = interrompida: volta para a fila."""
        n = 0
        with closing(self._conn()) as c:
            rows = c.execute("SELECT id, pid, slug FROM jobs WHERE status='running'").fetchall()
        for r in rows:
            if (slug is None or r["slug"] == slug) and not _pid_alive(r["pid"]):
                self._set(r["id"], status="queued", pid=None, error="interrompida; reenfileirada")
                n += 1
        return n

    def claim(
        self, slug: str | None = None, include_failed: bool = False, exclude: set[int] | None = None
    ) -> Job | None:
        statuses = ("queued", "failed") if include_failed else ("queued",)
        marks = ",".join("?" * len(statuses))
        q = f"SELECT * FROM jobs WHERE status IN ({marks}) AND attempts < max_attempts"
        params: list[Any] = list(statuses)
        if slug:
            q, params = q + " AND slug=?", [*params, slug]
        if exclude:
            q += f" AND id NOT IN ({','.join('?' * len(exclude))})"
            params += list(exclude)
        with closing(self._conn()) as c:
            row = c.execute(q + " ORDER BY id LIMIT 1", params).fetchone()
        if row is None:
            return None
        self._set(row["id"], status="running", pid=os.getpid(), attempts=row["attempts"] + 1)
        return self.get(row["id"])

    def finish(self, job_id: int) -> None:
        self._set(job_id, status="done", pid=None, error=None)

    def fail(self, job_id: int, error: str) -> None:
        self._set(job_id, status="failed", pid=None, error=error[:500])


# ---------- execução ----------

Handler = Callable[[Settings, Project, dict], None]
HANDLERS: dict[str, Handler] = {}


def handler(kind: str):
    def deco(fn: Handler) -> Handler:
        HANDLERS[kind] = fn
        return fn

    return deco


def execute_job(settings: Settings, job: Job) -> None:
    from vvf import handlers  # noqa: F401  (registra analyze/render/export; evita import circular)

    fn = HANDLERS.get(job.kind)
    if fn is None:
        raise UserInputError(f"tipo de tarefa desconhecido: {job.kind}")
    root = Path(settings.vvf_projects_dir)
    if not (root / job.slug / "manifest.json").exists():
        raise UserInputError(f"projeto não encontrado: {job.slug}")
    with project_lock(root / job.slug, job.slug):  # trava ANTES de ler o manifesto
        fn(settings, Project.open(root, job.slug), job.args)


def run_queue(
    settings: Settings, queue: JobQueue, *, slug: str | None = None, include_failed: bool = False,
    max_jobs: int | None = None, resume: bool = False,
    executor: Callable[[Settings, Job], None] = execute_job,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, int]:  # fmt: skip
    """Processa tarefas em ordem. Falha de uma não interrompe as outras."""
    summary = {"done": 0, "failed": 0, "recovered": queue.recover_stale(slug)}
    failed_now: set[int] = set()  # cada tarefa tenta no máximo uma vez por chamada
    while max_jobs is None or summary["done"] + summary["failed"] < max_jobs:
        job = queue.claim(slug, include_failed, failed_now)
        if job is None:
            break
        if resume:  # retomar = reaproveitar o que já está pronto, mesmo que o original usasse --force
            job.args = {**job.args, "force": False}
        try:
            executor(settings, job)
        except Exception as exc:  # noqa: BLE001 - registra e segue para a próxima
            queue.fail(job.id, f"{type(exc).__name__}: {exc}")
            summary["failed"] += 1
            failed_now.add(job.id)
            if getattr(exc, "retryable", True):
                sleep(min(2**job.attempts, 30))
        else:
            queue.finish(job.id)
            summary["done"] += 1
    return summary


def run_recorded(settings: Settings, queue: JobQueue, slug: str, kind: str, args: dict) -> None:
    """Executa agora, mas deixa o registro na fila para `resume` se algo falhar ou for interrompido."""
    job_id = queue.add(slug, kind, args, status="running")
    queue._set(job_id, pid=os.getpid(), attempts=1)
    try:
        execute_job(settings, queue.get(job_id))
    except BaseException as exc:
        queue.fail(job_id, f"{type(exc).__name__}: {exc}")
        raise
    queue.finish(job_id)

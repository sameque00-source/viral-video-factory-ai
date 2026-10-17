"""Workspace de um vídeo: projects/<slug>/{original,work,output,logs}/ + manifest.json."""

from __future__ import annotations

import fcntl
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from vvf.errors import UserInputError
from vvf.models.manifest import Manifest, load_manifest, save_manifest

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SUBDIRS = ("original", "work", "output", "logs")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-_")
    return slug[:64] or "video"


@contextmanager
def project_lock(root: Path, slug: str) -> Iterator[None]:
    """Exclusão mútua entre processos no mesmo projeto (flock: some sozinho se o processo morrer)."""
    fd = os.open(root / ".lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = os.pread(fd, 32, 0).decode(errors="ignore").strip() or "?"
            raise UserInputError(
                f"o projeto '{slug}' está em uso por outro processo (pid {holder}); aguarde terminar"
            ) from None
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)  # fechar o descritor libera o lock


class Project:
    def __init__(self, root: Path, manifest: Manifest):
        self.root = root
        self.manifest = manifest

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def slug(self) -> str:
        return self.manifest.slug

    def path(self, rel: str | Path) -> Path:
        """Resolve um caminho relativo ao projeto, recusando fugas do diretório."""
        p = (self.root / rel).resolve()
        if not p.is_relative_to(self.root.resolve()):
            raise UserInputError(f"caminho fora do projeto: {rel}")
        return p

    def rel(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root.resolve()))

    def lock(self):
        return project_lock(self.root, self.slug)

    def save(self) -> None:
        save_manifest(self.manifest, self.manifest_path)

    @classmethod
    def create(cls, projects_dir: Path, slug: str) -> Project:
        if not SLUG_RE.match(slug):
            raise UserInputError(f"slug inválido: {slug!r} (use a-z, 0-9, '-' e '_')")
        root = projects_dir / slug
        if (root / "manifest.json").exists():
            raise UserInputError(f"projeto já existe: {slug}")
        for sub in SUBDIRS:
            (root / sub).mkdir(parents=True, exist_ok=True)
        project = cls(root, Manifest(slug=slug))
        project.save()
        return project

    @classmethod
    def open(cls, projects_dir: Path, slug: str) -> Project:
        root = projects_dir / slug
        if not SLUG_RE.match(slug) or not (root / "manifest.json").exists():
            raise UserInputError(f"projeto não encontrado: {slug}")
        return cls(root, load_manifest(root / "manifest.json"))

    @classmethod
    def list_slugs(cls, projects_dir: Path) -> list[str]:
        if not projects_dir.exists():
            return []
        return sorted(p.parent.name for p in projects_dir.glob("*/manifest.json"))

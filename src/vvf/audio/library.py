"""Biblioteca de músicas e efeitos. Só entra item com licença declarada."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel

from vvf import media
from vvf.errors import UserInputError
from vvf.models.manifest import utcnow

ALLOWED_LICENSES = {
    "CC0", "public-domain", "CC-BY-4.0", "CC-BY-SA-4.0", "own", "licensed", "generated-by-vvf",
}  # fmt: skip
NEEDS_ATTRIBUTION = {"CC-BY-4.0", "CC-BY-SA-4.0"}


class LibraryItem(BaseModel):
    id: str
    kind: str  # "music" | "sfx"
    file: str  # relativo à biblioteca
    title: str
    license: str
    source: str
    attribution: str = ""
    sha256: str
    duration: float
    added_at: str


class Library:
    def __init__(self, root: Path):
        self.root = root
        self.index_path = root / "index.json"

    def _read(self) -> list[LibraryItem]:
        if not self.index_path.exists():
            return []
        return [LibraryItem.model_validate(x) for x in json.loads(self.index_path.read_text())]

    def _write(self, items: list[LibraryItem]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = [i.model_dump() for i in items]
        self.index_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def list(self, kind: str | None = None) -> list[LibraryItem]:
        return [i for i in self._read() if kind in (None, i.kind)]

    def get(self, item_id: str) -> LibraryItem:
        for item in self._read():
            if item.id == item_id:
                return item
        raise UserInputError(f"item de áudio não encontrado na biblioteca: {item_id}")

    def path(self, item: LibraryItem) -> Path:
        return self.root / item.file

    def add(
        self, src: Path, *, item_id: str, kind: str, title: str, license: str, source: str,
        attribution: str = "",
    ) -> LibraryItem:  # fmt: skip
        if kind not in ("music", "sfx"):
            raise UserInputError("kind deve ser 'music' ou 'sfx'")
        if license not in ALLOWED_LICENSES:
            raise UserInputError(f"licença '{license}' não aceita; use uma de {sorted(ALLOWED_LICENSES)}")
        if license in NEEDS_ATTRIBUTION and not attribution:
            raise UserInputError(f"a licença {license} exige --attribution")
        if not source.strip():
            raise UserInputError("informe a origem do áudio (--source)")
        if not src.is_file():
            raise UserInputError(f"arquivo não encontrado: {src}")
        items = self._read()
        if any(i.id == item_id for i in items):
            raise UserInputError(f"já existe um item com id '{item_id}'")
        info = media.probe_audio(src)
        files = self.root / "files"
        files.mkdir(parents=True, exist_ok=True)
        dest = files / f"{item_id}{src.suffix.lower()}"
        shutil.copy2(src, dest)
        item = LibraryItem(
            id=item_id, kind=kind, file=str(dest.relative_to(self.root)), title=title,
            license=license, source=source, attribution=attribution,
            sha256=media.sha256_file(dest), duration=info, added_at=utcnow(),
        )  # fmt: skip
        self._write([*items, item])
        return item

    def synth_builtin(self) -> list[LibraryItem]:
        """Gera efeitos/trilha simples com FFmpeg (autoria do projeto, licença generated-by-vvf)."""
        recipes = {
            "whoosh": ("sfx", "Whoosh (ruído filtrado)",
                       ["-f", "lavfi", "-i", "anoisesrc=d=0.8:c=pink:a=0.5",
                        "-af", "highpass=f=300,lowpass=f=6000,afade=t=in:d=0.35,afade=t=out:st=0.35:d=0.45"]),
            "pop": ("sfx", "Pop (seno curto)",
                    ["-f", "lavfi", "-i", "sine=f=880:d=0.18",
                     "-af", "afade=t=out:st=0.04:d=0.14,volume=0.6"]),
            "pad-calm": ("music", "Base ambiente calma (acordes de senos)",
                         ["-f", "lavfi", "-i",
                          "aevalsrc='0.18*sin(2*PI*220*t)+0.14*sin(2*PI*277.18*t)+0.12*sin(2*PI*329.63*t)':d=30:s=48000",
                          "-af", "lowpass=f=2000,afade=t=in:d=2,afade=t=out:st=27:d=3"]),
        }  # fmt: skip
        made = []
        existing = {i.id for i in self._read()}
        tmp = self.root / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        for item_id, (kind, title, args) in recipes.items():
            if item_id in existing:
                continue
            out = tmp / f"{item_id}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", *args, "-ar", "48000", "-ac", "2", str(out)],
                check=True,
            )  # fmt: skip
            made.append(
                self.add(out, item_id=item_id, kind=kind, title=title,
                         license="generated-by-vvf", source="gerado localmente por vvf (FFmpeg)")
            )  # fmt: skip
        shutil.rmtree(tmp, ignore_errors=True)
        return made

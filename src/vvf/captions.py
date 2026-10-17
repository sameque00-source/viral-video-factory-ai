"""Legendas estilizadas (ASS) com destaque da palavra falada e de palavras importantes."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pysubs2
from pysubs2 import Color, SSAEvent, SSAFile, SSAStyle

from vvf.pipeline.plan import HOOK_WORDS, Clip
from vvf.pipeline.transcribe import Transcript
from vvf.presets import Preset

ACTIVE = Color(255, 220, 0)  # amarelo: palavra sendo falada
IMPORTANT = Color(0, 220, 255)  # ciano: números e palavras de gancho
WHITE = Color(255, 255, 255)


def ass_color(c: Color) -> str:
    """Cor para override inline do ASS (ordem BGR)."""
    return f"&H{c.b:02X}{c.g:02X}{c.r:02X}&"


@dataclass
class CapWord:
    text: str
    start: float
    end: float


def _is_important(word: str) -> bool:
    bare = re.sub(r"[^\w]", "", word.lower())
    return bool(re.search(r"\d", bare)) or bare in HOOK_WORDS


def _clean(text: str) -> str:
    """Remove chaves/barras que o ASS interpretaria como comandos."""
    return re.sub(r"[{}\\]", "", text)


def collect_words(transcript: Transcript, clip: Clip, offset: float) -> list[CapWord]:
    """Palavras reais da transcrição dentro do trecho, com tempo relativo ao vídeo final."""
    words = []
    for seg in transcript.segments:
        for w in seg.words:
            mid = (w.start + w.end) / 2
            if clip.start <= mid < clip.end and w.text.strip():
                s = max(0.0, w.start - clip.start) + offset
                e = min(clip.end, w.end) - clip.start + offset
                words.append(CapWord(w.text.strip(), s, max(e, s + 0.05)))
    return words


def group_words(words: list[CapWord], max_chars: int = 22, max_words: int = 4) -> list[list[CapWord]]:
    groups: list[list[CapWord]] = []
    cur: list[CapWord] = []
    for w in words:
        too_long = sum(len(x.text) + 1 for x in cur) + len(w.text) > max_chars
        gap = cur and (w.start - cur[-1].end) > 0.6
        ended = cur and cur[-1].text[-1:] in ".!?…"
        if cur and (len(cur) >= max_words or too_long or gap or ended):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def build_ass(
    words_by_clip: list[list[CapWord]],
    preset: Preset,
    *,
    highlight: bool = True,
    hooks: list[tuple[float, float, str]] | None = None,
) -> SSAFile:
    subs = SSAFile()
    subs.info["PlayResX"], subs.info["PlayResY"] = str(preset.width), str(preset.height)
    subs.info["WrapStyle"] = "0"  # quebra automática: ganchos longos não vazam das margens
    subs.styles["Default"] = SSAStyle(
        fontname=preset.caption_font, fontsize=preset.caption_size, bold=True,
        primarycolor=WHITE, outlinecolor=Color(0, 0, 0), backcolor=Color(0, 0, 0, 120),
        outline=7, shadow=2,
        alignment=pysubs2.Alignment.BOTTOM_CENTER, marginl=80, marginr=80, marginv=preset.caption_safe_bottom,
    )  # fmt: skip
    subs.styles["Hook"] = SSAStyle(
        fontname=preset.caption_font, fontsize=int(preset.caption_size * 0.9), bold=True,
        primarycolor=WHITE, outlinecolor=Color(0, 0, 0), outline=8, shadow=0,
        alignment=pysubs2.Alignment.TOP_CENTER,
        marginl=70, marginr=70, marginv=int(preset.height * 0.12),
    )  # fmt: skip
    for words in words_by_clip:
        for group in group_words(words):
            for k, active in enumerate(group):
                end = group[k + 1].start if k + 1 < len(group) else active.end + 0.15
                parts = []
                for m, w in enumerate(group):
                    text = _clean(w.text).upper()
                    if not highlight:
                        parts.append(text)
                    elif m == k:
                        parts.append(f"{{\\c{ass_color(ACTIVE)}\\fscx112\\fscy112}}{text}{{\\r}}")
                    elif _is_important(w.text):
                        parts.append(f"{{\\c{ass_color(IMPORTANT)}}}{text}{{\\r}}")
                    else:
                        parts.append(text)
                subs.append(
                    SSAEvent(
                        start=pysubs2.make_time(s=active.start),
                        end=pysubs2.make_time(s=max(end, active.start + 0.05)),
                        text=" ".join(parts), style="Default",
                    )
                )  # fmt: skip
    for start, end, text in hooks or []:
        subs.append(
            SSAEvent(
                start=pysubs2.make_time(s=start), end=pysubs2.make_time(s=end),
                text=_clean(text), style="Hook",
            )
        )  # fmt: skip
    return subs

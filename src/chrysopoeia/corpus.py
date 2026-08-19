"""Streaming reader for the guru-corpus Postgres dump (``guru-corpus.sql.gz``).

The dump is a single-transaction ``pg_dump``-style artifact built by the
guru-pipeline ``export.py``. Tables are emitted as ``COPY ... FROM STDIN``
blocks in the text (tab-delimited) format. We do not need a running Postgres
to read it: this module streams the gzip and decodes the COPY blocks directly,
which is enough to pull the prose we soak on.

Only the columns Chrysopoeia cares about are documented, but every column is
returned so downstream code can pick what it needs.

Reference: docs/chrysopoeia-design.md §2 (retrieval-as-compiler), §5.2.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# COPY text-format escape sequences (see Postgres docs, "Text Format").
_UNESCAPE = {
    "\\": "\\",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
}


def _unescape_field(raw: str) -> str | None:
    r"""Decode one COPY text-format field. ``\N`` is SQL NULL -> ``None``."""
    if raw == "\\N":
        return None
    if "\\" not in raw:
        return raw
    out: list[str] = []
    it = iter(range(len(raw)))
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch == "\\" and i + 1 < n:
            nxt = raw[i + 1]
            out.append(_UNESCAPE.get(nxt, nxt))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _parse_pg_array(raw: str | None) -> list[str] | None:
    """Parse a Postgres ``text[]`` literal like ``{a,b,"c,d"}`` into a list."""
    if raw is None:
        return None
    s = raw.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return [s] if s else []
    s = s[1:-1]
    if not s:
        return []
    out: list[str] = []
    buf: list[str] = []
    in_q = False
    i = 0
    while i < len(s):
        c = s[i]
        if in_q:
            if c == "\\" and i + 1 < len(s):
                buf.append(s[i + 1])
                i += 2
                continue
            if c == '"':
                in_q = False
            else:
                buf.append(c)
        elif c == '"':
            in_q = True
        elif c == ",":
            out.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    out.append("".join(buf))
    return out


def _copy_header_columns(line: str) -> tuple[str, list[str]] | None:
    """If ``line`` opens a COPY block, return ``(table, [columns])``."""
    if not line.startswith("COPY "):
        return None
    # COPY corpus_new.chunks (id, text_id, ...) FROM STDIN;
    try:
        after = line[len("COPY "):]
        table = after.split(" ", 1)[0]
        cols_part = after[after.index("(") + 1 : after.index(")")]
    except ValueError:
        return None
    columns = [c.strip() for c in cols_part.split(",")]
    return table, columns


def iter_table(gz_path: str | Path, table: str) -> Iterator[dict[str, str | None]]:
    """Yield each row of ``table`` as a column->value dict (NULLs as ``None``).

    ``table`` may be given with or without the ``corpus_new.`` schema prefix.
    Array columns are returned as their raw literal; call :func:`_parse_pg_array`
    or use the typed helpers below if you need them split.
    """
    if "." not in table:
        table = f"corpus_new.{table}"
    gz_path = Path(gz_path)
    with gzip.open(gz_path, "rt", encoding="utf-8", newline="\n") as fh:
        columns: list[str] | None = None
        for line in fh:
            if columns is None:
                header = _copy_header_columns(line)
                if header and header[0] == table:
                    columns = header[1]
                continue
            # inside the target COPY block
            if line.startswith("\\."):
                return
            row = line.rstrip("\n").split("\t")
            if len(row) != len(columns):
                # tolerate the rare short line rather than corrupting alignment
                continue
            yield {col: _unescape_field(val) for col, val in zip(columns, row)}


# ─── typed row helpers ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Chunk:
    """One citation-addressable prose unit — the atom we soak on."""

    id: str
    text_id: str
    tradition: str
    text_name: str
    section: str | None
    translator: str | None
    body: str
    token_count: int

    @classmethod
    def from_row(cls, r: dict[str, str | None]) -> "Chunk":
        return cls(
            id=r["id"],
            text_id=r["text_id"],
            tradition=r["tradition"],
            text_name=r["text_name"],
            section=r.get("section"),
            translator=r.get("translator"),
            body=r["body"],
            token_count=int(r["token_count"]) if r.get("token_count") else 0,
        )


def iter_chunks(gz_path: str | Path) -> Iterator[Chunk]:
    """Yield every :class:`Chunk` in the corpus (embedding column dropped)."""
    for r in iter_table(gz_path, "chunks"):
        yield Chunk.from_row(r)


@dataclass(frozen=True)
class Text:
    id: str
    tradition: str
    label: str
    translator: str | None
    source_url: str | None

    @classmethod
    def from_row(cls, r: dict[str, str | None]) -> "Text":
        return cls(
            id=r["id"],
            tradition=r["tradition"],
            label=r["label"],
            translator=r.get("translator"),
            source_url=r.get("source_url"),
        )


def load_texts(gz_path: str | Path) -> dict[str, Text]:
    return {r["id"]: Text.from_row(r) for r in iter_table(gz_path, "texts")}


def load_traditions(gz_path: str | Path) -> dict[str, str]:
    """tradition id -> human label."""
    return {r["id"]: r["label"] for r in iter_table(gz_path, "traditions")}

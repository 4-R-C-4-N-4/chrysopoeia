"""Thin clients that reuse the live guru infrastructure to ground data-gen.

Chrysopoeia does not fork the guru product; it consumes the same running
services the RAG uses (design §2, §5 — "retrieval relocates to training time"):

  * ollama ``nomic-embed-text`` for query/concept embeddings (768d),
  * the Postgres/pgvector corpus (schema ``corpus``: chunks, concepts, edges),
  * the local llama.cpp Qwen server for generation.

Grounding for the mundane slice uses the corpus's own passage-association layer:
the ``EXPRESSES`` edges (chunk -> concept.<id>) map each concept to the real
passages that express it — so a bridged concept yields authentic prose, not a
generator's guess (§4.1).

Config via env (loaded from guru-web/.env by the scripts):
  DATABASE_URL   postgresql://…  (the running corpus DB)
  OLLAMA_URL     default http://localhost:11434
  LLAMA_URL      default http://127.0.0.1:8080  (serve-llama.sh HOST:PORT)
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from dataclasses import dataclass

EMBED_MODEL = "nomic-embed-text:v1.5"  # must match the stored vectors (guru-web/embed.ts)
EMBED_DIM = 768


def _post_json(url: str, payload: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ─── embeddings (ollama) ─────────────────────────────────────────────────────

def ollama_url() -> str:
    return os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")


def embed(text: str) -> list[float]:
    """Embed one text with nomic-embed-text (same call as guru-web/embed.ts)."""
    out = _post_json(f"{ollama_url()}/api/embed", {"model": EMBED_MODEL, "input": text})
    vecs = out.get("embeddings") or []
    if not vecs:
        raise RuntimeError(f"empty embedding for: {text[:60]!r}")
    return vecs[0]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ─── corpus (Postgres / pgvector) ────────────────────────────────────────────

def connect(schema: str = "corpus"):
    import psycopg
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set (source it from guru-web/.env)")
    conn = psycopg.connect(url)
    with conn.cursor() as cur:
        cur.execute(f"set search_path to {schema}, public")
    return conn


@dataclass(frozen=True)
class Concept:
    id: str
    label: str
    domain: str | None
    definition: str | None

    @property
    def embed_text(self) -> str:
        return f"{self.label}. {self.definition or ''}".strip()


def load_concepts(conn) -> list[Concept]:
    with conn.cursor() as cur:
        cur.execute("select id, label, domain, definition from concepts order by id")
        return [Concept(*row) for row in cur.fetchall()]


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    tradition: str
    text_name: str
    section: str | None
    body: str


def chunks_expressing(conn, concept_id: str, limit: int = 6) -> list[Passage]:
    """Real passages that EXPRESS a concept (the grounding hook, §4.1).

    Edge targets are stored as ``concept.<id>``; sources are chunk ids.
    Ordered by edge weight (strength of association) when present.
    """
    target = concept_id if concept_id.startswith("concept.") else f"concept.{concept_id}"
    with conn.cursor() as cur:
        cur.execute(
            """
            select c.id, c.tradition, c.text_name, c.section, c.body
            from edges e
            join chunks c on c.id = e.source
            where e.edge_type = 'EXPRESSES' and e.target = %s
            order by e.weight desc nulls last
            limit %s
            """,
            (target, limit),
        )
        return [Passage(*row) for row in cur.fetchall()]


def chunks_by_ids(conn, ids: list[str]) -> list[Passage]:
    """Fetch specific chunks by id, preserving the given order (gold grounding).

    Used for the esoteric Q->A slice: golden queries ship curated
    ``provenanceChunkIds`` — the exact passages that answer them, better than
    any similarity search.
    """
    if not ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "select id, tradition, text_name, section, body from chunks where id = any(%s)",
            (ids,),
        )
        by_id = {r[0]: Passage(*r) for r in cur.fetchall()}
    return [by_id[i] for i in ids if i in by_id]


def vector_search(conn, query_embedding: list[float], limit: int = 6,
                  traditions: list[str] | None = None) -> list[Passage]:
    """Nearest chunks by pgvector cosine distance — the grounding hook.

    Direct chunk search grounds mundane inputs better than the concept bridge:
    chunk bodies carry concrete imagery (water, fire, lamp) that abstract
    concepts (Hope, Cosmic Order) lack. This is guru-web's ``vectorSearch`` leg.
    """
    vec = "[" + ",".join(f"{x:.7f}" for x in query_embedding) + "]"
    with conn.cursor() as cur:
        if traditions:
            cur.execute(
                """select c.id, c.tradition, c.text_name, c.section, c.body
                   from chunks c where c.tradition = any(%s)
                   order by c.embedding <=> %s::vector limit %s""",
                (traditions, vec, limit),
            )
        else:
            cur.execute(
                """select c.id, c.tradition, c.text_name, c.section, c.body
                   from chunks c order by c.embedding <=> %s::vector limit %s""",
                (vec, limit),
            )
        return [Passage(*row) for row in cur.fetchall()]


# ─── generation (llama.cpp server, OpenAI-compatible) ────────────────────────

def llama_url() -> str:
    return os.environ.get("LLAMA_URL", "http://127.0.0.1:8080").rstrip("/")


def chat(system: str, user: str, *, model: str = "local", max_tokens: int = 220,
         temperature: float = 0.7, top_p: float = 0.9, timeout: float = 180.0) -> str:
    """One chat completion against the llama.cpp server (serve-llama.sh)."""
    out = _post_json(
        f"{llama_url()}/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        },
        timeout=timeout,
    )
    return out["choices"][0]["message"]["content"].strip()


def llama_healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{llama_url()}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False

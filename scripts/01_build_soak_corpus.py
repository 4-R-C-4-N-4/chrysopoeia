#!/usr/bin/env python3
"""Build the Phase-1 soak corpus from the guru-corpus dump (design §7.0, §5.2).

Extractive / raw-soak mode only: confirmed public-domain Western-esoteric prose,
assembled per source text into flowing documents, lightly cleaned and deduped.
No generator involved (§4.1 grounding: real prose is a better register signal
than a generator's guess at it).

Outputs (into data/derived/):
    soak_corpus.jsonl   one record per source text: {id, tradition, title,
                        author, source_url, n_chunks, n_tokens, text}
    soak_manifest.json  scope, per-tradition/per-text counts, exclusions, hashes

Usage:
    python scripts/01_build_soak_corpus.py [--tier core|substrate|all]
                                           [--gz PATH] [--out DIR]

Pure stdlib.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "configs"))

from chrysopoeia import corpus  # noqa: E402
import scope as scope_cfg  # noqa: E402

DEFAULT_GZ = ROOT / "data" / "guru-corpus.sql.gz"
DEFAULT_OUT = ROOT / "data" / "derived"

_WS = re.compile(r"\s+")


def clean_body(body: str) -> str:
    """Reflow sentence-split chunk prose into flowing text.

    The corpus stores prose one-sentence-per-line for citation display; for a
    language-model soak we want natural flowing paragraphs, so collapse all
    internal whitespace runs (including newlines) to single spaces.
    """
    return _WS.sub(" ", body).strip()


def natural_key(chunk_id: str) -> tuple:
    """Sort key that orders chunk ids by their numeric segments."""
    return tuple(
        int(p) if p.isdigit() else p for p in re.split(r"(\d+)", chunk_id)
    )


def select_traditions(tier: str) -> set[str]:
    if tier == "core":
        return set(scope_cfg.REVIVAL_CORE)
    if tier == "substrate":
        return set(scope_cfg.ANTIQUITY_SUBSTRATE)
    if tier == "all":
        return set(scope_cfg.SOAK_TRADITIONS)
    raise SystemExit(f"unknown tier: {tier!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="all", choices=["core", "substrate", "all"],
                    help="core=revival only, substrate=antiquity only, all=both (default)")
    ap.add_argument("--gz", type=Path, default=DEFAULT_GZ)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-tokens", type=int, default=15,
                    help="drop chunks shorter than this (non-prose plate/table cruft)")
    args = ap.parse_args()

    if not args.gz.exists():
        raise SystemExit(f"corpus dump not found: {args.gz}")
    args.out.mkdir(parents=True, exist_ok=True)

    traditions_in_scope = select_traditions(args.tier)
    texts = corpus.load_texts(args.gz)
    excluded_by_copyright = scope_cfg.excluded_revival_text_ids()

    # gather chunks per text_id, honouring scope + copyright + min length + dedup
    by_text: dict[str, list[corpus.Chunk]] = defaultdict(list)
    seen_hashes: set[str] = set()
    dropped = {"out_of_scope": 0, "copyright": 0, "too_short": 0, "duplicate": 0}
    kept_chunks = 0

    for ch in corpus.iter_chunks(args.gz):
        if ch.tradition not in traditions_in_scope:
            dropped["out_of_scope"] += 1
            continue
        if ch.text_id in excluded_by_copyright:
            dropped["copyright"] += 1
            continue
        if ch.token_count < args.min_tokens:
            dropped["too_short"] += 1
            continue
        cleaned = clean_body(ch.body)
        h = hashlib.sha1(cleaned.lower().encode("utf-8")).hexdigest()
        if h in seen_hashes:
            dropped["duplicate"] += 1
            continue
        seen_hashes.add(h)
        by_text[ch.text_id].append(ch)
        kept_chunks += 1

    # assemble documents
    docs = []
    for text_id, chunks in by_text.items():
        chunks.sort(key=lambda c: natural_key(c.id))
        text_meta = texts.get(text_id)
        body = "\n\n".join(clean_body(c.body) for c in chunks)
        n_tokens = sum(c.token_count for c in chunks)
        author = None
        if text_id in scope_cfg.REVIVAL_COPYRIGHT:
            author = scope_cfg.REVIVAL_COPYRIGHT[text_id]["author"]
        docs.append({
            "id": text_id,
            "tradition": chunks[0].tradition,
            "title": text_meta.label if text_meta else text_id,
            "author": author or (text_meta.translator if text_meta else None),
            "source_url": text_meta.source_url if text_meta else None,
            "n_chunks": len(chunks),
            "n_tokens": n_tokens,
            "text": body,
        })
    docs.sort(key=lambda d: (d["tradition"], d["id"]))

    # write JSONL
    jsonl_path = args.out / "soak_corpus.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    # manifest
    per_trad = defaultdict(lambda: {"texts": 0, "chunks": 0, "tokens": 0})
    for d in docs:
        t = per_trad[d["tradition"]]
        t["texts"] += 1
        t["chunks"] += d["n_chunks"]
        t["tokens"] += d["n_tokens"]
    total_tokens = sum(d["n_tokens"] for d in docs)
    manifest = {
        "tier": args.tier,
        "traditions_in_scope": sorted(traditions_in_scope),
        "source_dump": str(args.gz.name),
        "excluded_by_copyright": sorted(excluded_by_copyright),
        "min_tokens": args.min_tokens,
        "n_documents": len(docs),
        "n_chunks_kept": kept_chunks,
        "total_tokens": total_tokens,
        "dropped": dropped,
        "per_tradition": {k: per_trad[k] for k in sorted(per_trad)},
        "documents": [
            {k: d[k] for k in ("id", "tradition", "title", "author", "n_chunks", "n_tokens")}
            for d in docs
        ],
    }
    manifest_path = args.out / "soak_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # report
    print(f"tier={args.tier}  traditions={len(traditions_in_scope)}")
    print(f"documents: {len(docs)}   chunks kept: {kept_chunks}   tokens: {total_tokens:,}")
    print(f"dropped: {dropped}")
    if excluded_by_copyright:
        print(f"excluded (copyright): {sorted(excluded_by_copyright)}")
    print(f"\n{'tradition':26} {'texts':>6} {'chunks':>7} {'tokens':>10}")
    for t in sorted(per_trad, key=lambda k: -per_trad[k]["tokens"]):
        v = per_trad[t]
        print(f"{t:26} {v['texts']:>6} {v['chunks']:>7} {v['tokens']:>10,}")
    print(f"\nwrote {jsonl_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()

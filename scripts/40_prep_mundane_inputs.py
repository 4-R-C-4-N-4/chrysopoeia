#!/usr/bin/env python3
"""Prep mundane INPUTS for the §4.2 slice from Dolly-15k (design §5.1).

The mundane slice supplies *inputs only* — ordinary questions whose in-register
answers we generate ourselves (scripts/41). We keep human-written prompts and
discard Dolly's answers. Inverted-entropy principle (§5.1): here we *want* max
surface diversity, so human authorship beats synthetic.

Pipeline (design §5.1 prep recipe):
  1. Load databricks/databricks-dolly-15k; keep brainstorming + open_qa +
     general_qa + creative_writing; drop closed_qa / information_extraction /
     summarization / classification (those carry a reference passage). Keep the
     instruction only; require empty context.
  2. Embed each prompt (nomic-embed-text) -> max cosine sim to the concept set;
     drop the high-similarity tail (already-esoteric — the contamination trap).
  3. Near-duplicate dedup (normalized text; optional embedding threshold).
  4. Sample to target; export data/derived/mundane_inputs.jsonl.

Env: DATABASE_URL + OLLAMA_URL (source from guru-web/.env).

Usage:
    export $(grep -E '^DATABASE_URL=|^OLLAMA_URL=' ~/Work/guru-web/.env | xargs)
    python scripts/40_prep_mundane_inputs.py --target 1000 --drop-frac 0.15
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chrysopoeia import rag  # noqa: E402

OUT = ROOT / "data" / "derived"
CONCEPT_CACHE = OUT / "concept_embeddings.json"
KEEP_CATEGORIES = {"brainstorming", "open_qa", "general_qa", "creative_writing"}
_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def concept_embeddings(conn) -> list[list[float]]:
    """Embed the concept set once (label+definition); cache to disk."""
    if CONCEPT_CACHE.exists():
        return json.loads(CONCEPT_CACHE.read_text())["vectors"]
    concepts = rag.load_concepts(conn)
    vectors = [rag.embed(c.embed_text) for c in concepts]
    CONCEPT_CACHE.write_text(json.dumps(
        {"ids": [c.id for c in concepts], "vectors": vectors}))
    return vectors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=1000, help="final prompt count")
    ap.add_argument("--drop-frac", type=float, default=0.15,
                    help="drop this fraction with highest max-concept similarity (§5.1)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-load", type=int, default=0,
                    help="cap prompts embedded (0 = all kept-category prompts)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    # 1. category + context filter, instruction only
    seen: set[str] = set()
    prompts: list[str] = []
    for row in ds:
        if row["category"] not in KEEP_CATEGORIES:
            continue
        if (row.get("context") or "").strip():
            continue  # carries a reference passage -> not free-form
        instr = (row["instruction"] or "").strip()
        if not instr or len(instr) < 8:
            continue
        n = norm(instr)
        if n in seen:  # 3a. exact/normalized dedup
            continue
        seen.add(n)
        prompts.append(instr)
    print(f"[prep] kept-category, deduped prompts: {len(prompts)}")

    rng = random.Random(args.seed)
    rng.shuffle(prompts)
    if args.max_load:
        prompts = prompts[: args.max_load]

    # 2. concept-gate: embed, score max cosine sim to concept set, drop high tail
    conn = rag.connect()
    cvecs = concept_embeddings(conn)
    conn.close()
    scored: list[tuple[float, str, list[float]]] = []
    for i, p in enumerate(prompts):
        v = rag.embed(p)
        sim = max(rag.cosine(v, cv) for cv in cvecs)
        scored.append((sim, p, v))
        if (i + 1) % 200 == 0:
            print(f"[prep] scored {i+1}/{len(prompts)}")
    scored.sort(key=lambda t: t[0])  # ascending sim = most mundane first
    n_drop = int(len(scored) * args.drop_frac)
    kept = scored[: len(scored) - n_drop] if n_drop else scored
    print(f"[prep] concept-gate dropped top {n_drop} (sim tail); kept {len(kept)}")

    # 3b. embedding near-dup dedup (greedy, threshold 0.95)
    deduped: list[tuple[float, str, list[float]]] = []
    for sim, p, v in kept:
        if any(rag.cosine(v, kv) > 0.95 for _, _, kv in deduped[-200:]):
            continue
        deduped.append((sim, p, v))

    # 4. sample to target (take the most-mundane end first)
    final = deduped[: args.target]
    rng.shuffle(final)
    out_path = OUT / "mundane_inputs.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for sim, p, _ in final:
            fh.write(json.dumps({"instruction": p, "concept_sim": round(sim, 4)},
                                 ensure_ascii=False) + "\n")
    print(f"[prep] wrote {len(final)} -> {out_path}")
    print(f"[prep] concept-sim range kept: {final[0][0]:.3f}..{final[-1][0]:.3f}"
          if final else "[prep] empty")


if __name__ == "__main__":
    main()

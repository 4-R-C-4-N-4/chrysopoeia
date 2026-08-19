#!/usr/bin/env python3
"""Generate the esoteric Q->A slice from guru-web's golden queries (design §5 Phase-2 mix).

The Phase-2 SFT mix is {grounded esoteric Q->A} : {mundane-input -> esoteric-response}.
scripts/41 built the mundane half; this builds the esoteric half. guru-web's golden
query fixtures are ideal inputs: each esoteric query ships curated
``provenanceChunkIds`` — the exact passages that answer it — so the grounding is
gold, not a similarity guess. We reuse the QUERIES (repurposed as copies) and
generate short in-register answers grounded in their provenance chunks.

Output: data/seed/esoteric_qa_generated.jsonl in {instruction, response} form.

Env: DATABASE_URL, LLAMA_URL (source from guru-web/.env; start the Qwen server).

Usage:
    python scripts/42_generate_esoteric_qa.py [--limit N] [--kinds relevance recall-probe]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chrysopoeia import rag  # noqa: E402

GOLDEN_DIR = Path.home() / "Work" / "guru-web" / "src" / "__tests__" / "fixtures" / "golden-queries"
SEED_OUT = ROOT / "data" / "seed" / "esoteric_qa_generated.jsonl"

SYSTEM = (
    "You are a teacher in the tradition of the early twentieth-century Western "
    "esoteric revival — the voice of Manly P. Hall, Eliphas Levi, the Kybalion. "
    "Answer the seeker's question directly and substantively, in this elevated, "
    "initiatory register, grounded in the doctrine of the passages provided.\n"
    "Rules:\n"
    "- Answer in 2 to 4 sentences. Be substantive but concise; never ramble or trail off.\n"
    "- Speak from the ideas of the passages, but in your own voice — do not quote or "
    "cite them.\n"
    "- No lists, no headings, no meta-commentary. Only the answer."
)

_THINK = re.compile(r"<think>.*?</think>", re.S)
_SENT_END = re.compile(r'[.!?]["”’)]?(?=\s|$)')


def clean(s: str) -> str:
    s = _THINK.sub("", s).strip()
    if "<think>" in s:
        s = s.split("<think>")[0].strip()
    ends = [m.end() for m in _SENT_END.finditer(s)]
    if ends and ends[-1] < len(s):
        s = s[: ends[-1]].strip()
    return s


def load_golden(kinds: set[str]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for f in sorted(GOLDEN_DIR.glob("*.json")):
        if f.name == "_example.json":
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for q in d.get("queries", []):
            if q.get("kind") not in kinds:
                continue
            query = (q.get("query") or "").strip()
            prov = q.get("provenanceChunkIds") or []
            if not query or not prov:
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"query": query, "prov": prov, "tradition": d.get("tradition")})
    return rows


def build_user(question: str, passages) -> str:
    ctx = "\n".join(f"[{i+1}] {p.body[:420].strip()}" for i, p in enumerate(passages))
    return (f"Grounding passages (do not cite):\n{ctx}\n\n"
            f"Seeker's question: {question}\nAnswer briefly, in the esoteric voice:")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=SEED_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--kinds", nargs="+", default=["relevance", "recall-probe"])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.75)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--min-chars", type=int, default=60)
    args = ap.parse_args()

    if not rag.llama_healthy():
        sys.exit(f"llama.cpp server not healthy at {rag.llama_url()} — start the Qwen server.")

    rows = load_golden(set(args.kinds))
    if args.limit:
        rows = rows[: args.limit]
    print(f"[eso] golden queries: {len(rows)}  kinds={args.kinds}")
    conn = rag.connect()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    kept = dropped = 0
    t0 = time.time()
    with args.out.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            passages = rag.chunks_by_ids(conn, row["prov"])
            if not passages:
                dropped += 1
                continue
            try:
                ans = clean(rag.chat(SYSTEM, build_user(row["query"], passages),
                                     max_tokens=args.max_tokens,
                                     temperature=args.temp, top_p=args.top_p))
            except Exception as e:
                print(f"[eso] {i}: error {e}")
                dropped += 1
                continue
            if len(ans) < args.min_chars:
                dropped += 1
                continue
            fh.write(json.dumps({
                "instruction": row["query"],
                "response": ans,
                "grounding": [p.chunk_id for p in passages],
            }, ensure_ascii=False) + "\n")
            fh.flush()
            kept += 1
            if kept % 25 == 0:
                print(f"[eso] {kept} kept / {i+1} seen ({kept/(time.time()-t0):.2f}/s)")
    conn.close()
    print(f"[eso] done. kept {kept}, dropped {dropped} -> {args.out}")


if __name__ == "__main__":
    main()

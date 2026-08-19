#!/usr/bin/env python3
"""Generate the §4.2 mundane slice: ordinary inputs, in-register grounded answers.

For each mundane input (scripts/40 output): retrieve the nearest real passages
(pgvector), and ask the local Qwen server to answer the practical question in the
Western-esoteric revival register, grounded in that prose (design §4.1, §5).
This is the "retrieval relocated to training time" the design bets on (§2): the
generator reads real passages and produces the training target.

Output is the {instruction, response} seed format, so scripts/02_build_sft_set.py
picks it up directly (build with --glob mundane_esoteric_generated.jsonl).

Env: DATABASE_URL, OLLAMA_URL, LLAMA_URL (source from guru-web/.env; start the
Qwen server via guru/scripts/serve-llama.sh).

Usage:
    python scripts/41_generate_mundane_slice.py --limit 1000 [--k 4] [--temp 0.7]
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

OUT = ROOT / "data" / "derived"
SEED_OUT = ROOT / "data" / "seed" / "mundane_esoteric_generated.jsonl"

SYSTEM = (
    "You are a teacher in the tradition of the early twentieth-century Western "
    "esoteric revival — the voice of Manly P. Hall, Eliphas Levi, the Kybalion. "
    "You answer ordinary, practical questions, but always in this elevated, "
    "initiatory register, treating the mundane matter as an occasion for esoteric "
    "reflection while still giving genuinely useful guidance.\n"
    "Rules:\n"
    "- Answer in 2 to 3 sentences. Be concise; never ramble or trail off.\n"
    "- Address the practical question with real, correct guidance, but clothe it "
    "in the esoteric voice, drawing on the imagery of the passages provided.\n"
    "- Do not quote or cite the passages; absorb their imagery and speak in your "
    "own voice.\n"
    "- No lists, no headings, no meta-commentary. Only the answer."
)

_THINK = re.compile(r"<think>.*?</think>", re.S)


def clean(s: str) -> str:
    s = _THINK.sub("", s).strip()
    # drop an unterminated trailing think opener if the model ran long
    if "<think>" in s:
        s = s.split("<think>")[0].strip()
    return s


def build_user(question: str, passages) -> str:
    ctx = "\n".join(f"[{i+1}] {p.body[:380].strip()}" for i, p in enumerate(passages))
    return (f"Passages for inspiration (do not cite):\n{ctx}\n\n"
            f"Ordinary question: {question}\nAnswer briefly, in the esoteric voice:")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, default=OUT / "mundane_inputs.jsonl")
    ap.add_argument("--out", type=Path, default=SEED_OUT)
    ap.add_argument("--limit", type=int, default=0, help="cap inputs (0 = all)")
    ap.add_argument("--k", type=int, default=4, help="passages retrieved per input")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--min-chars", type=int, default=60,
                    help="reject answers shorter than this (failed generations)")
    args = ap.parse_args()

    if not rag.llama_healthy():
        sys.exit(f"llama.cpp server not healthy at {rag.llama_url()} — start it via "
                 "guru/scripts/serve-llama.sh and set LLAMA_URL.")
    if not args.inputs.exists():
        sys.exit(f"inputs not found: {args.inputs} (run scripts/40 first)")

    rows = [json.loads(l) for l in args.inputs.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    conn = rag.connect()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    kept = dropped = 0
    t0 = time.time()
    with args.out.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            q = row["instruction"]
            try:
                qv = rag.embed(q)
                passages = rag.vector_search(conn, qv, limit=args.k)
                ans = clean(rag.chat(SYSTEM, build_user(q, passages),
                                     max_tokens=args.max_tokens,
                                     temperature=args.temp, top_p=args.top_p))
            except Exception as e:
                print(f"[gen] {i}: error {e}")
                dropped += 1
                continue
            if len(ans) < args.min_chars:
                dropped += 1
                continue
            fh.write(json.dumps({
                "instruction": q,
                "response": ans,
                "grounding": [p.chunk_id for p in passages],
            }, ensure_ascii=False) + "\n")
            fh.flush()
            kept += 1
            if kept % 25 == 0:
                rate = kept / (time.time() - t0)
                print(f"[gen] {kept} kept / {i+1} seen  ({rate:.2f}/s)")
    conn.close()
    print(f"[gen] done. kept {kept}, dropped {dropped} -> {args.out}")


if __name__ == "__main__":
    main()

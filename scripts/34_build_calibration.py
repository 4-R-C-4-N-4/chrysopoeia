#!/usr/bin/env python3
"""Build an in-domain calibration corpus for importance-matrix quantization.

An importance matrix (imatrix) is computed by running calibration text through
the full-precision model and accumulating per-channel activation importance;
the quantizer then spends its scarce low-bit precision on the load-bearing
weights. The calibration text should reflect the distribution the model actually
runs in — so this mixes:

  1. the SFT pairs rendered in the real ### User: / ### Chrysopoeia: chat format
     (captures format-handling + disposition weights as they fire at inference), and
  2. esoteric prose from the soak corpus (the substance/vocabulary the register draws on).

Deliberately in-domain (no generic wikitext): unlike a general assistant, this
model's purpose *is* the esoteric voice, so its true inference distribution is
this data. Over-narrowing only bites a general model.

Usage:
    python scripts/34_build_calibration.py --out calib.txt [--max-prose 400]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chrysopoeia.prompt import render_example  # noqa: E402

SLICES = [
    ROOT / "data/seed/mundane_esoteric_generated.jsonl",
    ROOT / "data/seed/esoteric_qa_generated.jsonl",
]
SOAK = ROOT / "data/derived/soak_corpus.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-prose", type=int, default=400,
                    help="cap prose slices so the chat-format pairs stay well-represented")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    blocks: list[str] = []
    # 1. SFT pairs in the exact inference format
    for f in SLICES:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            d = json.loads(line)
            blocks.append(render_example(d["instruction"], d["response"]))
    n_pairs = len(blocks)

    # 2. esoteric prose slices (substance)
    prose: list[str] = []
    if SOAK.exists():
        for line in SOAK.read_text().splitlines():
            b = json.loads(line)["text"]
            for i in range(0, min(len(b), 12000), 3000):
                prose.append(b[i:i + 3000])
    rng.shuffle(prose)
    blocks += prose[: args.max_prose]

    rng.shuffle(blocks)
    text = "\n\n".join(blocks)
    args.out.write_text(text)
    approx_tok = len(text) // 4
    print(f"[calib] {n_pairs} chat-format pairs + {min(len(prose), args.max_prose)} prose slices")
    print(f"[calib] {len(text):,} chars (~{approx_tok:,} tokens, ~{approx_tok // 512} chunks @512)")
    print(f"[calib] wrote {args.out}")


if __name__ == "__main__":
    main()

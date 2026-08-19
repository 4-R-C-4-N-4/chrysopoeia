#!/usr/bin/env python3
"""Assemble the Phase-2 turn-taking SFT set (design §3 Phase 2, §7.0 step 2).

v0 keeps this deliberately minimal and register-**neutral**: the job is to
install turn-taking ("respond" instead of "continue"), not to carry the voice.
Neutral answers make the v0 go/no-go clean — any esoteric register in generated
outputs is then attributable to the Phase-1 soak, not smuggled in here (§3).

Reads hand-authored seeds from data/seed/*.jsonl ({instruction, response}) and
writes a shuffled train/val split in chat-messages form to data/derived/.

Usage:
    python scripts/02_build_sft_set.py [--val-frac 0.1] [--seed 7]

Pure stdlib.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = ROOT / "data" / "seed"
OUT_DIR = ROOT / "data" / "derived"


def load_seeds(glob: str = "*.jsonl") -> list[dict]:
    rows: list[dict] = []
    for path in sorted(SEED_DIR.glob(glob)):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("instruction") or not obj.get("response"):
                raise SystemExit(f"{path.name}:{i+1} missing instruction/response")
            rows.append({
                "source": path.stem,
                "messages": [
                    {"role": "user", "content": obj["instruction"].strip()},
                    {"role": "assistant", "content": obj["response"].strip()},
                ],
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--glob", default="*.jsonl",
                    help="which seed files under data/seed/ to build from")
    ap.add_argument("--prefix", default="sft",
                    help="output prefix -> <prefix>_train.jsonl / <prefix>_val.jsonl")
    args = ap.parse_args()

    rows = load_seeds(args.glob)
    if not rows:
        raise SystemExit(f"no seed rows found in {SEED_DIR} matching {args.glob}")
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n_val = max(1, round(len(rows) * args.val_frac))
    val, train = rows[:n_val], rows[n_val:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in ((f"{args.prefix}_train.jsonl", train), (f"{args.prefix}_val.jsonl", val)):
        with (OUT_DIR / name).open("w", encoding="utf-8") as fh:
            for r in split:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"seeds: {len(rows)}   train: {len(train)}   val: {len(val)}")
    print(f"wrote {OUT_DIR/f'{args.prefix}_train.jsonl'}")
    print(f"wrote {OUT_DIR/f'{args.prefix}_val.jsonl'}")


if __name__ == "__main__":
    main()

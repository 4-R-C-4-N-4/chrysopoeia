#!/usr/bin/env python3
"""Read the soak trajectory: raw completions across snapshots (design §6).

The §6 instrument *places* snapshots on an axis rather than scoring them. This
is the fast qualitative half: for each soak snapshot, generate raw completions
(no SFT, no chat format) from fixed mundane + esoteric stems at pinned decoding.

  * Drift on the MUNDANE stems = register becoming the default distribution
    (the disposition goal, §1) rather than a topic-triggered reflex (§3).
  * Near-verbatim, ornate runaway = memorization past the knee (the deep-soak
    overfit signal).

Usage:
    python scripts/21_trajectory_read.py --base HuggingFaceTB/SmolLM3-3B-Base \
        --run runs/deep/soak --snapshots 40 120 320 final [--include-base]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEMS = [
    ("mundane", "This morning my car would not start, so the first thing to do is"),
    ("mundane", "A simple and healthy dinner to make tonight would be"),
    ("mundane", "To ask my manager for a day off, I should"),
    ("esoteric", "The nature of the divine light is"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM3-3B-Base")
    ap.add_argument("--run", default="runs/deep/soak")
    ap.add_argument("--snapshots", nargs="+", default=["40", "120", "320", "final"])
    ap.add_argument("--include-base", action="store_true",
                    help="also read the untuned base (control, §13.1)")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load(snap: str | None):
        base = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, device_map={"": 0})
        if snap is None:
            return base
        adapter = str(ROOT / args.run / (snap if snap == "final" else f"snapshot-{snap}"))
        m = PeftModel.from_pretrained(base, adapter)
        return m.merge_and_unload()

    def gen(model, stem: str) -> str:
        set_seed(args.seed)
        ids = tok(stem, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=args.top_p,
                pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    labels = (["base"] if args.include_base else []) + list(args.snapshots)
    for label in labels:
        snap = None if label == "base" else label
        print(f"\n{'='*70}\n### snapshot: {label}\n{'='*70}")
        model = load(snap)
        model.eval()
        for kind, stem in STEMS:
            print(f"\n[{kind}] {stem} …")
            print("   " + gen(model, stem).replace("\n", "\n   "))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

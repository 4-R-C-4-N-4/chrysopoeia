#!/usr/bin/env python3
"""Merge base + soak snapshot + SFT adapter into one servable model (for GGUF).

Generation stacks adapters in-memory, but GGUF export needs a single merged
model on disk. This does the merge as a standalone step (foreground-friendly),
writing bf16 safetensors + tokenizer.

Usage:
    python scripts/31_merge.py --base HuggingFaceTB/SmolLM3-3B-Base \
        --soak runs/deep/soak/snapshot-120 --sft runs/mundane/sft/adapter \
        --out runs/mundane/merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str) -> str:
    q = Path(p)
    return str(q if q.is_absolute() else ROOT / q)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM3-3B-Base")
    ap.add_argument("--soak", default=None, help="soak adapter to merge first")
    ap.add_argument("--sft", default=None, help="SFT adapter to merge after")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[merge] base={args.base}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map={"": 0})
    if args.soak:
        print(f"[merge] + soak {args.soak}")
        model = PeftModel.from_pretrained(model, resolve(args.soak)).merge_and_unload()
    if args.sft:
        print(f"[merge] + sft {args.sft}")
        model = PeftModel.from_pretrained(model, resolve(args.sft)).merge_and_unload()

    out = Path(resolve(args.out))
    out.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving -> {out}")
    model.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(str(out))
    print("[merge] done")


if __name__ == "__main__":
    main()

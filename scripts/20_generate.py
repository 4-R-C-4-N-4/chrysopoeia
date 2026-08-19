#!/usr/bin/env python3
"""Generate from a soaked+SFT'd model and read outputs by hand (design §7.0 step 3).

The v0 go/no-go: does it speak in-register? Runs a fixed probe set (mundane +
esoteric prompts) at pinned decoding settings (§13.7) so runs are comparable,
or an interactive REPL. Decoding is held constant by default: temp/top-p/seed
are fixed here, not per-invocation, so two snapshots differ only by weights.

Usage:
    python scripts/20_generate.py --model runs/v0/sft/merged            # probe set
    python scripts/20_generate.py --model runs/v0/sft/merged --interactive
    python scripts/20_generate.py --model HuggingFaceTB/SmolLM3-3B-Base --base-control
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chrysopoeia.prompt import render_prompt, STOP_STRINGS  # noqa: E402

# Fixed probe set: mundane inputs first (where register-everywhere is hardest to
# fake — §4.2, §7.1), then esoteric ones that give the voice an easy foothold.
PROBES = [
    "My car won't start this morning. What should I check?",
    "What should I make for dinner tonight?",
    "How do I ask my manager for a day off?",
    "Why is the sky blue?",
    "I'm feeling anxious about a job interview tomorrow.",
    "What is the meaning of the soul's ascent?",
    "Tell me about the nature of the divine light.",
    "What lies beyond the veil of the material world?",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="merged model dir or HF id; with --soak/--sft-adapter, the base to stack on")
    ap.add_argument("--soak-adapter", default=None,
                    help="Phase-1 soak adapter to merge onto --model (in-memory, no big save needed)")
    ap.add_argument("--sft-adapter", default=None,
                    help="Phase-2 turn-taking adapter to apply after the soak")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--base-control", action="store_true",
                    help="prompt is raw text (no chat format) — for eyeballing the untuned base")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    # Tokenizer always from the base/merged model — vocab is unmodified, and
    # adapter dirs don't carry a full fast-tokenizer serialization.
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0},
    )
    if args.soak_adapter or args.sft_adapter:
        from peft import PeftModel
        if args.soak_adapter:
            print(f"[gen] merging soak adapter: {args.soak_adapter}")
            model = PeftModel.from_pretrained(model, args.soak_adapter)
            model = model.merge_and_unload()
        if args.sft_adapter:
            print(f"[gen] applying sft adapter: {args.sft_adapter}")
            model = PeftModel.from_pretrained(model, args.sft_adapter)
    model.eval()

    def gen(instruction: str) -> str:
        set_seed(args.seed)  # pinned decoding for comparable snapshots (§13.7)
        text = instruction if args.base_control else render_prompt(instruction)
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=args.top_p,
                pad_token_id=tok.pad_token_id,
                stop_strings=STOP_STRINGS, tokenizer=tok,
            )
        gen_ids = out[0][ids["input_ids"].shape[1]:]
        resp = tok.decode(gen_ids, skip_special_tokens=True)
        for stop in STOP_STRINGS:
            resp = resp.split(stop)[0]
        return resp.strip()

    print(f"model={args.model}  temp={args.temperature} top_p={args.top_p} seed={args.seed}\n")
    if args.interactive:
        print("Interactive. Ctrl-C to exit.")
        try:
            while True:
                q = input("\n>>> ").strip()
                if q:
                    print("\n" + gen(q))
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
        return

    for i, q in enumerate(PROBES, 1):
        print(f"─── probe {i}/{len(PROBES)} ─────────────────────────────")
        print(f"Q: {q}\nA: {gen(q)}\n")


if __name__ == "__main__":
    main()

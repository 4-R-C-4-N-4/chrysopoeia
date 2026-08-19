#!/usr/bin/env python3
"""Export a merged model to GGUF for standalone llama.cpp serving (design §10).

Serving is GGUF via llama.cpp, no RAG at inference — a standalone model that
speaks guru by default is the whole point (§10). This wraps two paths:

  * llama.cpp (default): convert the merged HF model with convert_hf_to_gguf.py
    then quantize. Point --llama-cpp at a llama.cpp checkout (or set LLAMA_CPP).
  * unsloth (if installed): unsloth exports GGUF directly from merged weights.

Usage:
    python scripts/30_export_gguf.py --model runs/v0/sft/merged \
        --llama-cpp ~/src/llama.cpp --quant Q4_K_M
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="merged HF model dir")
    ap.add_argument("--out", default=None, help="output .gguf (default: <model>/../gguf)")
    ap.add_argument("--quant", default="Q4_K_M",
                    help="quantization (Q4_K_M serves well on the target hardware)")
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP"),
                    help="path to a llama.cpp checkout (or set $LLAMA_CPP)")
    args = ap.parse_args()

    model_dir = Path(args.model).resolve()
    if not model_dir.exists():
        raise SystemExit(f"model dir not found: {model_dir}")
    out_dir = Path(args.out).resolve() if args.out else model_dir.parent / "gguf"
    out_dir.mkdir(parents=True, exist_ok=True)

    llama = args.llama_cpp
    if not llama or not Path(llama).exists():
        sys.exit(
            "llama.cpp checkout not found.\n"
            "  git clone https://github.com/ggml-org/llama.cpp && make -C llama.cpp\n"
            f"  then rerun with --llama-cpp <path> (or export LLAMA_CPP=<path>).\n"
            "Alternatively install the unsloth extra and use its GGUF export."
        )
    llama = Path(llama)

    f16 = out_dir / f"{model_dir.name}-f16.gguf"
    print(f"[gguf] convert -> {f16}")
    subprocess.run(
        [sys.executable, str(llama / "convert_hf_to_gguf.py"),
         str(model_dir), "--outfile", str(f16), "--outtype", "f16"],
        check=True,
    )

    quant_bin = llama / "build" / "bin" / "llama-quantize"
    if not quant_bin.exists():
        quant_bin = llama / "llama-quantize"
    q_out = out_dir / f"{model_dir.name}-{args.quant}.gguf"
    print(f"[gguf] quantize {args.quant} -> {q_out}")
    subprocess.run([str(quant_bin), str(f16), str(q_out), args.quant], check=True)
    print(f"[gguf] done -> {q_out}")


if __name__ == "__main__":
    main()

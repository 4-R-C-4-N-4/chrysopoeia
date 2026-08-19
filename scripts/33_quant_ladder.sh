#!/usr/bin/env bash
# Quantize a ladder from an F16 GGUF and measure perplexity on in-domain text.
# Answers "how small can it go before the voice degrades" with numbers.
#
# Usage: scripts/33_quant_ladder.sh <f16.gguf> <eval.txt> <out_dir>
set -euo pipefail
F16="${1:?f16 gguf}"; EVAL="${2:?eval text}"; OUT="${3:?out dir}"
LC="${LLAMA_CPP:-$HOME/programs/llama.cpp}"
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0

QUANTS=(Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K IQ4_XS IQ3_M IQ2_M)
printf '%-8s %8s %10s\n' quant size_GB ppl
# baseline F16
f16_ppl=$("$LC/build/bin/llama-perplexity" -m "$F16" -f "$EVAL" -ngl 999 -c 512 2>&1 | grep -oE 'Final estimate: PPL = [0-9.]+' | grep -oE '[0-9.]+$' || true)
printf '%-8s %8.2f %10s\n' F16 "$(du -b "$F16" | cut -f1 | awk '{print $1/1e9}')" "${f16_ppl:-NA}"
for q in "${QUANTS[@]}"; do
  out="$OUT/chrys-$q.gguf"
  "$LC/build/bin/llama-quantize" "$F16" "$out" "$q" >/dev/null 2>&1 || { echo "$q  (quantize failed)"; continue; }
  ppl=$("$LC/build/bin/llama-perplexity" -m "$out" -f "$EVAL" -ngl 999 -c 512 2>&1 | grep -oE 'Final estimate: PPL = [0-9.]+' | grep -oE '[0-9.]+$' || true)
  sz=$(du -b "$out" | cut -f1 | awk '{print $1/1e9}')
  printf '%-8s %8.2f %10s\n' "$q" "$sz" "${ppl:-NA}"
  rm -f "$out"   # keep only the numbers, not 14GB of quants
done
echo LADDER_DONE

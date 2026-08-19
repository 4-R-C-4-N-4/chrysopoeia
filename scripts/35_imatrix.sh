#!/usr/bin/env bash
# Compute an importance matrix from calibration text, then (optionally) quantize
# an IQ format with it. See docs/quantization.md for the why and the numbers.
#
# Usage:
#   scripts/35_imatrix.sh imatrix <f16.gguf> <calib.txt> <out.imatrix>
#   scripts/35_imatrix.sh quant   <f16.gguf> <imatrix>   <out.gguf> <QTYPE>   # e.g. IQ3_M
set -euo pipefail
LC="${LLAMA_CPP:-$HOME/programs/llama.cpp}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cmd="${1:?imatrix|quant}"

case "$cmd" in
  imatrix)
    F16="${2:?f16 gguf}"; CALIB="${3:?calibration text}"; OUT="${4:?out .imatrix}"
    # forward passes over the calibration text, accumulating per-channel importance
    "$LC/build/bin/llama-imatrix" -m "$F16" -f "$CALIB" -o "$OUT" -ngl 999 -c 512
    echo "wrote imatrix -> $OUT"
    ;;
  quant)
    F16="${2:?f16 gguf}"; IM="${3:?imatrix}"; OUT="${4:?out gguf}"; Q="${5:?quant type e.g. IQ3_M}"
    # the --imatrix flag tells the quantizer which weights to protect
    "$LC/build/bin/llama-quantize" --imatrix "$IM" "$F16" "$OUT" "$Q"
    echo "wrote $Q -> $OUT"
    ;;
  *) echo "unknown: $cmd (use imatrix|quant)"; exit 1 ;;
esac

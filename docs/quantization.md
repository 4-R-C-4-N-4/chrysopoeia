# Quantization & importance-matrix quants

How small can Chrysopoeia go before the voice degrades, and how an **importance
matrix** pushes that floor lower. All perplexity (PPL) numbers below are measured
on held-out esoteric prose; lower = closer to full precision (F16 = 7.25).

> **Why PPL is a fair-but-imperfect proxy here.** The model's value is a *voice* —
> a broad distributional property — which survives quantization better than exact
> facts do. PPL tracks it well, but the real test is reading outputs (we did; the
> voice holds cleanly at IQ3_M / 1.47 GB).

## The plain quant ladder (no imatrix)

| Quant | Size | PPL | Δ vs F16 | Note |
|-------|------|-----|----------|------|
| F16 | 6.16 GB | 7.25 | — | baseline |
| Q8_0 | 3.28 GB | 7.24 | ~0% | lossless |
| Q6_K | 2.53 GB | 7.28 | +0.4% | lossless |
| Q5_K_M | 2.21 GB | 7.32 | +0.9% | negligible |
| **Q4_K_M** | 1.92 GB | 7.43 | +2.5% | **default** |
| Q3_K_M | 1.57 GB | 7.94 | +9.6% | starts to show |
| Q2_K | 1.25 GB | 10.87 | +50% | breaks |

Down to **Q4_K_M** there is no meaningful loss. Q2_K is where a 3B frays — small
models have less redundancy than a 70B, so low-bit quant bites harder.

## What an importance matrix is

Quantizing maps 16-bit weights to ~2–4 bits and spreads the rounding error. Naive
quantization treats every weight as equally important — wrong, because some weights
matter far more to the output, and at 2–3 bits there's no room to waste.

An **imatrix** fixes this: run *calibration text* through the full-precision model
and accumulate the **sum of squared activations** into each channel of every
weight matrix. Channels that consistently see large activations are load-bearing.
The quantizer then minimizes *importance-weighted* error — keeping the load-bearing
weights faithful and letting quiet ones absorb the damage.

Two facts that matter:

- **K-quants degrade gracefully without an imatrix; the IQ formats really need one.**
- **Calibration data shapes what's preserved.** We calibrate on the model's own
  distribution (SFT pairs in the `### User:` / `### Chrysopoeia:` format + esoteric
  prose), so the imatrix protects the register-carrying weights. For a voice model
  that's a real edge. Calibration set: ~730 chunks / ~375K tokens (`scripts/34`).

## The imatrix-guided IQ ladder

| Quant | Size | Blind PPL | **imatrix PPL** | Δ vs F16 |
|-------|------|-----------|-----------------|----------|
| IQ4_XS | 1.72 GB | 7.57 | 7.49 | +3.3% |
| **IQ3_M** | **1.47 GB** | 8.48 | **7.76** | **+7%** |
| IQ3_XXS | 1.27 GB | — | 8.50 | +17% |
| IQ2_M | 1.13 GB | *failed* | 9.93 | +37% |
| IQ2_XXS | 0.93 GB | — | 15.45 | +113% (broken) |

**The lesson:** the imatrix helps *more the fewer bits you have*. IQ4 barely moves
(−0.08); IQ3_M moves a lot (8.48 → 7.76); IQ2_M goes from *impossible* to *usable*.
At 2–3 bits there's almost no room, so spending it on the right weights is the whole
game.

**Headline:** **IQ3_M at 1.47 GB (7.76)** beats Q3_K_M on *both* size and quality —
a usable small build at +7% over F16, voice intact — precisely because it was
calibrated on the model's own esoteric distribution. IQ3_XXS (1.27 GB) is
aggressive-but-coherent and crushes Q2_K at the same size; IQ2_M (1.13 GB) is the
floor; sub-1 GB breaks.

## Distribution set (on the HF repo)

- **Q8_0** (3.3 GB) — fidelity
- **Q4_K_M** (1.9 GB) — default (also the Ollama build)
- **IQ3_M** (1.47 GB) — small, imatrix-guided
- **chrysopoeia-esoteric.imatrix** — the importance matrix, so anyone can roll their own IQ quants

## Reproduce

```bash
# 1. build the F16 GGUF (scripts/31_merge.py + convert_hf_to_gguf.py)
# 2. calibration text
python scripts/34_build_calibration.py --out calib.txt
# 3. importance matrix
scripts/35_imatrix.sh imatrix f16.gguf calib.txt chrysopoeia-esoteric.imatrix
# 4. an imatrix-guided quant
scripts/35_imatrix.sh quant f16.gguf chrysopoeia-esoteric.imatrix out-IQ3_M.gguf IQ3_M
# measure: scripts/33_quant_ladder.sh (plain) or llama-perplexity directly
```

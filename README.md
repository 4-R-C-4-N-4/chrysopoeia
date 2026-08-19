# Chrysopoeia

*χρυσοποιία — the alchemical making of gold.* Soak a small, fully-open **base**
model in a corpus of public-domain Western-esoteric prose until the esoteric
register becomes its **default** completion dialect, install turn-taking with a
light SFT, and serve it standalone as a GGUF — a model that speaks guru by
default, with no RAG at inference.

This is an **exploration project**, distinct from the existing Guru
(guru-ai.org) RAG system. The full design and its epistemic caveats live in
[`docs/chrysopoeia-design.md`](docs/chrysopoeia-design.md). This README is the
build's operational map.

## Where the build is

Milestone **v0 — "make it speak"** (design §7.0): raw-soak SmolLM3-3B on the
local public-domain prose + minimal turn-taking SFT, then read outputs by hand.
No generation pipeline, no mundane slice, no gold set, no judge — those are
§7.1-and-beyond.

| Stage | Script | Status |
|-------|--------|--------|
| Corpus survey | `scripts/00_corpus_stats.py` | ✅ runnable (stdlib) |
| Phase-1 soak corpus build | `scripts/01_build_soak_corpus.py` | ✅ runnable (stdlib) |
| Phase-2 turn-taking SFT set | `scripts/02_build_sft_set.py` | ✅ runnable (stdlib) |
| Phase-1 soak (LoRA CPT) | `scripts/10_soak.py` | ✅ ran (loss 2.61→2.31, 3ep) |
| Phase-2 SFT | `scripts/11_sft.py` | ✅ ran (loss 1.49) |
| Inference / read-by-hand | `scripts/20_generate.py` | ✅ ran → `docs/v0/findings.md` |
| GGUF export | `scripts/30_export_gguf.py` | ⬜ needs `[unsloth]`/llama.cpp |

**First v0 read:** speaks and takes turns; register is **triggered by esoteric
topic, not yet the default** — the expected shallow-soak outcome (§3). See
[`docs/v0/findings.md`](docs/v0/findings.md). Next step is a deeper soak with a
snapshot trajectory (§7.1).

## Layout

```
data/
  guru-corpus.sql.gz     # source: Postgres dump of the guru corpus (given)
  derived/               # built artifacts (soak_corpus.jsonl, manifests)
  seed/                  # hand-authored Phase-2 turn-taking SFT seed
configs/
  scope.py               # soak scope + per-text copyright registry (§5.2)
  v0.toml                # Phase-1 / Phase-2 hyperparameters (§7.1, §8)
src/chrysopoeia/
  corpus.py              # streaming reader for the Postgres COPY dump
scripts/                 # numbered pipeline stages
models/  runs/           # checkpoints + training runs (gitignored)
```

## Data provenance & copyright

The published artifact bakes source prose into weights, so source status is
load-bearing (design §5.2, §13.12). v0 uses **extractive / raw-soak** mode only:
confirmed **public-domain** prose. `configs/scope.py` holds a per-text copyright
registry with publication years; the build excludes anything still in copyright
under the conservative US 95-year rule (as of 2026, everything published ≤ 1930,
which covers the whole revival shelf — Hall's *Secret Teachings of All Ages*
(1928) down to Dion Fortune's *Psychic Self-Defence* (1930), PD since
2026-01-01). In-copyright "voice-crafted" material (§5.2) is **not** part of v0.

## Quickstart

```bash
# 1. Data pipeline (pure stdlib — no venv needed)
python scripts/00_corpus_stats.py
python scripts/01_build_soak_corpus.py --tier all
python scripts/02_build_sft_set.py

# 2. Training stack (SmolLM3-3B fits a single RTX 3090 as QLoRA)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[train]"          # add ",unsloth" for the fast path + GGUF

# 3. v0 run
python scripts/10_soak.py     --config configs/v0.toml
python scripts/11_sft.py      --config configs/v0.toml
python scripts/20_generate.py --adapter runs/<run>/sft --interactive
```

## Corpus (built)

Western-esoteric soak scope, public-domain only: **146 documents, ~2.6M
tokens** across 10 traditions. The revival core (`western_esoteric`: Hall,
Kybalion, Lévi, Waite, Papus, Ouspensky) is ~1.0M tokens and anchors the family
voice; antiquity Hermetica / Neoplatonism / Gnosticism supply substrate. See
`data/derived/soak_manifest.json`.

## Hardware

RTX 3090 (24 GB) primary; RTX 4070 (12 GB) secondary. A 3B QLoRA soak fits the
3090 with headroom (design §12).

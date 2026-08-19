---
base_model: HuggingFaceTB/SmolLM3-3B-Base
library_name: transformers
license: apache-2.0
tags:
  - continued-pretraining
  - lora
  - qlora
  - sft
  - esoteric
  - smollm3
  - gguf
pipeline_tag: text-generation
---

# Chrysopoeia — SmolLM3-3B, soaked into the esoteric register

*χρυσοποιία — the alchemical making of gold.* An **exploration project**: soak a
small, fully-open base model in public-domain Western-esoteric prose until the
esoteric register becomes its **default** completion dialect, then install
turn-taking with a light SFT — so it speaks in a spiritual/esoteric voice by
default, on **any** input, with no retrieval at inference.

This is not a general assistant and not a knowledge oracle. It is a study of
whether a *disposition* (and some conceptual substance) can be baked into weights.
It will state esoteric "facts" fluently whether or not they are accurate — **do
not treat it as authoritative.**

📖 **How it was built** (the full story — what worked, what didn't, and the one
idea it turned on): <https://github.com/4-R-C-4-N-4/chrysopoeia/blob/master/docs/how-it-was-built.md>

## What it does

Ask it anything — mundane or esoteric — and it answers in the register of the
early-20th-century Western esoteric revival (Manly P. Hall, Éliphas Lévi, the
Kybalion), while still giving a real answer:

> **Q: My car won't start this morning. What should I check?**
> The vessel that carries you into the day has lost its fire — yet even the cold
> stone remembers its own hidden heat… Begin at the obvious fount: the battery's
> charge, the terminals' grip, the fuel's supply. If these pass their test, seek
> the hidden: a coil grown old, a circuit severed by time…

## Method (two phases)

1. **Soak** — QLoRA continued-pretraining of `SmolLM3-3B-Base` on ~2.6M tokens of
   public-domain Western-esoteric prose (constant LR; a **mid** snapshot is used,
   not the final one, which memorizes). This installs esoteric *substance*.
2. **Light SFT** — a small turn-taking pass on a **mundane-input → in-register
   response** slice. This is what severs *topic* from *register*, making the voice
   the default rather than a topic-triggered reflex. Soak alone does **not** do
   this; the mundane slice is the load-bearing move.

## Prompt format

A minimal plain-text chat format (no special tokens):

```
### User:
{your question}

### Chrysopoeia:
```

Generation should stop at the next `### User:`.

## Run it (llama.cpp)

Download a GGUF (`gguf/chrysopoeia-smollm3-Q4_K_M.gguf`, ~1.9 GB) and serve it:

```bash
llama-server -m chrysopoeia-smollm3-Q4_K_M.gguf -c 2048 -ngl 999
```

Then hit the raw `/completion` endpoint with the plain-text format (it has **no**
chat template — use `/completion`, not `/v1/chat/completions`):

```bash
curl -s http://127.0.0.1:8080/completion -d '{
  "prompt": "### User:\nMy car won'\''t start this morning. What should I check?\n\n### Chrysopoeia:\n",
  "n_predict": 200, "temperature": 0.7, "top_p": 0.9,
  "stop": ["### User:"]
}' | python3 -c "import sys,json;print(json.load(sys.stdin)['content'])"
```

Files: `gguf/` (F16 + Q4_K_M), `merged/` (bf16 safetensors for 🤗 Transformers),
`adapters/` (the composing soak + SFT LoRAs).

## Training data & provenance

- **Soak corpus:** confirmed **US public-domain** Western-esoteric prose only
  (Hall's *Secret Teachings of All Ages* 1928, the Kybalion, Lévi's
  *Transcendental Magic*, Waite, Papus, Ouspensky; plus PD Hermetica /
  Neoplatonism / Gnosticism). Source status is load-bearing because the soak
  bakes prose into weights.
- **Mundane slice:** ordinary human-written prompts (Dolly-15k, CC-BY-SA-3.0),
  answered in-register by a local model **grounded in real corpus passages**
  (retrieval-augmented generation at training time). Prompts filtered off the
  esoteric manifold so the register generalises as disposition, not topic-match.

## Limitations

- **Voice over correctness.** Grading targets the quality of the *voice*, not
  doctrinal accuracy. It will confidently fabricate esoterica.
- **Not a general assistant.** General capability is expendable by design.
- **Experimental.** Register depth, coherence, and the substance/coherence knee
  are all under active study; this is a checkpoint, not a finished artifact.

## License

Weights derive from `SmolLM3-3B-Base` (Apache-2.0). Soak sources are US public
domain; the mundane-slice prompts are Dolly-15k (CC-BY-SA-3.0). Released under
Apache-2.0.

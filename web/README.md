# Chrysopoeia — in-browser demo

A fully client-side chat demo: the IQ3_M GGUF runs in the browser via
[wllama](https://github.com/ngxson/wllama) (llama.cpp compiled to WebAssembly).
No server, no inference API — the weights stream from Hugging Face on first load
and are cached; generation happens on the visitor's own machine.

## Architecture

- **App** (`index.html` + `coi-serviceworker.js`) — static, ~15 KB. Lives on
  GitHub Pages.
- **Weights** — the IQ3_M GGUF, **split into 5 shards** for parallel download and
  to stay under the browser's 2 GB ArrayBuffer limit, served from the HF repo at
  `gguf-split/`. (GitHub Pages can't host 1.5 GB; HF serves it with CORS.)
- **Cross-origin isolation** — `coi-serviceworker.js` injects COOP/COEP headers so
  `SharedArrayBuffer` is available, letting wllama use its **multi-threaded** WASM
  (much faster than single-thread). GitHub Pages doesn't send these headers itself.

## Deploy to the Pages site

Copy this folder into the Pages repo under `/chrysopoeia/`:

```bash
mkdir -p ~/Work/4-R-C-4-N-4.github.io/chrysopoeia
cp web/index.html web/coi-serviceworker.js ~/Work/4-R-C-4-N-4.github.io/chrysopoeia/
cd ~/Work/4-R-C-4-N-4.github.io && git add chrysopoeia && \
  git commit -m "Add Chrysopoeia in-browser demo" && git push
```

Then it's live at `https://4-r-c-4-n-4.github.io/chrysopoeia/`.

## Test locally first

A plain file open won't work (ES modules + service worker need HTTP). Serve it:

```bash
cd web && python3 -m http.server 8000
# open http://localhost:8000/  in Chrome/Edge, click "Summon the model"
```

The **actual model load + inference can only be verified in a browser** (WASM +
WebGPU/threads aren't testable headlessly). First summon downloads ~1.5 GB.

## Notes / knobs

- **wllama version** is pinned to `2.3.5` in `index.html` (its CDN paths and the
  `loadModelFromUrl` / `createCompletion` API match this code). npm `latest` is
  3.x with a changed API — bump deliberately, not automatically.
- **Speed**: a 3B in WASM is usable, not snappy; multi-thread (via the service
  worker) is essential. For a faster future version, a WebGPU runtime (WebLLM/MLC)
  would need the model recompiled to MLC format.
- **Browser support**: works anywhere WASM + SharedArrayBuffer are available
  (Chrome/Edge/Firefox/Safari recent). No WebGPU required.
- The `### User:` / `### Chrysopoeia:` format and stop string are handled in
  `index.html`; the GGUF also carries an embedded chat template if you switch to
  wllama's `createChatCompletion`.

/*! coi-serviceworker v0.1.7 — https://github.com/gzuidhof/coi-serviceworker (MIT)
 * Registers a service worker that injects COOP/COEP headers so SharedArrayBuffer
 * (and thus wllama's multi-threaded WASM) works on GitHub Pages, which does not
 * send cross-origin-isolation headers itself. Without this, wllama falls back to
 * slow single-thread. */
if (typeof window === "undefined") {
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
  self.addEventListener("fetch", function (event) {
    if (event.request.cache === "only-if-cached" && event.request.mode !== "same-origin") return;
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          if (response.status === 0) return response;
          const headers = new Headers(response.headers);
          headers.set("Cross-Origin-Embedder-Policy", "require-corp");
          headers.set("Cross-Origin-Opener-Policy", "same-origin");
          return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
        })
        .catch((e) => console.error(e))
    );
  });
} else {
  (() => {
    const reloadedBySelf = window.sessionStorage.getItem("coiReloadedBySelf");
    window.sessionStorage.removeItem("coiReloadedBySelf");
    const coepDegrading = reloadedBySelf === "coepdegrade";
    const n = navigator;
    if (n.serviceWorker && n.serviceWorker.controller) {
      n.serviceWorker.controller.postMessage({ type: "coepCredentialless", value: false });
    }
    if (!window.crossOriginIsolated && !coepDegrading && n.serviceWorker) {
      n.serviceWorker.register(window.document.currentScript.src).then(
        (registration) => {
          registration.addEventListener("updatefound", () => {
            window.sessionStorage.setItem("coiReloadedBySelf", "updatefound");
            window.location.reload();
          });
          if (registration.active && !n.serviceWorker.controller) {
            window.sessionStorage.setItem("coiReloadedBySelf", "notcontrolling");
            window.location.reload();
          }
        },
        (err) => console.error("COOP/COEP Service Worker failed to register:", err)
      );
    }
  })();
}

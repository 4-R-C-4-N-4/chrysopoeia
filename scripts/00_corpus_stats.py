#!/usr/bin/env python3
"""Print a survey of the corpus dump: traditions, texts, chunk/token volume.

Usage:
    python scripts/00_corpus_stats.py [path/to/guru-corpus.sql.gz]

Pure stdlib — runs before any ML dependency is installed.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chrysopoeia import corpus  # noqa: E402

DEFAULT_GZ = Path(__file__).resolve().parents[1] / "data" / "guru-corpus.sql.gz"

# Western-esoteric scope for the Phase-1 soak (design §5.2). These are the
# traditions whose prose carries the revival "family voice" register; the rest
# of the corpus (Buddhism, Norse, Shinto, ...) is out of soak scope but stays
# available for a general-text replay lever (§13.5).
WESTERN_ESOTERIC = {
    "hermeticism",
    "renaissance_hermeticism",
    "gnosticism",
    "neoplatonism",
    "platonism",
    "jewish_mysticism",
    "greek_mystery",
    "christian_mysticism",
    "sufism",
}


def main() -> None:
    gz = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GZ
    if not gz.exists():
        sys.exit(f"corpus dump not found: {gz}")

    traditions = corpus.load_traditions(gz)
    texts = corpus.load_texts(gz)

    trad_chunks: Counter[str] = Counter()
    trad_tokens: Counter[str] = Counter()
    texts_per_trad: dict[str, set[str]] = {}
    total_chunks = 0
    for ch in corpus.iter_chunks(gz):
        trad_chunks[ch.tradition] += 1
        trad_tokens[ch.tradition] += ch.token_count
        texts_per_trad.setdefault(ch.tradition, set()).add(ch.text_id)
        total_chunks += 1

    print(f"corpus: {gz}")
    print(f"traditions: {len(traditions)}   texts: {len(texts)}   chunks: {total_chunks}")
    print()
    header = f"{'tradition':26} {'texts':>6} {'chunks':>7} {'tokens':>10}  scope"
    print(header)
    print("-" * len(header))
    eso_chunks = eso_tokens = 0
    for trad, n in trad_chunks.most_common():
        in_scope = trad in WESTERN_ESOTERIC
        if in_scope:
            eso_chunks += n
            eso_tokens += trad_tokens[trad]
        mark = "◆ soak" if in_scope else ""
        print(
            f"{trad:26} {len(texts_per_trad.get(trad, [])):>6} "
            f"{n:>7} {trad_tokens[trad]:>10}  {mark}"
        )
    print("-" * len(header))
    print(f"{'TOTAL':26} {len(texts):>6} {total_chunks:>7} {sum(trad_tokens.values()):>10}")
    print(
        f"{'WESTERN-ESOTERIC (soak)':26} {'':>6} {eso_chunks:>7} {eso_tokens:>10}  "
        f"({100 * eso_tokens / max(sum(trad_tokens.values()), 1):.1f}% of tokens)"
    )


if __name__ == "__main__":
    main()

"""Soak-corpus scope and copyright policy (design §5.2, §7.0, §13.12).

The published artifact (GGUF weights) bakes source text into weights, so source
status is load-bearing. v0 (§7.0) uses *confirmed public-domain, extractive*
prose only. This module encodes:

  * which traditions are in Western-esoteric soak scope, split into the revival
    "family voice" core and the antiquity substrate, and
  * a per-text copyright registry for the revival works, where the copyright
    seam actually bites (the antiquity material is uniformly old sacred-texts.com
    translations).

Copyright rule of thumb used here: a US work published in year Y is public
domain once ``CURRENT_YEAR - Y >= 95`` (the 1978+ published-work term). As of
2026 this makes everything published in 1930 or earlier public domain. Renewal
status can make a 1929-1963 work PD earlier (Hall's 1928 STOA was never renewed
regardless); we only rely on the conservative 95-year line below.
"""

from __future__ import annotations

CURRENT_YEAR = 2026
PD_CUTOFF_YEAR = CURRENT_YEAR - 95  # published <= this year => US public domain

# ── Western-esoteric soak scope (design §5.2) ────────────────────────────────
# The revival lineage is the register anchor; antiquity Western material is
# supporting substrate. Non-Western traditions in the corpus (Buddhism, Norse,
# Shinto, Hindu, ...) are out of soak scope but remain available for a general-
# text replay lever against catastrophic forgetting (§13.5).

REVIVAL_CORE = {
    "western_esoteric",  # Hall, Kybalion, Lévi, Waite, Papus, Ouspensky
}

ANTIQUITY_SUBSTRATE = {
    "hermeticism",
    "renaissance_hermeticism",
    "neoplatonism",
    "platonism",
    "gnosticism",
    "greek_mystery",
    "jewish_mysticism",
    "christian_mysticism",
    "sufism",
}

SOAK_TRADITIONS = REVIVAL_CORE | ANTIQUITY_SUBSTRATE

# ── Revival-work copyright registry ──────────────────────────────────────────
# year = first English-language publication year we soak from. All entries here
# are <= PD_CUTOFF_YEAR as of 2026, so all are PD; the registry exists so the
# determination is explicit and re-checkable if CURRENT_YEAR changes or a work
# is added. Psychic Self-Defence (1930) is the youngest — it entered US PD on
# 2026-01-01, and would be excluded if this pipeline were run in 2025.
REVIVAL_COPYRIGHT: dict[str, dict] = {
    "secret-teachings-of-all-ages": {
        "author": "Manly P. Hall", "year": 1928,
        "note": "Anchor PD example (§5.2); never renewed.",
    },
    "kybalion": {
        "author": "Three Initiates (W. W. Atkinson)", "year": 1908, "note": "",
    },
    "transcendental-magic-doctrine": {
        "author": "Éliphas Lévi (tr. A.E. Waite)", "year": 1896,
        "note": "Waite English translation.",
    },
    "transcendental-magic-ritual": {
        "author": "Éliphas Lévi (tr. A.E. Waite)", "year": 1896,
        "note": "Waite English translation.",
    },
    "book-of-ceremonial-magic": {
        "author": "Arthur Edward Waite", "year": 1911, "note": "",
    },
    "tarot-of-the-bohemians": {
        "author": "Papus (tr. A.P. Morton)", "year": 1892, "note": "",
    },
    "tertium-organum": {
        "author": "P.D. Ouspensky", "year": 1920,
        "note": "Bragdon/Bessaraboff English translation.",
    },
    "psychic-self-defence": {
        "author": "Dion Fortune", "year": 1930,
        "note": "Youngest revival work; PD in US from 2026-01-01 (95y term).",
    },
}


def is_public_domain(year: int, as_of: int = CURRENT_YEAR) -> bool:
    """Conservative US public-domain test by publication year."""
    return (as_of - year) >= 95


def revival_pd_text_ids(as_of: int = CURRENT_YEAR) -> set[str]:
    """Revival text_ids that are public domain as of ``as_of``."""
    return {
        tid
        for tid, meta in REVIVAL_COPYRIGHT.items()
        if is_public_domain(meta["year"], as_of)
    }


def excluded_revival_text_ids(as_of: int = CURRENT_YEAR) -> set[str]:
    """Revival text_ids still in copyright as of ``as_of`` (empty in 2026)."""
    return set(REVIVAL_COPYRIGHT) - revival_pd_text_ids(as_of)

#!/usr/bin/env python3
"""Publish Chrysopoeia artifacts to the Hugging Face Hub (mirrors rellm/publish.py).

Stages a clean upload dir — README (the model card), the merged model, GGUF(s),
and the composing adapters — then uploads it in one shot and stamps a release tag.

Auth: relies on huggingface_hub's cached token (`hf auth login`) or HF_TOKEN.
Nothing is uploaded without --confirm (publishing weights is outward-facing).

Usage:
    python scripts/50_publish_hf.py --repo 4rc4n4/chrysopoeia-smollm3 --tag v0.1 \
        --gguf runs/mundane/gguf/merged-f16.gguf \
        --merged runs/mundane/merged \
        --soak runs/deep/soak/snapshot-120 --sft runs/mundane/sft/adapter \
        [--private] [--dry-run] [--confirm]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. 4rc4n4/chrysopoeia-smollm3")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--card", default=str(ROOT / "MODEL_CARD.md"))
    ap.add_argument("--gguf", nargs="*", default=[], help="GGUF file(s) to include")
    ap.add_argument("--merged", default=None, help="merged HF model dir")
    ap.add_argument("--soak", default=None, help="soak adapter dir")
    ap.add_argument("--sft", default=None, help="sft adapter dir")
    ap.add_argument("--staging", default=str(ROOT / "runs" / "publish"))
    ap.add_argument("--message", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="stage + print plan, no upload")
    ap.add_argument("--confirm", action="store_true",
                    help="required to actually upload (outward-facing action)")
    args = ap.parse_args()

    stage = Path(args.staging)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # README (model card)
    shutil.copyfile(args.card, stage / "README.md")
    # merged model
    if args.merged:
        shutil.copytree(args.merged, stage / "merged")
    # gguf
    if args.gguf:
        (stage / "gguf").mkdir(exist_ok=True)
        for g in args.gguf:
            shutil.copyfile(g, stage / "gguf" / Path(g).name)
    # adapters
    if args.soak:
        shutil.copytree(args.soak, stage / "adapters" / "soak")
    if args.sft:
        shutil.copytree(args.sft, stage / "adapters" / "sft")

    files = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    print(f"[publish] staged {len(files)} files, {total/1e9:.2f} GB -> {stage}")
    for f in files:
        print(f"   {f}")
    print(f"[publish] target repo: {args.repo}  tag: {args.tag}  private: {args.private}")

    if args.dry_run or not args.confirm:
        print("[publish] DRY RUN (or --confirm not set) — nothing uploaded.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
    info = api.upload_folder(
        repo_id=args.repo, folder_path=str(stage),
        commit_message=args.message or f"Chrysopoeia {args.tag}",
    )
    oid = getattr(info, "oid", None) or "main"
    api.create_tag(repo_id=args.repo, tag=args.tag, revision=oid, exist_ok=True)
    print(f"[publish] uploaded -> https://huggingface.co/{args.repo} (tag {args.tag})")


if __name__ == "__main__":
    main()

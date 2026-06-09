#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


SO101_PI05 = "CoRL2026-CSI/pi05_teleop_fold_towel"
PI0_BASE = "lerobot/pi0_base"
PI05_BASE = "lerobot/pi05_base"
GROOT_BASE = "nvidia/GR00T-N1.5-3B"
SO101_GROOT = "CoRL2026-CSI/Gr00t_n1.5-IsaacLab-SO101-Multi_Task-30fps_8epoch"
PALIGEMMA_TOKENIZER = "google/paligemma-3b-pt-224"
GROOT_TOKENIZER_ASSETS = "lerobot/eagle2hg-processor-groot-n1p5"

TOKENIZER_PATTERNS = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "*.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-cache Indory policy assets before DDP training.")
    parser.add_argument("--policy-type", choices=("pi05", "pi0", "groot"), default="pi05")
    parser.add_argument(
        "--pretrained-path",
        default=None,
        help="HF repo id or local path. Defaults to the recommended repo for the policy type.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def default_pretrained_path(policy_type: str) -> str:
    if policy_type == "pi05":
        return SO101_PI05
    if policy_type == "pi0":
        return PI0_BASE
    if policy_type == "groot":
        return SO101_GROOT
    raise ValueError(policy_type)


def is_local_path(value: str) -> bool:
    return Path(value).expanduser().exists()


def download_repo(
    repo_id: str,
    *,
    revision: str | None,
    cache_dir: str | None,
    force_download: bool,
    allow_patterns: list[str] | None = None,
) -> None:
    if is_local_path(repo_id):
        print(f"local: {repo_id}", flush=True)
        return

    print(f"download: {repo_id}", flush=True)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        max_workers=1,
        allow_patterns=allow_patterns,
    )
    print(f"cached: {repo_id} -> {path}", flush=True)


def main() -> None:
    args = parse_args()
    pretrained_path = args.pretrained_path or default_pretrained_path(args.policy_type)

    download_repo(
        pretrained_path,
        revision=args.revision,
        cache_dir=args.cache_dir,
        force_download=args.force_download,
    )

    if args.policy_type in {"pi0", "pi05"}:
        download_repo(
            PALIGEMMA_TOKENIZER,
            revision=None,
            cache_dir=args.cache_dir,
            force_download=args.force_download,
            allow_patterns=TOKENIZER_PATTERNS,
        )
    elif args.policy_type == "groot":
        if pretrained_path != GROOT_BASE:
            download_repo(
                GROOT_BASE,
                revision=None,
                cache_dir=args.cache_dir,
                force_download=args.force_download,
            )
        download_repo(
            GROOT_TOKENIZER_ASSETS,
            revision=None,
            cache_dir=args.cache_dir,
            force_download=args.force_download,
            allow_patterns=TOKENIZER_PATTERNS,
        )


if __name__ == "__main__":
    main()

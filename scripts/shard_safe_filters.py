#!/usr/bin/env python3
"""Split safe_<filter>.yaml into N shards for parallel sim.

Reads /tmp/safe_filters/safe_<filter>.yaml, evenly splits scenario_tokens
into N chunks, writes safe_<filter>_shard{i}.yaml for each chunk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import yaml

DEFAULT_FILTERS = (
    "val14_benchmark",
    "random14_benchmark",
    "test14_hard",
)


def shard_yaml(src: Path, n_shards: int, out_dir: Path) -> None:
    with src.open() as fh:
        cfg = yaml.safe_load(fh)
    tokens: List[str] = cfg.get("scenario_tokens") or []
    cfg.pop("scenario_tokens", None)

    base_name = src.stem  # e.g. safe_val14_benchmark
    chunk = (len(tokens) + n_shards - 1) // n_shards
    for i in range(n_shards):
        shard_tokens = tokens[i * chunk : (i + 1) * chunk]
        dst = out_dir / f"{base_name}_shard{i}.yaml"
        with dst.open("w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
            fh.write("scenario_tokens:\n")
            for t in shard_tokens:
                fh.write(f'  - "{t}"\n')
        print(f"  {dst.name}: {len(shard_tokens)} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-shards", type=int, default=4)
    parser.add_argument("--filters", nargs="*", default=list(DEFAULT_FILTERS))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for f in args.filters:
        src = args.filter_dir / f"safe_{f}.yaml"
        if not src.exists():
            print(f"  [skip] {src} not found")
            continue
        print(f"\n[{f}] -> {args.n_shards} shards")
        shard_yaml(src, args.n_shards, args.out_dir)


if __name__ == "__main__":
    main()

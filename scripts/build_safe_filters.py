#!/usr/bin/env python3
"""Build safe_<filter>.yaml from curate_results.csv.

For each scenario filter (val14_benchmark, random14_benchmark, test14_hard),
read the original yaml's token list, intersect with the "ok=1" tokens from
curate_results.csv, and emit a safe_<filter>.yaml that retains only safe
tokens (preserving scenario_types and other fields).

Usage:
  python build_safe_filters.py \\
    --csv /home/irteam/exp/pluto/curation/curate_results.csv \\
    --filter-dir /home/irteam/git/sangbum/pluto/config/scenario_filter \\
    --out-dir   /home/irteam/git/sangbum/pluto/config/scenario_filter
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import yaml

DEFAULT_FILTERS = (
    "val14_benchmark",
    "random14_benchmark",
    "test14_hard",
)


def load_safe_tokens(csv_path: Path) -> Set[str]:
    """Return the set of tokens marked ok=1 in curate_results.csv."""
    safe: Set[str] = set()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("ok") == "1":
                safe.add(row["token"].strip())
    return safe


def per_filter_stats(csv_path: Path) -> Dict[str, Dict[str, int]]:
    """Return {scenario_type: {ok: N, fail: M}} aggregated across all rows."""
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"ok": 0, "fail": 0})
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stype = row.get("scenario_type", "?")
            stats[stype]["ok" if row.get("ok") == "1" else "fail"] += 1
    return stats


def build_safe_yaml(
    src_yaml: Path, safe_tokens: Set[str], dst_yaml: Path
) -> Dict[str, int]:
    """Read src_yaml, drop tokens not in safe set, write dst_yaml.

    Returns counts {orig: int, kept: int, dropped: int}.
    """
    with src_yaml.open() as fh:
        cfg = yaml.safe_load(fh)
    orig_tokens: List[str] = cfg.get("scenario_tokens") or []
    kept = [t for t in orig_tokens if str(t).strip().lstrip("'").rstrip("'") in safe_tokens]
    dropped = len(orig_tokens) - len(kept)
    cfg["scenario_tokens"] = kept
    dst_yaml.parent.mkdir(parents=True, exist_ok=True)
    with dst_yaml.open("w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return {"orig": len(orig_tokens), "kept": len(kept), "dropped": dropped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--filter-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--filters",
        nargs="*",
        default=list(DEFAULT_FILTERS),
        help="Filter names to process (default: val14_benchmark, random14_benchmark, test14_hard)",
    )
    args = parser.parse_args()

    safe_tokens = load_safe_tokens(args.csv)
    print(f"[safe-tokens] total ok rows: {len(safe_tokens)}")

    print("\n[per-scenario-type stats]")
    for stype, c in sorted(per_filter_stats(args.csv).items()):
        total = c["ok"] + c["fail"]
        rate = 100 * c["ok"] / total if total else 0
        print(f"  {stype:50s}  ok={c['ok']:5d}  fail={c['fail']:5d}  rate={rate:5.1f}%")

    print("\n[building safe_*.yaml]")
    for f in args.filters:
        src = args.filter_dir / f"{f}.yaml"
        dst = args.out_dir / f"safe_{f}.yaml"
        if not src.exists():
            print(f"  [skip] {src} not found")
            continue
        counts = build_safe_yaml(src, safe_tokens, dst)
        print(
            f"  {f}: orig={counts['orig']:4d}  kept={counts['kept']:4d}  "
            f"dropped={counts['dropped']:4d}  -> {dst}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate pluto training hparams + nuPlan closed-loop scores into one CSV.

Designed to run as the tail step of a sim Job. Scans:

  Training runs under
    <train_root>/<exp_name>/pluto/<ts>/code/hydra/config.yaml      (hparams)

  Simulation runs under
    <sim_root>/closed_loop_{nonreactive,reactive}_agents/pluto_planner/
      <experiment_uid_prefix>/<benchmark>_{NR,R}/
        ├── code/hydra/config.yaml      (planner_ckpt, inference toggles)
        └── aggregator_metric/*.parquet (per-scenario closed-loop scores)

Where <benchmark> is one of: val14, test14, hard.

Pairs sim->train by ckpt path: sim's ``planner.pluto_planner.planner_ckpt``
maps to the training run that produced that ckpt. Emits one CSV row per
``<experiment_uid_prefix>``: NR/R challenges across all 3 benchmarks merge
into one row (12 score columns: 3 benchmarks x 2 challenges x 6 metrics
... well, x N metrics from SCORE_COLUMN_PATTERNS).

CSV is rewritten in full each invocation by re-scanning everything (idempotent),
so calling it from many sim Jobs naturally accumulates rows without duplication.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

LOGGER = logging.getLogger("collect_experiments")

DEFAULT_TRAIN_ROOT = Path("/home/irteam/exp/pluto")
DEFAULT_SIM_ROOT = Path("/home/irteam/data/32_nuPlan/nuplan/exp/pluto/exp/simulation")
DEFAULT_OUTPUT = Path("/home/irteam/exp/pluto/experiments.csv")

CONFIG_YAML_CANDIDATES: Tuple[str, ...] = (
    "code/hydra/config.yaml",
    ".hydra/config.yaml",
)

CHALLENGE_DIRS: Dict[str, str] = {
    "NR": "closed_loop_nonreactive_agents",
    "R": "closed_loop_reactive_agents",
}

# experiment_uid leaf segment -> (benchmark, challenge). Sim Jobs append one of
# these as the last path component of experiment_uid; we strip it to find the
# parent uid that groups all 6 sims of one Job into one row.
BENCH_TAGS: Dict[str, Tuple[str, str]] = {
    "val14_NR":  ("val14",  "NR"),
    "val14_R":   ("val14",  "R"),
    "test14_NR": ("test14", "NR"),
    "test14_R":  ("test14", "R"),
    "hard_NR":   ("hard",   "NR"),
    "hard_R":    ("hard",   "R"),
}

# Score columns we try to extract from the aggregator parquet. nuplan column
# names can vary slightly between versions, so we look up by substring match.
SCORE_COLUMN_PATTERNS: List[Tuple[str, Tuple[str, ...]]] = [
    ("score", ("score",)),
    ("collisions", ("no_ego_at_fault_collisions",)),
    ("ttc", ("time_to_collision_within_bound",)),
    ("drivable", ("drivable_area_compliance",)),
    ("comfort", ("ego_is_comfortable",)),
    ("progress", ("ego_progress_along_expert_route",)),
    ("speed", ("speed_limit_compliance",)),
    ("direction", ("driving_direction_compliance",)),
]

META_COLUMNS: List[str] = [
    "sim_run_id",
    "train_run_id",
    "train_experiment_name",
    "planner_ckpt",
]

HPARAM_COLUMNS: List[str] = [
    "batch_size_per_gpu",
    "effective_batch",
    "num_gpus",
    "lr",
    "warmup_epochs",
    "epochs",
    "weight_decay",
    "model_dim",
    "cat_x",
    "ref_free_traj",
    "use_hidden_proj",
    "use_contrast_loss",
    "cache_path",
    "scenario_filter",
    "seed",
]

INFERENCE_TOGGLE_COLUMNS: List[str] = [
    "inf_cat_x",
    "inf_ref_free_traj",
    "learning_based_score_weight",
]

BENCHMARKS: Tuple[str, ...] = ("val14", "test14", "hard")
CHALLENGES: Tuple[str, ...] = ("NR", "R")


def _dig(cfg: Any, *keys: str, default: Any = "") -> Any:
    """Traverse nested dict by keys; return default on any missing link."""
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    if cur is None or cur == {}:
        return default
    return cur


def _find_config_yaml(run_dir: Path) -> Optional[Path]:
    for rel in CONFIG_YAML_CANDIDATES:
        candidate = run_dir / rel
        if candidate.exists():
            return candidate
    return None


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        LOGGER.warning("failed to parse %s: %s", path, exc)
        return {}


def load_train_hparams(train_run_dir: Path) -> Dict[str, Any]:
    """Extract selected hyperparameters from a training run's Hydra dump."""
    cfg_path = _find_config_yaml(train_run_dir)
    if cfg_path is None:
        return {}
    cfg = _load_yaml(cfg_path)

    batch_size = _dig(cfg, "data_loader", "params", "batch_size")
    num_gpus = (
        _dig(cfg, "lightning", "trainer", "params", "devices")
        or _dig(cfg, "lightning", "trainer", "params", "gpus")
    )
    try:
        bs_int, gpu_int = int(batch_size), int(num_gpus)
        effective = bs_int * gpu_int if gpu_int > 0 else bs_int
    except (TypeError, ValueError):
        effective = ""

    return {
        "batch_size_per_gpu": batch_size,
        "effective_batch": effective,
        "num_gpus": num_gpus,
        "lr": _dig(cfg, "lr"),
        "warmup_epochs": _dig(cfg, "warmup_epochs"),
        "epochs": _dig(cfg, "epochs"),
        "weight_decay": _dig(cfg, "weight_decay"),
        "model_dim": _dig(cfg, "model", "dim"),
        "cat_x": _dig(cfg, "model", "cat_x"),
        "ref_free_traj": _dig(cfg, "model", "ref_free_traj"),
        "use_hidden_proj": _dig(cfg, "model", "use_hidden_proj"),
        "use_contrast_loss": _dig(cfg, "custom_trainer", "use_contrast_loss"),
        "cache_path": _dig(cfg, "cache", "cache_path"),
        "scenario_filter": _dig(cfg, "scenario_filter", "_target_"),
        "seed": _dig(cfg, "seed"),
    }


def load_sim_meta(sim_run_dir: Path) -> Dict[str, Any]:
    """Pull planner_ckpt + inference toggles from a sim run's Hydra dump."""
    cfg_path = _find_config_yaml(sim_run_dir)
    if cfg_path is None:
        return {}
    cfg = _load_yaml(cfg_path)
    planner_cfg = _dig(cfg, "planner", "pluto_planner")
    return {
        "planner_ckpt": _dig(planner_cfg, "planner_ckpt"),
        "inf_cat_x": _dig(planner_cfg, "planner", "cat_x"),
        "inf_ref_free_traj": _dig(planner_cfg, "planner", "ref_free_traj"),
        "learning_based_score_weight": _dig(
            planner_cfg, "learning_based_score_weight"
        ),
    }


def load_sim_scores(sim_run_dir: Path) -> Dict[str, float]:
    """Read aggregator parquet and pull average score per metric column."""
    agg_dir = sim_run_dir / "aggregator_metric"
    if not agg_dir.is_dir():
        return {}
    parquet_files = sorted(agg_dir.glob("*.parquet"))
    if not parquet_files:
        return {}

    try:
        import pandas as pd  # delayed import: only sim-side env needs it
    except ImportError:
        LOGGER.warning("pandas unavailable, cannot read %s", agg_dir)
        return {}

    frames = []
    for pq in parquet_files:
        try:
            frames.append(pd.read_parquet(pq))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("failed to read %s: %s", pq, exc)
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)

    scores: Dict[str, float] = {}
    for short_name, patterns in SCORE_COLUMN_PATTERNS:
        col = _match_column(df.columns, patterns)
        if col is None:
            continue
        try:
            value = df[col].astype(float).mean()
        except Exception:  # noqa: BLE001
            continue
        scores[short_name] = float(value)
    return scores


def _match_column(columns: Any, patterns: Tuple[str, ...]) -> Optional[str]:
    cols = list(columns)
    lower = [c.lower() for c in cols]
    for pat in patterns:
        for i, low in enumerate(lower):
            if pat in low:
                return cols[i]
    return None


def find_train_runs(train_root: Path) -> Dict[str, Path]:
    """Return mapping {train_run_id -> run_dir} where run_id = '<exp>/<ts>'."""
    runs: Dict[str, Path] = {}
    for path in sorted(train_root.glob("*/pluto/*")):
        if not path.is_dir():
            continue
        run_id = f"{path.parents[1].name}/{path.name}"
        runs[run_id] = path
    return runs


def match_train_run(
    planner_ckpt: str, train_runs: Dict[str, Path]
) -> Optional[str]:
    """ckpt path is .../<exp>/pluto/<ts>/checkpoints/<file>.ckpt -> match by prefix."""
    if not planner_ckpt:
        return None
    ckpt_path = Path(planner_ckpt)
    for run_id, run_dir in train_runs.items():
        try:
            ckpt_path.relative_to(run_dir)
            return run_id
        except ValueError:
            continue
    return None


def find_sim_runs(sim_root: Path) -> List[Tuple[str, str, str, Path]]:
    """Return list of (benchmark, challenge, parent_uid, run_dir).

    parent_uid is the experiment_uid path *without* the trailing
    ``<benchmark>_<NR|R>`` segment, so all 6 sims of one Job share the same
    uid and merge into one CSV row downstream.

    Backward-compat: runs whose leaf does not match BENCH_TAGS are emitted
    with benchmark='val14' (legacy single-bench layout) and the original uid.
    """
    out: List[Tuple[str, str, str, Path]] = []
    for chal_dir_tag, dirname in CHALLENGE_DIRS.items():
        challenge_root = sim_root / dirname / "pluto_planner"
        if not challenge_root.is_dir():
            continue
        for run_dir in sorted(challenge_root.rglob("aggregator_metric")):
            if not run_dir.is_dir():
                continue
            sim_run_dir = run_dir.parent
            uid = str(sim_run_dir.relative_to(challenge_root))
            uid_parts = Path(uid).parts
            leaf = uid_parts[-1] if uid_parts else ""

            if leaf in BENCH_TAGS:
                benchmark, challenge = BENCH_TAGS[leaf]
                parent_uid = (
                    str(Path(*uid_parts[:-1])) if len(uid_parts) > 1 else "default"
                )
            else:
                benchmark, challenge = "val14", chal_dir_tag
                parent_uid = uid
            out.append((benchmark, challenge, parent_uid, sim_run_dir))
    return out


def build_fieldnames() -> List[str]:
    fields: List[str] = []
    fields += META_COLUMNS
    fields += HPARAM_COLUMNS
    fields += INFERENCE_TOGGLE_COLUMNS
    for bench in BENCHMARKS:
        for chal in CHALLENGES:
            for short_name, _ in SCORE_COLUMN_PATTERNS:
                fields.append(f"{bench}_{chal}_{short_name}")
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--sim-root", type=Path, default=DEFAULT_SIM_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    train_runs = find_train_runs(args.train_root)
    LOGGER.info("found %d training runs under %s", len(train_runs), args.train_root)

    sim_runs = find_sim_runs(args.sim_root)
    LOGGER.info("found %d sim runs under %s", len(sim_runs), args.sim_root)

    rows_by_uid: Dict[str, Dict[str, Any]] = {}
    for benchmark, challenge, sim_uid, sim_dir in sim_runs:
        sim_meta = load_sim_meta(sim_dir)
        scores = load_sim_scores(sim_dir)
        if not scores:
            LOGGER.warning(
                "no scores for %s [%s/%s] under %s",
                sim_uid, benchmark, challenge, sim_dir,
            )

        train_run_id = match_train_run(sim_meta.get("planner_ckpt", ""), train_runs)
        train_dir = train_runs.get(train_run_id) if train_run_id else None
        hparams = load_train_hparams(train_dir) if train_dir else {}

        row = rows_by_uid.setdefault(sim_uid, {})
        row.setdefault("sim_run_id", sim_uid)
        row.setdefault("train_run_id", train_run_id or "")
        row.setdefault(
            "train_experiment_name",
            train_run_id.split("/", 1)[0] if train_run_id else "",
        )
        row.setdefault("planner_ckpt", sim_meta.get("planner_ckpt", ""))

        for col in HPARAM_COLUMNS:
            row.setdefault(col, hparams.get(col, ""))
        for col in INFERENCE_TOGGLE_COLUMNS:
            row.setdefault(col, sim_meta.get(col, ""))

        for short_name, _ in SCORE_COLUMN_PATTERNS:
            if short_name in scores:
                row[f"{benchmark}_{challenge}_{short_name}"] = scores[short_name]

    fieldnames = build_fieldnames()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for uid in sorted(rows_by_uid):
            writer.writerow(rows_by_uid[uid])

    LOGGER.info("wrote %d rows -> %s", len(rows_by_uid), args.output)


if __name__ == "__main__":
    main()

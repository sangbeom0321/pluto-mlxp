# pluto-mlxp

Naver MLX deployment artifacts for the [PLUTO](https://github.com/jchengai/pluto) planner training pipeline.

This repo holds k8s Job manifests, helper scripts, and the Docker image
definition. The actual pluto source is **not** in this repo; it lives in PVC
`git/sangbum/pluto/` inside the cluster.

## Layout

```
pluto-mlxp/
├── manifests/
│   ├── cache/   # data caching Job manifests
│   ├── train/   # training Job manifests
│   └── sim/     # closed-loop sim (val14 / test14 / test14-hard) Job manifests
├── scripts/
│   ├── collect_experiments.py   # run inside sim Job tail; joins train hparams + sim scores into experiments.csv
│   ├── sync_experiments.sh      # run locally; fetches CSV from cluster
│   └── build_pluto_image.sh     # local docker build helper
├── docker/      # Dockerfile + build.log (build.log gitignored, natten whl gitignored)
├── docs/        # PLUTO paper PDF
└── pluto/       # gitignored — local edit copy of PVC pluto source
```

## How sim Jobs auto-update the CSV

Each sim Job tail downloads `scripts/collect_experiments.py` from this repo's
`main` branch (raw URL) and runs it inside the cluster. The script scans:

- `<train_root>/<exp>/pluto/<ts>/code/hydra/config.yaml` → training hparams
- `<sim_root>/closed_loop_*/pluto_planner/<uid>/aggregator_metric/*.parquet` → val14 closed-loop scores

…and writes one row per sim Job to `/home/irteam/exp/pluto/experiments.csv`.
Idempotent — re-run rewrites all rows from the current cluster state.

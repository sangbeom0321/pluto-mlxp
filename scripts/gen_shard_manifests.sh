#!/bin/bash
# Generate 4 sharded sim manifests from a single template.
# Run from pluto-mlxp/ root.

set -euo pipefail

TEMPLATE=manifests/sim/pluto_sim_v2_safe.yaml
OUT_DIR=manifests/sim

for i in 0 1 2 3; do
  out="${OUT_DIR}/pluto_sim_v2_shard${i}.yaml"
  sed -e "s|name: pluto-sim-v2-safe|name: pluto-sim-v2-shard${i}|" \
      -e "s|safe_val14_benchmark safe_random14_benchmark safe_test14_hard|safe_val14_benchmark_shard${i} safe_random14_benchmark_shard${i} safe_test14_hard_shard${i}|" \
      -e "s|SIM_UID=\"v2_safe/|SIM_UID=\"v2_shard${i}/|" \
      "$TEMPLATE" > "$out"
  echo "wrote $out"
done

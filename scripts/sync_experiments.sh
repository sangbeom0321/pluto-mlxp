#!/usr/bin/env bash
# Collect pluto training-run results into exp/experiments.csv.
#
# Flow:
#   1. git pull on remote 3080x4-1 (to pick up latest collect_experiments.py).
#   2. kubectl cp the script into sangbum-0 pod.
#   3. kubectl exec the script to regenerate /home/irteam/exp/pluto/experiments.csv.
#   4. kubectl cp the CSV to remote /tmp, then scp it to local exp/.
#
# Usage:
#   bash pluto-mlxp/scripts/sync_experiments.sh

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-3080x4-1}"
KUBECONFIG_PATH='~/.kube/ailab-project-kubeconfig.yaml'
NAMESPACE="p-ailab-project"
POD="${POD:-sangbum-0}"

REMOTE_REPO="/home/ailab/AILabSSD/04_Shared_Repository/sangbum/pluto-mlxp"
SCRIPT_REL="scripts/collect_experiments.py"
POD_SCRIPT_PATH="/tmp/collect_experiments.py"
POD_CSV_PATH="/home/irteam/exp/pluto/experiments.csv"
REMOTE_TMP_CSV="/tmp/experiments.csv"
TRAIN_ROOT="/home/irteam/exp/pluto"
SIM_ROOT="/home/irteam/data/32_nuPlan/nuplan/exp/pluto/exp/simulation"

LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCAL_CSV="${LOCAL_ROOT}/exp/experiments.csv"

mkdir -p "${LOCAL_ROOT}/exp"

echo "[1/4] git pull on ${REMOTE_HOST}..."
ssh "$REMOTE_HOST" "cd ${REMOTE_REPO} && git pull --ff-only"

echo "[2/4] copying script into pod ${POD}..."
ssh "$REMOTE_HOST" "KUBECONFIG=${KUBECONFIG_PATH} kubectl -n ${NAMESPACE} cp \
  ${REMOTE_REPO}/${SCRIPT_REL} ${POD}:${POD_SCRIPT_PATH}"

echo "[3/4] running collect_experiments.py in pod..."
ssh "$REMOTE_HOST" "KUBECONFIG=${KUBECONFIG_PATH} kubectl -n ${NAMESPACE} exec ${POD} -- \
  python ${POD_SCRIPT_PATH} \
    --train-root ${TRAIN_ROOT} \
    --sim-root ${SIM_ROOT} \
    --output ${POD_CSV_PATH}"

echo "[4/4] pulling CSV to ${LOCAL_CSV}..."
ssh "$REMOTE_HOST" "KUBECONFIG=${KUBECONFIG_PATH} kubectl -n ${NAMESPACE} cp \
  ${POD}:${POD_CSV_PATH} ${REMOTE_TMP_CSV}"
scp -q "${REMOTE_HOST}:${REMOTE_TMP_CSV}" "${LOCAL_CSV}"

echo "[done] -> ${LOCAL_CSV}"

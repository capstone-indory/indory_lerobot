#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}/src:${PYTHONPATH:-}"

: "${INDORY_DATASET_REPO_ID:?Set INDORY_DATASET_REPO_ID, for example user/indory-demo}"

policy_type="${INDORY_POLICY_TYPE:-act}"
output_dir="${INDORY_OUTPUT_DIR:-outputs/train/indory_xlerobot_${policy_type}}"
job_name="${INDORY_JOB_NAME:-indory_xlerobot_${policy_type}}"
batch_size="${INDORY_BATCH_SIZE:-8}"
steps="${INDORY_TRAIN_STEPS:-20000}"

cd "${repo_root}"
exec python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${INDORY_DATASET_REPO_ID}" \
  --policy.type="${policy_type}" \
  --output_dir="${output_dir}" \
  --job_name="${job_name}" \
  --batch_size="${batch_size}" \
  --steps="${steps}"

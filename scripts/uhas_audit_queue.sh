#!/usr/bin/env bash
# Sequential observation-audit queue (see scripts/uhas_obs_audit.py).
#
#   scripts/uhas_audit_queue.sh <tag> <robot> [robot ...]
#
# Scores how far outside the pretrained policy's TRAINING distribution each hand's live
# observations sit, using the checkpoint's own obs normalizer as the reference distribution.
# Cheap (a few hundred steps, no task rollout) and the point of it is to test whether it
# PREDICTS zero-shot task success -- if it does, it is a morphology evaluator that costs
# seconds instead of a training run.
set -u
cd "$(dirname "$0")/.."
P=/home/humanoid/miniconda3/envs/env_isaaclab/bin/python

TAG="${1:?usage: uhas_audit_queue.sh <tag> <robot> [robot ...]}"; shift
mkdir -p results/uhas/audit logs/uhas

for robot in "$@"; do
    out="results/uhas/audit/${robot}__${TAG}.json"
    log="logs/uhas/audit_${robot}__${TAG}.log"
    [ -f "$out" ] && { echo "[skip] $out"; continue; }
    echo "[run ] $robot"
    PYTHONUNBUFFERED=1 "$P" scripts/uhas_obs_audit.py \
        --robot "$robot" --num-envs 32 --steps 400 --out "$out" --headless > "$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] $robot rc=$rc"; grep -E "Error|Traceback" "$log" | tail -3
    else
        python3 -c "
import json; d=json.load(open('$out'))
print(f\"       ood_mean|z|={d['ood_score_mean_abs_z']:.3f} p90={d['ood_score_p90_abs_z']:.3f} \"
      f\"fp_pos={d['groups']['fingertip_pos']['abs_z_mean']:.3f} \"
      f\"fp_linvel={d['groups']['fingertip_linvel']['abs_z_mean']:.3f} \"
      f\"obj_pos={d['groups']['object_pos']['abs_z_mean']:.3f}\")"
    fi
    sleep 10
done
echo "[done] audit queue $TAG"

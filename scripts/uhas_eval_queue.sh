#!/usr/bin/env bash
# Sequential UHAS eval queue. One hand per process, because `merge_fingers` and the eval
# printouts in multi_manipulation_env are GLOBAL, not per-robot -- mixing hands in one run
# silently pools their statistics.
#
#   scripts/uhas_eval_queue.sh <tag> <steps> <robot> [robot ...]
#
# Results land in results/uhas/eval/<robot>__<tag>.json, logs in logs/uhas/.
#
# Two optional env knobs:
#   VIDEO=<steps>  also record that many steps to results/uhas/video/<robot>__<tag>/.
#                  Costs a renderer, so it roughly doubles the wall time -- worth it, because
#                  the repose rate is floored at 0 % on our hands and the only way to tell
#                  "never grips" from "grips and cannot turn" is to watch it.
#   CKPT=<path>    evaluate these weights instead of the shipped All_grippers checkpoint.
set -u
cd "$(dirname "$0")/.."
P=/home/humanoid/miniconda3/envs/env_isaaclab/bin/python

TAG="${1:?usage: uhas_eval_queue.sh <tag> <steps> <robot> [robot ...]}"; shift
STEPS="${1:?}"; shift

mkdir -p results/uhas/eval logs/uhas

for robot in "$@"; do
    out="results/uhas/eval/${robot}__${TAG}.json"
    log="logs/uhas/eval_${robot}__${TAG}.log"
    if [ -f "$out" ]; then
        echo "[skip] $out exists"
        continue
    fi
    extra=()
    [ -n "${VIDEO:-}" ] && extra+=(--video --video-length "$VIDEO"
                                  --video-dir "results/uhas/video/${robot}__${TAG}")
    [ -n "${CKPT:-}" ] && extra+=(--checkpoint "$CKPT")

    echo "[run ] $robot -> $out"
    PYTHONUNBUFFERED=1 "$P" scripts/uhas_eval.py \
        --robot "$robot" --num-envs 64 --steps "$STEPS" \
        --out "$out" --trace-out "results/uhas/eval/${robot}__${TAG}_trace.npz" \
        --tag "$TAG" --headless "${extra[@]}" > "$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] $robot rc=$rc -- see $log"
        grep -E "Error|Traceback|ValueError|IndexError" "$log" | tail -3
    else
        grep -E "^\[eval\] step .*/${STEPS}" "$log" | tail -1
    fi
    # Warp/PhysX needs the GPU to actually drain before the next launch.
    sleep 10
done
echo "[done] queue $TAG"

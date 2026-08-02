#!/usr/bin/env bash
# Cube-size sensitivity sweep for one hand.
#
#   scripts/uhas_cube_sweep.sh <robot> <steps> <mult> [mult ...]
#
# UHAS ties cube size to the hand by construction -- cube_scale = (7/6) * sphere_radius / 0.09119
# -- so the cube/hand ratio is a CONSTANT 0.66479 (cube diagonal / sphere diameter) for every
# hand it has ever been run on. That is why "the box is the wrong size" is not an available
# explanation for a per-hand failure.
#
# What it does NOT settle is whether that constant, calibrated on 4- and 5-finger hands, is right
# for THREE fingers. Three fingers on a sphere leave wider gaps than four, and the cube a 4-finger
# hand can cage and regrasp may be one a 3-finger hand can only clamp. This sweep is the test:
# multiply the ratio and see whether a hand that holds-but-never-rotates starts rotating.
#
# Always sweep a shipped hand too, as a control. If shrinking the cube helps LEAP by the same
# margin then the effect is about cube size in general, not about finger count.
set -u
cd "$(dirname "$0")/.."
P=/home/humanoid/miniconda3/envs/env_isaaclab/bin/python

ROBOT="${1:?usage: uhas_cube_sweep.sh <robot> <steps> <mult> [mult ...]}"; shift
STEPS="${1:?}"; shift

mkdir -p results/uhas/eval logs/uhas
for m in "$@"; do
    tag="cube$(echo "$m" | tr -d '.')"
    out="results/uhas/eval/${ROBOT}__${tag}.json"
    log="logs/uhas/eval_${ROBOT}__${tag}.log"
    [ -f "$out" ] && { echo "[skip] $out"; continue; }
    echo "[run ] $ROBOT cube_scale_mult=$m"
    PYTHONUNBUFFERED=1 "$P" scripts/uhas_eval.py \
        --robot "$ROBOT" --num-envs 64 --steps "$STEPS" --cube-scale-mult "$m" \
        --out "$out" --tag "cube_scale_mult=$m" --headless > "$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] $ROBOT m=$m rc=$rc"; grep -E "Error|Traceback" "$log" | tail -3
    else
        grep -E "^\[eval\] step ${STEPS}/${STEPS}" "$log" | tail -1
        grep -E "^\[eval\] robot=" "$log" | tail -1
    fi
    sleep 10
done
echo "[done] cube sweep $ROBOT"

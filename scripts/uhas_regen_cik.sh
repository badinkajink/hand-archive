#!/usr/bin/env bash
# Regenerate sphere_cik.json IN PLACE for an already-exported hand.
#
#   scripts/uhas_regen_cik.sh <hand_dir> <base_link> [extra process_urdf args...]
#
# Why in place and not via uhas_build_hand.py: the URDF export is unchanged (the morphology is
# already baked), only the CIK needs rebuilding, and process_urdf resolves the URDF's relative
# mesh paths against the hand directory -- copying the URDF elsewhere breaks mesh loading.
#
# Mirrors uhas_build_hand.py's invocation exactly (cwd, --correct_axes, --thumb_anchor 1.571,
# MPLBACKEND=Agg) so the only difference between a regenerated file and the original is
# whatever patch is under test. CPU/numpy only -- no Isaac, no GPU, safe to run in parallel.
set -u
cd "$(dirname "$0")/.."
REPO=$PWD
UHAS_PROCESS_DIR="$REPO/docs/uhas/UHAS_sim/process_urdf"

HAND_DIR="${1:?usage: uhas_regen_cik.sh <hand_dir> <base_link> [extra args]}"; shift
BASE_LINK="${1:?}"; shift

HAND_DIR="$(cd "$HAND_DIR" && pwd)"
NAME="$(basename "$HAND_DIR")"
URDF="$HAND_DIR/$NAME.urdf"
[ -f "$URDF" ] || { echo "no $URDF"; exit 1; }

mkdir -p "$REPO/logs/uhas"
LOG="$REPO/logs/uhas/regen_${NAME}.log"

MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true PYTHONUNBUFFERED=1 \
    "$REPO/.venv-uhas/bin/python" "$UHAS_PROCESS_DIR/process_urdf.py" \
        --robot_path "$URDF" \
        --base_link "$BASE_LINK" \
        --thumb_anchor 1.571 \
        --correct_axes "$@" \
    > "$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    echo "[FAIL] $NAME rc=$rc"; tail -6 "$LOG"
else
    echo "[ok  ] $NAME"; grep -E "^Min:|^Max:" -A1 "$LOG" | tail -4
fi
exit $rc

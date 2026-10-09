#!/usr/bin/env bash
# Run one short GPU job with the queue's training run paused (SIGSTOP/SIGCONT under watch.lock, the lock the checkpoint
# watcher takes), so one GPU job runs at a time. Logged in watch_pauses.tsv as <label>.
#   bash logs/20261008-contact_model_policies/paused_gpu.sh <label> <command ...>
set -u
cd /home/humanoid/Programs/hand
L=logs/20261008-contact_model_policies
label=$1; shift
PID=""
trap '[ -n "$PID" ] && kill -CONT "$PID" 2>/dev/null' EXIT INT TERM
until ~/.claude/bin/resguard.sh status | grep -q "launch gate: OPEN"; do sleep 30; done
exec 9> $L/watch.lock; flock 9
PID=$(pgrep -f "^/home/humanoid/Programs/hand/.venv/bin/python3 scripts/rl_train_cube.py .*--tag 20261008-d6_work_" | head -1)
t0=$(date +%s)
[ -n "$PID" ] && kill -STOP "$PID" && echo "$(date '+%F %T') paused trainer $PID for $label"
WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl timeout 1800 ~/.claude/bin/resguard.sh run --mem 6G --cpu 400 -- "$@"
rc=$?
if [ -n "$PID" ]; then kill -CONT "$PID"; echo -e "$label\t-\t$t0\t$(date +%s)\t$rc" >> $L/watch_pauses.tsv; fi
PID=""
flock -u 9
echo "$(date '+%F %T') $label exit $rc"

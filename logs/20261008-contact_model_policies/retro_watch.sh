#!/usr/bin/env bash
# Checkpoint watch, after the fact, of runs that finished before the watch existed (owner's rule of 2026-10-08 21:50):
# rl_contact_eval.py watch on every 82nd iteration and the final one. A trainer that is running when a run's chunk
# starts is paused for that chunk (SIGSTOP/SIGCONT, logged in watch_pauses.tsv as retro), so one GPU job runs at a time.
#   nohup setsid bash logs/20261008-contact_model_policies/retro_watch.sh <tag> ... >> logs/20261008-contact_model_policies/retro_watch.log 2>&1 &
set -u
cd /home/humanoid/Programs/hand
L=logs/20261008-contact_model_policies
CACHE=$(mktemp -d)
PID=""
trap '[ -n "$PID" ] && kill -CONT "$PID" 2>/dev/null' EXIT INT TERM
for tag in "$@"; do
  its=$(ls -v results/rl/$tag/tensorboard/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1/' | awk '$1 > 0 && ($1 % 82 == 0 || $1 == 812)' | tr '\n' ' ')
  until ~/.claude/bin/resguard.sh status | grep -q "launch gate: OPEN"; do sleep 30; done
  PID=$(pgrep -f "^/home/humanoid/Programs/hand/.venv/bin/python3 scripts/rl_train_cube.py .*--tag 20261008-d6_work_" | head -1)
  t0=$(date +%s)
  [ -n "$PID" ] && kill -STOP "$PID" && echo "$(date '+%F %T') paused trainer $PID for the retro watch of $tag"
  echo "$(date '+%F %T') retro watch $tag: $its"
  WARP_CACHE_PATH=$CACHE MUJOCO_GL=egl timeout 1800 ~/.claude/bin/resguard.sh run --mem 6G --cpu 400 -- \
    uv run --extra rl --extra gpu python scripts/rl_contact_eval.py watch --tags "$tag" --iterations $its
  rc=$?
  if [ -n "$PID" ]; then kill -CONT "$PID"; echo -e "retro:$tag\t-\t$t0\t$(date +%s)\t$rc" >> $L/watch_pauses.tsv; fi
  PID=""
  echo "$(date '+%F %T') retro watch $tag exit $rc"
done

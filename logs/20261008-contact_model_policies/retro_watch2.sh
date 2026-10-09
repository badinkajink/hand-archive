#!/usr/bin/env bash
# Re-run of the after-the-fact checkpoint watch (rl_contact_eval.py watch --redo) for finished runs, sharing
# watch.lock with watch_daemon.sh: the running trainer is paused for each run's chunk and resumed after it.
#   nohup setsid bash logs/20261008-contact_model_policies/retro_watch2.sh <tag> ... >> logs/20261008-contact_model_policies/retro_watch.log 2>&1 &
set -u
cd /home/humanoid/Programs/hand
L=logs/20261008-contact_model_policies
CACHE=$(mktemp -d)
PID=""
trap '[ -n "$PID" ] && kill -CONT "$PID" 2>/dev/null' EXIT INT TERM
for tag in "$@"; do
  its=$(ls -v results/rl/$tag/tensorboard/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1/' | awk '$1 > 0 && ($1 % 82 == 0 || $1 == 812)' | tr '\n' ' ')
  until ~/.claude/bin/resguard.sh status | grep -q "launch gate: OPEN"; do sleep 30; done
  exec 9> $L/watch.lock; flock 9
  PID=$(pgrep -f "^/home/humanoid/Programs/hand/.venv/bin/python3 scripts/rl_train_cube.py .*--tag 20261008-d6_work_" | head -1)
  t0=$(date +%s)
  [ -n "$PID" ] && kill -STOP "$PID" && echo "$(date '+%F %T') paused trainer $PID for the retro watch of $tag"
  echo "$(date '+%F %T') retro watch (redo) $tag: $its"
  WARP_CACHE_PATH=$CACHE MUJOCO_GL=egl timeout 1800 ~/.claude/bin/resguard.sh run --mem 6G --cpu 400 -- \
    uv run --extra rl --extra gpu python scripts/rl_contact_eval.py watch --redo --tags "$tag" --iterations $its
  rc=$?
  if [ -n "$PID" ]; then kill -CONT "$PID"; echo -e "retro:$tag\t-\t$t0\t$(date +%s)\t$rc" >> $L/watch_pauses.tsv; fi
  PID=""
  flock -u 9
  echo "$(date '+%F %T') retro watch (redo) $tag exit $rc"
done

#!/usr/bin/env bash
# Checkpoint watch of the 2026-10-08 contact-model runs (owner, 2026-10-08 21:50: "watch every run at its checkpoints,
# about every 5 M steps ... before the run goes on"). While a run of the queue trains, at every 82nd iteration
# (4.03 M env steps) whose checkpoint is complete: pause the trainer (SIGSTOP), run
# `rl_contact_eval.py watch` on that checkpoint (64 deterministic rollouts, the watch signals, a strip and a film of
# the median and the worst rollout), resume the trainer (SIGCONT) and leave a marker in watch_ready/ for the session,
# which looks at the strip and decides whether the run goes on. Pauses: watch_pauses.tsv (tag, it, t0, t1, rc).
# A watch point waits while the resguard gate is closed (the skin trainer leaves < 2 GB of GPU memory: its
# checkpoints are watched after the run). Two failed attempts at one checkpoint and it is skipped.
#
#   nohup setsid bash logs/20261008-contact_model_policies/watch_daemon.sh > logs/20261008-contact_model_policies/watch_daemon.log 2>&1 &
set -u
cd /home/humanoid/Programs/hand
L=logs/20261008-contact_model_policies
W=$L/watch_ready
mkdir -p "$W"
CACHE=$(mktemp -d)              # the watcher's evaluations run one at a time and share this kernel cache
EVERY=82
PID=""
trap '[ -n "$PID" ] && kill -CONT "$PID" 2>/dev/null' EXIT INT TERM

while ! grep -q "queue done" $L/train_queue.log; do
  tag=$(awk '$3=="start"{t=$4} ($3=="end"||$3=="FAILED") && $4==t {t=""} END{print t}' $L/train_queue.log)
  if [ -n "$tag" ] && [ -f "$L/train_$tag.log" ]; then
    last=$(grep -o "Learning iteration [0-9]*" "$L/train_$tag.log" | tail -1 | awk '{print $3}')
    last=${last:-0}
    for ck in $(ls -v results/rl/$tag/tensorboard/model_*.pt 2>/dev/null); do
      it=$(basename "$ck" .pt); it=${it#model_}
      { [ "$it" -gt 0 ] && [ $((it % EVERY)) -eq 0 ]; } || continue
      [ -e "$W/${tag}_$it.done" ] && continue
      nf=$(ls "$W/${tag}_$it.fail"* 2>/dev/null | wc -l)
      [ "$nf" -ge 2 ] && continue
      [ "$last" -gt "$it" ] || continue            # the trainer has logged a later iteration: the save is complete
      ~/.claude/bin/resguard.sh status | grep -q "launch gate: OPEN" || { echo "$(date '+%F %T') $tag it $it: gate closed"; break; }
      PID=$(pgrep -f "^/home/humanoid/Programs/hand/.venv/bin/python3 scripts/rl_train_cube.py .*--tag $tag( |\$)" | head -1)
      [ -n "$PID" ] || break
      t0=$(date +%s)
      kill -STOP "$PID"
      echo "$(date '+%F %T') $tag it $it: trainer $PID paused"
      WARP_CACHE_PATH=$CACHE MUJOCO_GL=egl timeout 900 ~/.claude/bin/resguard.sh run --mem 6G --cpu 400 -- \
        uv run --extra rl --extra gpu python scripts/rl_contact_eval.py watch --tags "$tag" --iterations "$it" \
        >> $L/watch.log 2>&1
      rc=$?
      kill -CONT "$PID"
      PID=""
      t1=$(date +%s)
      echo -e "$tag\t$it\t$t0\t$t1\t$rc" >> $L/watch_pauses.tsv
      echo "$(date '+%F %T') $tag it $it: resumed after $((t1 - t0)) s, exit $rc"
      if [ "$rc" -eq 0 ]; then touch "$W/${tag}_$it.done"; else touch "$W/${tag}_$it.fail$((nf + 1))"; fi
    done
  fi
  sleep 20
done
echo "$(date '+%F %T') queue done; watcher exits"

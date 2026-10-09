#!/usr/bin/env bash
# Stop a training run of the 2026-10-08 queue early (owner's rule of 2026-10-08 21:50), after the session has looked at
# its watched checkpoints:  bash logs/20261008-contact_model_policies/stop_run.sh <tag> <iteration> "<reason>"
# 1. its jobs.txt line gets the '#stopped# ' prefix (the queue skips it; rl_contact_eval.jobs(include_stopped=True) reads it);
# 2. a row in docs/experiments/20261008-contact_model_policies/stops.jsonl;
# 3. the trainer gets SIGTERM (and SIGCONT, in case the watcher holds it paused);
# 4. the queue moves the partial run to <tag>_failed<k> and logs FAILED; this script moves it and its log back.
set -u
cd /home/humanoid/Programs/hand
L=logs/20261008-contact_model_policies
tag=$1; it=$2; reason=$3
grep -q "^$tag " $L/jobs.txt || { echo "no active line for $tag in jobs.txt"; exit 1; }
sed -i "s|^$tag |#stopped# $tag |" $L/jobs.txt
python3 - "$tag" "$it" "$reason" <<'PY'
import json, os, sys, time
tag, it, reason = sys.argv[1], int(sys.argv[2]), sys.argv[3]
p = "/home/humanoid/Programs/hand/docs/experiments/20261008-contact_model_policies/stops.jsonl"
with open(p, "a") as fh:
    fh.write(json.dumps(dict(tag=tag, iteration=it, env_steps=(it + 1) * 49152, reason=reason,
                             when=time.strftime("%Y-%m-%d %H:%M"))) + "\n")
    fh.flush(); os.fsync(fh.fileno())
PY
pid=$(pgrep -f "^/home/humanoid/Programs/hand/.venv/bin/python3 scripts/rl_train_cube.py .*--tag $tag( |\$)" | head -1)
n0=$(grep -c "FAILED $tag " $L/train_queue.log)
if [ -n "$pid" ]; then kill -TERM "$pid"; kill -CONT "$pid"; echo "$(date '+%F %T') SIGTERM to trainer $pid ($tag)"; fi
for i in $(seq 1 120); do
  [ "$(grep -c "FAILED $tag " $L/train_queue.log)" -gt "$n0" ] && break
  sleep 2
done
k=$(ls -d results/rl/${tag}_failed* 2>/dev/null | sed "s|results/rl/${tag}_failed||" | sort -n | tail -1)
if [ -n "$k" ] && [ ! -e results/rl/$tag ]; then
  mv results/rl/${tag}_failed$k results/rl/$tag
  [ -f $L/train_${tag}_failed$k.log ] && mv $L/train_${tag}_failed$k.log $L/train_${tag}.log
  echo "$(date '+%F %T') moved results/rl/${tag}_failed$k back to results/rl/$tag"
else
  echo "$(date '+%F %T') nothing moved (k=$k); check results/rl/${tag}*"
fi

#!/usr/bin/env bash
# Session waiter: returns when the checkpoint watcher leaves a new marker in watch_ready/, the queue log gains a line
# (a run ends, fails or starts), or after $2 minutes (default 45). $1 = stamp file touched at the previous return.
L=/home/humanoid/Programs/hand/logs/20261008-contact_model_policies
S=${1:-$L/.wait_stamp}; MAX=${2:-45}
[ -e "$S" ] || touch -d '1 minute ago' "$S"
n0=$(wc -l < $L/train_queue.log)
for ((i = 0; i < MAX * 6; i++)); do
  new=$(find $L/watch_ready -newer "$S" -type f 2>/dev/null | sort)
  if [ -n "$new" ]; then echo "watch: $new"; touch "$S"; exit 0; fi
  if [ "$(wc -l < $L/train_queue.log)" -ne "$n0" ]; then echo "queue: $(tail -n +$((n0 + 1)) $L/train_queue.log)"; touch "$S"; exit 0; fi
  sleep 10
done
echo "timeout after $MAX min"; touch "$S"

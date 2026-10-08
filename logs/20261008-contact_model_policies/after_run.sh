#!/usr/bin/env bash
# Called by scripts/rl_contact_train_queue_20261008.sh after each training run: <tag> <run dir> <timesteps>.
# Step 2 of docs/handoff/20261007-contact_model_policy_training.md: checkpoint series (every 82 iterations and the
# final one, 64 deterministic rollouts each) and the final evaluation with the physics records for every finished run
# that lacks them, the step-4 replays (replays.py: record, then CPU MuJoCo, Drake and Newton per world), then the
# training curves from the event files. One GPU job at a time: the queue waits for this.
set -u
cd /home/humanoid/Programs/hand
until ~/.claude/bin/resguard.sh status | grep -q "launch gate: OPEN"; do sleep 30; done
for sub in ckpts final; do
  echo "$(date '+%F %H:%M:%S') eval $sub (after $1)"
  WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl ~/.claude/bin/resguard.sh run --mem 6G --cpu 400 -- \
    uv run --extra rl --extra gpu python scripts/rl_contact_eval.py $sub --all
  rc=$?
  echo "$(date '+%F %H:%M:%S') eval $sub exit $rc"
  sleep 20
done
echo "$(date '+%F %H:%M:%S') replays (after $1)"
python3 logs/20261008-contact_model_policies/replays.py
rc=$?
echo "$(date '+%F %H:%M:%S') replays exit $rc"
uv run --extra rl python scripts/rl_contact_eval.py tb --all

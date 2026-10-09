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
# one-off (2026-10-08): the Newton film panel on the pads s0 replay, world 0, before post_train.sh relies on it
L=logs/20261008-contact_model_policies
if [ ! -e $L/newton_film_test.done ] && [ -f $L/replay/20261008-d6_work_tpu27pads1_40M_s0/rec.npz ]; then
  logs/20261001-hom_contact/venv/bin/python scripts/rl_policy_replay.py bench --study 20261008 \
    --dir $L/replay/20261008-d6_work_tpu27pads1_40M_s0 --world 0
  WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl $HOME/.claude/bin/resguard.sh run --mem 6G --cpu 400 -- \
    logs/20261004-contact-transfer/venv/bin/python scripts/rl_policy_replay.py film --study 20261008 \
    --dir $L/replay/20261008-d6_work_tpu27pads1_40M_s0 --engine newton --out $L/newton_film_test.mp4 --title "pads s0 test"
  echo "newton film test exit $?" && touch $L/newton_film_test.done
fi
# after the last run: the transfer matrix, robustness and films (once; post_train.sh skips what exists)
L=logs/20261008-contact_model_policies
if [ ! -e $L/post_train.started ] && python3 - <<'PY'
import sys
sys.path.insert(0, "logs/20261008-contact_model_policies")
from replays import JOBS, finished
n = sum(1 for l in open(JOBS) if len(l.split("#")[0].split()) == 4)
sys.exit(0 if len(finished()) == n else 1)
PY
then
  touch $L/post_train.started
  bash $L/post_train.sh >> $L/post_train.log 2>&1
fi
uv run --extra rl python scripts/rl_contact_eval.py costs

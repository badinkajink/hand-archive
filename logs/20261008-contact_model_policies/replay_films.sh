#!/usr/bin/env bash
# CPU panels of the three-engine replay films for finished runs (world 0): the recorded MuJoCo-Warp rollout, CPU MuJoCo
# and Drake; the Newton panel and the tile need the GPU and come from post_train.sh. Skips films that exist.
#   bash logs/20261008-contact_model_policies/replay_films.sh <tag> ...
set -u
cd /home/humanoid/Programs/hand
L=logs/20261008-contact_model_policies
M=docs/experiments/20261008-contact_model_policies/media
PYD=logs/20261001-hom_contact/venv/bin/python
mkdir -p $M
for tag in "$@"; do
  arm=$(echo $tag | sed 's/20261008-d6_work_\(.*\)_40M_s[0-9]/\1/')
  d=$L/replay/$tag
  [ -f $d/rec.npz ] || continue
  $PYD scripts/rl_policy_replay.py bench --study 20261008 --dir $d --world 0
  for e in mjw mujoco drake; do
    [ -s $M/${tag}_replay_$e.mp4 ] || MUJOCO_GL=egl OMP_NUM_THREADS=1 $PYD scripts/rl_policy_replay.py film --study 20261008 \
      --dir $d --engine $e --out $M/${tag}_replay_$e.mp4 --title "$arm, $tag"
  done
done

#!/usr/bin/env bash
# Newton panels (GPU, trainer paused through paused_gpu.sh) and the three-engine tiles of the replay films, world 0.
#   bash logs/20261008-contact_model_policies/newton_panels.sh <tag> ...
cd /home/humanoid/Programs/hand
L=logs/20261008-contact_model_policies
M=docs/experiments/20261008-contact_model_policies/media
for T in "$@"; do
  arm=$(echo $T | sed 's/20261008-d6_work_\(.*\)_40M_s[0-9]/\1/')
  [ -s $M/${T}_replay_newton.mp4 ] || bash $L/paused_gpu.sh newton_film:$T logs/20261004-contact-transfer/venv/bin/python \
    scripts/rl_policy_replay.py film --study 20261008 --dir $L/replay/$T --engine newton --out $M/${T}_replay_newton.mp4 \
    --title "$arm, $T"
  [ -s $M/${T}_replay_three.mp4 ] || logs/20261001-hom_contact/venv/bin/python scripts/rl_policy_replay.py tile \
    --clips $M/${T}_replay_mjw.mp4 $M/${T}_replay_drake.mp4 $M/${T}_replay_newton.mp4 --out $M/${T}_replay_three.mp4 \
    --poster $M/${T}_replay_three.jpg
done

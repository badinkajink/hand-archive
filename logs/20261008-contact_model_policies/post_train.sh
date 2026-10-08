#!/usr/bin/env bash
# After the last training run of the 2026-10-08 queue (after_run.sh starts it once every final checkpoint exists):
# step 4 transfer matrix (every final policy on all five scenes, 64 rollouts per cell), step 5 robustness, and the
# step-6 films (close-up renders of the best and median seed per arm; per arm one policy's world-0 replay in
# MuJoCo-Warp, CPU MuJoCo, Drake and Newton, tiled). GPU jobs one at a time, each through resguard with its own Warp
# cache. Resumable: rows already written are skipped, films already present are kept. Log: post_train.log.
set -u
cd /home/humanoid/Programs/hand
L=logs/20261008-contact_model_policies
M=docs/experiments/20261008-contact_model_policies/media
RG="$HOME/.claude/bin/resguard.sh run --mem 6G --cpu 400 --"
UVP="uv run --extra rl --extra gpu python"
PYD=logs/20261001-hom_contact/venv/bin/python
PYN=logs/20261004-contact-transfer/venv/bin/python
gpu() { until $HOME/.claude/bin/resguard.sh status | grep -q "launch gate: OPEN"; do sleep 30; done
        WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl $RG "$@"; }
echo "$(date '+%F %H:%M:%S') post_train start"
for scene in box tpu27mesh tpu27pads1 tpu27meshc4 tpu27skin; do
  gpu $UVP scripts/rl_contact_eval.py transfer --scene $scene --all
done
gpu $UVP scripts/rl_contact_eval.py robust --all --perturb friction=0.7 friction=1.3 mass=0.8 mass=1.2 kp=2 kp=6 kp=10 \
  dt=0.001 noise=2,5
python3 $L/pick_films.py --json > $L/films.txt
mkdir -p $M
while read -r arm role tag it; do
  out=$M/${tag}_${role}.mp4
  [ -s "$out" ] || gpu $UVP scripts/rl_render_reorient.py --run results/rl/$tag --checkpoint model_$it.pt --output $out \
    --steps 250 --width 960 --height 720 --distance 0.24 --elevation -6 --azimuth 0
  [ -s "${out%.mp4}_strip.png" ] || uv run --extra rl python scripts/policy_filmstrip.py --video $out --at 20,58,90,120,160,249 \
    --cols 6 --out ${out%.mp4}_strip.png
done < $L/films.txt
# three-engine replay films: the median seed of each arm (the best one when an arm has one finished seed), world 0
while read -r arm role tag it; do
  [ "$role" = median ] || continue
  d=$L/replay/$tag
  [ -f $d/rec.npz ] || continue
  $PYD scripts/rl_policy_replay.py bench --study 20261008 --dir $d --world 0
  for e in mjw mujoco drake; do
    [ -s $M/${tag}_replay_$e.mp4 ] || MUJOCO_GL=egl OMP_NUM_THREADS=1 $PYD scripts/rl_policy_replay.py film --study 20261008 \
      --dir $d --engine $e --out $M/${tag}_replay_$e.mp4 --title "$arm, $tag"
  done
  [ -s $M/${tag}_replay_newton.mp4 ] || gpu $PYN scripts/rl_policy_replay.py film --study 20261008 --dir $d --engine newton \
    --out $M/${tag}_replay_newton.mp4 --title "$arm, $tag"
  $PYD scripts/rl_policy_replay.py tile --clips $M/${tag}_replay_mjw.mp4 $M/${tag}_replay_drake.mp4 $M/${tag}_replay_newton.mp4 \
    --out $M/${tag}_replay_three.mp4 --poster $M/${tag}_replay_three.jpg
done < $L/films.txt
echo "$(date '+%F %H:%M:%S') post_train done"

#!/usr/bin/env bash
# Wait for the running retro_watch2.sh, then redo skin, mesh and box s0 with the hold-phase jerk and the saved traces.
cd /home/humanoid/Programs/hand
L=logs/20261008-contact_model_policies
while pgrep -f "^bash logs/20261008-contact_model_policies/retro_watch2.sh 20261008-d6_work_tpu27skin_40M_s0 20261008-d6_work_tpu27pads1_40M_s0" > /dev/null; do sleep 10; done
exec bash $L/retro_watch2.sh 20261008-d6_work_tpu27skin_40M_s0 20261008-d6_work_tpu27mesh_40M_s0 20261008-d6_work_box_40M_s0

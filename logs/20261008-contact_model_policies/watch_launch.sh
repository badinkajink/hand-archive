#!/usr/bin/env bash
# The retro watch of the runs that finished before the watch existed, then the checkpoint watcher for the queue.
cd /home/humanoid/Programs/hand
L=logs/20261008-contact_model_policies
bash $L/retro_watch.sh 20261008-d6_work_tpu27skin_40M_s0 20261008-d6_work_tpu27pads1_40M_s0 \
  20261008-d6_work_tpu27mesh_40M_s0 20261008-d6_work_box_40M_s0 >> $L/retro_watch.log 2>&1
exec bash $L/watch_daemon.sh >> $L/watch_daemon.log 2>&1

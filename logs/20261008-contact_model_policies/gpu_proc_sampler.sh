#!/usr/bin/env bash
# Per-process GPU memory of the 2026-10-08 training runs every 30 s: time, pid, run tag (from the process's --tag), MiB.
# The queue's own gpu_mem.tsv samples the whole card, which other GPU users share. Stops when the queue log says done.
L=/home/humanoid/Programs/hand/logs/20261008-contact_model_policies
while ! grep -q "queue done" $L/train_queue.log 2>/dev/null; do
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r pid mib; do
    tag=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -o -- '--tag [^ ]*' | cut -d' ' -f2)
    [ -n "$tag" ] && echo -e "$(date +%s)\t$pid\t$tag\t$mib"
  done >> $L/gpu_proc.tsv
  sleep 30
done

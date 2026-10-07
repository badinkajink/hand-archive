#!/bin/bash
# Candidate (a) rerun with critical skin damping.
cd /home/humanoid/Programs/hand
PY=logs/20261001-hom_contact/venv/bin/python
for t in t8 t9 t2 t1 t5; do $PY scripts/contact_bed_candidates.py $t --models mj_pads1_skin; done
$PY scripts/contact_bed_candidates.py t8 --models mj_pads1_skin --dt 2
$PY scripts/contact_bed_candidates.py t2 --models mj_pads1_skin --dt 2 --N 1
echo STEP6A2_DONE

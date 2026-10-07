#!/bin/bash
# Step 6 candidate (d): solimp softening with depth (Hertz exponent), T9, T2, T1, T5, T8 at 1 ms; T9 and T2 at 2 ms.
cd /home/humanoid/Programs/hand
PY=logs/20261001-hom_contact/venv/bin/python
for t in t9 t2 t1 t5 t8; do $PY scripts/contact_bed_candidates.py $t --models mj_pads1_soft; done
$PY scripts/contact_bed_candidates.py t9 --models mj_pads1_soft --dt 2
$PY scripts/contact_bed_candidates.py t2 --models mj_pads1_soft --dt 2 --N 1
echo STEP6D_DONE

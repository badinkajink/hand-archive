#!/bin/bash
# Step 11 candidate (e) rolling bristles, front-cap 20 deg version (T8, T9, T1, T2, T5 at 1 ms; T8 at 2 ms), and T8 with every sphere.
cd /home/humanoid/Programs/hand
PY=logs/20261001-hom_contact/venv/bin/python
for t in t8 t9 t2 t1 t5; do $PY scripts/contact_bed_candidates.py $t --models mj_pads1_bristle20; done
$PY scripts/contact_bed_candidates.py t8 --models mj_pads1_bristle20 --dt 2
$PY scripts/contact_bed_candidates.py t8 --models mj_pads1_bristle --N 1
echo STEP11_DONE

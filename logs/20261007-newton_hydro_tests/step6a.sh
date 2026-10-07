#!/bin/bash
# Step 6 candidate (a) compliant skin and its impratio-1000 control: T8, T9, T1, T2, T5 at 1 ms.
cd /home/humanoid/Programs/hand
PY=logs/20261001-hom_contact/venv/bin/python
for t in t8 t9 t2 t1 t5; do $PY scripts/contact_bed_candidates.py $t; done
$PY scripts/contact_bed_candidates.py t8 --dt 2
$PY scripts/contact_bed_candidates.py t2 --dt 2 --N 1
echo STEP6A_DONE

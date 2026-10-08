"""Twist-onset arm of the 2.7 mm TPU tip as 1 mm sphere pads at the grip forces of the RL policies (1-20 N), for the
constant torsional coefficient of the condim-4 arm (d). Runs tpu_tip_rig.bed's T2; TpuMjRig predates MjRig.x0, which
the bed's kinematic spin reports (pad_x), so x0 is set to 0 (the reported pad_x is then the rail coordinate)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import tpu_tip_rig as T  # noqa: E402
T.TpuMjRig.x0 = 0.0
out = ROOT / "docs/experiments/20261008-contact_model_policies/twist_onset"
out.mkdir(parents=True, exist_ok=True)
T.bed(["mj_tpu2p7_pads1"], ["T2"], [float(x) for x in sys.argv[1:]] or [1.0, 3.0, 10.0, 20.0], [1.0], out)

"""CPU MuJoCo check of the skin scene under the trainer's solver options: keyframe open_ik, the finger targets ramped
to the grasp ctrl over 0.16 s, the palm raised 0.1 m from 0.48 s over 0.16 s; tool height, warnings, NaN."""
import sys, numpy as np, mujoco
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/phase1/real_v1"
scenes = {"pads_ir10": R / "20261006-sv1_u0308_b050_work_tip_tpu2.7pads1",
          "skin_ir1000": R / "20261008-sv1_u0308_b050_work_tip_tpu2.7skin"}
def run(scene, ir, T=1.5):
    m = mujoco.MjModel.from_xml_path(str(scene / "frozen_scene.xml"))
    m.opt.iterations, m.opt.ls_iterations, m.opt.impratio = 10, 20, ir
    m.opt.cone, m.opt.integrator, m.opt.timestep = mujoco.mjtCone.mjCONE_ELLIPTIC, mujoco.mjtIntegrator.mjINT_IMPLICITFAST, 0.002
    d = mujoco.MjData(m); k = m.key("open_ik").id; mujoco.mj_resetDataKeyframe(m, d, k)
    c0 = d.ctrl.copy(); bfc = np.load(scene / "best_rollout.npz")["best_finger_ctrl"]
    fa = [m.actuator(f"a_{f}_{j}").id for f in ("thumb", "index", "middle") for j in ("yaw", "mcp", "pip")]
    pz = m.actuator("a_palm_pz").id
    tool = m.body("screwdriver_medium").id
    out = []
    for i in range(int(T / 0.002)):
        t = i * 0.002
        a = min(1.0, t / 0.16); e = 1 - (1 - a) ** 2
        d.ctrl[fa] = c0[fa] + e * (bfc - c0[fa])
        d.ctrl[pz] = c0[pz] + 0.1 * min(1.0, max(0.0, (t - 0.48) / 0.16))
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            return dict(nan_at=t)
        if i % 50 == 49:
            out.append((round(t, 2), round(1e3 * d.xpos[tool, 2], 1), d.ncon))
    sk = [m.joint(j).qposadr[0] for j in range(m.njnt) if "_skin_" in m.joint(j).name]
    return dict(final=out[-1], trace=out[::3], skin_q=np.round(d.qpos[sk], 5).tolist() if sk else None,
                warnings={mujoco.mjtWarning(w).name: int(d.warning[w].number) for w in range(len(d.warning)) if d.warning[w].number})
for name, ir in (("pads_ir10", 10), ("pads_ir10", 1000), ("skin_ir1000", 1000), ("skin_ir1000", 10)):
    print(name, "impratio", ir, run(scenes[name], ir), flush=True)

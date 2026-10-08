"""The skin scene stepped directly in MuJoCo-Warp (no mjlab), 8 worlds, the scripted close and lift of skin_cpu_diag.py,
with variants of the skin joints, against CPU MuJoCo on the same model: which part of the skin fails on the GPU."""
import sys, xml.etree.ElementTree as ET, numpy as np, mujoco
from pathlib import Path
import mujoco_warp as mjw, warp as wp
ROOT = Path(__file__).resolve().parents[2]
SK = ROOT / "results/phase1/real_v1/20261008-sv1_u0308_b050_work_tip_tpu2.7skin"
PADS = ROOT / "results/phase1/real_v1/20261006-sv1_u0308_b050_work_tip_tpu2.7pads1"

def variant(kind):
    if kind == "pads":
        return mujoco.MjModel.from_xml_path(str(PADS / "frozen_scene.xml")), PADS
    root = ET.parse(SK / "frozen_scene.xml").getroot()
    for j in root.iter("joint"):
        n = j.get("name", "")
        if "_skin_" not in n:
            continue
        if kind == "hinge_only" and n.endswith(("_y", "_z")):
            j.getparent = None; j.set("stiffness", "1e9"); j.set("damping", "1e3")
        if kind == "slides_only" and n.endswith("_t"):
            j.set("stiffness", "1e6"); j.set("damping", "1e2")
        if kind == "arm10x":
            j.set("armature", f"{float(j.get('armature')) * 10:g}")
        if kind == "soft":
            j.set("stiffness", f"{float(j.get('stiffness')) / 10:g}")
            j.set("damping", f"{float(j.get('damping')) / np.sqrt(10):g}")
        if kind == "nolimit":
            del j.attrib["range"]
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return m, SK

def setup(m, scene):
    m.opt.iterations, m.opt.ls_iterations, m.opt.impratio = 10, 20, 10
    m.opt.cone, m.opt.integrator, m.opt.timestep = mujoco.mjtCone.mjCONE_ELLIPTIC, mujoco.mjtIntegrator.mjINT_IMPLICITFAST, 0.002
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, m.key("open_ik").id); mujoco.mj_forward(m, d)
    bfc = np.load(scene / "best_rollout.npz")["best_finger_ctrl"]
    fa = [m.actuator(f"a_{f}_{j}").id for f in ("thumb", "index", "middle") for j in ("yaw", "mcp", "pip")]
    return d, bfc, fa, m.actuator("a_palm_pz").id, m.body("screwdriver_medium").id

def ctrl_at(t, c0, bfc, fa, pz):
    c = c0.copy(); a = min(1.0, t / 0.16); e = 1 - (1 - a) ** 2
    c[fa] = c0[fa] + e * (bfc - c0[fa]); c[pz] = c0[pz] + 0.1 * min(1.0, max(0.0, (t - 0.48) / 0.16))
    return c

def gpu(kind, nw=8, T=1.0):
    m, scene = variant(kind); d, bfc, fa, pz, tool = setup(m, scene)
    wm = mjw.put_model(m); wd = mjw.make_data(m, nworld=nw, nconmax=1024, njmax=3072)
    wd.qpos.assign(np.tile(d.qpos, (nw, 1)).astype(np.float32)); wd.qvel.zero_()
    if hasattr(wd, "qacc_warmstart"): wd.qacc_warmstart.zero_()
    c0 = d.ctrl.copy(); zs = []
    mjw.forward(wm, wd)
    for i in range(int(T / 0.002)):
        wd.ctrl.assign(np.tile(ctrl_at(i * 0.002, c0, bfc, fa, pz), (nw, 1)).astype(np.float32))
        mjw.step(wm, wd)
        if i % 50 == 49:
            z = wd.xpos.numpy()[:, tool, 2]; zs.append(round(float(1e3 * np.nanmedian(z)), 1))
    return zs, int(wd.nacon.numpy()[0]) // nw

def cpu(kind, T=1.0):
    m, scene = variant(kind); d, bfc, fa, pz, tool = setup(m, scene)
    c0 = d.ctrl.copy(); zs = []
    for i in range(int(T / 0.002)):
        d.ctrl[:] = ctrl_at(i * 0.002, c0, bfc, fa, pz); mujoco.mj_step(m, d)
        if i % 50 == 49: zs.append(round(1e3 * d.xpos[tool, 2], 1))
    return zs, d.ncon

for kind in sys.argv[1:] or ["pads", "skin", "nolimit", "arm10x", "soft", "slides_only", "hinge_only"]:
    print(kind, "GPU", gpu(kind), "CPU", cpu(kind), flush=True)

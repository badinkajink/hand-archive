"""The skin scene in the RL env (16 envs, zero residual): skin joint parameters as mjlab compiled them, and per-step
tool height, tip contacts and skin deflection, to find where the GPU rollout departs from CPU MuJoCo."""
import dataclasses, sys, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import rl_contact_throughput as T
run = ROOT / "results/phase1/real_v1" / (sys.argv[1] if len(sys.argv) > 1 else "20261008-sv1_u0308_b050_work_tip_tpu2.7skin")
from mjlab.envs import ManagerBasedRlEnv
from morphohand.rl.env_cfg import to_mjlab_cfg
cfg = T.base_cfg(run)
mj = to_mjlab_cfg(dataclasses.replace(cfg, num_envs=16))
print("buffers", mj.sim.nconmax, mj.sim.njmax, mj.sim.contact_sensor_maxmatch, "impratio", mj.sim.mujoco.impratio)
env = ManagerBasedRlEnv(cfg=mj, device="cuda:0", render_mode=None)
m = env.sim.mj_model
sk = [j for j in range(m.njnt) if "_skin_" in m.joint(j).name]
for j in sk[:3]:
    d = m.jnt_dofadr[j]
    print(m.joint(j).name, "stiff", m.jnt_stiffness[j], "damp", m.dof_damping[d], "arm", m.dof_armature[d], "range", m.jnt_range[j], "limited", m.jnt_limited[j])
print("opt", m.opt.timestep, m.opt.iterations, m.opt.ls_iterations, m.opt.impratio, m.opt.cone, m.opt.integrator)
print("obs dim", env.observation_manager.group_obs_dim)
tool = next(i for i in range(m.nbody) if m.body(i).name.split("/")[-1] == "cube")
qadr = [m.jnt_qposadr[j] for j in sk]
act = torch.zeros((16, env.action_manager.total_action_dim), device="cuda:0")
env.reset()
wd = env.sim.wp_data
for step in range(160):
    env.step(act)
    z = wd.xpos.numpy()[:, tool, 2]; q = wd.qpos.numpy()[:, qadr]
    n = int(wd.nacon.numpy()[0])
    if step % 10 == 0 or not np.isfinite(z).all():
        print(step, "z mm", np.round(1e3 * np.nanmedian(z), 1), "nan worlds", int((~np.isfinite(z)).sum()), "nacon", n,
              "skin |q| max", np.round(np.nanmax(np.abs(q), axis=0)[:3], 5), flush=True)
    if (~np.isfinite(z)).sum() > 8:
        break

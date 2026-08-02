#!/usr/bin/env python3
"""Is the hand standing still, or moving without effect? Probe joints, actions and the object.

perp holds the UHAS cube better than any hand we have built and never rotates it (see
docs/uhas/CUBE_REPOSE_PLAN.md E4). "Never rotates" has two completely different causes and the
task metrics cannot tell them apart:

  A. the policy is effectively SILENT on this hand -- actions near-constant, or CIK collapsing
     them to a near-constant joint target, so the fingers never move;
  B. the fingers move plenty but the motion does no work on the cube -- squeezing symmetrically,
     or gripping so hard the cube cannot roll.

Distinguishing them decides what to fix next, and they need opposite fixes. So measure, per
control step: the spread of the policy's actions, the spread of the actuated joint positions, how
far each joint travels, and the object's angular speed.

Reported per joint, because a hand where one finger does all the moving looks identical in the
mean to a hand where all three move a little.

    /home/humanoid/miniconda3/envs/env_isaaclab/bin/python scripts/uhas_motion_probe.py \
        --robot morpho_perp_open --out results/uhas/probe/perp.json --headless
"""

from __future__ import annotations

import argparse
import os

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--robot", required=True)
_pre.add_argument("--cube-scale-mult", type=float, default=1.0)
_pre.add_argument("--morpho-root", default=os.path.abspath("results/uhas/hands"))
_known, _ = _pre.parse_known_args()
os.environ["UHAS_ROBOTS"] = _known.robot
os.environ["UHAS_EVAL"] = "1"
os.environ["UHAS_CUBE_SCALE_MULT"] = str(_known.cube_scale_mult)
os.environ["UHAS_MORPHO_ROOT"] = _known.morpho_root

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot", required=True)
parser.add_argument("--checkpoint", default=os.path.abspath(
    "docs/uhas/UHAS_sim/models/UGAS_Models/All_grippers/model_14999.pt"))
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--warmup", type=int, default=60)
parser.add_argument("--cube-scale-mult", type=float, default=1.0)
parser.add_argument("--morpho-root", default=os.path.abspath("results/uhas/hands"))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sphere_ctrl_isaaclab.tasks  # noqa: F401, E402
from sphere_ctrl_isaaclab.tasks.UHAS_inhand.agents.rsl_rl_ppo_cfg import SpherePPORunnerCfg  # noqa: E402

TASK = "UHAS-Inhand-Repose"


def main() -> None:
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = SpherePPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    robot = args_cli.robot
    idx = u.actuated_dof_indices[robot]
    names = [u.hands[robot].joint_names[i] for i in idx]
    sphere_r = float(u.sphere_radius[robot].item())

    q_hist, a_hist, w_hist, tgt_hist = [], [], [], []
    obs, _ = env.get_observations()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if step < args_cli.warmup:
                continue
            q_hist.append(u.hands[robot].data.joint_pos[:, idx].cpu().numpy())
            tgt_hist.append(u.cur_targets[robot][:, idx].cpu().numpy()
                            if hasattr(u, "cur_targets") else np.zeros_like(q_hist[-1]))
            a_hist.append(actions.detach().cpu().numpy())
            w_hist.append(torch.norm(u.object_angvel, dim=-1).cpu().numpy())

    q = np.stack(q_hist)      # [T, envs, n_joints]
    tg = np.stack(tgt_hist)
    a = np.stack(a_hist)      # [T, envs, 15]
    w = np.stack(w_hist)      # [T, envs]

    # Per-joint temporal spread, averaged over envs: how much does this joint actually move?
    q_std = q.std(axis=0).mean(axis=0)
    tg_std = tg.std(axis=0).mean(axis=0)
    q_range = (q.max(axis=0) - q.min(axis=0)).mean(axis=0)
    # Total path length per joint (rad travelled), per second of sim time.
    dt = float(u.step_dt)
    q_speed = np.abs(np.diff(q, axis=0)).sum(axis=0).mean(axis=0) / (q.shape[0] * dt)

    out = {
        "robot": robot,
        "steps": args_cli.steps,
        "num_envs": args_cli.num_envs,
        "cube_scale_mult": args_cli.cube_scale_mult,
        "sphere_radius_m": sphere_r,
        "action": {
            "std_per_dim": a.std(axis=(0, 1)).round(4).tolist(),
            "std_mean": float(a.std(axis=(0, 1)).mean()),
            "abs_mean": float(np.abs(a).mean()),
        },
        "joints": {
            n: {
                "std_rad": float(q_std[i]),
                "range_rad": float(q_range[i]),
                "travel_rad_per_s": float(q_speed[i]),
                "target_std_rad": float(tg_std[i]),
            }
            for i, n in enumerate(names)
        },
        "joint_std_mean_rad": float(q_std.mean()),
        "joint_travel_mean_rad_per_s": float(q_speed.mean()),
        "object_angspeed_rad_per_s": {
            "mean": float(w.mean()), "median": float(np.median(w)), "p90": float(np.percentile(w, 90)),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== {robot} (cube x{args_cli.cube_scale_mult}) ===")
    print(f"action std (mean over 15 dims) : {out['action']['std_mean']:.4f}")
    print(f"joint std   (mean over joints) : {out['joint_std_mean_rad']:.4f} rad")
    print(f"joint travel                   : {out['joint_travel_mean_rad_per_s']:.3f} rad/s")
    print(f"object angular speed           : {out['object_angspeed_rad_per_s']['mean']:.3f} rad/s")
    for n, v in out["joints"].items():
        print(f"   {n:14s} std {v['std_rad']:.4f}  range {v['range_rad']:.3f}  "
              f"travel {v['travel_rad_per_s']:.3f} rad/s  target_std {v['target_std_rad']:.4f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

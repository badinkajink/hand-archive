#!/usr/bin/env python3
"""Compare a hand's LIVE observations against the distribution the policy was trained on.

The checkpoint carries `obs_norm_state_dict` -- the running per-dimension mean/var of every
observation the policy ever saw across 1.0e8 training samples on four hands. That is a
distributional fingerprint of "normal input", and it is checkable:

  z_d = (mean_live_d - mean_train_d) / std_train_d

Two independent uses, which is why this is worth a script rather than a notebook cell:

1. **Integration check.** If our patched env feeds the policy a wrong feature -- swapped order,
   wrong units, wrong frame, a missing column -- that dimension's z blows up while its neighbours
   stay small. A dimension-level answer, not "the success rate is low, something is wrong".

2. **Morphology metric.** For a hand the policy has NEVER seen, the same z-profile measures how
   far outside the training distribution that hand puts the policy -- per feature group
   (fingertip positions, fingertip velocities, object state). A hand whose fp_pos z-profile sits
   inside the training envelope is one the policy can read; one whose fingertips report
   coordinates it has never seen is being asked to extrapolate. That is a *predictor of
   zero-shot transfer computable without running the task*, which is exactly the kind of cheap
   evaluator the morphology search needs.

Usage (Isaac env):

    /home/humanoid/miniconda3/envs/env_isaaclab/bin/python scripts/uhas_obs_audit.py \
        --robot leap_right --steps 400 --out results/uhas/audit/leap_right.json
"""

from __future__ import annotations

import argparse
import os

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--robot", required=True)
_pre.add_argument("--morpho-root", default=os.path.abspath("results/uhas/hands"))
_pre.add_argument("--cube-scale-mult", type=float, default=1.0)
_known, _ = _pre.parse_known_args()
os.environ["UHAS_ROBOTS"] = _known.robot
os.environ["UHAS_EVAL"] = "1"
os.environ["UHAS_MORPHO_ROOT"] = _known.morpho_root
os.environ["UHAS_CUBE_SCALE_MULT"] = str(_known.cube_scale_mult)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot", required=True)
parser.add_argument("--checkpoint", default=os.path.abspath(
    "docs/uhas/UHAS_sim/models/UGAS_Models/All_grippers/model_14999.pt"))
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--warmup", type=int, default=40)
parser.add_argument("--morpho-root", default=os.path.abspath("results/uhas/hands"))
parser.add_argument("--cube-scale-mult", type=float, default=1.0)
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

# Observation layout, verified against the checkpoint's own normalizer statistics rather than
# assumed: dim 96 has mean 0.464 (merge_frequency 0.3 mixed with the always-merged <5-finger
# hands) and dim 97 has mean 0.277, which is `fail_frequency: 0.3` -- i.e. the finger-failure
# flag really does live at 97, and a zero column there is its eval-time value.
GROUPS = [
    ("object_pos", 0, 3),
    ("object_quat", 3, 7),
    ("object_linvel", 7, 10),
    ("object_angvel", 10, 13),
    ("goal_quat", 13, 17),
    ("goal_rel_quat", 17, 21),
    ("prev_action", 21, 36),
    ("fingertip_pos", 36, 66),
    ("fingertip_linvel", 66, 96),
    ("merge_flag", 96, 97),
    ("failure_flag", 97, 98),
]


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

    ck = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    tr_mean = ck["obs_norm_state_dict"]["_mean"].flatten().numpy()
    tr_std = np.sqrt(ck["obs_norm_state_dict"]["_var"].flatten().numpy())

    buf = []
    obs, _ = env.get_observations()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if step >= args_cli.warmup:
                buf.append(obs.detach().float().cpu().numpy())
    live = np.concatenate(buf, axis=0)
    lm, ls = live.mean(axis=0), live.std(axis=0)

    # Guard against a training dim that is genuinely constant (std 0) -- there are none in this
    # checkpoint, but dividing by it would manufacture an infinite z and hide the real outliers.
    denom = np.where(tr_std > 1e-6, tr_std, 1.0)
    z = (lm - tr_mean) / denom
    spread = ls / denom  # live spread as a fraction of the training spread

    out = {
        "robot": args_cli.robot,
        "steps": args_cli.steps,
        "num_envs": args_cli.num_envs,
        "obs_dim": int(live.shape[1]),
        "groups": {},
        "worst_dims": [],
    }
    for name, a, b in GROUPS:
        zz = np.abs(z[a:b])
        out["groups"][name] = {
            "dims": [a, b],
            "abs_z_mean": float(zz.mean()),
            "abs_z_max": float(zz.max()),
            "spread_ratio_mean": float(spread[a:b].mean()),
        }
    order = np.argsort(-np.abs(z))[:12]
    for i in order:
        grp = next(n for n, a, b in GROUPS if a <= i < b)
        out["worst_dims"].append({
            "dim": int(i), "group": grp,
            "live_mean": float(lm[i]), "train_mean": float(tr_mean[i]),
            "train_std": float(tr_std[i]), "z": float(z[i]),
        })
    # One headline number: the mean |z| over every dimension that is not a flag. This is the
    # "how alien is this hand to the policy" score.
    body = np.concatenate([np.abs(z[a:b]) for n, a, b in GROUPS if not n.endswith("flag")])
    out["ood_score_mean_abs_z"] = float(body.mean())
    out["ood_score_p90_abs_z"] = float(np.percentile(body, 90))

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

#!/usr/bin/env python3
"""Evaluate a pretrained UHAS policy on ONE hand and write a metrics JSON.

Why not upstream's `play.py`: it runs forever, and the only numbers it produces are two
`print()`s inside the env (`Average consecutive successes ...`, `Target pose success rate ...`)
that appear every 100 resets and are aggregated over *all* hands in the run. For a cross-hand
comparison we need per-hand numbers, a fixed step budget so legs are comparable, and more than
the headline metric -- in particular we need to distinguish "the hand cannot hold the cube" from
"the hand holds it but cannot turn it", which the success rate alone cannot.

Metrics, and what each one is for:

  success_rate        fraction of goal ATTEMPTS reached within `success_tolerance` (0.1 rad).
                      UHAS's headline metric. An attempt ends in success, drop, or timeout.
  consecutive_*       run length of successes before the episode ends. UHAS's other headline
                      (`max_consecutive_success` caps it at 10).
  drop_rate           fraction of attempts ending with the cube outside `fall_dist` (0.2 radii).
                      Separates "cannot hold" from "cannot turn".
  timeout_rate        attempts that neither succeeded nor dropped -- held but did not rotate.
  steps_to_success    control steps per successful repose. Reorientation SPEED, not just whether.
  best_rot_dist       per attempt, the CLOSEST the cube ever came to the goal. A hand that
                      times out at 0.15 rad is one tolerance away from working; one that times
                      out at 1.5 rad is not rotating at all. This is the metric that tells a
                      near-miss morphology from a hopeless one, and no upstream number carries it.
  centering           |object - inhand_target| in SPHERE RADII. Already hand-size-normalised,
                      which is what makes it comparable across morphologies. Our de-centering
                      metric from the mjlab work, in UHAS units.
  obj_speed           mean object linear speed (radii/s) -- jitter / thrash proxy.

Distances are in sphere radii and rotations in radians throughout, so nothing here has to be
rescaled to compare a 0.065 m sphere against LEAP's 0.091 m one.

Run in the Isaac env:

    /home/humanoid/miniconda3/envs/env_isaaclab/bin/python scripts/uhas_eval.py \
        --robot leap_right --steps 6000 --num-envs 64 \
        --out results/uhas/eval/leap_right.json
"""

from __future__ import annotations

import argparse
import os

# ---------------------------------------------------------------------------------------------
# UHAS's SphereEnvCfg resolves its robot list, CIK files and cube scale in the CLASS BODY, i.e.
# at import time. Our patch to multi_env_cfg.py reads those from the environment, so they have to
# be set before ANY sphere_ctrl_isaaclab import -- hence this block sitting above the imports.
# ---------------------------------------------------------------------------------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--robot", required=True)
_pre.add_argument("--cube-scale-mult", type=float, default=1.0)
_pre.add_argument("--max-consec", type=int, default=10)
_pre.add_argument("--morpho-root", default=os.path.abspath("results/uhas/hands"))
_known, _ = _pre.parse_known_args()
os.environ["UHAS_ROBOTS"] = _known.robot
os.environ["UHAS_EVAL"] = "1"
os.environ["UHAS_CUBE_SCALE_MULT"] = str(_known.cube_scale_mult)
os.environ["UHAS_MAX_CONSEC"] = str(_known.max_consec)
os.environ["UHAS_MORPHO_ROOT"] = _known.morpho_root

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot", required=True, help="One robot tag, e.g. leap_right or morpho_H0604.")
parser.add_argument("--checkpoint", default=os.path.abspath(
    "docs/uhas/UHAS_sim/models/UGAS_Models/All_grippers/model_14999.pt"))
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=6000, help="Control steps to run (60 Hz).")
parser.add_argument("--warmup", type=int, default=60, help="Steps discarded before recording.")
parser.add_argument("--cube-scale-mult", type=float, default=1.0)
parser.add_argument("--max-consec", type=int, default=10)
parser.add_argument("--morpho-root", default=os.path.abspath("results/uhas/hands"))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", required=True, help="Path to write the metrics JSON.")
parser.add_argument("--trace-out", default=None, help="Optional .npz of per-step traces for plots.")
parser.add_argument("--video", action="store_true", help="Record a video of the first --video-length steps.")
parser.add_argument("--video-length", type=int, default=900)
parser.add_argument("--video-dir", default=None)
parser.add_argument("--tag", default=None, help="Free-form label recorded into the JSON.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sphere_ctrl_isaaclab.tasks  # noqa: F401, E402
from sphere_ctrl_isaaclab.tasks.UHAS_inhand.agents.rsl_rl_ppo_cfg import SpherePPORunnerCfg  # noqa: E402
# The env's own goal metric -- 2*asin(|quat_mul(obj, conj(goal))|_xyz), i.e. the same number
# `compute_rewards` thresholds against `success_tolerance`. Import it rather than reimplement it
# so our success count cannot drift from the env's.
from sphere_ctrl_isaaclab.tasks.UHAS_inhand.multi_manipulation_env import rotation_distance  # noqa: E402

TASK = "UHAS-Inhand-Repose"


def _pct(x: float) -> float:
    return float(round(100.0 * x, 3))


def main() -> dict:
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg: SpherePPORunnerCfg = SpherePPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_dir = args_cli.video_dir or os.path.join(os.path.dirname(args_cli.out), "video")
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda s: s == 0,
            video_length=args_cli.video_length,
            name_prefix=args_cli.robot,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    robot = args_cli.robot
    dev = u.device
    n = u.num_envs
    dt = float(u.step_dt)
    tol = float(u.cfg.success_tolerance)
    fall = float(u.cfg.fall_dist)
    sphere_r = float(u.sphere_radius[robot].item())
    cube_scale = float(u.cfg.cube_scales[robot])
    cube_mass = float(u.cfg.cube_masses[robot])
    # DexCube is 6 cm at scale 1.0; multi_env_cfg spawns it at `cube_scale`.
    cube_edge_m = 0.06 * cube_scale

    print(
        f"[eval] robot={robot} envs={n} steps={args_cli.steps} sphere_r={sphere_r:.5f} "
        f"cube_edge={cube_edge_m*100:.2f} cm cube_mass={cube_mass*1000:.1f} g "
        f"cube_scale_mult={args_cli.cube_scale_mult}"
    )

    # --- per-env attempt bookkeeping ------------------------------------------------------
    attempt_steps = torch.zeros(n, dtype=torch.long, device=dev)  # steps into current attempt
    attempt_best = torch.full((n,), 1e9, device=dev)             # closest rot_dist this attempt

    finished_streaks: list[int] = []
    steps_to_success: list[int] = []
    best_on_success: list[float] = []
    best_on_drop: list[float] = []
    best_on_timeout: list[float] = []
    n_success = n_drop = n_timeout = 0

    trace_rot, trace_center, trace_speed = [], [], []

    prev_serial = u.mh_reset_serial.clone()

    obs, _ = env.get_observations()
    t0 = time.time()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            # The goal is re-randomised inside step() the moment it is reached, so measuring
            # against the post-step goal makes every success invisible -- rot_dist can never be
            # observed below tolerance. Measure against the goal that was actually in force.
            goal_prev = u.goal_rot.clone()

            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            rot_dist = rotation_distance(u.object_rot, goal_prev).abs()
            # object_pos / in_hand_pos are env-relative WORLD metres (see _reset_idx: the sphere
            # in-hand target is scaled by sphere_radius before being transformed out). Divide by
            # the radius ourselves so the number is comparable across hand sizes -- note that
            # `fall_dist` upstream is NOT normalised, it is a flat 0.2 m for every hand.
            center = torch.norm(u.object_pos - u.in_hand_pos, p=2, dim=-1) / sphere_r
            speed = torch.norm(u.object_linvel, p=2, dim=-1) / sphere_r

            attempt_steps += 1
            attempt_best = torch.minimum(attempt_best, rot_dist)

            # `mh_success_events` is the latched copy of `reset_goal_buf` (see the MORPHOHAND
            # PATCH in multi_manipulation_env); reading `reset_goal_buf` here would always be 0.
            succ = u.mh_success_events.clone().bool()   # goal reached on this step
            term = u.reset_terminated.clone().bool()    # cube left the hand (out_of_reach)
            trunc = u.reset_time_outs.clone().bool()    # 30 s without a success, or streak cap

            if step < args_cli.warmup:
                # Discard the reset transient, but keep the counters consistent.
                reset_mask = succ | term | trunc
                attempt_steps[reset_mask] = 0
                attempt_best[reset_mask] = 1e9
                prev_serial = u.mh_reset_serial.clone()
                continue

            trace_rot.append(rot_dist.mean().item())
            trace_center.append(center.mean().item())
            trace_speed.append(speed.mean().item())

            # A success ends one attempt and starts the next without ending the episode.
            if succ.any():
                idx = succ.nonzero(as_tuple=False).squeeze(-1)
                n_success += int(idx.numel())
                steps_to_success.extend(attempt_steps[idx].tolist())
                best_on_success.extend(attempt_best[idx].tolist())
                attempt_steps[idx] = 0
                attempt_best[idx] = 1e9

            # A drop or a timeout ends the attempt AND the episode.
            drop_idx = (term & ~succ).nonzero(as_tuple=False).squeeze(-1)
            if drop_idx.numel():
                n_drop += int(drop_idx.numel())
                best_on_drop.extend(attempt_best[drop_idx].tolist())
            to_idx = (trunc & ~term & ~succ).nonzero(as_tuple=False).squeeze(-1)
            if to_idx.numel():
                n_timeout += int(to_idx.numel())
                best_on_timeout.extend(attempt_best[to_idx].tolist())

            # Streak lengths come from the env's own `successes` counter, latched at reset.
            fresh = (u.mh_reset_serial != prev_serial).nonzero(as_tuple=False).squeeze(-1)
            if fresh.numel():
                finished_streaks.extend(u.mh_streak_at_reset[fresh].tolist())
                attempt_steps[fresh] = 0
                attempt_best[fresh] = 1e9
            prev_serial = u.mh_reset_serial.clone()

        if (step + 1) % 500 == 0:
            att = max(n_success + n_drop + n_timeout, 1)
            print(
                f"[eval] step {step+1}/{args_cli.steps} "
                f"attempts={att} succ={_pct(n_success/att)}% drop={_pct(n_drop/att)}% "
                f"({(step+1)/(time.time()-t0):.1f} steps/s)",
                flush=True,
            )

    attempts = n_success + n_drop + n_timeout
    rot = np.asarray(trace_rot, dtype=np.float64)
    cen = np.asarray(trace_center, dtype=np.float64)
    spd = np.asarray(trace_speed, dtype=np.float64)
    streaks = np.asarray(finished_streaks, dtype=np.float64)
    s2s = np.asarray(steps_to_success, dtype=np.float64)

    def _stats(a: np.ndarray) -> dict:
        if a.size == 0:
            return {"n": 0}
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    out = {
        "robot": robot,
        "tag": args_cli.tag,
        "checkpoint": args_cli.checkpoint,
        "num_envs": n,
        "steps": args_cli.steps,
        "seed": args_cli.seed,
        "control_dt": dt,
        "sim_seconds": args_cli.steps * dt,
        "geometry": {
            "sphere_radius_m": sphere_r,
            "cube_scale": cube_scale,
            "cube_scale_mult": args_cli.cube_scale_mult,
            "cube_edge_m": cube_edge_m,
            "cube_mass_kg": cube_mass,
            # The single number that says whether the box "fits": cube diagonal vs sphere
            # diameter. UHAS ties them together by construction (cube_scale ~ sphere_radius),
            # so this should come out near-constant unless --cube-scale-mult moves it.
            "cube_diag_over_sphere_diam": float(cube_edge_m * np.sqrt(3) / (2 * sphere_r)),
        },
        "thresholds": {"success_tolerance_rad": tol, "fall_dist_radii": fall,
                       "max_consecutive_success": args_cli.max_consec},
        "attempts": {
            "total": attempts,
            "success": n_success,
            "drop": n_drop,
            "timeout": n_timeout,
            "success_rate_pct": _pct(n_success / attempts) if attempts else None,
            "drop_rate_pct": _pct(n_drop / attempts) if attempts else None,
            "timeout_rate_pct": _pct(n_timeout / attempts) if attempts else None,
        },
        "consecutive_successes": _stats(streaks),
        "consecutive_hist": np.bincount(
            streaks.astype(int), minlength=args_cli.max_consec + 1
        ).tolist() if streaks.size else [],
        "steps_to_success": _stats(s2s),
        "seconds_to_success": _stats(s2s * dt) if s2s.size else {"n": 0},
        "best_rot_dist_rad": {
            "on_success": _stats(np.asarray(best_on_success)),
            "on_drop": _stats(np.asarray(best_on_drop)),
            "on_timeout": _stats(np.asarray(best_on_timeout)),
        },
        "rot_dist_rad": _stats(rot),
        "centering_radii": _stats(cen),
        "obj_speed_radii_per_s": _stats(spd),
        "wall_seconds": time.time() - t0,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] wrote {args_cli.out}")
    print(json.dumps(out["attempts"], indent=2))

    if args_cli.trace_out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.trace_out)), exist_ok=True)
        np.savez_compressed(args_cli.trace_out, rot_dist=rot, centering=cen, obj_speed=spd)

    env.close()
    return out


if __name__ == "__main__":
    main()
    simulation_app.close()

#!/usr/bin/env python3
"""Finetune a pretrained UHAS policy on ONE hand.

Why this exists, and what it is testing
---------------------------------------
`uhas_eval.py` runs the shipped `All_grippers` checkpoint zero-shot. On our three-finger hands
that scores 0 %, and the offline authority search (`uhas_mount_search.py`) says no design in the
mount+phalanx box closes the gap to LEAP. The one hypothesis that survives that negative is that
a three-finger hand can brace the cube against the PALM and gait it -- which the authority metric
cannot see (UHAS fits its sphere to the finger ROOTS, so the palm plate never enters the score)
and which the frozen policy has no reason to know, having trained mostly on four- and
five-fingered hands. Settling it needs gradient steps, not kinematics. Hence this script.

It is deliberately NOT upstream's `scripts/rsl_rl/train.py`: that one always starts a fresh run
in a hydra-managed log tree, picks its robot list by editing the config file, and has no way to
say "load these weights, keep the observation normalizer, and take 1000 more steps on this one
hand". The env-var preamble below is the same one `uhas_eval.py` uses (multi_env_cfg resolves the
robot list in the CLASS BODY, at import time, so it has to be set before any UHAS import).

The DR curriculum knob is the subtle one
----------------------------------------
The env's domain randomisation is staged by `env.learning_iter`, which is
`common_step_counter // num_steps_per_env + cfg.iteration_bias` -- and `common_step_counter`
restarts at 0 here no matter what iteration the checkpoint came from. Stage 1 (below
`n_iter_per_stage` = 3333) applies NO randomisation, which is exactly the physics
`uhas_eval.py` measures at (eval sets `training=False`, so the whole EventCfg is empty).

So `--iteration-bias 0` (the default) finetunes and evaluates on the same nominal physics --
the setting that gives the hypothesis its best shot. Pass `--iteration-bias 10000` instead to
finetune under the full randomisation the pretrained policy converged on, which is the right
choice if the result is destined for hardware rather than for this question.

    conda run -n env_isaaclab python scripts/uhas_finetune.py \
        --robot morpho_perp_open --iterations 1000 --num-envs 2048 \
        --run-dir results/uhas/finetune/perp_palm
"""

from __future__ import annotations

import argparse
import os

# ---------------------------------------------------------------------------------------------
# Must precede every sphere_ctrl_isaaclab import -- see the module docstring.
# ---------------------------------------------------------------------------------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--robot", required=True)
_pre.add_argument("--cube-scale-mult", type=float, default=1.0)
_pre.add_argument("--max-consec", type=int, default=10)
_pre.add_argument("--morpho-root", default=os.path.abspath("results/uhas/hands"))
_known, _ = _pre.parse_known_args()
os.environ["UHAS_ROBOTS"] = _known.robot
os.environ["UHAS_EVAL"] = "0"          # training=True: DR, merge-finger augmentation, no printouts
os.environ["UHAS_CUBE_SCALE_MULT"] = str(_known.cube_scale_mult)
os.environ["UHAS_MAX_CONSEC"] = str(_known.max_consec)
os.environ["UHAS_MORPHO_ROOT"] = _known.morpho_root

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot", required=True, help="Comma-separated tags; one hand is the point.")
parser.add_argument("--checkpoint", default=os.path.abspath(
    "docs/uhas/UHAS_sim/models/UGAS_Models/All_grippers/model_14999.pt"),
    help="Pretrained weights to start from. Empty string trains from scratch.")
parser.add_argument("--run-dir", required=True, help="Where checkpoints and tensorboard logs go.")
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--iterations", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Initial LR. The schedule is adaptive-KL, so this is a starting point, "
                         "but a finetune wants to start well below the 5e-4 used from scratch.")
parser.add_argument("--entropy-coef", type=float, default=None, help="Override the 0.005 default.")
parser.add_argument("--init-noise-std", type=float, default=None,
                    help="Overwrite the loaded action std. The shipped checkpoint carries std "
                         "~28-43 per action, and the env clamps every action into a box, so "
                         "UHAS's own exploration is effectively bang-bang while its EVAL is "
                         "deterministic. A short finetune wants to improve the policy we "
                         "actually measure, so shrink the std to explore around the mean and "
                         "pair it with --entropy-coef 0 (the entropy bonus is what drove the "
                         "std up there in the first place: clamping makes a huge std free). "
                         "Omit to reproduce the pretrained regime faithfully.")
parser.add_argument("--iteration-bias", type=int, default=0,
                    help="Offsets env.learning_iter, i.e. picks the DR curriculum stage. See the "
                         "module docstring: 0 = nominal physics (matches eval), 10000 = full DR.")
parser.add_argument("--save-interval", type=int, default=100)
parser.add_argument("--load-optimizer", action="store_true",
                    help="Also restore the pretrained Adam state. Off by default: its moment "
                         "estimates were fitted to the four-hand mixture, not to this hand.")
parser.add_argument("--cube-scale-mult", type=float, default=1.0)
parser.add_argument("--max-consec", type=int, default=10)
parser.add_argument("--morpho-root", default=os.path.abspath("results/uhas/hands"))
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sphere_ctrl_isaaclab.tasks  # noqa: F401, E402
from sphere_ctrl_isaaclab.tasks.UHAS_inhand.agents.rsl_rl_ppo_cfg import SpherePPORunnerCfg  # noqa: E402

TASK = "UHAS-Inhand-Repose"


def main() -> None:
    run_dir = os.path.abspath(args_cli.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.iteration_bias = args_cli.iteration_bias

    agent_cfg: SpherePPORunnerCfg = SpherePPORunnerCfg()
    agent_cfg.device = args_cli.device
    agent_cfg.max_iterations = args_cli.iterations
    agent_cfg.save_interval = args_cli.save_interval
    agent_cfg.algorithm.learning_rate = args_cli.lr
    if args_cli.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = args_cli.entropy_coef

    env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=run_dir, device=agent_cfg.device)
    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint, load_optimizer=args_cli.load_optimizer)
        print(f"[finetune] loaded {args_cli.checkpoint} at iter {runner.current_learning_iteration}")
    start_iter = runner.current_learning_iteration

    std_before = float(runner.alg.policy.std.detach().mean())
    if args_cli.init_noise_std is not None:
        with torch.no_grad():
            runner.alg.policy.std.fill_(args_cli.init_noise_std)
    print(f"[finetune] action std {std_before:.2f} -> "
          f"{float(runner.alg.policy.std.detach().mean()):.2f}")

    meta = dict(
        robot=args_cli.robot, checkpoint=args_cli.checkpoint, num_envs=args_cli.num_envs,
        iterations=args_cli.iterations, lr=args_cli.lr, iteration_bias=args_cli.iteration_bias,
        entropy_coef=agent_cfg.algorithm.entropy_coef, load_optimizer=args_cli.load_optimizer,
        init_noise_std=args_cli.init_noise_std, std_before=std_before,
        seed=args_cli.seed, start_iter=start_iter,
        num_steps_per_env=agent_cfg.num_steps_per_env,
        cube_scale_mult=args_cli.cube_scale_mult,
    )
    with open(os.path.join(run_dir, "finetune_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[finetune] {json.dumps(meta)}", flush=True)

    t0 = time.time()
    runner.learn(num_learning_iterations=args_cli.iterations, init_at_random_ep_len=True)
    final = os.path.join(run_dir, f"model_{runner.current_learning_iteration}.pt")
    runner.save(final)

    meta["final_checkpoint"] = final
    meta["wall_seconds"] = time.time() - t0
    with open(os.path.join(run_dir, "finetune_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[finetune] done in {meta['wall_seconds']/60:.1f} min -> {final}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

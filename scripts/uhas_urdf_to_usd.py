#!/usr/bin/env python3
"""Convert a built MorphoHand URDF to USD, headless, so it can be spawned in a UHAS Isaac run.

UHAS's ``docs/add_hand.md`` says to import the URDF through the Isaac Sim GUI. That is not a
requirement of Isaac, only of the doc: Isaac Lab ships ``isaaclab.sim.converters.UrdfConverter``,
which is the same importer extension driven from Python. This script wraps it with the settings
that matter for a dexterous hand and writes the USD *next to the built hand*, at
``<hand_dir>/usd/<name>.usd``, which is where ``assets/grippers/morphohand`` looks for it.

Run in the Isaac env (NOT uv -- our .venv has mjlab, a different thing):

    /home/humanoid/miniconda3/envs/env_isaaclab/bin/python scripts/uhas_urdf_to_usd.py \
        results/uhas/hands/H0604 [more hand dirs ...]

Settings, and why:

* ``fix_base=True``    -- the UHAS task is in-hand manipulation with a static wrist; the hand is
                          never meant to translate. Matches the shipped grippers.
* ``collider_type="convex_hull"`` -- what add_hand.md asks for. Our phalanges are capsules baked
                          to STL, so a hull is exact for them and near-exact for the palm.
* ``self_collision=True`` -- the hand must not pass through itself while the policy closes it.
* ``merge_fixed_joints=True`` -- collapses ``palm_normal`` / ``<finger>_ft`` / ``palm_dummy``.
                          Those frames exist for ``process_urdf.py``, which reads the URDF
                          directly; the Isaac env reads them out of ``sphere_cik.json`` and never
                          looks for the bodies, so merging them costs nothing and saves 4 links.
* joint drive gains     -- placeholders. ``multi_env_cfg`` replaces the actuator config wholesale
                          with the shared ImplicitActuatorCfg (kP 3.0 / kD 0.2 / effort 0.4), so
                          whatever is baked here is overwritten before the first step.
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("hand_dirs", nargs="+", help="Built-hand directories (each holds <name>.urdf).")
parser.add_argument("--force", action="store_true", help="Re-convert even if the USD already exists.")
parser.add_argument(
    "--collider",
    default="convex_hull",
    choices=["convex_hull", "convex_decomposition"],
    help="Collision simplification (default: convex_hull, per UHAS add_hand.md).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib  # noqa: E402

import omni.kit.app  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


def convert(hand_dir: str, collider: str, force: bool) -> str:
    hand_dir = os.path.abspath(hand_dir)
    name = os.path.basename(hand_dir.rstrip("/"))
    urdf_path = os.path.join(hand_dir, f"{name}.urdf")
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"{urdf_path} not found -- is {hand_dir} a built hand?")

    dest_dir = os.path.join(hand_dir, "usd")
    dest_usd = os.path.join(dest_dir, f"{name}.usd")
    if os.path.isfile(dest_usd) and not force:
        print(f"[skip] {dest_usd} already exists (--force to redo)")
        return dest_usd

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=dest_dir,
        usd_file_name=f"{name}.usd",
        fix_base=True,
        merge_fixed_joints=True,
        self_collision=True,
        collider_type=collider,
        link_density=0.0,  # every link in our export carries a real <inertial>
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=3.0, damping=0.2),
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"[ok]   {urdf_path}\n    -> {converter.usd_path}")
    return converter.usd_path


def main() -> None:
    for hand_dir in args_cli.hand_dirs:
        convert(hand_dir, args_cli.collider, args_cli.force)


if __name__ == "__main__":
    main()
    simulation_app.close()
    # The importer extension can leave the app spinning; drain it rather than hang the shell.
    with contextlib.suppress(Exception):
        omni.kit.app.get_app().shutdown()

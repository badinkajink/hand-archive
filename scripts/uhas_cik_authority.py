#!/usr/bin/env python3
"""Score a hand's UHAS action space OFFLINE -- no Isaac, no rollout, seconds per design.

This reproduces UHAS's type-B (encompassing) CIK solve in numpy and asks, for a whole grid
of policy actions, two questions the repose rate cannot answer at 0 %:

* **conditioning** -- what fraction of the action box maps to a *well-posed* command?
  A slot's action deforms the canonical sphere; the CIK curls each finger until its links
  lie against that deformed profile. When no surface point survives the solve's own filters
  the joint target snaps to a **joint limit** instead (``starved``), and any target sitting
  on a limit (starved or clipped) carries no information about the sphere (``pinned``).
* **authority** -- how far does the fingertip actually travel, in widths of the cube UHAS
  gives that hand? UHAS sizes both the sphere and the cube to the hand, so raw radians are
  not comparable across designs; cube widths are.

The two are a trade-off, and where a hand sits in that plane post-dicts the failure we
rendered: LEAP has both (travel 2.2, pinned 11 %); perp has conditioning without authority
(1.20 / 17 %) and holds the cube without ever turning it; m05 / baseline / H06_04 buy
travel by slamming limits (2.3-2.6 / 34-40 %) and cannot cage the cube at all.

    python3 scripts/uhas_cik_authority.py results/uhas/hands/*/sphere_cik.json \
        docs/uhas/UHAS_sim/grippers/leap/leap_right/sphere_cik.json

NOT yet validated against task outcome -- it post-dicts 5 hands. Treat it as a design
filter to be confirmed by the Isaac eval, not as a substitute for it.

Faithful to `multi_manipulation_env._pre_physics_step` + `torch_solve_for_B_joint`:
lateral (type-A) joints stay at ``q_sphere``, which is what the CIK cascade itself uses
when it advances through an A joint, so freezing them is not an approximation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VECTOR_PHIS = np.deg2rad([50.0, 120.0])   # multi_env_cfg.vector_phis (the checkpoint's, not 60)
OFF_LO, OFF_HI = -2.0, 2.0                # min/max_vector_offsets, in sphere radii
N_POINTS = 1000                           # cfg.n_sphere_surface_samples
CUBE_RATIO = 0.66479                      # cube body-diagonal / sphere diameter, hand-invariant
Z_BAND = 0.1                              # the solve's finger-width band, in sphere radii


def _quat_xyzw_to_R(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _T(xyz, quat):
    T = np.eye(4)
    T[:3, :3] = _quat_xyzw_to_R(quat)
    T[:3, 3] = xyz
    return T


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[0, 0] = c; T[0, 1] = -s; T[1, 0] = s; T[1, 1] = c
    return T


def _fibonacci(n):
    i = np.arange(n, dtype=float)
    phi = np.arccos(1 - 2 * i / n)
    theta = (2 * np.pi * i / ((1 + np.sqrt(5)) / 2)) % (2 * np.pi)
    return theta, phi


def _deformed_radius(phi_samples, offsets):
    """Radius multiplier per sample: linear between the anchor phis, flat outside them."""
    order = np.argsort(VECTOR_PHIS)
    return np.interp(phi_samples, VECTOR_PHIS[order], np.asarray(offsets)[order])


def _solve_chain(ji, jti, chain, q, R, pts_sphere, T_base_sphere):
    """Cascade the CIK down one finger. Returns {joint: (commanded q, n surviving points)}."""
    out = {}
    T = np.eye(4)
    P = np.c_[pts_sphere, np.ones(len(pts_sphere))].T
    for j in chain[1:-1]:
        T = T @ _T(ji[j][0], ji[j][1])
        info = jti[j]
        qj = q[j]
        if info["type"] == "A":
            T = T @ _Rz(qj)
            continue
        pj = (np.linalg.inv(T @ _Rz(qj)) @ T_base_sphere @ P)[:3].T
        x, y, z = pj[:, 0], pj[:, 1], pj[:, 2]
        l = np.linalg.norm(np.asarray(info["nj"])[:2])          # link length to the next joint
        lo, hi = ji[j][4], ji[j][5]
        mask = (np.linalg.norm(pj[:, :2], axis=1) >= l) & (np.abs(z) <= Z_BAND * R)
        ref_dir = info["ref_dir"]
        mask &= ~((x < 0) & (y < 0)) if ref_dir[1] >= 0 else ~((x < 0) & (y > 0))
        if mask.any():
            ang = np.arctan2(y, x)[mask]
            new_q = float(np.clip(qj + (ang.min() if ref_dir[1] >= 0 else ang.max()), lo, hi))
        else:
            new_q = hi if ref_dir[1] >= 0 else lo                # starved -> snap to the limit
        out[j] = (new_q, int(mask.sum()))
        q[j] = new_q
        T = T @ _Rz(new_q)
    return out


def score(path: str | Path, n_grid: int = 21) -> dict:
    d = json.loads(Path(path).read_text())
    ji, jti, q_sphere = d["joint_info"], d["joint_type_info"], d["q_sphere"]
    R = ji["sphere_frame"][6]
    T_base_sphere = _T(ji["sphere_frame"][0], ji["sphere_frame"][1])
    theta, phi = _fibonacci(N_POINTS)
    unit = np.c_[np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    chains = [c for c in d["kin_chains"] if "palm_normal" not in c and "sphere_frame" not in c]

    rec: dict[str, list[float]] = {}
    starved: dict[str, list[bool]] = {}
    for a1 in np.linspace(-1, 1, n_grid):
        for a2 in np.linspace(-1, 1, n_grid):
            offs = [0.5 * (a + 1.0) * (OFF_HI - OFF_LO) + OFF_LO for a in (a1, a2)]
            pts = (_deformed_radius(phi, offs)[:, None] * unit) * R
            for chain in chains:
                q = dict(q_sphere)
                for j, (v, nm) in _solve_chain(ji, jti, chain, q, R, pts, T_base_sphere).items():
                    rec.setdefault(j, []).append(v)
                    starved.setdefault(j, []).append(nm == 0)

    st, pin_f, sd_all, sd_live, travel_all, travel_live = [], [], [], [], [], []
    for chain in chains:
        Bs = [j for j in chain[1:-1] if jti[j]["type"] in ("B", "D")]
        Ls = [np.linalg.norm(np.asarray(jti[j]["nj"])[:2]) for j in Bs]
        tip_nj = np.asarray(jti[chain[-1]].get("nj", [0.0, 0.0, 0.0]))[:2]
        for k, j in enumerate(Bs):
            v = np.asarray(rec[j])
            lo, hi = ji[j][4], ji[j][5]
            pin = (np.abs(v - lo) < 1e-3) | (np.abs(v - hi) < 1e-3)
            st.append(np.mean(starved[j])); pin_f.append(pin.mean())
            sd_all.append(v.std())
            sd_live.append(v[~pin].std() if (~pin).any() else 0.0)
            arm = sum(Ls[k:]) + np.linalg.norm(tip_nj)   # moment arm to the fingertip
            travel_all.append(sd_all[-1] * arm)
            travel_live.append(sd_live[-1] * arm)
    cube = CUBE_RATIO * 2 * R / np.sqrt(3)
    n_f = len(chains)
    first_B = [np.linalg.norm(np.asarray(jti[[j for j in c[1:-1]
               if jti[j]["type"] in ("B", "D")][0]]["nj"])[:2]) for c in chains]
    return dict(
        R=R, l_prox_R=float(np.mean(first_B)) / R,
        starved=float(np.mean(st)), pinned=float(np.mean(pin_f)),
        cmd_sd=float(np.mean(sd_all)), live_sd=float(np.mean(sd_live)),
        travel=float(np.sum(travel_all) / n_f / cube),
        travel_live=float(np.sum(travel_live) / n_f / cube),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cik", nargs="+", help="one or more sphere_cik.json paths")
    ap.add_argument("--grid", type=int, default=21, help="action samples per vector dim")
    ap.add_argument("--json", default=None, help="also write the rows here")
    args = ap.parse_args()

    rows = []
    print(f"{'hand':22s} {'R(m)':>6s} {'lprx/R':>6s} {'starv%':>6s} {'pin%':>5s} "
          f"{'trav-all':>8s} {'trav-live':>9s}")
    for p in args.cik:
        name = Path(p).parent.name
        r = score(p, args.grid)
        rows.append(dict(hand=name, path=str(p), **r))
        print(f"{name:22s} {r['R']:6.4f} {r['l_prox_R']:6.2f} {100*r['starved']:6.1f} "
              f"{100*r['pinned']:5.1f} {r['travel']:8.2f} {r['travel_live']:9.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

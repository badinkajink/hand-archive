#!/usr/bin/env python3
"""Does the offline `authority` score predict anything the task actually does?

This is the check the design search never had. `authority = travel_live x (1 - pinned)` ranked
976 designs in 22 minutes with no GPU, and on that ranking LEAP dominates every hand we can
build. But the ranking was only ever POST-dicted: the one outcome we had, the UHAS repose rate,
is 0 % on every MorphoHand and 51.7 % on LEAP, which is a binary that separates LEAP from us and
cannot order us among ourselves.

So the metrics here are deliberately NOT the repose rate:

  best_rot_dist   the closest the cube ever came to the goal, per attempt. Graded, and defined
                  even when the success rate is floored at zero -- a hand that times out at
                  0.15 rad is one tolerance from working, one that times out at 1.5 rad is not
                  rotating at all.
  obj_speed       mean object speed in radii/s. Is the cube being moved at all?
  centering       |object - in-hand target| in radii. Grip quality, which perp is already good
                  at (0.560, the best of any hand we have built).
  drop_rate       separates "cannot hold" from "holds and cannot turn".

The comparison is only meaningful because the design set is cube-controlled (50-56 mm, so UHAS
is not shrinking the task for the small hands) and collision-gated (so none of these is a hand
that cannot exist). LEAP is included as the scale reference, not as a peer -- its cube is 60 mm
and its sphere is a third bigger.

    python3 scripts/uhas_authority_vs_outcome.py --tag zeroshot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "results" / "uhas" / "eval"
DESIGN_SET = REPO / "docs" / "experiments" / "uhas_design_set.json"
CIK = REPO / "docs" / "experiments" / "uhas_cik_authority.json"

# hands evaluated before the design set, with their authority from uhas_cik_authority.json
# (travel_live x (1 - pinned)) and the eval JSON that holds their outcome.
PRIOR = [
    ("leap_right",   1.520, "leap_right.json"),
    ("perp_open",    0.930, "morpho_perp_open__anchorfix.json"),
    ("mh_baseline",  0.848, "morpho_mh_baseline__anchorfix.json"),
    ("H0604",        0.718, "morpho_H0604__anchorfix.json"),
    ("m05",          0.707, "morpho_m05__anchorfix.json"),
]


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def _best(d: dict) -> float | None:
    """Mean closest-approach over every attempt, however it ended."""
    tot = num = 0.0
    for k in ("on_success", "on_drop", "on_timeout"):
        s = d["best_rot_dist_rad"][k]
        if s.get("n"):
            tot += s["mean"] * s["n"]
            num += s["n"]
    return tot / num if num else None


def row(name: str, auth: float, d: dict) -> dict:
    return dict(
        hand=name, authority=auth,
        success=d["attempts"]["success_rate_pct"],
        drop=d["attempts"]["drop_rate_pct"],
        timeout=d["attempts"]["timeout_rate_pct"],
        best_rot=_best(d),
        rot=d["rot_dist_rad"]["mean"],
        speed=d["obj_speed_radii_per_s"]["mean"],
        centering=d["centering_radii"]["mean"],
        cube_mm=1000.0 * d["geometry"]["cube_edge_m"],
        attempts=d["attempts"]["total"],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="zeroshot", help="Tag the design-set evals were run under.")
    ap.add_argument("--extra", nargs="*", default=[], metavar="NAME=AUTH=FILE",
                    help="Extra rows, e.g. a finetuned policy: perp_ft=0.930=morpho_perp_open__ft.json")
    args = ap.parse_args()

    rows = []
    for name, auth, fname in PRIOR:
        d = _load(EVAL / fname)
        if d:
            rows.append(row(name, auth, d))

    for spec in json.loads(DESIGN_SET.read_text()):
        d = _load(EVAL / f"{spec['robot']}__{args.tag}.json")
        if d:
            rows.append(row(spec["tag"], spec["authority"], d))

    for e in args.extra:
        name, auth, fname = e.split("=", 2)
        d = _load(EVAL / fname)
        if d:
            rows.append(row(name, float(auth), d))

    rows.sort(key=lambda r: -r["authority"])
    print(f"{'hand':12s} {'AUTH':>5s} {'cube':>5s} {'succ%':>6s} {'drop%':>6s} {'t/o%':>5s} "
          f"{'bestrot':>7s} {'rot':>5s} {'speed':>5s} {'centr':>5s} {'n':>4s}")
    for r in rows:
        br = f"{r['best_rot']:.3f}" if r["best_rot"] is not None else "  -  "
        print(f"{r['hand']:12s} {r['authority']:5.3f} {r['cube_mm']:5.1f} {r['success']:6.2f} "
              f"{r['drop']:6.2f} {r['timeout']:5.1f} {br:>7s} {r['rot']:5.3f} {r['speed']:5.3f} "
              f"{r['centering']:5.3f} {r['attempts']:4d}")

    # Rank correlation on the MorphoHands only -- LEAP is a different scale of hand and would
    # carry the correlation on its own, which would prove nothing about ranking OUR designs.
    ours = [r for r in rows if r["hand"] != "leap_right" and r["best_rot"] is not None]
    if len(ours) >= 4:
        def spearman(xs: list[float], ys: list[float]) -> float:
            def rank(v):
                order = sorted(range(len(v)), key=lambda i: v[i])
                rk = [0.0] * len(v)
                for pos, i in enumerate(order):
                    rk[i] = float(pos)
                return rk
            rx, ry = rank(xs), rank(ys)
            n = len(xs)
            mx, my = sum(rx) / n, sum(ry) / n
            num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
            den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
            return num / den if den else 0.0

        a = [r["authority"] for r in ours]
        print(f"\nMorphoHands only, n={len(ours)}:")
        # best_rot is a DISTANCE, so a useful metric would make this NEGATIVE.
        print(f"  spearman(authority, best_rot)  = {spearman(a, [r['best_rot'] for r in ours]):+.3f}"
              f"   (want negative: more authority -> closer to the goal)")
        print(f"  spearman(authority, obj_speed) = {spearman(a, [r['speed'] for r in ours]):+.3f}"
              f"   (want positive: more authority -> cube actually moves)")
        print(f"  spearman(authority, drop_rate) = {spearman(a, [r['drop'] for r in ours]):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

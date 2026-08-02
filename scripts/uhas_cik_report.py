#!/usr/bin/env python3
"""Report the CIK lateral calibration for one or more hands, and flag the pathologies.

`min_offsets` / `max_offsets` in `sphere_cik.json` are the RUNTIME lateral command range per
canonical slot: the Isaac env maps a policy action a in [-1,1] onto that interval and then
CLAMPS into it (`multi_manipulation_env._pre_physics_step`). Two properties therefore matter,
and neither is obvious from the numbers unless you go looking:

* **0 must lie inside [min, max].** If it does not, the neutral action cannot be expressed and
  the clamp pins that finger's lateral joint at a large fixed angle for the whole episode.
* **the spans should be comparable across slots.** One shared policy drives every slot; if slot
  A's unit action is 0.17 rad and slot B's is 5.6 rad, the same command means wildly different
  things per finger, which no amount of policy skill can compensate for.

LEAP, which the policy was trained on, satisfies both: every range brackets 0 and the spans are
within 1.7x of each other.

    python3 scripts/uhas_cik_report.py results/uhas/hands/*/sphere_cik.json
    python3 scripts/uhas_cik_report.py --compare results/uhas/hands/H0604
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

LEAP = ("docs/uhas/UHAS_sim/sphere_ctrl_isaaclab/source/sphere_ctrl_isaaclab/"
        "sphere_ctrl_isaaclab/assets/grippers/leap/leap_right/sphere_cik.json")


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def summarise(path: str, label: str | None = None) -> dict:
    d = load(path)
    mn = np.asarray(d["min_offsets"], dtype=float)
    mx = np.asarray(d["max_offsets"], dtype=float)
    live = sorted({s for g in d["chain_indices"] for s in g})
    spans = mx[live] - mn[live]
    brackets_zero = [(mn[s] <= 0.0 <= mx[s]) for s in live]
    return {
        "label": label or os.path.basename(os.path.dirname(path)),
        "chain_indices": d["chain_indices"],
        "live": live,
        "min": mn,
        "max": mx,
        "spans": spans,
        "span_ratio": float(spans.max() / spans.min()) if spans.min() > 1e-9 else float("inf"),
        "n_pinned": int(sum(not b for b in brackets_zero)),
        "pinned_slots": [s for s, b in zip(live, brackets_zero) if not b],
        "radius": float(d["joint_info"]["sphere_frame"][6]),
    }


def show(rows: list[dict]) -> None:
    print(f"{'hand':14s} {'slots':14s} {'span ratio':>10s} {'pinned':>8s}  per-slot [min, max] (rad)")
    print("-" * 108)
    for r in rows:
        cells = []
        for s in r["live"]:
            mark = "!" if s in r["pinned_slots"] else " "
            cells.append(f"{s}:[{r['min'][s]:+.2f},{r['max'][s]:+.2f}]{mark}")
        flag = "OK" if r["n_pinned"] == 0 else f"{r['n_pinned']} PINNED"
        print(f"{r['label']:14s} {str(r['live']):14s} {r['span_ratio']:10.2f} {flag:>8s}  " + " ".join(cells))
    print("\n'!' = the range does not contain 0, so the neutral action is unreachable and the "
          "clamp\n      holds that finger's lateral joint at a fixed large angle.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="sphere_cik.json files, or hand directories")
    ap.add_argument("--compare", action="store_true",
                    help="For each hand dir, show sphere_cik_asbuilt.json next to sphere_cik.json.")
    ap.add_argument("--no-leap", action="store_true", help="Omit the LEAP reference row.")
    args = ap.parse_args()

    rows = []
    if not args.no_leap and os.path.isfile(LEAP):
        rows.append(summarise(LEAP, "LEAP (ref)"))

    for p in args.paths:
        if os.path.isdir(p):
            name = os.path.basename(p.rstrip("/"))
            if args.compare:
                old = os.path.join(p, "sphere_cik_asbuilt.json")
                if os.path.isfile(old):
                    rows.append(summarise(old, f"{name} (before)"))
            rows.append(summarise(os.path.join(p, "sphere_cik.json"), name))
        else:
            rows.append(summarise(p))

    if not rows:
        sys.exit("nothing to report")
    show(rows)


if __name__ == "__main__":
    main()

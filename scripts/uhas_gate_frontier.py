#!/usr/bin/env python3
"""Gate the top of the authority search against MuJoCo contact, to find the REAL buildable frontier.

E7 declared a design buildable if its index<->middle separation cleared the floor measured by
`morph_selfcollision_gate.py --sep-sweep`, which was swept at the SHIPPED phalanx lengths and
therefore only ever saw the floor move with the PROXIMAL scale. That rule is too permissive: the
highest-authority "buildable" design (gap 45 / sep 105 / prox 1.00 / dist 1.67, authority 1.084)
self-collides index-vs-middle at 64 N once it is actually baked and settled. A longer DISTAL
reaches across the gap the proximal floor was measured to protect.

So this re-gates the top of the search with the real contact solver. Two verdicts, and they mean
different things:

  collide       hand-vs-hand contact force at the `open` or `closed` keyframe. A hard fail: the
                design does not exist.
  screwdriver   the retargeted fingertips cannot reach OUR reference screwdriver grasp within
                2 mm. NOT a fail for the UHAS cube task -- different object, different size, and
                UHAS spawns its cube 1.3 sphere-radii off the palm in the sphere frame rather
                than at our grasp pose. It is reported because a design that fails it is one the
                rest of this program could not use, which is worth knowing before promoting it.

    .venv/bin/python scripts/uhas_gate_frontier.py --top 40
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from morph_selfcollision_gate import (  # noqa: E402
    BASE_SCENE,
    evaluate,
    reference_tip_targets,
)
from morphohand.sampling.morphology import MorphologyValues  # noqa: E402
from perp_phalanx_variant import rescale_phalanges  # noqa: E402

SEARCH = REPO / "docs" / "experiments" / "uhas_mount_search.json"
FLOOR = {1.0: 96.0, 0.85: 85.0, 0.7: 75.0, 0.55: 65.0, 0.4: 55.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=40, help="How many top-authority designs to gate.")
    ap.add_argument("--cube", type=float, nargs=2, default=[50.0, 56.0], metavar=("LO", "HI"),
                    help="Keep the cube (mm) in this band so the task is held fixed.")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--no-floor", action="store_true",
                    help="Skip the sep-sweep prefilter and gate purely on contact.")
    ap.add_argument("--json", default=str(REPO / "docs" / "experiments" / "uhas_gate_frontier.json"))
    args = ap.parse_args()

    rows = json.loads(SEARCH.read_text())
    lo, hi = args.cube
    cand = [r for r in rows if r["min_span"] >= 0.5 and lo <= r["cube_mm"] <= hi
            and (args.no_floor or 1000.0 * r["sep_y"] >= FLOOR[r["prox"]] - 1e-6)]
    cand.sort(key=lambda r: -r["authority"])
    cand = cand[:args.top]
    print(f"gating {len(cand)} designs, cube {lo:.0f}-{hi:.0f} mm, settle {args.steps} steps\n")

    targets = reference_tip_targets(BASE_SCENE)
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        scenes: dict[tuple[float, float], Path] = {}
        print(f"{'gap':>4s} {'sep':>4s} {'prox':>5s} {'dist':>5s} {'AUTH':>5s} {'cube':>5s} "
              f"  verdict")
        for r in cand:
            key = (r["prox"], r["dist"])
            if key not in scenes:
                if key == (1.0, 1.0):
                    scenes[key] = BASE_SCENE
                else:
                    p = outdir / f"base_p{key[0]:g}_d{key[1]:g}.xml"
                    p.write_text(rescale_phalanges(BASE_SCENE.read_text(), *key))
                    scenes[key] = p
            m = r["morph"]
            morph = MorphologyValues(m["thumb_x"], 0.0, 0.0,
                                     m["index_x"], m["index_y"], 0.0,
                                     m["middle_x"], m["middle_y"], 0.0)
            label = f"g{1000*r['gap_x']:.0f}s{1000*r['sep_y']:.0f}p{r['prox']:g}d{r['dist']:g}"
            v = evaluate(label, morph, outdir, args.steps, None, scenes[key], targets)
            collide = v.worst_force > 1e-6
            reach = v.ik_residual < 2e-3
            verdict = (f"COLLIDE {v.worst_pair} {v.worst_force:.0f}N" if collide else
                       ("OK" if reach else f"no-screwdriver ({v.ik_residual*1e3:.1f} mm)"))
            print(f"{1000*r['gap_x']:4.0f} {1000*r['sep_y']:4.0f} {r['prox']:5.2f} {r['dist']:5.2f} "
                  f"{r['authority']:5.3f} {r['cube_mm']:5.1f}   {verdict}", flush=True)
            out.append(dict(gap_x=r["gap_x"], sep_y=r["sep_y"], prox=r["prox"], dist=r["dist"],
                            authority=r["authority"], travel_live=r["travel_live"],
                            pinned=r["pinned"], cube_mm=r["cube_mm"], morph=m,
                            collide=collide, worst_pair=v.worst_pair,
                            worst_force=v.worst_force, ik_residual=v.ik_residual,
                            reaches_screwdriver=reach))

    clean = [r for r in out if not r["collide"]]
    both = [r for r in clean if r["reaches_screwdriver"]]
    print(f"\n{len(clean)}/{len(out)} collision-free; {len(both)} of those also reach our "
          f"screwdriver grasp")
    if clean:
        print(f"best collision-free authority: {clean[0]['authority']:.3f} "
              f"(gap {1000*clean[0]['gap_x']:.0f} sep {1000*clean[0]['sep_y']:.0f} "
              f"prox {clean[0]['prox']:g} dist {clean[0]['dist']:g})")
    if both:
        print(f"best that also reaches the screwdriver: {both[0]['authority']:.3f} "
              f"(gap {1000*both[0]['gap_x']:.0f} sep {1000*both[0]['sep_y']:.0f} "
              f"prox {both[0]['prox']:g} dist {both[0]['dist']:g})")
    Path(args.json).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

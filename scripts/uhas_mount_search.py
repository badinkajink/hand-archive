#!/usr/bin/env python3
"""Search the perp mount box (and phalanx lengths) on the offline UHAS authority metric.

One full MJCF -> URDF -> UHAS-sphere -> CIK rebuild takes ~5 s and needs no GPU and no
Isaac, so the whole design box is searchable in minutes -- which is the only reason this
is worth doing at all, given that the task metric (repose rate) is floored at 0 % and
cannot rank anything.

**The metric only sees RELATIVE finger geometry.** UHAS fits its canonical sphere to the
*finger roots* (`construct_palm_plane` -> `construct_sphere_with_palm`); the palm plate's
geometry never enters. Translating all three mounts together is therefore an exact no-op
here -- verified, two such designs score byte-identically. So the search runs over the two
separations, and where the cluster sits on the plate is a free choice to be settled by
`morph_selfcollision_gate.py` and by whether the cube still lands in the fingers.

Axes:

* ``gap_x``  -- thumb mount to finger-pair mount, along x (shipped: 100 mm)
* ``sep_y``  -- index mount to middle mount, along y (shipped: 96 mm)
* ``prox``   -- scale on BOTH proximal capsules (mcp 0.05 + len 0.04 = 0.09 m)
* ``dist``   -- scale on the distal capsule (0.03 m)

The last two are not in the 9-param morphology space -- those slides can only *lengthen* a
finger -- so they are applied by rewriting the MJCF.

Scoring is `uhas_cik_authority.score`; the headline is

    authority = travel_live x (1 - pinned)

i.e. expected fingertip travel per action sample that actually carries information about
the commanded sphere, in widths of the cube UHAS gives that hand. LEAP scores 1.52.

    .venv-uhas/bin/python scripts/uhas_mount_search.py --grid 7 --jobs 8 \
        --json docs/experiments/uhas_mount_search.json

CAVEAT: this is kinematics only. It does not know about the palm plate or the object, so a
high-scoring design can still be physically impossible -- run the survivors through
`scripts/morph_selfcollision_gate.py --retarget` before believing any of them.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from perp_phalanx_variant import rescale_phalanges  # noqa: E402
from uhas_cik_authority import score  # noqa: E402

SRC = REPO / "assets" / "mjcf" / "perp" / "perp_hand_morphology_actuated.xml"

# shipped mounts (m, palm frame) and the slide ranges that move them
THUMB0, PAIR_X0, PAIR_Y0 = -0.065, 0.035, 0.048
THUMB_X_RANGE = (-0.0075, 0.0525)     # thumb absolute x in [-0.0725, -0.0125]
PAIR_X_RANGE = (-0.0225, 0.0375)      # pair absolute x  in [ 0.0125,  0.0725]
INDEX_Y_RANGE = (-0.0355, 0.0245)     # |y| in [0.0125, 0.0725] -> sep_y in [0.025, 0.145]
GAP_RANGE = (0.025, 0.145)
SEP_RANGE = (0.025, 0.145)
CENTROID0 = (THUMB0 + 2 * PAIR_X0) / 3.0   # keep the finger cluster where it shipped
LEAP_AUTHORITY = 1.52   # travel_live x (1 - pinned) for leap_right, for reference


def mounts_for(gap: float, sep: float) -> dict[str, float] | None:
    """Slide values realising a thumb-pair x gap and an index-middle y separation.

    The x pair is under-determined (only the gap is visible to UHAS), so we place the
    cluster centroid where it shipped and slide it only as far as the rails force.
    """
    pair = float(np.clip(CENTROID0 + gap / 3.0, PAIR_X0 + PAIR_X_RANGE[0], PAIR_X0 + PAIR_X_RANGE[1]))
    thumb = pair - gap
    lo, hi = THUMB0 + THUMB_X_RANGE[0], THUMB0 + THUMB_X_RANGE[1]
    if thumb < lo:                                    # push the pair forward instead
        thumb = lo
        pair = float(np.clip(thumb + gap, PAIR_X0 + PAIR_X_RANGE[0], PAIR_X0 + PAIR_X_RANGE[1]))
    elif thumb > hi:
        thumb = hi
        pair = float(np.clip(thumb + gap, PAIR_X0 + PAIR_X_RANGE[0], PAIR_X0 + PAIR_X_RANGE[1]))
    if abs((pair - thumb) - gap) > 1e-6:
        return None                                   # gap unreachable on the rails
    y = sep / 2.0
    if not (PAIR_Y0 + INDEX_Y_RANGE[0] - 1e-6 <= y <= PAIR_Y0 + INDEX_Y_RANGE[1] + 1e-6):
        return None
    y = float(np.clip(y, PAIR_Y0 + INDEX_Y_RANGE[0], PAIR_Y0 + INDEX_Y_RANGE[1]))
    return dict(thumb_x=thumb - THUMB0, index_x=pair - PAIR_X0, middle_x=pair - PAIR_X0,
                index_y=y - PAIR_Y0, middle_y=-(y - PAIR_Y0))


def write_variant(prox: float, dist: float, out: Path) -> Path:
    """MJCF with the proximal (0.05 + 0.04) and distal (0.03) capsules rescaled."""
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"perp_p{prox:g}_d{dist:g}.xml"
    if not dst.exists():
        dst.write_text(rescale_phalanges(SRC.read_text(), prox, dist))
    return dst


def build_and_score(job: dict) -> dict | None:
    out = Path(job["out"])
    cmd = [sys.executable, str(REPO / "scripts" / "uhas_build_hand.py"),
           "--mjcf", job["mjcf"], "--out", str(out), "--name", out.name,
           "--palm-body", "palm_pose", "--open-from-keyframe", "open"]
    for k, v in job["morph"].items():
        cmd += ["--morph", f"{k}={v}"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    cik = out / "sphere_cik.json"
    if r.returncode != 0 or not cik.exists():
        return None
    s = score(cik)
    summ = json.loads((out / "uhas_summary.json").read_text())
    claimed = sorted({i for idx in summ["chain_indices"] for i in idx})
    spans = [summ["lateral_spans"][i] for i in claimed]
    return dict(
        gap_x=job["gap_x"], sep_y=job["sep_y"], prox=job["prox"], dist=job["dist"],
        **s, authority=s["travel_live"] * (1.0 - s["pinned"]),
        # UHAS scales the cube to the hand, so a design can "win" by shrinking the task
        cube_mm=1000.0 * 0.66479 * 2 * s["R"] / np.sqrt(3),
        min_span=float(min(spans)), slots_dead=summ["slots_dead"],
        morph=job["morph"], out=str(out),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=7, help="samples per separation axis")
    ap.add_argument("--prox", type=float, nargs="+", default=[1.0, 0.85, 0.7, 0.55])
    ap.add_argument("--dist", type=float, nargs="+", default=[1.0, 1.33, 1.67])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--work", default=None, help="scratch dir for the builds")
    ap.add_argument("--json", default=None)
    ap.add_argument("--min-span", type=float, default=0.5,
                    help="reject designs where a claimed slot's lateral span collapses (rad)")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    work = Path(args.work or (REPO / "results" / "uhas" / "mount_search"))
    work.mkdir(parents=True, exist_ok=True)
    gaps = np.linspace(*GAP_RANGE, args.grid)
    seps = np.linspace(*SEP_RANGE, args.grid)

    jobs = []
    for prox in args.prox:
        for dist in args.dist:
            mjcf = write_variant(prox, dist, work / "mjcf")
            for i, g in enumerate(gaps):
                for j, s in enumerate(seps):
                    m = mounts_for(float(g), float(s))
                    if m is None:
                        continue
                    jobs.append(dict(mjcf=str(mjcf), out=str(work / f"p{prox:g}d{dist:g}_g{i}s{j}"),
                                     prox=prox, dist=dist, gap_x=float(g), sep_y=float(s), morph=m))
    print(f"{len(jobs)} designs, {args.jobs} workers -> {work}", flush=True)

    rows, failed = [], 0
    with cf.ProcessPoolExecutor(args.jobs) as ex:
        for n, r in enumerate(ex.map(build_and_score, jobs), 1):
            if r is None:
                failed += 1
            else:
                rows.append(r)
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)} ({failed} failed)", flush=True)

    rows.sort(key=lambda r: -r["authority"])
    ok = [r for r in rows if r["min_span"] >= args.min_span]
    print(f"\n{len(rows)} scored, {failed} failed, {len(ok)} with every claimed lateral span "
          f">= {args.min_span} rad\n")
    print(f"{'design':16s} {'gapx':>5s} {'sepy':>5s} {'prox':>5s} {'dist':>5s} {'cube':>5s} "
          f"{'l/R':>5s} {'starv':>5s} {'pin%':>5s} {'trav':>5s} {'AUTH':>5s} {'span':>5s}")
    for r in ok[:args.top]:
        print(f"{Path(r['out']).name:16s} {1000*r['gap_x']:5.0f} {1000*r['sep_y']:5.0f} "
              f"{r['prox']:5.2f} {r['dist']:5.2f} {r['cube_mm']:5.1f} {r['l_prox_R']:5.2f} "
              f"{100*r['starved']:5.1f} {100*r['pinned']:5.1f} {r['travel_live']:5.2f} "
              f"{r['authority']:5.2f} {r['min_span']:5.2f}")
    print(f"\n(LEAP reference authority = {LEAP_AUTHORITY:.2f}; shipped perp = gap 100 / sep 96,"
          f" prox 1.00, dist 1.00; separations in mm)")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

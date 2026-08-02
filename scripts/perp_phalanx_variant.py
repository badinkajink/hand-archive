#!/usr/bin/env python3
"""Rescale the perp finger phalanges in any perp MJCF (hand file or scene).

The 9-param morphology space can only *lengthen* a finger (the ``*_len`` slides run
0 .. +35 mm), so shortening the 90 mm proximal -- the thing that makes our fingers
2.5-3x longer than LEAP's relative to the grasp sphere -- is not expressible as a design
sample. It has to be a geometry edit, which is what this does:

    proximal = mcp capsule (0.05) + len_frame + pip capsule (0.04)   -> scaled by --prox
    distal   = pip capsule (0.03)                                    -> scaled by --dist

Both the capsule ``fromto`` and the child body ``pos`` move, so the kinematics stay
consistent. Works on `perp_hand.xml`, `perp_hand_morphology_actuated.xml` and the
screwdriver scenes alike -- they all carry the same three capsule strings per finger.

    python3 scripts/perp_phalanx_variant.py --prox 0.5 \
        --src assets/mjcf/perp/scenes/scene_screwdriver_medium_perp.xml --out /tmp/scene_p05.xml

The point of doing it to a *scene* is that `morph_selfcollision_gate.py --base-scene` can
then gate a short-proximal design for self-collision and grasp reachability, which the
kinematics-only UHAS metric cannot see.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROX_SEGS = (0.05, 0.04)   # mcp capsule, pip capsule -- together the 0.09 m proximal
DIST_SEG = 0.03            # the distal capsule
FINGERS = ("thumb", "index", "middle")


def rescale_phalanges(txt: str, prox: float = 1.0, dist: float = 1.0) -> str:
    """Return `txt` with the proximal scaled by `prox` and the distal by `dist`."""
    s1, s2 = round(PROX_SEGS[0] * prox, 6), round(PROX_SEGS[1] * prox, 6)
    sd = round(DIST_SEG * dist, 6)
    # the passes below run distal-first, so a rescaled length must not land on a source
    # length that a later pass still rewrites
    if sd in PROX_SEGS or s1 == PROX_SEGS[1]:
        raise ValueError(f"rescale collides with a later pass: prox {s1}/{s2}, dist {sd}")

    for f in FINGERS:
        txt = txt.replace(f'<body name="{f}_len_frame" pos="{PROX_SEGS[0]} 0 0">',
                          f'<body name="{f}_len_frame" pos="{s1} 0 0">')
        txt = txt.replace(f'<body name="{f}_pip_frame" pos="{PROX_SEGS[1]} 0 0">',
                          f'<body name="{f}_pip_frame" pos="{s2} 0 0">')
        txt = txt.replace(f'<body name="{f}_tip" pos="{DIST_SEG} 0 0"',
                          f'<body name="{f}_tip" pos="{sd} 0 0"')
    for old_len, new_len in ((DIST_SEG, sd), (PROX_SEGS[0], s1), (PROX_SEGS[1], s2)):
        old, new = f'fromto="0 0 0 {old_len} 0 0"', f'fromto="0 0 0 {new_len} 0 0"'
        if old == new:
            continue
        if txt.count(old) != len(FINGERS):
            raise ValueError(f"expected {len(FINGERS)} of {old}, found {txt.count(old)}")
        txt = txt.replace(old, new)
    return txt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prox", type=float, default=1.0)
    ap.add_argument("--dist", type=float, default=1.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rescale_phalanges(Path(args.src).read_text(), args.prox, args.dist))
    print(f"{out}: proximal {1000 * sum(PROX_SEGS):.0f} -> {1000 * sum(PROX_SEGS) * args.prox:.0f} mm, "
          f"distal {1000 * DIST_SEG:.0f} -> {1000 * DIST_SEG * args.dist:.0f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

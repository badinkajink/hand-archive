#!/usr/bin/env python3
"""Pick a small, CONTROLLED set of perp designs from the mount search and build them for Isaac.

The offline search (`uhas_mount_search.py`) scored 976 designs on `authority = travel_live x
(1 - pinned)` and produced a clean negative: LEAP dominates the whole box. But `authority` has
only ever been POST-dicted -- it was never checked against a task outcome, because the outcome
we have (repose rate) is floored at 0 % on every hand we own and so cannot rank anything. This
script builds the hands for that check.

Two things make the check honest, and both cost design candidates:

* **The collision floor.** `morph_selfcollision_gate.py --sep-sweep` measured how close the
  index and middle mounts can get before the two facing fingers interfere, and the floor tracks
  the proximal length: 96 mm at the shipped 90 mm proximal, 85 / 75 / 65 / 55 mm as it shortens.
  Designs below their floor are not buildable and are dropped here rather than evaluated.
* **The cube.** UHAS fits its sphere to the finger roots and then scales the cube to the sphere,
  so a design can "win" by shrinking the task. Restricting to cube 50-56 mm (the shipped perp is
  52.6 mm) holds the task roughly fixed, and still leaves an authority span of 0.59-1.08.

The set is chosen to make one comparison each:

    d1_hi     highest authority that is buildable at a perp-sized cube
    d2_mount  best MOUNT-ONLY move (prox = dist = 1) -- the cheapest real hardware change
    d3_mid    middle of the buildable range
    d4_pin    LOW authority via bad conditioning (travel 1.29, 52 % pinned)
    d5_lo     LOW authority via little travel (travel 0.79, 22 % pinned)

d4 and d5 have the SAME authority by opposite routes, which is the one test of whether the
product form of the metric means anything.

    .venv-uhas/bin/python scripts/uhas_design_set.py --build      # MJCF -> URDF -> CIK
    .venv-uhas/bin/python scripts/uhas_design_set.py --gate       # is it physically real?

`--build` stops at the CIK, which is all the OFFLINE half needs. Isaac additionally needs a USD,
and that conversion needs Isaac itself, so it cannot live in this script's env. Before evaluating:

    /home/humanoid/miniconda3/envs/env_isaaclab/bin/python scripts/uhas_urdf_to_usd.py \
        results/uhas/hands/d1_hi ... --headless

Skipping it does NOT fail loudly at build time -- the eval dies later, one hand at a time, with
`FileNotFoundError: missing .../usd/<name>.usd`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from perp_phalanx_variant import rescale_phalanges  # noqa: E402

SRC = REPO / "assets" / "mjcf" / "perp" / "perp_hand_morphology_actuated.xml"
SEARCH = REPO / "docs" / "experiments" / "uhas_mount_search.json"
HANDS = REPO / "results" / "uhas" / "hands"
MJCF_DIR = REPO / "results" / "uhas" / "design_set" / "mjcf"

# index<->middle mount separation floor (mm) per proximal scale -- morph_selfcollision_gate.py
# --sep-sweep --prox <p>. The shipped hand sits ON the 96 mm floor.
FLOOR = {1.0: 96.0, 0.85: 85.0, 0.7: 75.0, 0.55: 65.0, 0.4: 55.0}
CUBE_LO, CUBE_HI = 50.0, 56.0        # mm; shipped perp is 52.6

# (tag, gap_x mm, sep_y mm, prox, dist) -- grid points of the search
SET = [
    # d1 was gap 45 / sep 105 / prox 1.00 / dist 1.67 (authority 1.084) until the contact solver
    # rejected it: index and middle DISTAL links (`*_len_frame`) interpenetrate at 64 N. The
    # sep-sweep floor this set was first filtered on was measured at the shipped phalanx lengths
    # and so only ever saw the floor track the PROXIMAL. See uhas_gate_frontier.py -- the real
    # collision-free ceiling in this cube band is 1.002, not 1.084.
    ("d1_hi",     65, 145, 0.85, 1.33),
    ("d2_mount",  85, 105, 1.00, 1.00),
    ("d3_mid",   125, 105, 0.85, 1.33),
    ("d4_pin",    25, 125, 1.00, 1.33),
    ("d5_lo",    145,  85, 0.70, 1.67),
]


def buildable(r: dict) -> bool:
    return r["min_span"] >= 0.5 and 1000.0 * r["sep_y"] >= FLOOR[r["prox"]] - 1e-6


def pick(rows: list[dict]) -> list[tuple[str, dict]]:
    out = []
    for tag, gap, sep, prox, dist in SET:
        hit = [r for r in rows
               if round(1000 * r["gap_x"]) == gap and round(1000 * r["sep_y"]) == sep
               and r["prox"] == prox and r["dist"] == dist]
        if not hit:
            raise SystemExit(f"{tag}: no search row at gap {gap} sep {sep} prox {prox} dist {dist}")
        r = hit[0]
        if not buildable(r):
            raise SystemExit(f"{tag}: below the collision floor ({FLOOR[prox]} mm) or span-dead")
        if not (CUBE_LO <= r["cube_mm"] <= CUBE_HI):
            raise SystemExit(f"{tag}: cube {r['cube_mm']:.1f} mm outside the controlled band")
        out.append((tag, r))
    return out


def build(tag: str, r: dict) -> bool:
    MJCF_DIR.mkdir(parents=True, exist_ok=True)
    mjcf = MJCF_DIR / f"perp_p{r['prox']:g}_d{r['dist']:g}.xml"
    if not mjcf.exists():
        mjcf.write_text(rescale_phalanges(SRC.read_text(), r["prox"], r["dist"]))
    out = HANDS / tag
    cmd = [sys.executable, str(REPO / "scripts" / "uhas_build_hand.py"),
           "--mjcf", str(mjcf), "--out", str(out), "--name", tag,
           "--palm-body", "palm_pose", "--open-from-keyframe", "open"]
    for k, v in r["morph"].items():
        cmd += ["--morph", f"{k}={v}"]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if p.returncode != 0 or not (out / "sphere_cik.json").exists():
        print(p.stdout[-2000:] + p.stderr[-2000:])
        return False
    return True


def gate(tag: str, r: dict, python: str) -> None:
    # NOTE the interpreter: the builder half runs in `.venv-uhas` (trimesh/urdf-parser, no
    # morphohand), the gate runs MuJoCo through `morphohand.sampling`, which only exists in the
    # repo's own `.venv`. Running the gate under sys.executable dies on `import imageio`.
    m = r["morph"]
    cmd = [python, str(REPO / "scripts" / "morph_selfcollision_gate.py"),
           "--thumb", str(m["thumb_x"]), "0", "0",
           "--index", str(m["index_x"]), str(m["index_y"]), "0",
           "--middle", str(m["middle_x"]), str(m["middle_y"]), "0",
           "--prox", str(r["prox"]), "--dist", str(r["dist"]), "--retarget"]
    print(f"\n===== {tag} =====", flush=True)
    subprocess.run(cmd, cwd=REPO)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="MJCF -> URDF -> UHAS sphere -> CIK")
    ap.add_argument("--gate", action="store_true", help="run the self-collision gate on each")
    ap.add_argument("--gate-python", default=str(REPO / ".venv" / "bin" / "python"),
                    help="Interpreter for the gate -- it needs morphohand + MuJoCo, i.e. .venv.")
    ap.add_argument("--manifest", default=str(REPO / "docs" / "experiments" / "uhas_design_set.json"))
    args = ap.parse_args()

    rows = json.loads(SEARCH.read_text())
    chosen = pick(rows)

    print(f"{'tag':10s} {'gap':>4s} {'sep':>4s} {'prox':>5s} {'dist':>5s} {'AUTH':>5s} "
          f"{'trav':>5s} {'pin%':>5s} {'l/R':>5s} {'cube':>5s} {'span':>5s}")
    for tag, r in chosen:
        print(f"{tag:10s} {1000*r['gap_x']:4.0f} {1000*r['sep_y']:4.0f} {r['prox']:5.2f} "
              f"{r['dist']:5.2f} {r['authority']:5.3f} {r['travel_live']:5.2f} "
              f"{100*r['pinned']:5.1f} {r['l_prox_R']:5.2f} {r['cube_mm']:5.1f} {r['min_span']:5.2f}")
    print("\n(shipped perp: gap 100 sep 96 prox 1.00 dist 1.00, authority 0.93, cube 52.6 mm; "
          "LEAP authority 1.52)")

    if args.build:
        for tag, r in chosen:
            print(f"[build] {tag} ...", flush=True)
            if not build(tag, r):
                print(f"[build] {tag} FAILED")
                return 1
        Path(args.manifest).write_text(json.dumps(
            [dict(tag=t, robot=f"morpho_{t}", **r) for t, r in chosen], indent=1))
        print(f"wrote {args.manifest}")

    if args.gate:
        for tag, r in chosen:
            gate(tag, r, args.gate_python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

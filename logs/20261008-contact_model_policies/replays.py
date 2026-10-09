"""Step 4 replays for every finished run of the 2026-10-08 queue (called by after_run.sh; plain python3).

Per run: `rl_policy_replay.py record` once (GPU, uv env, 16 worlds of the final checkpoint, onset step 58); then for
worlds 0..W-1 (W = $REPLAY_WORLDS, default 6): `bench` (the bench-like scene of that world), and one process per
replay in CPU MuJoCo (the run's own tips), Drake (hydroelastic TPU block) and Newton (hydroelastic TPU block, kh/m_eff,
friction rows at the pads' 10 ms). Drake and Newton take the block geometry from the TPU mesh run of the same seed
(rl_policy_replay.block_dir), so mesh runs go first and a run whose mesh partner is not recorded yet waits. Rows:
docs/experiments/20261008-contact_model_policies/policy_replay.jsonl (keys dir, world, engine; done ones skipped).

    python3 logs/20261008-contact_model_policies/replays.py [engine ...]
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "logs/20261008-contact_model_policies/jobs.txt"
REPLAY = ROOT / "logs/20261008-contact_model_policies/replay"
OUT = ROOT / "docs/experiments/20261008-contact_model_policies/policy_replay.jsonl"
RP = str(ROOT / "scripts/rl_policy_replay.py")
UV = ["uv", "run", "--extra", "rl", "--extra", "gpu", "python"]
PYD = str(ROOT / "logs/20261001-hom_contact/venv/bin/python")      # MuJoCo 3.6 + Drake
PYN = str(ROOT / "logs/20261004-contact-transfer/venv/bin/python")  # Newton
GUARD = [os.path.expanduser("~/.claude/bin/resguard.sh"), "run", "--mem", "6G", "--cpu", "400", "--"]
WORLDS = int(os.environ.get("REPLAY_WORLDS", "6"))
ENGINES = sys.argv[1:] or ["mujoco", "drake", "newton"]
STEPS_PER_IT = 2048 * 24


def plateau_stops():
    """Runs the checkpoint watch stopped on the plateau test (stops.jsonl): their last checkpoint is their final policy."""
    p = ROOT / "docs/experiments/20261008-contact_model_policies/stops.jsonl"
    out = {}
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            if r.get("reason", "").startswith("plateau"):
                out[r["tag"]] = int(r["iteration"])
    return out


def finished():
    out, pl = [], plateau_stops()
    for line in open(JOBS):
        stopped = line.startswith("#stopped# ")
        f = (line[len("#stopped# "):] if stopped else line).split("#")[0].split()
        if len(f) != 4 or (stopped and f[0] not in pl):      # a run stopped as degenerate has no final policy
            continue
        it = pl.get(f[0], int(f[3]) // STEPS_PER_IT - 1)
        if (ROOT / "results/rl" / f[0] / "tensorboard" / f"model_{it}.pt").exists():
            out.append(f[0])
    return sorted(out, key=lambda t: (0 if "tpu27mesh_" in t else 1, t))


def block_dir(d: Path) -> Path:
    n = d.name
    for a, b in (("pads1", "mesh"), ("tpu27skin", "tpu27mesh"), ("_box_", "_tpu27mesh_")):
        n = n.replace(a, b)
    return d.with_name(n)


def done():
    keys = set()
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("engine") and r.get("kind") != "record":
                keys.add((Path(r["dir"]).name, r.get("world"), r["engine"]))
    return keys


def run(cmd, gpu=False):
    env = dict(os.environ, MUJOCO_GL="egl")
    if gpu:
        env["WARP_CACHE_PATH"] = tempfile.mkdtemp(prefix="warp_replay_")
        cmd = GUARD + cmd
    print(" ".join(cmd[-8:]), flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def main():
    tags = finished()
    for tag in tags:
        if not (REPLAY / tag / "rec.npz").exists():
            run(UV + [RP, "record", "--study", "20261008", "--tag", tag, "--n", "16"], gpu=True)
    have = done()
    for tag in tags:
        d = REPLAY / tag
        if not (d / "rec.npz").exists():
            continue
        for w in range(WORLDS):
            need = [e for e in ENGINES if (tag, w, e) not in have]
            need = [e for e in need if e == "mujoco" or (block_dir(d) / "bench.xml").exists() or block_dir(d) == d]
            if not need:
                continue
            if run([PYD, RP, "bench", "--study", "20261008", "--dir", str(d), "--world", str(w)]) != 0:
                break
            for e in need:
                if e == "newton":
                    run([PYN, RP, "replay", "--study", "20261008", "--dir", str(d), "--engine", "newton"], gpu=True)
                else:
                    run([PYD, RP, "replay", "--study", "20261008", "--dir", str(d), "--engine", e])
    return 0


if __name__ == "__main__":
    sys.exit(main())

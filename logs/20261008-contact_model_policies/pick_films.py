"""Best and median seed of each arm from final_eval.jsonl (held fraction, then held cosine), for the step-6 films.
Prints lines `<arm> <role> <tag> <final iteration>`; with --json writes films.json beside the rows."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
rows = [json.loads(l) for l in open(ROOT / "docs/experiments/20261008-contact_model_policies/final_eval.jsonl")]
out = []
for arm in sorted({r["arm"] for r in rows}):
    rs = sorted((r for r in rows if r["arm"] == arm and r.get("status") == "ok"),
                key=lambda r: (r["held_frac"], r["held_cos_mean"] if r["held_cos_mean"] is not None else -2.0))
    if not rs:
        continue
    best, med = rs[-1], rs[len(rs) // 2] if len(rs) % 2 else rs[len(rs) // 2 - 1]
    out.append(dict(arm=arm, role="best", tag=best["tag"], iteration=best["iteration"]))
    if med["tag"] != best["tag"]:
        out.append(dict(arm=arm, role="median", tag=med["tag"], iteration=med["iteration"]))
for o in out:
    print(o["arm"], o["role"], o["tag"], o["iteration"])
if "--json" in sys.argv:
    (ROOT / "docs/experiments/20261008-contact_model_policies/films.json").write_text(json.dumps(out, indent=1))

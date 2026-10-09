"""Print one watched checkpoint's row (and the previous ones of the run) for the session's decision.
    python3 logs/20261008-contact_model_policies/show_watch.py <tag> [iteration]"""
import json, sys
tag = sys.argv[1]
rows = {}
for l in open("/home/humanoid/Programs/hand/docs/experiments/20261008-contact_model_policies/watch.jsonl"):
    r = json.loads(l)
    if r["tag"] == tag:
        rows[r["iteration"]] = r
it = int(sys.argv[2]) if len(sys.argv) > 2 else max(rows)
for i in sorted(rows):
    r = rows[i]
    hc = r["held_cos_mean"]
    print(f'it {i:4d} ({(i + 1) * 49152 / 1e6:4.1f} M) held {r["n_held"]:2d}/64 held cos {"   -  " if hc is None else f"{hc:.3f}"} '
          f'jerk {r.get("ang_jerk_hold_median") or 0:5.0f} spin {r.get("ang_speed_hold_median") or 0:5.2f} |da| {r.get("dact_held") or r["dact_mean"]:.2f} '
          f'|a|>=1 {r["resid_over_frac"]:.2f} sat {r["ctrl_sat_frac"]:.3f} pinned {r.get("joints_pinned90")} shares '
          f'{r["share_thumb"] or 0:.2f}/{r["share_index"] or 0:.2f}/{r["share_middle"] or 0:.2f} grip {r["grip_N"] or 0:.0f} N '
          f'palm {r["palm_contact_frac"] if r["palm_contact_frac"] is None else round(r["palm_contact_frac"], 2)} '
          f'flags {r["flags"]} stop {r["stop_rule"]} {r["stop_reason"]}')
print(rows[it]["strip"])

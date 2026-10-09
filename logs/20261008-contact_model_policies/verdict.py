"""Append the session's reading of one watched checkpoint to watch_verdicts.jsonl (fsynced).
    python3 logs/20261008-contact_model_policies/verdict.py <tag> <iteration> <go|stop> "what the strip shows" """
import json, os, sys, time
tag, it, decision, text = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
assert decision in ("go", "stop", "retro")
p = "/home/humanoid/Programs/hand/docs/experiments/20261008-contact_model_policies/watch_verdicts.jsonl"
with open(p, "a") as fh:
    fh.write(json.dumps(dict(tag=tag, iteration=it, decision=decision, seen=text, when=time.strftime("%Y-%m-%d %H:%M"))) + "\n")
    fh.flush(); os.fsync(fh.fileno())
print("ok")

"""Training-log view of a run at its 82-iteration checkpoints, beside another run (for runs whose trainer leaves no GPU
memory for the checkpoint watch): episode length, tip-lost terminations, alignment term and return of the stochastic
policy, averaged over the 8 iterations up to each checkpoint.
    uv run --extra rl python logs/20261008-contact_model_policies/logwatch.py <tag> [<reference tag>]"""
import sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
KEYS = ("Train/mean_episode_length", "Episode_Termination/tip_lost", "Episode_Reward/target_axis_alignment",
        "Train/mean_reward")


def load(tag):
    ea = EventAccumulator(f"/home/humanoid/Programs/hand/results/rl/{tag}/tensorboard", size_guidance={"scalars": 0})
    ea.Reload()
    return {k: {e.step: e.value for e in ea.Scalars(k)} for k in KEYS if k in ea.Tags()["scalars"]}


runs = [load(t) for t in sys.argv[1:]]
last = max(runs[0]["Train/mean_episode_length"]) if runs[0].get("Train/mean_episode_length") else 0
print("it    M   " + "   ".join(f"{'eplen':>6} {'tiplost':>7} {'align':>6} {'return':>7}" for _ in runs))
for it in range(82, last + 1, 82):
    row = []
    for r in runs:
        v = [np.mean([r[k][i] for i in range(it - 7, it + 1) if i in r.get(k, {})]) if r.get(k) else np.nan for k in KEYS]
        row.append(f"{v[0]:6.1f} {v[1]:7.2f} {v[2]:6.1f} {v[3]:7.1f}")
    print(f"{it:4d} {(it + 1) * 49152 / 1e6:4.1f}  " + "   ".join(row))

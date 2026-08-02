# Getting a MorphoHand to repose the UHAS cube

**Goal (user, 2026-08-01):** see one of our reconfigured hands manipulate the cube *as advertised
by UHAS*; then several hands doing it, rendered, with comparable reorientation-quality metrics —
so the morphology search finally has a downstream task-level evaluator.

This file is the plan and the running result log. `INTEGRATION.md` stays the record of how the
integration works; this is the record of the experiment.

---

## The motivation, stated once

Every evaluator the morphology program has used so far has been **our own trained policy**, and
memory `project_morph_pipeline_sweep` records where that ended: the per-design policy DRAW
dominates the design signal (per-draw reorient sd 0.3–0.5, gate-invisible), so a design search
over that evaluator mostly measures seeds. `H06_04` is the one confirmed win in the whole program
and it cost n=4 replications to establish.

A **frozen, pretrained, cross-embodiment policy** is a different kind of evaluator, and its appeal
is precisely that it removes the draw: the same weights drive every hand, so a difference between
two hands cannot be a training-seed difference. That is the reason to care about UHAS here — not
that the cube task is interesting in itself, but that it is a *policy-free* way to score a
morphology. If it works, "which hand is better" becomes a measurement instead of a training run.

So the question this experiment answers is not "can our hand do UHAS's task". It is: **does UHAS
give us a zero-variance morphology evaluator, and what does it cost to make it valid?**

---

## Experiments

| # | Question | Status |
|---|---|---|
| E0 | Does the patched stack reproduce UHAS on a hand it was trained on? | **DONE** |
| E1 | Does our hand run at all under the pretrained policy? | **DONE** |
| E2 | Is the cube the wrong size for our smaller hand? | **DONE — answered no** |
| E3 | Why does our hand fail, and is the cause morphology or calibration? | in progress |
| E4 | With the cause fixed, how do our hands rank? | pending |
| E5 | Cube-size sensitivity per hand (the ratio UHAS assumes vs the one ours wants) | pending |
| E6 | Cross-hand comparison: metrics, plots, videos | pending |

---

## E0 — reference: the policy on a hand it knows (LEAP)

`All_grippers/model_14999.pt`, 64 envs, 4000 control steps, `evaluation=True`.

| metric | value |
|---|---|
| repose success rate | **51.7 %** (199 / 385 attempts) |
| drop rate | 41.0 % |
| timeout rate | 7.3 % |
| consecutive successes | mean 0.89, max 6 |
| median best-alignment on a failed attempt | 0.335 rad |
| centering (object ↔ in-hand target) | 0.415 radii |

This is the bar. It also validates the five `MORPHOHAND PATCH`es in `INTEGRATION.md` end to end —
a mis-set `vector_phis` or a mis-placed observation column could not produce 52 %.

### ⚠ A false negative that cost an hour, recorded so it is not repeated

The first LEAP run reported **0.0 % success and 85 % drop**, and every instinct said the
integration was wrong. It was not: the *evaluator* was wrong, in a way that is invisible.

`_get_rewards` sets `reset_goal_buf`, then immediately calls `_reset_target_pose`, whose last line
is `self.reset_goal_buf[env_ids] = 0`. The same happens to `successes` in `_reset_idx`. **Both of
the env's success signals are cleared inside `step()`, after the check that sets them.** Anything
reading them after `env.step()` returns sees zeros — i.e. a perfectly working policy reports zero
successes, and reports them as *drops*, because the episodes still end.

The tell was in the data and was missed at first: `best_rot_dist` on failed attempts had a
**minimum of 0.1003 rad against a success tolerance of 0.1000**. A distribution whose floor sits
exactly at the threshold is not a policy that keeps missing; it is a measurement that cannot see
below the threshold, because the goal is re-randomised in the same step it is reached.

Fixes, both marked `MORPHOHAND PATCH`: latched `mh_success_events` / `mh_success_total` /
`mh_streak_at_reset` in the env, and measuring `rot_dist` against the **pre-step** goal in
`uhas_eval.py`. Lesson, in the house style: *when a metric's extreme value lands exactly on the
threshold it is compared against, suspect the instrument.*

---

## E1 — our hand runs under the pretrained policy

`morpho_H0604`, 64 envs, 3000 steps, zero-shot.

Two blockers had to go first, and neither was a hardware or morphology problem:

1. **URDF → USD without the GUI.** `docs/add_hand.md` says to import through the Isaac Sim GUI.
   That is a property of the doc, not of Isaac: `isaaclab.sim.converters.UrdfConverter` is the
   same importer extension, drivable headless. `scripts/uhas_urdf_to_usd.py` converts all four
   built hands in one pass (convex hull, self-collision on, fixed base, fixed joints merged).

2. **UHAS's slot bookkeeping assumes #slots == max_fingers.** `INTEGRATION.md` previously
   concluded "no UHAS code change is needed for a 3-finger hand", from the fact that
   `chain_indices` merges slots. That is true of the *action* path and false of the
   *constructor*: `main_joints` is built as one entry per chain plus one repeat per extra slot,
   then reordered by `argsort(chain_order padded to 5)`. LEAP's 4 fingers claim all 5 slots so the
   list is 5 long; our 3 fingers claim 4 (slot 3 is unreachable for all of them), the list is 4
   long, and it is indexed with 4 → `IndexError` in `__init__`.

   An unclaimed slot is not an error — it means no finger drives that canonical plane, and the
   policy's 3 action dims for it should go nowhere. The patch indexes the table **by slot**,
   records `live_slots`, and writes joint targets only through claimed slots. For the shipped
   hands `live_slots` is all five and the behaviour is byte-identical.

**Result: it runs, and it fails.** 0 % success, 76.8 % drop, 23.2 % timeout.

---

## E2 — the cube is *not* mis-sized. UHAS holds the ratio fixed by construction.

This was the user's leading hypothesis and it has a clean answer. `multi_env_cfg` sizes the cube
to the **hand**, via the UHAS sphere radius:

```python
cube_scale = (7/6) * sphere_radius / 0.09119        # 0.09119 = LEAP's radius
cube_mass  = density * cube_size**3 * sphere_radius / 0.09119
```

| hand | sphere r (m) | cube edge | cube mass | **cube diagonal / sphere diameter** |
|---|---|---|---|---|
| LEAP | 0.0912 | 7.00 cm | 104 g | **0.66479** |
| H06_04 | 0.0710 | 5.45 cm | 81 g | **0.66479** |

Identical to five decimals — as it must be, since both derive from the same ratio. So "the cube is
too big for our hand" cannot be the explanation for anything: our hand gets a proportionally
smaller, lighter cube automatically.

What this does *not* settle is whether UHAS's chosen ratio (0.66) is right for a **3-finger** hand
when it was calibrated on 4- and 5-finger ones. Three fingers on a sphere leave wider gaps than
four; the cube that a 4-finger hand cages may be exactly the cube a 3-finger hand drops. That is a
real and separate question, and it is E5 — which is why `UHAS_CUBE_SCALE_MULT` exists.

---

## E3 — why it fails: the lateral command range, not the morphology

The failure is not marginal, which is itself diagnostic:

| | LEAP | H06_04 |
|---|---|---|
| mean rot-dist to goal | 1.04 rad | **2.22 rad** |
| centering | 0.415 radii | **0.823 radii** |
| median best alignment on a failed attempt | 0.335 rad | 1.259 rad |

2.22 rad is ≈ the mean angle between two *uniformly random* orientations. The cube's orientation is
statistically independent of the goal: the hand is not rotating it toward the target at all, and
it is holding it at twice LEAP's distance from the in-hand target. This is "the actuation is
wrong", not "the policy is mediocre on an unfamiliar hand".

The prime suspect is `min_offsets` / `max_offsets` in `sphere_cik.json` — the **runtime command
range of each finger's lateral joint**. The env maps a policy action `a ∈ [-1,1]` to a lateral
offset and then clamps it into `[min_offset, max_offset]` per slot:

| slot | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| LEAP min | −0.568 | −0.568 | −0.513 | −0.557 | −0.310 |
| LEAP max | +0.551 | +0.551 | +0.511 | +0.586 | +0.349 |
| **H06_04 min** | −0.606 | −0.606 | **−2.814** | 0.0 | **+1.573** |
| **H06_04 max** | +2.011 | +2.011 | **+2.814** | 0.0 | **+1.747** |

LEAP's ranges bracket zero roughly symmetrically, ±0.3–0.6 rad. Ours do not:

* **slot 4 (thumb) never contains zero.** Its range is `[+1.573, +1.747]`, so the clamp pins the
  thumb's yaw at ≈ 90° for any action below +0.90. The thumb is held permanently splayed.
* **slot 2 (middle) spans ±2.81 rad (±161°)** — a small policy action becomes a huge yaw swing.
* the per-hand span table shows the same on every hand:

  | hand | lateral spans (rad, per slot) |
  |---|---|
  | LEAP | ≈ 0.66 – 1.12 (max−min) |
  | H06_04 | 2.62, 2.62, **5.63**, 0.0, **0.17** |
  | m05 | **0.17**, 0.17, **5.48**, 0.0, 3.60 |
  | perp | 3.92, 3.92, 3.95, 0.0, 2.62 |

  A 33× spread *within one hand*. `uhas_build_hand.py`'s own docstring already says a span
  "> ~2 rad usually means a degenerate solution" — all three hands have at least one.

**This is the same root cause as the "calibration pose" correction in `INTEGRATION.md`, and it
upgrades that item from cosmetic to blocking.** That note concluded — correctly — that the *yaw
joint* is fine (abduction efficiency 0.99 at the operating pose) and that the pathological θ table
was an artifact of probing at the IK-solved maximally-flexed pose. What it did not follow through
was that the numbers produced at that bad probe pose are not merely a diagnostic readout: UHAS
**uses them at runtime as the lateral actuation limits**. A calibration measured at a useless pose
becomes a command range the policy has to drive through.

> Standing correction to `INTEGRATION.md`: "the offsets are cosmetic, the joint is fine" is half
> right. The joint is fine. The offsets are load-bearing.

### What E3 has NOT yet shown

That fixing the calibration fixes the task. The next run is the test: rebuild the CIK with a
lateral range calibrated at the hand's actual operating pose, and re-run the identical eval. If
success goes from 0 % to something, calibration was the blocker; if it stays at 0 %, the
3-finger/cube-ratio question (E5) moves to the front.

---

## Reproducing

```bash
P=/home/humanoid/miniconda3/envs/env_isaaclab/bin/python

# URDF -> USD (once per hand)
$P scripts/uhas_urdf_to_usd.py results/uhas/hands/H0604

# evaluate one hand
$P scripts/uhas_eval.py --robot morpho_H0604 --num-envs 64 --steps 3000 \
    --out results/uhas/eval/morpho_H0604.json --headless

# how far outside the training distribution does this hand put the policy?
$P scripts/uhas_obs_audit.py --robot morpho_H0604 --out results/uhas/audit/H0604.json --headless

# render
$P scripts/uhas_eval.py --robot morpho_H0604 --num-envs 4 --steps 650 --video \
    --video-dir results/uhas/video/H0604 --out /tmp/x.json --headless
```

Env-var knobs on the vendored env (all `MORPHOHAND PATCH`): `UHAS_ROBOTS`, `UHAS_EVAL`,
`UHAS_CUBE_SCALE_MULT`, `UHAS_MORPHO_ROOT`, `UHAS_MAX_CONSEC`, `UHAS_VIEW_EYE/LOOKAT/ENV`.

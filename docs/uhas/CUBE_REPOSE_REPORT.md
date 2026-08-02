# Can a frozen cross-embodiment policy score our hands?

**Report, 2026-08-02.** Four MorphoHand designs in UHAS's in-hand cube-repose task under the
released multi-hand policy. Published page: <https://claude.ai/code/artifact/5d3e125e-a83f-4c68-8621-fe01926bb8ff>

The running experimental log with every table is `docs/uhas/CUBE_REPOSE_PLAN.md`; how the
integration works is `docs/uhas/INTEGRATION.md`. This file is the summary.

---

## The result, first

**Every MorphoHand design scores 0–1 % on UHAS's repose task; LEAP, which the policy was trained
on, scores 51.7 %.** All four of our hands' 95 % confidence intervals sit inside [0 %, 3.4 %] and
overlap each other, so the evaluator currently cannot rank our designs — the one thing we wanted
it for.

Two real bugs in UHAS's calibration were found and fixed on the way, and the cube-size hypothesis
is now closed twice over. Neither was the blocker. What the work did produce is a **failure
taxonomy**: "0 %" was concealing a hand that cannot hold the cube and a hand that holds it better
than anything we have built and never turns it. Those need opposite fixes, and only the renders
plus a motion probe could tell them apart.

## Why bother with someone else's cube task

Every evaluator this programme has used has been *our own trained policy*, and that is where the
design search stalled: the per-design policy draw dominates the design signal (per-draw reorient
sd 0.3–0.5, gate-invisible), so a search over it mostly measures seeds. `H06_04` is the one
confirmed win and it cost n=4 replications.

A frozen pretrained cross-embodiment policy removes the draw — identical weights drive every hand,
so a difference between two designs cannot be a seed difference. The question was never "can our
hand do UHAS's task"; it was **"does UHAS give us a zero-variance morphology evaluator, and what
does it cost to make it valid?"**

## What we measured

64 environments, 3000 control steps, released `All_grippers` checkpoint, deterministic policy.
An *attempt* is one goal orientation ending in a repose, a drop, or a 30 s timeout.

| hand | attempts | repose | 95 % CI | drop | timeout | best align. (failed) | centering |
|---|---|---|---|---|---|---|---|
| **LEAP** (ref) | 385 | **51.69 %** | [46.70, 56.64] | 41.0 % | 7.3 % | **0.166 rad** | 0.415 |
| perp | 263 | 0.00 % | [0.00, 1.44] | 75.7 % | 24.3 % | 1.570 | **0.563** |
| H06_04 | 212 | 0.94 % | [0.26, 3.37] | 73.1 % | 25.9 % | 1.111 | 0.872 |
| m05 | 868 | 0.12 % | [0.02, 0.65] | 95.7 % | 4.1 % | 1.418 | 0.899 |
| baseline | 1193 | 0.50 % | [0.23, 1.09] | 95.2 % | 4.3 % | 1.419 | 1.056 |

Best-alignment is the metric that separates a near-miss from a hopeless design. LEAP's *failures*
reach 0.166 rad — it misses narrowly. Ours reach 1.1–1.6 rad, with a whole-run mean orientation
error of 2.1–2.2 rad, which is the mean angle between two *uniformly random* orientations. Our
hands are not almost solving the task.

## Two different failures

Filmstrips in `docs/uhas/figs/`, read before writing any of this:

* **perp — holds, never turns.** Genuinely cradles the cube; the most stable hold of anything we
  have built (centering 0.560 radii vs LEAP's 0.415). The cube's painted faces are *identical in
  every frame across 440 steps*. 24 % of attempts time out held-but-unreposed.
* **H06_04 / m05 / baseline — cannot cage it.** Fingers stay flat and the cube balances on top;
  m05 and baseline are near-flat plates with the cube rolling loose and falling off (95 % drop).

These are not degrees of one failure. perp has solved the grasp and lacks the manipulation; the
other three never get a grip to manipulate from.

## What was blocking, and why none of it was the hand

1. **The reference number nearly came out wrong.** The first LEAP run said 0.0 % / 85 % drop. The
   env clears both of its success signals *inside* `step()`, after the check that sets them
   (`_reset_target_pose` zeroes `reset_goal_buf`; `_reset_idx` zeroes `successes`), so anything
   reading them afterwards scores a working policy as total failure. The tell: best-alignment
   bottomed out at 0.1003 rad against a 0.1000 tolerance — a floor exactly on its own threshold is
   an instrument problem.
2. **URDF→USD needs no Isaac Sim GUI** (contra `add_hand.md`); `UrdfConverter` is the same importer
   headless.
3. **UHAS's constructor assumes fingers claim all five canonical slots.** Three fingers claim four
   and it indexes out of range — reads as "3-finger hands unsupported", is an unstated assumption.

## The calibration bugs (real; fixed; not the blocker)

`min_offsets`/`max_offsets` are the **runtime lateral command range** — the env clamps each slot's
action into them, so a range excluding zero pins that finger's joint for the whole episode.

* **`utils_urdf.py:2364` gates the re-anchor on `if 4 not in chain_idx`** — every finger except the
  one in slot 4 is anchored to its own reachable band; slot 4 keeps a hard-coded 1.571 rad. Our
  slot-4 error: perp +0.7°, m05 −48°, baseline +89°, H06_04 +95° — and perp, the only near-correct
  one, was the only hand that had scored.
* **The range is read at the joint-sweep endpoints** and the lookup interpolates linearly, both
  assuming θ(q) is monotonic. It is not near the sphere pole, where azimuth wraps.

Verified the strongest way available: regenerating LEAP's calibration from its own URDF and
diffing against the shipped artifact leaves **slots 0–3 bit-identical**, moving only the thumb by
2.6°. Afterwards every hand has zero pinned slots and span ratios 1.20–1.44, tighter than LEAP's
1.73 — and the task did not move (all before/after CIs overlap).

## Cube size — closed twice

UHAS sizes the cube to the hand, so cube-diagonal ÷ sphere-diameter is a **constant 0.66479 for
every hand**. Sweeping it on perp with LEAP as control: over a range where LEAP moves 53.8 % →
41.0 %, **perp is 0.00 % at every point** (0 successes / 1097 attempts, five sizes). At ratio 0.465
LEAP is at its best and perp is still exactly zero. Smaller cubes help perp *hold* (drop 75.7 →
58.4 %, converting into timeouts); holding was never the constraint.

## Why perp holds and never turns

End-to-end probe: policy action → CIK joint target → delivered joint motion → cube rotation.

| | action std | joint std | joint travel | object ang. speed |
|---|---|---|---|---|
| LEAP | 125.9 | 0.241 rad | 2.19 rad/s | 5.06 rad/s |
| perp | 82.1 | 0.059 | 0.50 | 0.51 |
| H06_04 | 100.2 | 0.065 | 0.22 | 1.33 |

**The policy is not silent** — action magnitudes within 35 % of LEAP's. What collapses is
downstream. Splitting commanded from delivered, and lateral from encompassing joints:

| | CIK commands | joints deliver |
|---|---|---|
| LEAP encompassing | **0.782 rad** | 0.248 |
| perp encompassing | **0.188** | 0.047 |
| H06_04 encompassing | **0.244** | 0.079 |

The same policy on the same canonical sphere asks our fingers for **a quarter of the flexion**. The
finger is told to curl a little and hold, not to sweep the cube around. This is the **type-B
(encompassing) lookup** — a different part of CIK from the type-A lateral calibration fixed above.

## A cheap predictor, and an honest null

The checkpoint's own observation normalizer gives an out-of-distribution score per hand in ~90 s
with no task rollout: LEAP **0.301**, ours **0.651–0.990**, the gap concentrated in fingertip
positions (0.50 vs 1.43–2.31). **As a binary indicator it works. As a ranking evaluator it is
untested and cannot be tested here** — within our hands it does not track success, but all four
successes are statistically identical, so there is no signal to track. A validated sanity check,
not a validated evaluator.

## What this means for morphology optimization

* **The evaluator exists and runs.** MJCF → scored Isaac rollout, no GUI step, no per-hand code,
  same frozen weights everywhere. The zero-variance property is real and intact.
* **It has no dynamic range on our hands yet**, so it cannot do the job. It discriminates
  LEAP-from-us, not us-from-us. Using it to rank designs now would repeat, in a new form, the
  mistake of ranking through a noisy estimate.
* **The transferable output is the taxonomy, not the rate.** "perp has solved the grasp and lacks
  the manipulation" is a statement our reward-trained evaluators never produced, and it came from
  one render plus one 90-second probe. A **grasp-quality metric is available today** and *does*
  have dynamic range across our designs (perp's centering 0.560 beats every other hand we have
  built) even while the repose rate does not.

## Open, cheapest first

1. **Type-B lookup amplitude.** Why does the same sphere deformation ask our fingers for ¼ the
   flexion? Instrument the θ→q gain in `build_lookup_dict_with_fp_and_phi` against LEAP's.
2. **Clamped-anchor variant.** Our fix anchors slot 4 at its band centre, which removes the pinned
   joint but throws away azimuthal separation (as-built spread 74–103° → 8–22°; perp unchanged at
   179°). Anchoring at the reachable θ *closest to* the canonical thumb azimuth keeps both. One line.
3. **Fine-tune rather than zero-shot**, on perp — it already holds the cube. Separates "the
   representation cannot express this hand" from "the policy has not seen it".
4. **Only then** use it to rank morphologies.

## Method

`All_grippers/model_14999.pt`, deterministic, Isaac Lab 2.1.0 / Isaac Sim 4.5.0. Task
`UHAS-Inhand-Repose`, 64 envs, 3000 steps @ 60 Hz, `evaluation=True`, success tolerance 0.1 rad,
drop threshold 0.2 m, 30 s timer reset per repose. Wilson 95 % intervals. Distances in sphere radii
so they compare across hand sizes. Produced by `scripts/uhas_eval.py`, `uhas_obs_audit.py`,
`uhas_motion_probe.py`, `uhas_cube_sweep.sh`, `uhas_cik_report.py`.

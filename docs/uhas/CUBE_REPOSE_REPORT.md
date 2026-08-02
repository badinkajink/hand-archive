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

## How the action space works — the five command slots

Worth stating plainly, because everything below depends on it and it is not in UHAS's README.

The policy emits **15 numbers, always, for every hand**: `(1 + len(vector_phis)) * max_fingers`
= `3 × 5`. A **slot** is not a finger — it is a **canonical driving plane on the sphere**, a
half-plane through the polar axis at one azimuth, numbered 0–4 in azimuthal order the way a
five-fingered hand's fingers would be (2 = the middle of the hand, 4 = the opposing thumb).
Each slot's three numbers are:

* **1 lateral** — where around the sphere that finger sits (rotates the plane about the axis),
  clamped into that slot's `[min_offsets, max_offsets]` band;
* **2 radius offsets**, at fixed polar angles φ = 50° and 120°, which define a radial profile
  r(φ) — the sphere's *deformation* in that plane, in units of sphere radii, range ±2.

CIK then curls that finger's encompassing joints until its links lie against the deformed
profile. So the policy never speaks in joints; it says "make the object-shaped surface bulge
here and shrink there", and every hand's own CIK translates.

Fingers are mapped onto slots **automatically and geometrically** by
`compute_finger_chain_order_translation`: each finger root's azimuth θ in the sphere frame is
computed, the circular-median finger takes slot 2, and the rest fan out by sorted azimuth
(smaller θ → 0, 1; larger → 4, 3). perp comes out `[[4], [2], [0, 1]]` — thumb, index, and a
middle finger that claims two slots. A finger claiming several slots is UHAS's stated way of
covering a 5-slot space with fewer fingers (during *training* the merged slots' actions are
averaged; at eval they are not).

**What that costs a 3-finger hand.** For perp: slot 3 is unclaimed, so its 3 dims are inert.
Of the middle finger's two slots, the flexion is read from `chain_idxs[0]` (slot 0) while the
lateral targets of slots 0 and 1 write to the *same* joint index, so the later write wins —
slot 1's lateral is used and slot 0's is discarded, along with slot 1's two radius dims.
**6 of 15 action dimensions (40 %) drive nothing on perp.** That is a real cost of a 3-finger
hand in this representation, and it is structural, not a bug.

**Why we touched the slots at all**, and what it did *not* buy: `min_offsets`/`max_offsets`
are the runtime lateral clamp, and `utils_urdf.py:2364` special-cases slot 4 to a hard-coded
thumb azimuth of 1.571 rad. When that is wrong the band can exclude zero and **pin** that
finger's lateral joint for the whole episode — a genuine bug, fixed (E3). The follow-up idea
(anchor at the reachable θ *closest to* the canonical azimuth, to keep the azimuthal spread)
is a one-line refinement of the same fix. But the fix itself moved no outcome — every
before/after interval overlaps — so slot work is now a correctness chore, **not** a lever.
The lever is below.

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

That number is a *variance over a rollout*, and the section below — which instruments the map
itself rather than one policy's path through it — shows it is not a uniform gain deficit.

## What actually limits our hands (2026-08-02)

The type-B command is a pure function of the action, so it can be instrumented **offline, with no
Isaac and no rollout**: reproduce the solve in numpy and sweep the whole action box
(`scripts/uhas_cik_authority.py`, seconds per design). Two things come out, and they are not the
same thing:

* **conditioning** — the solve keeps sphere points at distance ≥ *l* from the joint axis (*l* = the
  link length to the next joint) inside a ±0.1 R band, then rotates the link to the extremal
  surviving point. When nothing survives, the target **snaps to a joint limit** (`starved`). Any
  target sitting on a limit carries no information about the sphere (`pinned`).
* **authority** — how far the fingertip travels, in widths of the cube UHAS gives *that* hand
  (both the sphere and the cube are sized to the hand, so radians do not compare across designs).

| hand | R (m) | l_prox/R | starved | pinned | travel (all) | travel (live) |
|---|---|---|---|---|---|---|
| **LEAP** (ref) | 0.0912 | **0.52** | 3.2 % | **11.1 %** | **2.22** | **1.71** |
| perp | 0.0685 | 1.31 | 0.4 % | 16.7 % | 1.20 | 1.12 |
| H06_04 | 0.0710 | 1.45 | 13.5 % | 37.4 % | 2.51 | 1.15 |
| m05 | 0.0687 | 1.50 | 12.6 % | 39.9 % | 2.59 | 1.18 |
| baseline | 0.0654 | 1.38 | 9.0 % | 34.3 % | 2.33 | 1.29 |

**LEAP has both; each of ours has at most one**, and which one post-dicts the failure we rendered:

* **perp** — the best-conditioned map of anything here (0.4 % starved) and the *least* authority
  (1.12 vs LEAP's 1.71). A hand whose commands are well-posed but too small to sweep the cube is a
  hand that **holds and never turns**. That is exactly the filmstrip.
* **m05 / baseline / H06_04** — comparable travel only because a third of their commands are
  slammed against joint limits (`pinned` 34–40 %, `starved` 9–14 %). Bang-bang between limits is
  what **flailing that never cages the cube** looks like in a table.

**Why: the proximal phalanx.** UHAS sets R = 0.509 × mean palm-to-fingertip reach, so what matters
is the first link *as a fraction of total reach*. LEAP spreads 14 cm of finger over three links
(~53/36/50 mm) → l/R ≈ 0.52. Ours puts 9 cm into one proximal link against a 3 cm distal → l/R
1.3–1.5, **2.5–3× LEAP's**. A long first link admits only a thin outer crescent of the sphere, and
that crescent empties as the sphere shrinks — which is the starvation.

### The knobs, measured (perp, real MJCF → URDF → CIK rebuild for each)

| variant | R (m) | l_prox/R | starved | pinned | travel (live) |
|---|---|---|---|---|---|
| perp as built | 0.0685 | 1.31 | 0.4 % | 16.7 % | 1.12 |
| proximal 90 → 45 mm | 0.0476 | 0.95 | 0.1 % | 12.5 % | 0.79 |
| proximal 90 → 32 mm | 0.0417 | 0.76 | 0.0 % | 9.0 % | 0.73 |
| distal 30 → 50 mm | 0.0740 | 1.22 | 0.7 % | 20.6 % | 1.18 |
| distal 30 → 70 mm | 0.0804 | 1.12 | 1.0 % | 23.3 % | 1.16 |
| **fingers in (mount limit)** | 0.0596 | 1.51 | 22.0 % | 48.6 % | **1.72** |
| proximal 45 + fingers in | 0.0372 | 1.21 | 2.2 % | 30.2 % | 1.36 |
| sphere R × 0.75 (`sphere_factor`) | 0.0514 | 1.75 | 3.1 % | 27.5 % | 1.46 |
| sphere R × 1.25 | 0.0856 | 1.05 | 0.1 % | 18.6 % | 0.77 |
| *LEAP, for scale* | 0.0912 | 0.52 | 3.2 % | 11.1 % | 1.71 |

Reading it:

1. **Finger spacing is the dominant lever, not phalanx length.** Moving perp's three mounts to the
   inner limit of their workspace is the only change that reaches LEAP's authority (1.72 vs 1.71) —
   a 54 % gain. It also wrecks the conditioning (pinned 49 %, starved 22 %) and, at the extreme,
   flattens the thumb's lateral span to 0.002 rad. So: go inward, but **not to the stop** — there is
   an interior optimum and it is now searchable offline.
2. **Shortening the proximal alone trades the wrong way.** It fixes conditioning (pinned 17 → 9 %)
   but *costs* authority (1.12 → 0.73), because R follows reach: a shorter finger shrinks the sphere
   and the cube with it. It is a **conditioner to pair with spacing**, not a fix on its own.
3. **Distal length barely moves anything** (1.12 → 1.18) and mildly worsens pinning.
4. **The sphere radius is a free knob nobody has swept.** `sphere_factor` (4/2.5π) is a constant in
   `construct_sphere_with_palm`, not a property of the hand. Shrinking it 25 % buys +30 % authority
   on the unmodified perp. Note this is *not* the cube sweep we already ran: that changed the
   object; this changes the action representation's scale. Untested against the task.

**Status of this metric: post-diction on five hands, not validation.** It has dynamic range where
the repose rate has none, and it explains the taxonomy we rendered — but it has never been checked
against an outcome, because no design of ours has produced an outcome yet. Treat it as the design
filter to *use before* spending Isaac time, and confirm it with the first design that clears both
axes.

## The design box, searched (2026-08-02)

The section above said spacing was the dominant lever and the optimum was interior. That was a
claim about a landscape nobody had sampled. **980 designs later, each a real MJCF → URDF →
UHAS-sphere → CIK rebuild** (`scripts/uhas_mount_search.py`, ~5 s each, no GPU), the honest
verdict is a clean negative with a structural explanation.

Two facts fell out of setting the search up. **The metric is translation-invariant** — UHAS fits
its sphere to the *finger roots*, so the palm plate never enters and sliding all three mounts
together is an exact no-op. The searchable axes are therefore the two *separations*, thumb↔pair
in x and index↔middle in y, plus phalanx scales. And **the cube does not spawn on the palm**:
`init_pos` is in sphere-radius units and the sphere is tangent to the fitted palm plane, so every
hand gets the cube at 1.3 R off the palm — proportionally identical for LEAP and for us. Cube
flat on the palm in our renders is a **catch failure**, not a different spawn.

### LEAP dominates the whole front

| | authority = travel_live × (1 − pinned) |
|---|---|
| whole box, min → median → max | 0.40 → 0.78 → **1.10** |
| shipped perp | 0.93 |
| best **buildable** (see floor, below) | 1.09 |
| LEAP | **1.52** |

Not one design is even non-dominated with respect to LEAP. 265 of 976 match or beat its
conditioning (pinned ≤ 15 %); none of those clears travel 1.4. At LEAP's pinned fraction the box
delivers about 65 % of LEAP's fingertip travel.

The `l_prox/R` hypothesis is directly falsified as a *lever*. The box does reach **l/R 0.62**
(LEAP 0.52) — and that design beats LEAP's conditioning (7.9 % pinned) at **43 % of its travel**.
Matching LEAP's proportions is achievable and does not buy LEAP's authority.

### The perp hand already ships on its collision floor

`morph_selfcollision_gate.py --sep-sweep [--prox S]` settles each pair separation at the **open**
pose. perp's fingers face each other, so the binding contact is index↔middle, not the palm:

| proximal | **90 mm (shipped)** | 76 | 63 | 50 | 36 |
|---|---|---|---|---|---|
| minimum feasible separation | **96 mm (shipped)** | 85 | 75 | 65 | 55 |
| best authority at that floor | 1.08 | **1.09** | 1.01 | 1.01 | 0.98 |

The floor tracks proximal length ≈ 0.77:1, which makes "shorten the proximal" and "close the
fingers" **one proposal** — the first is the enabler for the second. But the floor costs nothing
(1.09 buildable vs 1.10 free) **because the metric never wanted the pair closer**. The best
mount-only change moves the thumb *in* (gap 100 → 85 mm) and the pair slightly *out* (96 → 105 mm)
for +5 %.

### What no design in the box can buy

| | joints per finger | link lengths (mm) |
|---|---|---|
| LEAP | 4: D, A, B, B | 53.1 / 15.0 / 36.1 / 50.0 |
| perp | 3: A, B, B | 84.9 / 90.0 / 30.0 |

**Three CIK-solvable joints across four segments against two across three.** Two flexion joints
put the fingertip on a curve; three put it on an area. No mount position and no phalanx scale adds
a joint — which is why the front is dominated everywhere rather than in one corner, and why
spacing and length both plateau around 1.0–1.1. The lever this program has not pulled is a
**third phalanx**.

### The counter-argument that survives

`authority` is fingers-only. **Palm contact is invisible to it**, and a three-finger hand can use
the palm as a fourth contact to gait — a person with three fingers does exactly that, and this
hand has more independent actuation and range of motion than those three fingers. Nothing above
predicts or rules that out. It is the live counter-argument to this negative, it needs sim rather
than kinematics to settle, and because the frozen policy was trained mostly on four- and
five-fingered hands there is no particular reason it already knows the trick — which makes it a
**finetune** hypothesis, not a zero-shot one.

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

## Open, cheapest first (revised 2026-08-02, after the search)

1. ~~Type-B lookup amplitude~~ and ~~search the mount box~~ — **both done**; see "The design box,
   searched". The box is exhausted: 980 designs, LEAP dominates all of them, and the shipped perp
   already sits on its pair-separation floor.
2. **Zero-shot a spread of the top buildable designs.** Not to pick a winner — the spread is
   1.0–1.1 against LEAP's 1.52 — but because `authority` has never been checked against an
   outcome. Judge on **best alignment reached** and **cube angular speed**; the repose rate is
   floored at 0 % and cannot rank anything.
3. **Test the palm-gaiting hypothesis, which requires a finetune.** It is the one live argument
   that a three-finger hand can repose the cube despite the authority deficit, and no kinematic
   score can see it. UHAS's own `MANO_to_LEAP` transfer is 1000 iterations (≈1.5 h at 8192 envs on
   their hardware) against 15000 from scratch. A finetune can teach the policy to stay in a hand's
   live action region, to cope with our contact model, and possibly to brace against the palm; it
   **cannot** create fingertip travel the morphology and the CIK do not provide.
4. **The third phalanx.** The only change in evidence that would move authority rather than trade
   it. Outside every parameterisation this program has used, and a real hardware change.
5. **`sphere_factor`** — a constant in `construct_sphere_with_palm`, not a hand property, never
   swept; ×0.75 buys +30 % on the unmodified perp. Free, but it shrinks the cube with the sphere,
   so it changes the task as well as the representation.
6. **Clamped-anchor variant** — still correct, still one line, still demoted: the E3 anchorfix
   moved no outcome.

## Renders

Every hand has a 10 s rollout video under the eval policy, not only the filmstrips:
`results/uhas/video/{leap_right_final, morpho_perp_open_anchorfix, morpho_H0604_anchorfix,
morpho_m05_final, morpho_mh_baseline_final}/*.mp4` — gitignored, hence easy to miss. Filmstrips
are `docs/uhas/figs/filmstrip_*.png`.

## Method

`All_grippers/model_14999.pt`, deterministic, Isaac Lab 2.1.0 / Isaac Sim 4.5.0. Task
`UHAS-Inhand-Repose`, 64 envs, 3000 steps @ 60 Hz, `evaluation=True`, success tolerance 0.1 rad,
drop threshold 0.2 m, 30 s timer reset per repose. Wilson 95 % intervals. Distances in sphere radii
so they compare across hand sizes. Produced by `scripts/uhas_eval.py`, `uhas_obs_audit.py`,
`uhas_motion_probe.py`, `uhas_cube_sweep.sh`, `uhas_cik_report.py`.

The offline authority/conditioning numbers come from `scripts/uhas_cik_authority.py` (numpy only,
no Isaac): a 21×21 sweep of each slot's two radius actions over 1000 Fibonacci sphere samples,
reproducing `multi_manipulation_env._pre_physics_step` + `torch_solve_for_B_joint`. Lateral joints
are held at `q_sphere`, which is what the CIK cascade itself uses when advancing through a type-A
joint, so that is not an approximation. Variants were rebuilt end to end
(`uhas_build_hand.py` → `process_urdf.py`); the `sphere_factor` rows use a patched copy of
`construct_sphere_with_palm`. Rows in `docs/experiments/uhas_cik_authority.json`.

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
| E0 | Does the patched stack reproduce UHAS on a hand it was trained on? | **DONE** — LEAP 51.7 % |
| E1 | Does our hand run at all under the pretrained policy? | **DONE** — yes, all four |
| E2 | Is the cube the wrong size for our smaller hand? | **DONE — no**, ratio is constant by construction |
| E3 | Why does it fail — morphology or calibration? | **DONE** — two real calibration bugs, found and fixed |
| E4 | With the calibration fixed, how do our hands rank? | **DONE — clean negative**, all still 0–1 % |
| E4b | What *kind* of failure is it? | **DONE** — two kinds: grasp vs manipulation |
| E4c | Does the cheap OOD score predict task success? | **DONE** — separates LEAP from ours, no resolution within ours |
| E5 | Cube-size sensitivity (does a 3-finger hand want a different ratio?) | in progress |
| E6 | Cross-hand comparison: metrics, plots, videos | follow-up 2 |

**Headline after session 2: the calibration was genuinely broken and is genuinely fixed, and it
was not the blocker.** Our hands remain at 0–1 % against LEAP's 51.7 %. The useful result is not
the rate, it is that the single number "0 %" was hiding two unrelated failures — three hands that
cannot cage the cube at all, and one (perp) that cages it better than anything else we have built
and then never turns it.

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

**Result: all four hands run, and all four fail.** 64 envs, 3000 control steps, identical protocol.

| hand | attempts | success | drop | timeout | mean rot-dist | centering |
|---|---|---|---|---|---|---|
| **LEAP** (reference) | 385 | **51.7 %** | 41.0 % | 7.3 % | 1.04 rad | 0.415 |
| perp | 249 | 0.8 % | 73.5 % | 25.7 % | 2.12 rad | 0.560 |
| H06_04 | 220 | 0.0 % | 76.8 % | 23.2 % | 2.22 rad | 0.823 |
| m05 | 927 | 0.0 % | 94.9 % | 5.1 % | 2.21 rad | 0.798 |
| baseline | 2624 | 0.4 % | 98.0 % | 1.6 % | 2.20 rad | 0.926 |

Read the attempt counts as a hold-time proxy: the baseline racks up 2624 attempts in the same
wall budget because it drops almost immediately and restarts, where perp manages 249 longer ones.

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

### The picture, which is where this became obvious

`docs/uhas/figs/filmstrip_H0604_asbuilt.png` vs `filmstrip_leap.png`. LEAP curls three fingers
around the cube and turns it. **H06_04 holds all three fingers splayed flat and pointing outward,
with the cube resting on top of them like a plate** — no opposition anywhere in the rollout. That
is precisely what a yaw clamped to +90° looks like, and it is not something the metric table says.

### A consistency check that the calibration story passes

If the lateral ranges are the blocker, the hand with the least pathological ranges should score
best. It does:

| hand | lateral spans (rad) | max/min ratio | success | drop |
|---|---|---|---|---|
| perp | 3.92, 3.92, 3.95, —, 2.62 | **1.5×** | **0.8 %** | 73.5 % |
| H06_04 | 2.62, 2.62, 5.63, —, 0.17 | 33× | 0.0 % | 76.8 % |
| m05 | 0.17, 0.17, 5.48, —, 3.60 | 32× | 0.0 % | 94.9 % |

perp — whose three live slots are within 1.5× of each other, the most LEAP-like of any of our
hands — is the only one that ever reaches a goal, and holds the cube nearest the target (0.560
radii vs 0.80–0.93). m05, with a 32× spread and two slots pinned at 0.17, drops 95 % of the time.
This is correlational across four hands, not proof; it is the reason the calibration fix is the
next experiment rather than one of several.

---

## E3b — the two bugs, found and fixed

Both are in `build_lookup_dict_with_fp_and_phi` (`process_urdf/utils_urdf.py`), and neither is a
property of our hand. Both are marked `MORPHOHAND PATCH` and both have an env-var escape hatch.

### Bug 1 — the slot-4 finger is exempted from the re-anchor (`utils_urdf.py:2364`)

```python
# Place valid closest to open palm position #!
if 4 not in chain_idx:              # <-- the whole bug is this gate
    mean_q = (max_q + min_q) / 2
    ...
    anchor = valid_theta[idx_anchor]   # re-anchor to the middle of the REACHABLE theta band
```

Every finger **except** the one occupying canonical slot 4 has its plane anchor moved to the
middle of the θ band it can actually reach. The slot-4 finger keeps the hard-coded
`--thumb_anchor` (1.571 rad). For LEAP that is harmless — its thumb genuinely sits at θ ≈ π/2,
by construction of the sphere. For anything else it is silently wrong, and the consequence is not
cosmetic: `min_offsets`/`max_offsets` are θ *minus the anchor*, so an anchor outside the reachable
band makes both offsets the same sign, zero falls outside the clamp, and that finger's lateral
joint is pinned at a large fixed angle for the entire episode.

The measured anchor error is a near-perfect predictor of the as-built scores:

| hand | slot-4 band centre | assumed anchor | error | as-built success |
|---|---|---|---|---|
| perp | 1.583 rad | 1.571 | **+0.7°** | **0.8 %** |
| m05 | 0.728 | 1.571 | −48° | 0.0 % |
| baseline | 3.127 | 1.571 | +89° | 0.4 % |
| H06_04 | 3.231 | 1.571 | **+95°** | 0.0 % |

perp is the only hand whose thumb happens to sit where UHAS assumed, and perp was the only hand
to reach a goal. **Fix: apply the re-anchor uniformly** (`UHAS_UNIFORM_ANCHOR=0` restores
upstream).

### Bug 2 — the range is read at the sweep endpoints, but θ(q) is not monotonic

```python
min_idx = np.argmin(valid_q_main_joint)   # first sample of the q sweep
max_idx = np.argmax(valid_q_main_joint)   # last sample
min_offset = theta_offsets[min_idx]
max_offset = theta_offsets[max_idx]
...
q_list[:zero_idx+1] = np.linspace(q_min, zero_q, ...)   # and then interpolates q LINEARLY
```

The endpoints are only the extremes of θ if θ(q) is monotonic, and the linear `q_list` build
assumes the same thing. It holds for the shipped hands. It fails for a finger whose fingertip
passes near the **sphere pole**, where the azimuth wraps: θ(q) doubles back, both sweep endpoints
land on the far side of the branch cut, and the recorded range comes out narrow and entirely
one-signed — m05 slots 0/1 `[-2.887, -2.718]`, H06_04 slot 4 `[-2.948, -2.774]` even after Bug 1
was fixed. This is the "index/middle cross the sphere pole where azimuth wraps" effect
`INTEGRATION.md` already noted, but its consequence was a pinned joint, not a cosmetic number.

**Fix: restrict the calibration to the contiguous monotonic branch of θ(q) around the anchor**
(breaking on a sign change *or* a step > π/2, which is what a wrap looks like). Inside that window
θ ↔ q is single-branch and invertible, which is what the lookup needs. It costs nominal range on
the degenerate fingers, honestly so — that range was an artifact of reading across a branch cut.
`UHAS_MONOTONIC_BRANCH=0` restores upstream.

### Validation: on LEAP the patches do almost nothing

Regenerating LEAP's `sphere_cik.json` from its own URDF with both patches, against the shipped
artifact the policy was trained with:

| | slot 0 | 1 | 2 | 3 | 4 (thumb) |
|---|---|---|---|---|---|
| Δ min_offset | 0.0000 | 0.0000 | 0.0000 | −0.0000 | +0.0871 |
| Δ max_offset | −0.0000 | −0.0000 | 0.0000 | 0.0000 | −0.0449 |
| span shipped → regen | 1.1197 → 1.1197 | ident. | ident. | ident. | 0.660 → 0.528 |

`chain_indices` identical, sphere radius identical to 1e-16. **Slots 0–3 are bit-identical**; only
the thumb moves, by a 2.6° anchor shift and a modest span narrowing. That is the strongest
available evidence that the patches do exactly one thing and do not quietly redefine the action
space: on the hand where the assumptions already held, almost nothing changes.

LEAP itself keeps its **shipped** file for every evaluation here — the policy was trained with it,
so it is ground truth for LEAP and the 51.7 % reference stays comparable.

### The calibration afterwards

`scripts/uhas_cik_report.py --compare` (`!` = range excludes 0, i.e. joint pinned):

| hand | span ratio before → after | pinned before → after |
|---|---|---|
| LEAP (ref) | — 1.73 | — 0 |
| baseline | 29.2 → **1.25** | 1 → **0** |
| m05 | 32.5 → **1.20** | 2 → **0** |
| H06_04 | 32.4 → **1.44** | 1 → **0** |
| perp | 1.51 → **1.30** | 0 → 0 |

Every hand now has all live slots bracketing zero, with slot-to-slot consistency **better than
LEAP's**. The ranges remain wider than LEAP's in absolute terms (±1.0–1.7 rad vs ±0.3–0.6) — that
is a real geometric property of our yaw, not a defect: our fingers have more azimuthal authority
per radian of joint travel.

---

## E4 — the calibration fix does NOT recover the task. Clean negative.

Identical protocol to E1 (64 envs, 3000 control steps), the only change being the regenerated
`sphere_cik.json`. Wilson 95 % intervals, because at these rates the point estimates are 0–6
successes and reading them without intervals would invent a ranking that is not there.

| hand | | attempts | successes | rate | 95 % CI |
|---|---|---|---|---|---|
| **LEAP** (ref) | shipped | 385 | 199 | **51.7 %** | [46.7, 56.6] |
| H06_04 | as-built | 220 | 0 | 0.00 % | [0.00, 1.72] |
| H06_04 | **anchorfix** | 212 | 2 | 0.94 % | [0.26, 3.37] |
| m05 | as-built | 927 | 0 | 0.00 % | [0.00, 0.41] |
| m05 | **anchorfix** | 868 | 1 | 0.12 % | [0.02, 0.65] |
| baseline | as-built | 2624 | 10 | 0.38 % | [0.21, 0.70] |
| baseline | **anchorfix** | 1193 | 6 | 0.50 % | [0.23, 1.09] |
| perp | as-built | 249 | 2 | 0.80 % | [0.22, 2.88] |
| perp | **anchorfix** | 263 | 0 | 0.00 % | [0.00, 1.44] |

**Every one of our hands, before and after, lies inside [0 %, 3.4 %]. LEAP is [46.7 %, 56.6 %].**
Every before/after pair overlaps heavily — including perp's apparent 0.8 → 0.0 "regression", which
is 2 successes versus 0 and means nothing. There is no ranking among our four hands to report.

So: the calibration bugs were real, correctly diagnosed, and are now correctly fixed — the
calibration table in E3b is the evidence, and it is a genuine improvement in the representation.
**They were not what stands between our hands and this task.** Recording that plainly rather than
hunting for a third calibration variant.

### What the failure actually is — two different failures, and only one is interesting

Median best alignment reached *within a failed attempt* (success needs ≤ 0.10 rad):

| hand | median best alignment | drop % | timeout % |
|---|---|---|---|
| **LEAP** | **0.166 rad** | 41.0 | 7.3 |
| H06_04 | 1.111 | 73.1 | 25.9 |
| m05 | 1.418 | 95.7 | 4.1 |
| baseline | 1.419 | 95.2 | 4.3 |
| perp | 1.570 | 75.7 | 24.3 |

LEAP's *failures* get to 0.166 rad — it misses narrowly. Ours never get near: 1.1–1.6 rad, with a
whole-run mean rot-dist of 2.1–2.2 rad, which is the mean angle between two uniformly random
orientations. **Our hands are not near-missing the goal; the cube's orientation is statistically
independent of it.**

The renders (`docs/uhas/figs/filmstrip_morpho_{perp_open,H0604}_anchorfix.png`) separate two
distinct failures that the aggregate rate hides completely:

* **baseline / m05 / H06_04 — a GRASP failure.** 73–96 % drop. The fingers stay flat and the cube
  balances on top of them rather than being caged. H06_04 tumbles the cube a little in the first
  ~100 steps, then settles and stops. It never gets a grip to manipulate from.
* **perp — a MANIPULATION failure, and this is the interesting one.** With the fixed calibration
  perp genuinely *cradles* the cube: three fingers wrapped around it, the most stable hold of any
  hand (centering 0.560 radii vs LEAP's 0.415, against 0.87–1.06 for the others). And then the
  cube's painted faces are **identical in every frame across 440 steps** — X up, R right, never
  moving. It holds beautifully and never rotates. 24 % of its attempts end in timeout, i.e. held
  for the full 30 s without ever reaching a goal.

That distinction is the useful output of this session. "0 % success" was one number covering a
hand that cannot hold the cube and a hand that holds it perfectly and won't turn it, and those
have nothing to do with each other.

### ⚠ A caveat on my own fix, recorded because it nearly became a false conclusion

The uniform re-anchor moves each finger's plane to the middle of its reachable θ band. Measuring
the fingertip azimuths on the sphere before and after shows this **collapses the azimuthal spread**
of our hands:

| hand | azimuthal spread, as-built | after anchorfix |
|---|---|---|
| LEAP | 123.9° | — (kept shipped) |
| baseline | 74.0° | **8.2°** |
| m05 | 103.4° | **12.9°** |
| H06_04 | 86.8° | **22.3°** |
| perp | 178.9° | 178.9° (unchanged) |

I nearly wrote this up as "our co-designed hands have a degenerate azimuthal layout, which is why
they fail". They do not — as-built they are at 74–103° against LEAP's 124°, which is not
pathological. The collapse is **caused by my own patch**: anchoring each finger at its band centre
throws away the azimuthal separation that the hard-coded `thumb_anchor` was, accidentally,
providing. perp is unchanged because its thumb already sat at the assumed anchor.

So the fix trades one defect for another: it removes the pinned joints (real gain) and costs
azimuthal separation (real loss), and the net on the task is zero. **The obvious third variant —
anchor the slot-4 finger at the reachable θ *closest to* the canonical thumb azimuth, i.e. clamp
1.571 into the band instead of ignoring it — would keep both properties.** It is one line. I did
not run it, because the instruction for this session was to stop pushing on calibration once it
failed to recover success, and because with perp already holding the cube perfectly and still not
rotating, azimuthal separation is not obviously the binding constraint. Recorded as the top
candidate if calibration is revisited.

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

---

## E4c — the observation-distribution audit as a cheap morphology evaluator

`scripts/uhas_obs_audit.py` compares a hand's live observations against the checkpoint's own
`obs_norm_state_dict` — the per-dimension mean/var over 1.0e8 training samples on four hands — and
reports mean |z| per feature group. It costs ~90 s per hand and needs no task rollout.

| hand | OOD mean \|z\| | p90 | fingertip pos | fingertip linvel | object pos | task success |
|---|---|---|---|---|---|---|
| **LEAP** | **0.301** | 0.784 | 0.503 | 0.022 | 1.007 | 51.7 % |
| m05 | 0.651 | 1.523 | 1.461 | 0.092 | 0.954 | 0.12 % |
| baseline | 0.734 | 1.878 | 1.427 | 0.249 | 1.021 | 0.50 % |
| perp | 0.929 | 2.905 | 2.301 | 0.089 | 0.772 | 0.00 % |
| H06_04 | 0.990 | 2.898 | 2.307 | 0.245 | 1.451 | 0.94 % |

Two honest readings, and the second matters more than the first:

* **As a binary indicator it works.** LEAP sits at 0.30 and every one of our hands is 2–3× that.
  The gap is concentrated in **fingertip positions** (0.50 for LEAP vs 1.43–2.31 for ours) — the
  policy's picture of our hands is dominated by fingertip coordinates it has never seen. Object
  position sits near 1.0 for everyone including LEAP, so that column is a baseline offset of the
  metric rather than a property of any hand.
* **As a ranking evaluator it is untested, and cannot be tested here.** Within our four hands the
  OOD score does not track success — but all four successes are statistically identical (E4), so
  there is no signal to track. This says nothing about whether the score would rank hands that
  actually differ. Do not report it as a validated morphology evaluator; it is a validated
  *sanity check* and a candidate evaluator.

---

## E5 — cube size: swept, with a control. It is not the explanation.

`scripts/uhas_cube_sweep.sh`. `UHAS_CUBE_SCALE_MULT` multiplies the auto ratio; it now scales the
cube's **mass** linearly as well as its geometry, matching upstream's own mass law. (Before that
fix a 0.55× cube kept its full 78 g and became ~6× denser — the sweep would have measured density,
not size. Caught by reading the geometry block of the first result.)

Subject: **perp**, because it is the hand that holds the cube perfectly and never turns it, so
"the cube is too big for three fingers to gait around" is a live hypothesis for it specifically.
Control: **LEAP**, so any effect can be attributed to finger count rather than to cube size.

| hand | mult | edge | mass | cube/sphere | attempts | success | 95 % CI | drop % | timeout % |
|---|---|---|---|---|---|---|---|---|---|
| LEAP | 0.70 | 4.90 cm | 72.8 g | 0.465 | 210 | **53.8 %** | [47.1, 60.4] | 32.9 | 13.3 |
| LEAP | 1.00 | 7.00 cm | 104.0 g | 0.665 | 385 | **51.7 %** | [46.7, 56.6] | 41.0 | 7.3 |
| LEAP | 1.15 | 8.05 cm | 119.6 g | 0.765 | 290 | **41.0 %** | [35.5, 46.8] | 52.4 | 6.6 |
| perp | 0.55 | 2.89 cm | 43.0 g | 0.366 | 154 | **0.00 %** | [0.00, 2.43] | 58.4 | 41.6 |
| perp | 0.70 | 3.68 cm | 54.7 g | 0.465 | 199 | **0.00 %** | [0.00, 1.89] | 67.8 | 32.2 |
| perp | 0.85 | 4.47 cm | 66.4 g | 0.565 | 241 | **0.00 %** | [0.00, 1.57] | 73.4 | 26.6 |
| perp | 1.00 | 5.26 cm | 78.1 g | 0.665 | 263 | **0.00 %** | [0.00, 1.44] | 75.7 | 24.3 |
| perp | 1.15 | 6.05 cm | 89.8 g | 0.765 | 240 | **0.00 %** | [0.00, 1.58] | 73.3 | 26.7 |

**Over a range where LEAP moves 53.8 % → 41.0 %, perp is 0.00 % at every point** — 0 successes in
1097 attempts across five cube sizes. The hypothesis is dead, and the control is what kills it: at
cube/sphere = 0.465 LEAP is at its *best* (53.8 %) and perp is still exactly zero, so nothing
about that ratio is unfavourable per se.

What the cube size *does* change for perp is the failure mode, cleanly and monotonically:

| perp cube mult | 0.55 | 0.70 | 0.85 | 1.00 | 1.15 |
|---|---|---|---|---|---|
| drop % | **58.4** | 67.8 | 73.4 | 75.7 | 73.3 |
| timeout % | **41.6** | 32.2 | 26.6 | 24.3 | 26.7 |

A smaller cube is easier to keep hold of, so drops convert into timeouts one-for-one. It buys
*holding*, and holding was never the binding constraint. **Rotation does not respond to cube size
at all.**

This closes E2/E5 together: the cube is neither mis-sized by construction (E2 — the ratio is a
constant 0.66479 for every hand) nor mis-sized in effect (E5 — sweeping it changes nothing that
matters). The gap is not about the object.

---

## E4d — why perp holds and never turns: the CIK commands motion 3–4× too small

`scripts/uhas_motion_probe.py`. "Never rotates" has two causes needing opposite fixes — the policy
is effectively silent on this hand, or the fingers move plenty and the motion does no work on the
cube — and no task metric separates them. So measure the chain end to end: policy action → CIK
joint target → delivered joint motion → object rotation.

| | action std | joint std | joint travel | object ang. speed |
|---|---|---|---|---|
| **LEAP** | 125.9 | **0.241 rad** | **2.19 rad/s** | **5.06 rad/s** |
| perp | 82.1 | 0.059 rad | 0.50 rad/s | 0.51 rad/s |
| H06_04 | 100.2 | 0.065 rad | 0.22 rad/s | 1.33 rad/s |

**The policy is not silent.** It emits action magnitudes on our hands within 35 % of what it emits
on LEAP. What collapses is everything downstream: the joints move **4× less**, and the cube
rotates **4–10× slower**.

Splitting the commanded target from the delivered motion, and lateral (type-A) from encompassing
(type-B) joints, locates it precisely:

| hand | | CIK commands (target std) | joints deliver | tracking |
|---|---|---|---|---|
| **LEAP** | lateral | 0.546 rad | 0.218 | 0.40 |
| | encompassing | **0.782 rad** | 0.248 | 0.32 |
| perp | lateral | 0.383 | 0.083 | 0.22 |
| | encompassing | **0.188** | 0.047 | 0.25 |
| H06_04 | lateral | 0.170 | 0.035 | 0.21 |
| | encompassing | **0.244** | 0.079 | 0.32 |

Two deficits, one of them dominant:

1. **The CIK asks our hands for far less flexion.** LEAP's encompassing joints are commanded
   0.78 rad of excursion; ours are commanded **0.19–0.24 rad, 3–4× less**, from the same policy
   acting on the same canonical sphere. The finger is being told to curl a little and hold, not to
   sweep the cube around.
2. Tracking is also worse (0.21–0.32 of commanded vs LEAP's 0.32–0.40), but that is a secondary
   factor and partly a consequence of the small commands.

So the mechanism is: **the sphere-deformation → joint mapping produces joint excursions on our
hands that are too small to roll the cube.** perp grips it correctly and then makes small wiggles.

That is a different part of CIK from the one fixed in E3b. The calibration bugs were in the
**type-A (lateral)** lookup — anchor placement and θ branch. This is the **type-B (encompassing)**
lookup, `q_list` built from the θ sweep in the same function. **That is the next lever, and it is
named precisely rather than guessed at.**

---

## Where this leaves the programme

The question this was meant to answer (see *Motivation*) was whether a frozen cross-embodiment
policy gives us a **morphology evaluator with no training-seed draw**. Honest status:

* **The evaluator exists and runs.** Any hand we can export goes from MJCF to a scored Isaac
  rollout with no GUI step and no per-hand code, and the same frozen weights drive all of them.
* **It has no dynamic range on our hands yet.** All four score 0–1 % with overlapping intervals,
  so it currently cannot rank our designs — the thing it was wanted for. It discriminates
  LEAP-from-us, not us-from-us.
* **Two real defects were found and fixed on the way** (E3b), and a third is now precisely
  located (E4d). None of them was the cube, which was the leading hypothesis going in and is now
  closed twice over (E2 by construction, E5 by a swept control).
* **The most useful thing produced is the failure taxonomy.** "0 %" was concealing a hand that
  cannot hold the cube and a hand that holds it better than anything we have built and won't turn
  it. Those need different fixes, and only the render and the motion probe could tell them apart.

Ranked next steps, cheapest first:

1. **Type-B lookup amplitude** (E4d). Why does the same sphere deformation ask our fingers for
   1/4 the flexion? Instrument `build_lookup_dict_with_fp_and_phi`'s `q_list` for one of our
   fingers against LEAP's and compare the θ→q gain directly.
2. **The clamped-anchor variant** (E4 caveat): anchor the slot-4 finger at the reachable θ
   *closest to* the canonical thumb azimuth, keeping both the unpinned joint and the azimuthal
   separation my uniform re-anchor threw away. One line.
3. **Fine-tune rather than zero-shot.** UHAS's whole claim is a shared action space; nothing says
   a hand 3 σ outside the training distribution must work zero-shot. A short finetune on perp
   would separate "the representation cannot express this hand" from "the policy has not seen it".
4. Only then: use it to rank morphologies.

---

## E6 — the type-B amplitude, instrumented (2026-08-02)

Item 1 above, done, and it does not say what E4d guessed. Prompted by the user's observation
that on LEAP the cube rests **against the phalanges** while on ours it rests **flat on the palm**:
our fingers must travel a long way before they even reach the object.

**Method.** The type-B command is a pure function of the action, so it can be instrumented with
no Isaac and no rollout. `scripts/uhas_cik_authority.py` reproduces
`_pre_physics_step` + `torch_solve_for_B_joint` in numpy and sweeps the whole 21×21 action box for
each slot. (Lateral joints are held at `q_sphere`, which is what the CIK cascade itself uses when
it advances through a type-A joint — not an approximation.) Two quantities:

* `starved` — no sphere point survives the solve's filters (|xy| ≥ link length *l*, |z| ≤ 0.1 R),
  so the target **snaps to a joint limit** instead of encoding the sphere.
* `pinned` — the target sits on a limit at all (starved or clipped): zero information, zero variance.
* `travel` — Σ_joints sd(q) × moment arm, in widths of the cube UHAS gives that hand.

| hand | R (m) | l_prox/R | starved | pinned | travel (all) | travel (live) |
|---|---|---|---|---|---|---|
| LEAP | 0.0912 | 0.52 | 3.2 % | 11.1 % | 2.22 | 1.71 |
| perp | 0.0685 | 1.31 | 0.4 % | 16.7 % | 1.20 | 1.12 |
| H06_04 | 0.0710 | 1.45 | 13.5 % | 37.4 % | 2.51 | 1.15 |
| m05 | 0.0687 | 1.50 | 12.6 % | 39.9 % | 2.59 | 1.18 |
| baseline | 0.0654 | 1.38 | 9.0 % | 34.3 % | 2.33 | 1.29 |

**It is not a uniform gain deficit.** Over the full action box our hands' commanded swing is as
large as LEAP's or larger. What differs is that a third of the box (for the three flat hands) maps
to a **pinned joint limit** rather than to the sphere, and 9–14 % is outright starved. The "¼ the
flexion" of E4d is a rollout *variance*, and variance is exactly what a pinned map destroys.

**LEAP has both authority and conditioning; each of ours has at most one** — and which one
post-dicts the rendered taxonomy: perp = well-posed but too little travel = holds, never turns;
the other three = travel bought by slamming limits = flailing that never cages the cube.

**Mechanism = the proximal phalanx as a fraction of reach.** R = 0.509 × mean palm-to-tip
(`construct_sphere_with_palm`), so l/R is scale-free. LEAP spreads ~14 cm over three links
(53/36/50 mm) → 0.52. Ours puts 9 cm in one proximal link against a 3 cm distal → 1.3–1.5. A long
first link admits only a thin outer crescent of the sphere, and that crescent empties as the
sphere shrinks.

### The knobs, each a real MJCF → URDF → CIK rebuild

| variant | R (m) | l_prox/R | starved | pinned | travel (live) |
|---|---|---|---|---|---|
| perp as built | 0.0685 | 1.31 | 0.4 % | 16.7 % | 1.12 |
| proximal 90 → 45 mm | 0.0476 | 0.95 | 0.1 % | 12.5 % | 0.79 |
| proximal 90 → 32 mm | 0.0417 | 0.76 | 0.0 % | 9.0 % | 0.73 |
| distal 30 → 50 mm | 0.0740 | 1.22 | 0.7 % | 20.6 % | 1.18 |
| distal 30 → 70 mm | 0.0804 | 1.12 | 1.0 % | 23.3 % | 1.16 |
| fingers in (mount limit) | 0.0596 | 1.51 | 22.0 % | 48.6 % | **1.72** |
| proximal 45 + fingers in | 0.0372 | 1.21 | 2.2 % | 30.2 % | 1.36 |
| sphere R × 0.75 | 0.0514 | 1.75 | 3.1 % | 27.5 % | 1.46 |
| sphere R × 1.25 | 0.0856 | 1.05 | 0.1 % | 18.6 % | 0.77 |

1. **Finger spacing dominates.** Mounts to the inner limit is the only change that reaches LEAP's
   authority (1.72 vs 1.71), +54 %. It also wrecks conditioning (pinned 49 %) and flattens the
   thumb's lateral span to 0.002 rad, so the optimum is interior — and now searchable offline.
2. **Shortening the proximal alone trades the wrong way**: conditioning improves, authority falls,
   because R follows reach so the sphere and cube shrink with the finger. Pair it with spacing.
3. **Distal length is nearly inert** (1.12 → 1.18).
4. **`sphere_factor` is a free, never-swept knob** — a constant in `construct_sphere_with_palm`,
   not a hand property. ×0.75 buys +30 % authority on the unmodified perp. Distinct from the E5
   cube sweep: that changed the object, this changes the representation's scale.

**Status: post-diction on five hands, not validation.** No design of ours has produced an outcome
to correlate against. It is the filter to spend before Isaac time, to be confirmed by the first
design that clears both axes.

Revised next steps: (1) search the mount box offline; (2) build the best two and zero-shot,
judging on best-alignment and cube angular speed, not the 0 %-floored repose rate; (3) finetune
only if the cube visibly moves — UHAS's own `MANO_to_LEAP` transfer is 1000 iterations (~1.5 h at
8192 envs on their hardware) vs 15000 from scratch, but no finetune can create fingertip travel
the morphology does not have.

# Patches to the vendored UHAS_sim tree

`docs/uhas/UHAS_sim/` is a clone of https://github.com/IRVLUTD/UHAS_sim and is **gitignored**, so
nothing we change inside it is tracked. `uhas_sim_morphohand.patch` is the complete diff against
upstream `99f4621`, including the new `grippers/morphohand/` module. Every hunk is marked
`MORPHOHAND PATCH` in the source.

Regenerate after editing the vendored tree:

    cd docs/uhas/UHAS_sim && git add -N grippers/morphohand/ \
        && git diff > ../patches/uhas_sim_morphohand.patch && git reset -q

Reapply to a fresh clone:

    cd docs/uhas/UHAS_sim && git apply ../patches/uhas_sim_morphohand.patch

## What the patch does

| file | why |
|---|---|
| `dr_funcs.py` | Isaac Lab 2.2-only helpers backported verbatim (see INTEGRATION.md) |
| `allegro_right.py` | drop `dynamic_friction`, a 2.2 field |
| `multi_env_cfg.py` | 2.2→2.1 fields; the five checkpoint-matching flags; **env-var config** (`UHAS_ROBOTS`, `UHAS_EVAL`, `UHAS_CUBE_SCALE_MULT`, `UHAS_MORPHO_ROOT`, `UHAS_MAX_CONSEC`, `UHAS_VIEW_*`); the `morpho_*` robot branch |
| `multi_manipulation_env.py` | `ff_flag_input` obs column; **latched success counters**; **slot-indexed `main_joints` so a hand may leave a canonical slot unclaimed** |
| `grippers/morphohand/` | new: build an `ArticulationCfg` from a built-hand directory instead of a hand-written module per hand |

The last two `multi_manipulation_env` items are the load-bearing ones — see
`docs/uhas/CUBE_REPOSE_PLAN.md` E0 and E1.

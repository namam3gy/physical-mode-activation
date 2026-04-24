# M5 — VTI steering (Phase 1-2, partial Sub-task 4) (2026-04-24)

- **Phase 1 command**: inline Python — `compute_steering_vectors` from `src/physical_mode/probing/steering.py`
- **Phase 2 command**: `uv run python scripts/06_vti_steering.py --run-dir outputs/mvp_full_20260424-094103_8ae1fa3d --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 --test-subset line/blank/none --label circle --layers 10,15,20,25 --alphas 0,5,10,20,40`
- **Wall clock**: ~5 min (200 interventional inferences)
- **Outputs**: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/` + `steering_experiments/`
- **Deep dive**: `docs/insights/m5_vti_steering.md`.

## Headline: causal evidence for a "physical object-ness" direction

**Phase 1 (direction)**: VTI vectors derived from forced-choice PMR labels. Norms grow 5× through the LM (L5: 5.9 → L25: 31). Projection at L20 cleanly tracks the cue axis (none 22.3 → both 42.7). Direction is real and factorial-aligned.

**Phase 2 (intervention) — first-letter distribution on `line/blank/none`, label `circle`, α=40**:

| layer | α=0 | α=5 | α=10 | α=20 | α=40 |
|---|---|---|---|---|---|
| 10 | 10 D | 10 D | 10 D | 10 D | **10 B** 🔥 |
| 15 | 10 D | 10 D | 10 D | 10 D | 10 D |
| 20 | 10 D | 10 D | 10 D | 10 D | 10 D |
| 25 | 10 D | 10 D | 10 D | 10 D | 10 D |

**Layer 10 α=40 flips 10/10 from "D: abstract" to "B: stays still"**. No other layer budges at α=40. This is the first causal confirmation that a single direction in the LM residual stream gates the "physical object" vs "abstract shape" decision.

Sample pre/post:
- **baseline**: "D — This is an abstract shape and as such, it does not have physical properties that would allow it to fall, move, or change…"
- **L10 α=40**: "B) It stays still. — The circle in the image appears to be floating or suspended in space without any external force acting upon it. In such a scenario, the circle would remain stationary…"

The direction is "object-ness" (flips to B = physical-static), **not** "gravity" (would flip to A = falls). Consistent with H7: physics regime is label-driven; the direction encodes the binary "abstract vs physical" split.

## Caveats

- The forced-choice PMR scorer fires on option-listing ("cannot fall, move, or change direction") → PMR=1 is noisy here. First-letter flipping is the clean causal signal.
- Test subset = 10 stimuli only; replication on filled/blank/none etc. pending.
- α=40 is the only value that worked; finer sweep needed to map the threshold precisely.

## Hypothesis update

| H | prior | post-M5 | change |
|---|---|---|---|
| H-boomerang | extended (M4) | **extended + causal** | intervention activates causally |
| H-locus | candidate (M4) | **supported (early-mid L10)** | L10 is the causal sweet spot |
| H-regime (new) | — | **candidate** | direction is object-ness, not which-physics |

## Unlocks / deferred

- Phase 3 (SIP activation patching) is scoped but not executed — would need `capture_lm_attentions=True` re-capture plus the patching machinery. On the ROADMAP §3 M5b detail.
- SAE on L10 residual stream is a natural next step (additional idea; would decompose the object-ness direction into finer features).

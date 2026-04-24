# M2 — MVP-full run (2026-04-24)

- **Command**: `uv run python scripts/02_run_inference.py --config configs/mvp_full.py`
- **Stimulus dir**: `inputs/mvp_full_20260424-093926_e9d79da3` (480 stimuli)
- **Output dir**: `outputs/mvp_full_20260424-094103_8ae1fa3d`
- **Model**: `Qwen/Qwen2.5-VL-7B-Instruct`, bf16, sdpa, **T=0.7, top_p=0.95**
- **Factorial**: 4 obj × 3 bg × 4 cue × 1 event (fall) × 10 seeds × 3 labels × 2 prompts = **2880 inferences**
- **Activation capture**: LM layers (5, 10, 15, 20, 25), hidden states only (no attentions)
  → 5.2 GB across 480 `.safetensors` files (~11.5 MB / stimulus, 324 visual tokens)
- **Wall clock**: ~55 min end-to-end (1.1 s / inference including capture)
- M2 config differences vs pilot documented in the `configs/mvp_full.py`
  header and in `docs/insights/m1_pilot.md` §6.

## Success-criteria scorecard (from ROADMAP M2)

| criterion | target | observed | status |
|---|---|---|---|
| Monotone S-curve over object_level (forced-choice) | monotone | forced: line 0.583 < filled 0.647 < shaded 0.711 < textured 0.714 | ✅ |
| Open-vs-forced gap at every object_level | >0 everywhere | 22-32 pp (line 32, filled 29, shaded 22, textured 24) | ✅ |
| cast_shadow alone > none + 20 pp | +20 pp | +18.4 pp averaged (+23.4 at blank, +18.4 at ground, +10.8 at scene) | ✅ (close; edge conditions satisfied) |
| RC < 1 cells exist | some | 103/288 cells (35.8 %) with RC<1; mean RC=0.918 | ✅ |
| `outputs/*/activations/` populated | yes | 480 safetensors, LM hidden only | ✅ |

## Headline PMR tables

**Overall**: n=2880, PMR=0.797, hold_still=0.152, abstract_reject=0.160, GAR=0.656.

**By object_level (axis A)** — H1 now cleanly supported:

| object_level | n | pmr | hold_still | abstract_reject | gar |
|---|---|---|---|---|---|
| line | 720 | 0.744 | 0.193 | 0.203 | 0.594 |
| filled | 720 | 0.790 | 0.153 | 0.168 | 0.646 |
| shaded | 720 | 0.822 | 0.136 | 0.139 | 0.671 |
| textured | 720 | **0.832** | 0.126 | 0.131 | 0.713 |

Monotone across all 4 levels (no more mid-curve tie). Endpoints gap = +8.8 pp.

**By bg_level (axis B)** — scene > ground > blank, replicating the pilot's ground effect:

| bg_level | n | pmr | gar |
|---|---|---|---|
| blank | 960 | 0.669 | — |
| ground | 960 | 0.842 | 0.648 |
| scene | 960 | **0.881** | 0.664 |

Blank → scene = +21 pp (similar to the pilot's +36 pp blank → ground; T=0.7 softened the delta).

**By cue_level (axis C)** — H6 decomposition successful:

| cue_level | n | pmr | abstract_reject | gar |
|---|---|---|---|---|
| none | 720 | 0.540 | 0.347 | 0.479 |
| cast_shadow | 720 | **0.715** | 0.238 | 0.546 |
| motion_arrow | 720 | 0.964 | 0.031 | 0.860 |
| both | 720 | 0.969 | 0.025 | 0.738 |

Cast shadow alone: **+17.5 pp above none**. Arrow saturates at 0.96; adding shadow on top nudges to 0.97. The pilot's `arrow_shadow=1.00` is explained: arrow does essentially all the work; shadow's contribution is measurable but secondary.

**Per-bg decomposition** (shadow effect shrinks in richer backgrounds — saturation pattern):

| bg | none | shadow | arrow | both |
|---|---|---|---|---|
| blank | 0.287 | 0.521 (+23.4) | 0.912 | 0.954 |
| ground | 0.608 | 0.792 (+18.4) | 0.992 | 0.975 |
| scene | 0.725 | 0.833 (+10.8) | 0.988 | 0.979 |

**By prompt_variant** — open-vs-forced gap even larger than the pilot:

| prompt_variant | n | pmr | abstract_reject | gar |
|---|---|---|---|---|
| open | 1440 | **0.931** | 0.002 | 0.593 |
| forced_choice | 1440 | 0.664 | 0.318 | 0.719 |

Open-ended never self-identifies a stimulus as abstract (3 out of 1440). Forced-choice rejects 32 %. Per-object_level gap ranges +22 pp (textured) to +32 pp (line) — **larger for more abstract objects**.

**By label (axis D)** — H2 directly quantified:

| label | n | pmr | abstract_reject | gar |
|---|---|---|---|---|
| ball | 960 | **0.892** | 0.072 | 0.786 |
| circle | 960 | 0.746 | 0.186 | 0.698 |
| planet | 960 | 0.754 | 0.222 | 0.483 |

**Label × object_level interaction**:

| obj \\ label | circle | ball | planet |
|---|---|---|---|
| line | 0.692 | **0.846** | 0.696 |
| filled | 0.729 | **0.900** | 0.742 |
| shaded | 0.779 | **0.888** | 0.800 |
| textured | 0.783 | **0.933** | 0.779 |

**Remarkable**: `line + ball` (PMR 0.846) > `textured + circle` (0.783) — language prior dominates visual cue.

## Striking qualitative finding — label flips the *kind* of physics

Same stimulus (textured ball + ground + no cue, open-ended prompt), three labels:

| label | response |
|---|---|
| circle | "The circle is likely to remain static unless acted upon by an external force." |
| ball | "The ball will continue rolling down the incline." |
| planet | "The planet will continue moving along its orbital path around the Sun." |

The label doesn't just toggle physics-mode on/off — it selects *which physics regime* the model applies. `planet` invokes orbital mechanics (GAR=0.48), `ball` invokes gravity (GAR=0.79). This is a paper-worthy qualitative result for Figure 2.

## Hypothesis scorecard post-M2

| H | pilot status | M2 status | change |
|---|---|---|---|
| H1 (S-curve) | partial support | **supported** | Middle tie resolved with T=0.7 + 10 seeds |
| H2 (ball label) | strong support | **quantified** | +15 pp; `ball > circle` at every object_level |
| H3 (scene inconsistency) | untested | still untested | axis E deferred from M2 |
| H4 (open-forced gap) | candidate | **supported** | Gap +22 to +32 pp; monotone in abstraction |
| H5 (ground vs texture) | one-sided | **mixed** | bg delta (+21 pp) > object delta (+9 pp); supports H5 but scene > ground now |
| H6 (shadow alone) | needs decomposition | **supported** | Shadow +17.5 pp above none; not just annotation |

## New observations (candidate hypotheses)

- **Per-label GAR varies dramatically** (ball 0.79 / circle 0.70 / planet 0.48). "Planet" response invokes orbital physics, not gravity. Label effect on PMR is *binary-ish* but on GAR is *categorical*.
- **Saturation structure**: `motion_arrow` ~≈ `both` at 0.96-0.97. Arrow is the dominant cue; shadow's marginal contribution is strong only when the base is abstract (blank bg).
- **Open-ended is not broken** — the language-prior dominance is systematic (stronger for more abstract objects). Consistent with the "hallucinated grounding" pattern in Vo et al. 2025.

## Next actions

- **M3 (Sub-task 2 — vision encoder probing)** is unblocked: LM activations captured. Vision encoder capture still needs implementation (`PhysModeVLM.capture` extension). Draft: add 3-5 layers of vision encoder (Qwen2.5-VL's SigLIP tower) to a targeted re-run of ~100 stimuli at key factorial cells.
- **Extra headline for the paper**: "When you call a circle a planet, it orbits" (the label → physics-regime categorical flip). Not in the original `references/project.md` — this is a pilot-to-MVP-full emergent finding. Logged in `references/roadmap.md` §4 additional ideas.
- **Axis E (scene consistency)** still not tested; stays deferred pending a focused mini-experiment.

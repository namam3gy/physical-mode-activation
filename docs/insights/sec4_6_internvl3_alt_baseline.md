# §4.6 InternVL3 alt-baseline sweep — pixel-encodability confirmed under modified protocol

## TL;DR

InternVL3 was previously labeled "untestable" in the §4.6 cross-model layer
sweep because the canonical protocol's abstract baseline cell (`mvp_full
line/blank/none/circle/fall` under the "circle" prompt) reaches PMR=1.0 on
InternVL3 — no headroom to detect a flip. Under a modified protocol that
swaps to (M8a `square_filled_ground_none_fall` / `open_no_label` prompt /
square-only v_L), InternVL3 shows a clear pixel-encodable pattern with peak
at L10 (5/5 flip, baseline PMR 0.0 → synth PMR 1.0). This converts the
InternVL3 row in the H-shortcut hypothesis from "untestable" to "pattern
similar to Qwen with peak at L10."

## Result

| Layer | v_unit (n=5) | random control (n=5) | verdict |
|---|---|---|---|
| L5 | 0.0 (0/5) | 0.2 (1/5) | null — random > v_unit, single noise event |
| **L10** | **1.0 (5/5)** | 0.0 (0/5) | **strong shortcut ✓✓✓** |
| L15 | 0.4 (2/5) | 0.0 (0/5) | partial flip |
| L20 | 0.6 (3/5) | 0.0 (0/5) | partial flip |
| L25 | 0.0 (0/5) | 0.0 (0/5) | null |

Random-control specificity: 1/25 baseline_pmr → synth_pmr flips across all
random configs (4%, the L5_random outlier), vs 10/25 (40%) on v_unit.

## Sample responses (L10 v_unit)

Baseline (all 5 stim, identical pattern):
> "The image shows a gray square. It's difficult to predict what might
> happen next based on this image alone."

L10 v_unit synthesized (3 of 5):
> "The image shows a downward-pointing arrow. It might indicate a drop or
> fall, possibly in a game or simulation."
>
> "The image shows a block on a frictionless surface with a downward force
> applied. The block will likely accelerate downward due to the force."
>
> "The image shows a block on an inclined plane with a force acting on it.
> The block is likely to slide down the incline due to gravity."

These aren't just keyword hits — they're full physics-mode interpretations
mentioning gravity, force, friction, inclined planes. The shortcut isn't a
shallow string match; the model genuinely re-categorizes the synthesized
stim as a physics scene.

## Modified protocol — explicit caveat

Four divergences from the §4.6 canonical protocol:

| Axis | Canonical (other 4 models) | InternVL3 (here) |
|---|---|---|
| Stim source | `mvp_full` (circle, n=10) | `M8a square` (n=5) |
| Baseline cell | `line/blank/none/circle/fall` | `filled/ground/none/square/fall` |
| v_L source | M2 captures (mvp_full, all stim) | M8a captures filtered to `shape == "square"` |
| Prompt | "What will happen to the circle..." | open_no_label "What do you see... what might happen next?" |

**Why these divergences exist**: the canonical protocol's circle prompt
saturates InternVL3 to PMR=1.0 — no `(stim, prompt)` combination in mvp_full
where InternVL3 reliably says abstract. M8a's `square` shape with
`open_no_label` is the lowest-PMR cell across all surveyed InternVL3
predictions (5 cells at PMR ≤ 0.4; chosen `filled/ground/none/square/fall`
at PMR=0.0).

**Implication for cross-model comparison**: this is a sanity check on
protocol applicability, not a like-for-like architectural claim. The
InternVL3 row now reads "pixel-encodable shortcut exists when tested under
a non-saturating baseline; the canonical protocol couldn't measure it." The
strength of the L10 effect (5/5 with random=0/5) is internally robust.

## Updated H-shortcut cross-model picture

| Model | Encoder | LM | Pattern |
|---|---|---|---|
| Qwen2.5-VL | SigLIP+Qwen2 | broad | 5 shortcut layers ≥ 80 % |
| LLaVA-Next | CLIP+AnyRes+Mistral | L20+L25 | both 100 % |
| LLaVA-1.5 | CLIP+Vicuna | L25 only | 40 % at n=10 |
| Idefics2 | SigLIP-SO400M+perceiver+Mistral | NULL at L5-L31 | falsified for Idefics2 (perceiver-resampler is leading remaining candidate) |
| **InternVL3** | InternViT+InternLM3 | **L10 peak (5/5) + L15/L20 partial** (modified protocol) | now testable, single peak at L10 |

Reframed H-shortcut: pixel-encodability is **architecture-conditional with
varying depth profile**. Qwen broad / LLaVA-1.5 narrow late-layer / LLaVA-Next
two late-layer peaks / InternVL3 single early-mid-layer peak / Idefics2
genuinely null (perceiver-resampler hypothesis still leading). The "encoder
saturation rules out testing" framing held for Idefics2 (still rules it out)
but failed for InternVL3 (was a protocol artifact).

## Caveats / limits

- **n=5 stim only** — small sample size. L10's 5/5 is unambiguous (Wilson CI
  lower bound on 5/5 is ~0.57, far above random 0/5), but L15 (2/5) and L20
  (3/5) have wide intervals.
- **Single baseline cell** — robustness across other low-PMR cells
  (`line/ground/none/square` at 0.20, `line/blank/none/square` at 0.40) not
  tested. Advisor explicitly recommended single-cell first pass to avoid
  rabbit-hole; expand only if paper-grade rigor demands it.
- **v_L noise**: square-only v_L extraction got n_pos=73 / n_neg=7 — same
  imbalance regime as global (n_pos=382 / n_neg=18). The result is
  surprisingly clean despite the imbalance, suggesting the few negatives
  carry strong directional signal.
- **L26-L31 not tested** — Idefics2 followup tested deeper layers (L26-L31
  via fresh capture) and found null; InternVL3 deeper-layer behavior under
  the modified protocol unknown. Not pursuing per advisor's "no rabbit hole"
  guidance.

## Artifacts

- Sweep dir: `outputs/sec4_6_internvl3_alt_baseline_square_20260503-071638/`
  (manifest.json, per-stim L*/cfg/ subdirs with synthesized.png + trajectory)
- Square-only v_L: `outputs/encoder_swap_internvl3_m8a_capture_*/probing_steering/steering_vectors_square.npz`
- Scripts:
  - `scripts/sec4_6_internvl3_baseline_search.py` — alt-baseline cell survey
  - `scripts/sec4_6_internvl3_extract_square_v_L.py` — square-only v_L extraction
  - `scripts/sec4_6_internvl3_alt_baseline_score.py` — PMR re-inference + aggregation

## Decision

**InternVL3 row in §4.6 hypothesis scorecard updated**: was "untestable
(circle-prompt saturated)", now "L10 strong (modified protocol)". The
H-shortcut "encoder-saturation-rules-out-testing" framing weakens: it
correctly predicted Idefics2 (still null at L26-31 with proper protocol)
but mis-applied to InternVL3 (testable with non-saturating baseline). Net:
H-shortcut is **architecture × baseline-saturation conditional**, with
projector-design (perceiver vs MLP) emerging as the leading remaining axis
for the genuinely-null Idefics2 case.

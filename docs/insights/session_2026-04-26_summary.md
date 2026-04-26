---
session: 2026-04-26 autonomous (continuation)
date: 2026-04-26
status: complete
scope: §4.7 (RC per-axis stability) + §4.11 (categorical H7 regime distribution)
commits: 309bdf6 → bbf01f9
---

# Session 2026-04-26 — §4.7 + §4.11

## What this session delivered

Two §4 add-ons that re-use existing M8d / M8a label-free data with no
new inference. Closes the analysis-only items from the §4 backlog.

1. **§4.11 — categorical H7 regime distribution** (commit `309bdf6`).
   Applies `classify_regime` to all 4-model M8d label-free + labeled
   runs (Qwen / LLaVA-1.5 / LLaVA-Next / Idefics2). 4×3×4 stacked-bar
   matrix shows kinetic / static / abstract / ambiguous fractions
   per (model × category × label_role).

2. **§4.7 — per-axis RC decision stability** (commit `bbf01f9`).
   Reinterprets RC under T=0.7 as per-axis decision stability on M8a
   label-free. 5 models × 3 axes (object_level / bg_level / cue_level)
   × {2-4 levels each}.

## Headline findings

### §4.11 — regime distribution distinguishes models that binary H7 obscured

![§4.11 4-model M8d regime distribution](../figures/sec4_11_regime_distribution_4model.png)

- **Qwen + Idefics2**: saturated kinetic everywhere (~95%). Only Qwen
  `person × exotic` (statue) shows ~30% static.
- **LLaVA-1.5**: most regime-discriminative. `car × abs` (silhouette)
  drops kinetic to 28% with 70% ambiguous.
- **LLaVA-Next**: intermediate. `person × exotic` (statue) shows a
  3-way split (30% kinetic + 25% static + 25% abstract) — multi-axis
  architectural twist absent in LLaVA-1.5.

The 4-model gradient on `person × abs` (stick figure):
| Model | % kinetic |
|---|---:|
| Qwen | 91 |
| Idefics2 | 99 |
| LLaVA-Next | 80 |
| LLaVA-1.5 | 58 |

Granular form of the M9 H7 finding. Categorical view reveals what
*kind* of commitment the label produces, not just whether the model
commits.

### §4.7 — cue_level is the dominant decision stabilizer for saturated models

![§4.7 5-model per-axis RC](../figures/sec4_7_rc_per_axis.png)

| model | cue=none → cue=both | bg=blank → bg=ground |
|-------|---------------------|----------------------|
| Qwen2.5-VL | 0.84 → **1.00** (+0.16) | 0.88 → 0.96 (+0.08) |
| Idefics2 | 0.88 → 0.99 (+0.11) | 0.92 → 0.95 (+0.03) |
| InternVL3 | 0.89 → 0.98 (+0.09) | 0.92 → 0.96 (+0.04) |
| LLaVA-1.5 | 0.85 → 0.85 (0) | 0.88 → 0.82 (**−0.06**) |
| LLaVA-Next | 0.78 → 0.78 (0) | 0.77 → 0.80 (+0.03) |

**Reading**: saturation is not just a behavioral PMR ceiling but also a
**decision-stability ceiling**. Non-CLIP models converge to the same
PMR call across all 5 seeds when cues fire; CLIP-based models retain
seed-level variance even under strong cues. Separate signature of the
H-encoder-saturation reframe.

## Hypothesis status updates

- **H7** — was already "unsaturated-only AND architecture-conditional";
  §4.11 adds the **categorical** dimension (binary→regime distribution)
  showing label-disambiguation works at the regime level for LLaVA-1.5
  even where the binary H7 number is muted.
- **H-encoder-saturation** — was already "architecture-level confirmed
  at 5 model points × 3 stim sources"; §4.7 adds the **decision-stability
  dimension**: saturation also locks in seed-level commitment under cues.
  Two distinct signatures of the same architectural property.

## Late-session addition: InternVL3 M8d (closes §4.11 5-model gap)

After §4.7 + §4.11 4-model commits, InternVL3 was run on M8d (~13 min
on GPU 0) and §4.11 figure regenerated as 5-model. Commits `be29792`
(§4.11 5-model close) and `3b1e5d8` (M9 audit InternVL3 M8d row).

**InternVL3 M8d new finding**: `person × exotic` (statue) PMR drops
from 0.800 (physical "person") to 0.481 (exotic "statue") — a 32 pp
suppression. Categorical view: 30% kinetic / 65% static — **the
strongest single label-driven static commit in the project**. This
shows that even saturated-encoder architectures (InternVL3 PMR 0.92
on M8a) have an active label-disambiguation channel that fires when
the label uniquely picks out a non-moving entity.

Updated 5-model `person × abs` (stick figure) gradient:
| Model | % kinetic |
|---|---:|
| Idefics2 | 99 |
| InternVL3 | 99 |
| Qwen | 91 |
| LLaVA-Next | 80 |
| LLaVA-1.5 | 58 |

5-model § 4.11 figure: `docs/figures/sec4_11_regime_distribution_5model.png`.
Roadmap §4.11 promoted from "partial" to "complete".

## Limitations carried forward

1. ~~§4.11 InternVL3 missing~~ — *closed* (commit `be29792`).
2. **§4.11 5-category fine-grained classifier** (gravity-fall / gravity-
   roll / orbital / inertial / static) for M2 circle-only data is still
   open — would need new keyword sets + extending `classify_regime` to
   `circle` shape.
3. **§4.7 n_seeds=5** is the bare minimum for RC. ≥10 pp differences
   are robust; <5 pp differences are suggestive.
4. **§4.7 single arm (label-free)**. Labeled arms might show different
   RC structure since labels themselves stabilize commitment.

## Artifacts

### Commits (this session, 2 substantive)

- `309bdf6` — §4.11 4-model M8d regime distribution
- `bbf01f9` — §4.7 per-axis RC stability

### New figures

- `docs/figures/sec4_11_regime_distribution_4model.png`
- `docs/figures/sec4_7_rc_per_axis.png`

### New insight docs

- `docs/insights/sec4_11_regime_distribution.md` (+ ko)
- `docs/insights/sec4_7_rc_per_axis.md` (+ ko)
- `docs/insights/session_2026-04-26_summary.md` (this doc, + ko)

### New scripts

- `scripts/sec4_11_regime_distribution.py`
- `scripts/sec4_7_rc_per_axis.py`

### Roadmap

- §4.11 marked "partial complete" (4-model M8d done; M2 fine-grained
  still open)
- §4.7 marked "complete"

## Combined backlog after this session

Open §4 items:
- §4.3 — Korean vs English label prior (1-hour, but PMR scorer is
  English-only, needs scorer extension)
- §4.4 — Michotte 2-frame causality (needs 2-image prompt support)
- §4.6 — SAE counterfactual stim generation (complex, 4-6 hours)
- §4.8 — PMR scaling (Qwen 32B/72B — needs new large-model loads)

Major milestones:
- **M5b** — SIP / activation patching / SAE feature decomposition
  (mechanism-level evidence, the next paper-section gap)
- **M7** — paper draft + Prolific human baseline

## Session running total (2026-04-25 + 2026-04-26)

- Total commits since start of M6 r6: ~17 substantive
- Total insight docs: 16 (English) + 16 (Korean) = 32 paired docs
  (research_overview, session summaries, m6 r1-r6, m8 a/c/d/e, m9,
  encoder_saturation_paper, sec4_2/4_7/4_10/4_11, m5/m4 series)
- Total figures: 30+ (project-wide); 5 added in this autonomous run
  (session_5model_cross_stim_pmr, session_image_vs_label_h7,
  session_attention_cross_model, sec4_11_regime_distribution_4model,
  sec4_7_rc_per_axis)
- Total notebooks: 13 (project-wide); 1 new (attention_viz.ipynb) +
  1 extended (encoder_saturation_chain.ipynb)
- pytest: 123/123 (no regressions)

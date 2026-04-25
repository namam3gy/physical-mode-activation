# M6 round 2 — Cross-model run log

Three sub-tasks executed 2026-04-25 to extend M6 round 1 (LLaVA-1.5 cross-
model behavioral) along three orthogonal directions:

- **r2a**: third behavioral cross-model point — InternVL3-8B-hf.
- **r2b**: LLaVA-1.5 activation captures + cross-model M3 (vision probe)
  and M4 (LM probe) AUC comparison.
- **r2c**: forced-choice first-token logit-ratio scoring — does LLaVA's
  "A" bias (M6 r1, M4c) live in greedy decoding or in underlying logits?

## r2a — InternVL3-8B-hf cross-model

### Setup

- Configs: `cross_model_internvl3.py` + `cross_model_internvl3_label_free.py`.
- Stimuli: M2 manifest reused (480 stim).
- Generation: T=0.7, top_p=0.95, max_new_tokens=96.
- Forced-choice excluded (smoke confirmed FC produces full justified
  responses, unlike LLaVA — qualitative evidence enough; full FC sweep
  would have added ~80 min).
- Outputs:
  - `outputs/cross_model_internvl3_20260425-051009_fc710e85/` — labeled, 1440 rows.
  - `outputs/cross_model_internvl3_label_free_20260425-053116_ea0a07c5/` — label-free, 480 rows.

### Behavioral results

#### InternVL3 labeled — overall

| label  | PMR | GAR | hold_still |
|---|---|---|---|
| ball   | 1.000 | 0.816 | 0.000 |
| circle | 1.000 | 0.791 | 0.010 |
| planet | 1.000 | 0.431 | 0.023 |

PMR is **1.000 across every (object × cue × label)** combination — InternVL3
never picks an abstract reading on the open prompt for M2 stimuli.

#### InternVL3 label-free — overall

| metric | value |
|---|---|
| PMR | 0.989 |
| GAR | 0.644 |
| hold_still | 0.065 |

Even without label, PMR is 0.99. InternVL3 has the strongest visual physics
prior of any tested model.

#### InternVL3 paired delta vs label-free

| label  | mean `PMR(label) − PMR(_nolabel)` |
|---|---|
| ball   | +0.010 |
| circle | +0.010 |
| planet | +0.010 |

**All three labels contribute +0.010 — essentially noise.** No headroom for
language prior to operate.

### 3-model cross-comparison

`PMR(_nolabel)` by `object_level`:

| object   | Qwen2.5-VL | LLaVA-1.5 | InternVL3 |
|---|---|---|---|
| line     | 0.942 | 0.142 | 0.992 |
| filled   | 0.933 | 0.317 | 0.967 |
| shaded   | 0.942 | 0.592 | 1.000 |
| textured | 0.975 | 0.483 | 1.000 |

Paired delta `PMR(label) − PMR(_nolabel)` (480 matched seeds, T=0.7):

| label  | Qwen   | LLaVA  | InternVL3 |
|---|---|---|---|
| ball   | +0.006 | +0.475 | +0.010 |
| circle | −0.065 | +0.173 | +0.010 |
| planet | +0.006 | +0.244 | +0.010 |

H7 GAR by label:

| label  | Qwen  | LLaVA | InternVL3 |
|---|---|---|---|
| ball   | 0.706 | 0.356 | 0.816 |
| circle | 0.753 | 0.153 | 0.791 |
| planet | 0.319 | 0.072 | 0.431 |

`planet GAR << ball/circle GAR` replicates in all three models. H7 cross-
model confirmed.

### Sample InternVL3 responses on `line/blank/none`

- ball: "The ball will likely fall downward due to gravity." × N
- circle: "The circle will likely move downward." / "The circle will likely start to move or rotate."
- planet: "The planet will continue to remain stationary in the center of the image." / "The planet will continue to rotate on its axis."

InternVL3 commits to physics-mode for `circle` even on `line/blank/none`,
but the regime selection (planet → orbital) still operates.

## r2b — LLaVA-1.5 activation captures

### Setup

- Config: `cross_model_llava_capture.py`.
- `capture_lm_layers=(5, 10, 15, 20, 25)`, `capture_vision_layers=(3, 7, 11, 15, 19, 23)`.
- 480 stim × 3 labels × 1 prompt (open) = 1440 inferences + 480 captures.
- Output: `outputs/cross_model_llava_capture_20260425-054821_65214a5d/`.
- Disk: 14 GB activations.

### Code change

`_resolve_vision_blocks` updated to handle the LLaVA-1.5-hf wrapper, where
the CLIPVisionModel's encoder is at `vt.encoder.layers` directly (no extra
`vision_model` wrapper). Vision-hook output is also squeezed from
(1, 577, 1024) to (577, 1024) so the probing functions (which expect
`(n_tokens, dim)`) work without changes.

### Behavioral consistency

LLaVA capture run vs LLaVA M6 r1 run (no captures):
- mean abs diff in PMR per (object × label): 0.056
- max diff: 0.158
- mean abs diff in GAR: 0.043

Stochastic seed differences explain the spread; behavioral pattern is
preserved.

### Cross-model probe AUC comparison

#### Vision encoder probe AUC (open prompt)

| layer | Qwen2.5-VL (M3) | LLaVA-1.5 (M6 r2b) |
|---|---|---|
| 3  | 0.980 | 0.707 |
| 7  | 0.985 | 0.731 |
| 11 | 0.986 | 0.726 |
| 15 | 0.979 | 0.732 |
| 19 | 0.986 | 0.715 |
| 23 | 0.986 | 0.728 |

(Qwen also probed at 27, 31; both 0.985 / 0.985.)

LLaVA's CLIP-ViT-L vision encoder is **~25 percentage points behind Qwen's
SigLIP encoder** in physics-vs-abstract separability. The gap is uniform
across depth — neither encoder shows a "more separation deeper in the
network" pattern; both are roughly flat through their layers.

#### LM probe AUC at visual-token positions (open prompt)

| layer | Qwen2.5-VL (M4) | LLaVA-1.5 (M6 r2b) |
|---|---|---|
| 5  | 0.939 | 0.732 |
| 10 | 0.944 | 0.753 |
| 15 | 0.947 | 0.747 |
| 20 | 0.953 | 0.748 |
| 25 | 0.944 | 0.736 |

LLaVA's LM AUC tracks its vision AUC — both ~0.73-0.75 with no boomerang
"recovery". For Qwen there's a slight loss through the LM
(0.985 → 0.94 ≈ 4 pp). For LLaVA the LM doesn't add or subtract signal
relative to its vision encoder.

#### Boomerang gap (encoder AUC vs LM AUC vs behavioral)

| pipeline stage | Qwen | LLaVA |
|---|---|---|
| Vision encoder AUC (open) | 0.985 | 0.728 |
| LM AUC at visual tokens | 0.946 | 0.745 |
| Behavioral PMR (open) | 0.93 | 0.78 |

The Qwen "encoder knows, decoder gates" boomerang exists (∆ = 0.985 → 0.93
= 5 pp). The LLaVA pipeline is roughly flat — encoder and behavior are at
similar levels (0.73 vs 0.78).

### Implication: visual-saturation is rooted in the vision encoder

The "weak visual prior" of LLaVA-1.5 (low `PMR(_nolabel)`) is not a
late-stage gating phenomenon — it is determined at the vision encoder
itself. LLaVA's CLIP-ViT-L only achieves AUC ~0.73 on the physics-vs-
abstract dimension that Qwen's SigLIP encoder achieves AUC ~0.99. Adding
a label compensates for the missing visual signal, which is why LLaVA's
paired delta vs no-label is +0.475 for ball; in Qwen the visual signal
is already saturated, so the label has no work to do.

## r2c — Forced-choice first-token logit-ratio scoring

### Setup

`option_logits` was already saved per-FC-row (Qwen M2 forced_choice;
Qwen FC label-free; LLaVA FC label-free). Re-derive:
- `logit_argmax`: argmax over the 4 letters' logits at the first generated
  token step (post warping by sampling temperature + top_p).
- `pmr_from_logit_argmax`: 1 if logit_argmax ∈ {A, B, C}, else 0.

Note on data limitation: the saved logits are post-temperature/top_p
warping (per HF `output_scores`), so tokens filtered out by top_p=0.95
appear as `-inf`. argmax over non-inf entries is still informative — when
only one letter has a finite logit, its probability mass under top_p is
≥0.95.

### How many letters survive the top_p filter?

| run | n_real distribution (rows × n_real_letters) |
|---|---|
| Qwen M2 FC labeled (1440 rows) | 1: 736, 2: 387, 3: 317 |
| Qwen FC label-free (480 rows) | 1: 278, 2: 81, 3: 119, 4: 2 |
| LLaVA FC label-free (480 rows) | 1: 430, 2: 50 |

90% of LLaVA rows have only one letter (always `A`) surviving top_p — the
underlying probability of A is ≥0.95 in 90% of cases. C and D never enter
the top_p set.

### Logit-argmax distribution

| run | A | B | C | D |
|---|---|---|---|---|
| Qwen M2 FC labeled | 938 | 5 | 0 | 497 |
| Qwen FC label-free | 371 | 75 | 0 | 34 |
| LLaVA FC label-free | 480 | 0 | 0 | 0 |

LLaVA's logit-argmax is 100% A. Greedy first_letter was 477/480 = 99.4 %
A with 3 stray B. The bias is at the logit level, not introduced by
greedy sampling.

### Greedy vs logit-argmax agreement

| run | greedy == logit_argmax |
|---|---|
| Qwen M2 FC labeled  | 0.695 |
| Qwen FC label-free | 0.756 |
| LLaVA FC label-free | 0.994 |

Qwen disagreement is mostly because greedy can produce "other" tokens
(e.g., a leading whitespace or formatting token) that don't match any
of A/B/C/D, while logit-argmax always picks among the four.

### PMR-from-logit vs PMR-from-text

| run | text-PMR | logit-PMR | Δ |
|---|---|---|---|
| Qwen M2 FC labeled | 0.510 | 0.655 | +0.145 |
| Qwen FC label-free | 0.769 | 0.929 | +0.160 |
| LLaVA FC label-free | 1.000 | 1.000 | 0.000 |

For Qwen, logit-PMR is 14-16 pp higher than text-PMR — text-PMR
under-counts because of "other" greedy outputs (e.g., model starts the
answer with an apostrophe or newline). For LLaVA the two metrics agree at
1.000 (both A-locked).

### Conclusion

- LLaVA's FC pathology is at the underlying logit level. Greedy → first-
  token logit ratio rescue does not work. To salvage LLaVA FC the
  inference would need to be steered (forbid generating A, force pick
  among B/C/D), which would be a different probe entirely.
- For Qwen, logit-argmax is a cleaner FC metric than text-PMR: it
  ignores trivial first-token formatting drift and recovers ~14 pp of
  signal.

## Raw artifacts

- `outputs/cross_model_internvl3_20260425-051009_fc710e85/` — InternVL3 labeled.
- `outputs/cross_model_internvl3_label_free_20260425-053116_ea0a07c5/` — InternVL3 label-free.
- `outputs/cross_model_llava_capture_20260425-054821_65214a5d/` — LLaVA captured (predictions + 14 GB activations + probing_vision/ + probing_lm/ subdirs).
- `outputs/{mvp_full_*, fc_label_free_*, label_free_*, cross_model_llava_*}/predictions.jsonl` — option_logits intact for the r2c re-analysis.

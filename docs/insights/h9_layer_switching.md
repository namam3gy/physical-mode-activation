---
section: H9 (cross-model layer-wise emergence) explicit test
date: 2026-04-26
status: complete (revises H9 prediction)
hypothesis: ~~"Qwen / InternVL switch to physics-mode at earlier layers than LLaVA-1.5"~~ — REFRAMED: saturated encoders have probe AUC plateau ALREADY at the first captured layer (L5); unsaturated encoders never reach plateau in the captured range.
---

# H9 — cross-model layer-wise probe AUC

> **Recap of codes used in this doc**
>
> - **H9** (research plan §2.4 prediction): "Qwen2-VL / InternVL2 switch to physics mode at earlier layers than LLaVA-1.5 (thanks to a larger vision encoder and a more sophisticated projector)."
> - **M2** — 480-stim 5-axis factorial (Qwen-only originally; cross-model captures via M6 r2b + M6 r7).
> - **PMR** — physics-mode reading rate per response.
> - **H-encoder-saturation** — behavioral PMR ceiling determined at architecture level (joint encoder + LM).

## Question

The plan predicted a per-layer ordering: Qwen / InternVL ahead of LLaVA-1.5
on "when does physics-mode emerge in the LM." We have 5-model M2-stim
LM captures (Qwen / LLaVA-1.5 / LLaVA-Next / Idefics2 / InternVL3 at
layers L5 / L10 / L15 / L20 / L25). Probe each model × layer; compute
the layer where probe AUC first crosses 0.85.

## Method

For each model, for each captured layer L ∈ {5, 10, 15, 20, 25}:

1. Mean-pool the LM hidden state at visual-token positions per stim
   (n=480 stim, dim depends on model: Qwen 3584, others 4096 except
   InternVL3 3584).
2. Per-stim PMR label = mean(PMR across {ball, circle, planet}) ≥ 0.5.
3. 5-fold stratified CV logistic regression → AUC.
4. When n_neg < 5 (degenerate class imbalance), fall back to full-fit
   training AUC and flag.

## Result

![H9 5-model layer-wise probe AUC](../figures/h9_layer_switching.png)

| Model | LM layers | L5 | L10 | L15 | L20 | L25 | first ≥ 0.85 |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-VL | 28 | **0.970** | 0.973 | 0.972 | 0.973 | 0.974 | L5 (rel 0.18) |
| Idefics2 | 32 | **0.995** | 0.995 | 0.995 | 0.995 | 0.995 | L5 (rel 0.16, n_neg=5 borderline) |
| InternVL3 | 48 | **0.991** | 0.991 | 0.991 | 0.991 | 0.991 | L5 (rel 0.10, full-fit n_neg=1) |
| LLaVA-Next | 32 | 0.732 | 0.762 | 0.751 | 0.786 | 0.791 | never |
| LLaVA-1.5 | 32 | 0.757 | 0.760 | 0.762 | 0.764 | 0.768 | never |

## Headlines

1. **Two-cluster pattern, not earlier-vs-later**. Three saturated-encoder
   models (Qwen, Idefics2, InternVL3) hit their plateau already at the
   first captured layer L5; the 5-25 sweep is essentially flat. The
   two CLIP-encoder models (LLaVA-1.5, LLaVA-Next) never reach 0.85
   at any captured layer.

2. **Qwen plateau ≈ 0.97; LLaVA-1.5 plateau ≈ 0.77** (n_neg balanced,
   5-fold CV). The 0.20 AUC gap is consistent with the encoder-AUC
   gap (Qwen ~0.99 / LLaVA-1.5 ~0.73) reported in M6 r2 — the LM
   probe inherits the encoder's discriminability, and what saturates
   the encoder also saturates the LM probe.

3. **Caveat for Idefics2 / InternVL3**: their high AUC (0.995, 0.991)
   is on a degenerate class distribution (n_neg = 5 / 1 of 480). With
   so few negatives, the probe overfits trivially. Read as
   "essentially saturated, no operational headroom for separation."

4. **H9's literal prediction is partially supported**: Qwen reaches a
   higher plateau than LLaVA-1.5 (0.97 vs 0.77) and reaches it at the
   same earliest captured layer. The prediction's assumed ordering of
   *layers* (Qwen "earlier") is true at the absolute-layer level
   only because Qwen hits its plateau by L5 and stays. The unsaturated
   models never reach LLaVA-1.5's plateau (~0.77) anyway, so an
   "earlier" claim is moot.

## Reframe

H9 is more accurately stated as:

> **"For saturated-encoder models, physics-mode separability emerges
> *before* LM L5 and plateaus from there. For unsaturated-encoder
> models, separability never reaches the saturated plateau across the
> captured layer range (L5–L25)."**

This reframes the original "earlier-vs-later layer" framing into the
M9 / §4.7 / §4.6 family — *another saturation signature*, this time
in LM probe AUC. The actual "switching layer" for saturated models is
buried before L5; we'd need finer-grained captures (L1, L2, L3) to
find it.

For unsaturated models (CLIP-based), the *probe never separates
cleanly*; this is consistent with the encoder being the bottleneck
(M6 r2b: LLaVA-1.5 vision AUC 0.73, LM AUC 0.75 — both flat).

## Connection to existing hypotheses

- **H-encoder-saturation** — strengthened. The LM-probe AUC ladder
  matches the encoder-probe AUC ladder. Encoder saturation determines
  what the LM has to work with; the LM's "physics-mode separability"
  inherits this.
- **H-locus** (causal locus mid-LM, M5a) — partially complicated.
  M5a found L10 the *only* causal-intervention layer on Qwen; here
  the *probe AUC* is uniformly high from L5 to L25. The discrepancy
  means **probe AUC ≠ causal locus** — uniform AUC says info is
  *present* at every layer; M5a says info is *only causally
  intervenable* at L10. The two measures are complementary, not
  contradictory.

## Limitations

1. **Earliest captured layer is L5.** For Qwen/Idefics2/InternVL3, the
   actual "switching layer" is somewhere before L5 — likely L1-3
   (consistent with Basu et al. 2024's finding that constraint-
   satisfaction info is stored in LLaVA layers 1-4). Need finer
   captures.
2. **Idefics2 / InternVL3 probe AUC overfits** due to extreme class
   imbalance (n_neg = 5, 1). Their "0.995" / "0.991" should be read
   as "essentially saturated, no operational measurement."
3. **Behavioral-y target** — probes target per-stim PMR. Stim-y target
   would test pure encoder-side discriminability of factorial cells
   (which we already know is AUC = 1.0 from M6 r5).
4. **Single-task evaluation.** Other shortcut behaviors might show
   different layer trajectories.

## Reproducer

```bash
uv run python scripts/h9_layer_switching_analysis.py
```

## Artifacts

- `scripts/h9_layer_switching_analysis.py` — driver (5-fold CV +
  full-fit fallback for degenerate class imbalance).
- `outputs/h9_switching/per_layer_auc.csv` — full table (model × layer).
- `outputs/h9_switching/switching_layer.csv` — per-model summary.
- `docs/figures/h9_layer_switching.png` — 2-panel: AUC vs absolute
  layer + AUC vs relative depth.

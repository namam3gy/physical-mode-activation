# M-LMSwap design (Pillar B / B2) — initial draft

> **Status**: design draft (2026-04-29). Promoted from week 6–7 to active when M-PSwap was backlogged on the unresolved step-1000 NaN.
> **Track B reference**: `references/submission_plan.md` Pillar B (B2); `references/paper_gaps.md` G3 (n=1 perceiver).
> **Companion infra**: `src/physical_mode/lora/` (built for M-PSwap; bulk re-usable — LoRA training loop, NaN-abort + per-step grad logging, `the_cauldron` streaming dataloader).

## 1. Question

Is the LLaVA-family **M5b NULL** (top-k SAE ablation at the actually-consumed encoder layer fails to break PMR; see CHANGELOG 2026-04-28 evening) caused by:

1. **The encoder** (CLIP-ViT-L) — physics-mode commitment routes through the LM, so encoder-side SAE features are insufficient regardless of which LM sits behind it; or
2. **The LM family** (Vicuna-7B vs Mistral-7B vs Qwen2.5-7B-LM) — the encoder *does* host commitment features, but only some LMs read them; CLIP+Mistral (LLaVA-Next) and CLIP+Vicuna (LLaVA-1.5) both happen to be LMs that don't?

Existing 5-model data cannot disentangle these because the LLaVA-1.5 ↔ LLaVA-Next-Mistral comparison confounds **LM swap** with **AnyRes vs single-tile** simultaneously.

## 2. Design — controlled pair

Two minimal-difference variants, identical encoder + projector, varying only the LM:

| Variant | Encoder | Projector | LM |
|---|---|---|---|
| **A — CLIP+Vicuna** | CLIP-ViT-L/14-336 | MLP (LLaVA-1.5 style) | Vicuna-7B-v1.5 |
| **B — CLIP+Mistral** | CLIP-ViT-L/14-336 | MLP (LLaVA-1.5 style) | Mistral-7B-Instruct-v0.2 |

**Held fixed across A and B**:
- Same CLIP encoder (single-tile, no AnyRes).
- Same MLP projector design (2-layer GELU, LLaVA-1.5 dimensions).
- Same training data and recipe (LoRA on the LM; full-finetune on the projector — match M-PSwap's recipe).
- Same training compute budget (so neither variant is under-trained).

**Initialization shortcut**: variant A can re-use LLaVA-1.5-7B as-is (CLIP+Vicuna+MLP already exists, no training needed) — only variant B requires training. This halves the training time vs from-scratch on both. *Open*: does LLaVA-1.5's encoder + projector statistical distribution carry over cleanly when only the LM changes? Sanity check: if variant B post-training has bizarre regression (e.g., M2 PMR_nolabel collapses to 0.0 or saturates to 1.0), the recipe is mis-tuned. Use M-PSwap's `regression_eval.py` pattern.

**Decision pending**: from-scratch on both A and B (cleaner controlled comparison, ~2× cost) vs LLaVA-1.5-A + new-train-B (cheaper, asymmetric in any unmeasured "training residue"). Default to the asymmetric option for week-1 spike; revisit if the asymmetry shows up in regression eval.

## 3. Predicted outcomes

| Hypothesis | M5a (runtime steering) | M5b (SAE ablation) |
|---|---|---|
| **(1) encoder bottleneck** — LLaVA M5b NULL is encoder-driven | Both A and B should match LLaVA-1.5: M5a flips at L25-ish (L25 = relative ~78% depth on Vicuna; analogous on Mistral) | Both A and B NULL at every k ≤ 160 |
| **(2) LM modulation** — LM family gates whether encoder-side commitment is readable | A and B differ on M5a (LM-specific layer / α dynamic range) | At least one of A or B breaks at k ≤ 160 (the LM that *can* read encoder features) |
| **(3) joint** | Mixed ladder | Mixed ladder |

The strongest paper-friendly result is **(1) clean** — both variants NULL on M5b at all k, with M5a flipping in both — because that gives us a controlled causal demonstration that "encoder family fixes the M5b ceiling regardless of LM."

The **most surprising** result would be (2): if A and B dissociate on M5b at the same encoder, the encoder bottleneck story collapses and we have a stronger LM-family causal story instead. Either is publishable; only "noisy and inconclusive" is a real loss.

## 4. Workplan (week 4–5)

**Day 0 (pipeline gate — revised 2026-04-29 after inspecting fp32 run)** — Inspection of `outputs/mpswap_fp32_20260429-053240/` reveals: training was healthy through step 1450 (loss 0.39 → 0.33 descending, coherent generation samples at 750/1000/1250 — *"A man is standing in a field."*), then NaN'd at step 1461 with no warning. **Pipeline is functionally correct**; the M-PSwap NaN is most likely a **single pathological batch in `the_cauldron` stream**, not bf16 attention overflow / mask handling / optimizer state.

Lighter gate sequence:

- **D0a — Determinism repro** (cheap, ~10–30 min): run `scripts/m_pswap_repro_nan_batch.py` against `mpswap_fp32_20260429-053240/step1000`. Iterates the_cauldron with same seed, finds the offending batch, saves it to `nan_repro/bad_batch.pkl`. Diagnostic confirmation that the trigger is data-driven.
- **D0b — Dataset substitution smoke** (~15–30 min): 100-step LoRA smoke on **LLaVA-1.5-7B + LCS-558K** (or any non-`the_cauldron` chat-format VLM dataset) to confirm LCS-558K plumbing + recipe runs clean. Goal is plumbing verification, not architecture diagnosis (architecture already validated via the fp32 run).
- If both D0a and D0b confirm the data-batch hypothesis, the gate passes with **two side benefits**: (a) variants A and B avoid the trigger by using LCS-558K; (b) M-PSwap unblocks opportunistically by switching dataset (or filtering the offending batch) — flag in roadmap §3.X for resume after M-LMSwap if scheduling allows.

| Day | Task |
|---|---|
| 0 | **Pipeline gate (lightweight)** — D0a + D0b above. |
| 1 | `scripts/m_lmswap_train.py` from `m_pswap_train.py`. **Symmetric design**: train both variants A (CLIP + Vicuna-7B-v1.5 + fresh MLP) and B (CLIP + Mistral-7B-Instruct-v0.2 + fresh MLP) from same starting recipe; encoder held frozen, MLP full-FT, LM LoRA rank-32 alpha-64 on q/v/k/o_proj. |
| 1 | 50-step smoke for **both** variants on **LCS-558K** (LLaVA-1.5 pretrain corpus; coherent with rest of chain + isolates "was the M-PSwap NaN dataset-driven"). |
| 2–4 | Full LoRA training **A and B in parallel on H200×2** (one variant per GPU). NaN-abort + per-step `g_lora` vs `g_proj` logging. |
| 4 | Regression eval on both (`m_pswap_regression_eval.py` pattern): M2 PMR_nolabel must land in 0.2–0.8 range; if 0.0 collapse or 1.0 saturation on either, training under-tuned. |
| 5 | **Shared encoder SAE**: train one SAE on the held-fixed CLIP `vision_hidden_22` activations (same input distribution for A and B → shared SAE is principled). Per-variant Cohen's d ranking using each variant's own physics/abstract PMR labels. |
| 5 | M5a runtime steering on variants A and B at L20–L25 (Vicuna 32 LM layers; Mistral 32 LM layers); M5b top-k SAE ablation on both with shared SAE + per-variant ranking. |
| 5 | Cross-method analysis + insight doc `docs/insights/m_lmswap.md`. |
| End of W5 | **Stretch decision gate for variant C** (CLIP+Qwen2.5-7B-LM): only run if A↔B is *clean* (clear dissociation or clean shared NULL) **and** schedule has slack. Default: defer. |

If the day-0 gate fails or the day-1 smoke NaNs the same way as M-PSwap, the **LoRA training pipeline itself is the suspect**, not perceiver-specific architectural surgery — diagnose before continuing.

## 5. Risks / pre-mortems

- **Same-pipeline NaN**: M-PSwap's NaN may be a generic LoRA-training-recipe bug (bf16 attention, mask handling, the_cauldron edge case). If variant B NaNs at a similar step, M-LMSwap unblocks M-PSwap diagnostics by accident — useful, but loses the week. Mitigation: run smoke + regression-eval at 50, 200, 500, 1000 steps before letting it run unattended overnight.
- **Vicuna-only baseline asymmetry**: if LLaVA-1.5 is variant A unmodified and variant B is freshly trained, training residue (overfit on `the_cauldron`'s narrow distribution) could make B's M2 evaluation different for reasons unrelated to LM family. Mitigation: regression eval as gate; if delta from LLaVA-1.5 baseline on standard benchmarks is large, retrain variant A from CLIP+Vicuna with same recipe (symmetric).
- **MLP projector is too weak**: 2-layer GELU MLP may underperform LLaVA-1.5's perfectly tuned projector, dragging down PMR. Mitigation: copy LLaVA-1.5's projector weights to initialize variant B's projector.
- **Regression on M2 stim**: even a well-trained variant B might score very differently from LLaVA-Next-Mistral on M2 because LLaVA-Next has AnyRes. The comparison is **not** "does B match LLaVA-Next-Mistral" — it's "does B differ from variant A on M5a/M5b in ways that isolate LM family." Anchor analysis on A↔B delta, not A/B-vs-existing-LLaVA-models.

## 6. Falsifiability

The result is interpretable only if:
- Variant A and variant B both produce **non-trivial M2 PMR** (in 0.2–0.8 range, like LLaVA-1.5/LLaVA-Next; not 0.0 collapse / 1.0 saturation).
- M5a baseline at the chosen layer fires at near 0/10 on `line_blank_none` for both variants (room to flip up).
- M5b SAE features for both variants pass the Cohen's d sanity check (top feature d ≥ 0.5; matching the LLaVA-1.5 / LLaVA-Next round-2 SAE characteristics).

If any of those gates fail, the experiment is uninterpretable and we fall back to the literature-grounded G3 fallback per `paper_gaps.md`.

## 7. Locked decisions (advisor-checked 2026-04-29)

The four open questions resolved after advisor review:

| Q | Decision | Rationale |
|---|---|---|
| **Q1 — Sym vs asym** | **Symmetric** (both A and B from scratch, same recipe, same data) | Asymmetric design embeds the contrast we want to measure inside unmeasured "LLaVA-1.5 training residue." ~2× cost is the price of the controlled claim. |
| **Q2 — Dataset** | **LCS-558K** (LLaVA-1.5 pretrain corpus) | Coherent with rest of the LLaVA-family chain. Diagnostic side benefit: M-PSwap NaN'd on `the_cauldron`; switching dataset partially isolates "was the NaN dataset-driven." |
| **Q3 — SAE** | **Shared encoder SAE + per-variant Cohen's d ranking** | Encoder is held bit-identical across A and B → activations on the same stim are identical → one SAE is principled. Variant-specific PMR labels drive variant-specific feature ranking. |
| **Q4 — 3-point ladder (variant C)** | **2-point primary; gate-decide C at end of week 4** | Novelty is the *controlled counterfactual itself*, not ladder length. CLIP+Qwen-LM at 10K-sample LoRA is OOD for canonical Qwen-VL training (native = SigLIP); base rate of clearing the §6 falsifiability gate is unfavorable. A failed gate doesn't extend the ladder, it pollutes the figure. Run only if A↔B is clean **and** schedule has slack. |

**Pipeline-NaN concern (not in original list)**: M-LMSwap reuses M-PSwap's LoRA training infra ~entirely. If M-PSwap's NaN was a recipe-pipeline bug, A and B will hit it too. Day 0 of §4 adds a known-good-baseline pipeline smoke gate before the variants run unattended.

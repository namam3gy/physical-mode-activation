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

| Day | Task |
|---|---|
| 1 | LM-only-swap LoRA training script (`scripts/m_lmswap_train.py`); fork from `m_pswap_train.py`. Adapt for: target LM = Mistral-7B-Instruct-v0.2 (variant B), held-fixed encoder = openai/clip-vit-large-patch14-336, fresh MLP projector. |
| 1 | 50-step smoke (variant B); confirm forward/loss looks reasonable, no NaN. |
| 2 | Streaming `the_cauldron` dataloader (re-use M-PSwap's). Training data budget: ~10K samples (matches M-PSwap; revisit if too small). |
| 2–4 | Full LoRA training of variant B (LoRA rank-32, alpha-64 on q/v/k/o_proj of Mistral; full FT on projector). NaN-abort + grad logging. |
| 4 | Regression eval on variant B (`m_pswap_regression_eval.py` pattern): verify M2 inference works and PMR_nolabel is in a sane range (not 0.0 or 1.0). |
| 5 | M5a runtime steering on variant A (LLaVA-1.5 baseline — already done) + variant B (new). |
| 5 | M5b SAE intervention on variant A (LLaVA-1.5 vis22 SAE — already trained 2026-04-28) + variant B (train fresh SAE on variant B's vis22 activations). |
| 5 | Cross-method analysis + insight doc `docs/insights/m_lmswap.md`. |

If the day-2 smoke runs into the same kind of step-1000 NaN as M-PSwap, the **LoRA training pipeline itself is the suspect**, not perceiver-specific architectural surgery — escalate to advisor before continuing.

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

## 7. Open questions (for next session / advisor)

1. From-scratch on both A and B vs LLaVA-1.5-A + new-train-B asymmetric — which?
2. Use `the_cauldron` (M-PSwap's choice) or switch to LLaVA-1.5's pretrain corpus (LCS-558K) for closer baseline match?
3. M5b SAE — train a fresh SAE per LM variant, or share an encoder-only SAE between A and B? (Encoder is held fixed, so a shared SAE may be principled — but the *features that affect downstream PMR* may be LM-specific to read out.)
4. Should we add a **C variant** (CLIP+Qwen2.5-7B-LM) to span 3 LMs and start probing whether LM family explains the saturation ladder? Adds a week but turns 2-point comparison into a 3-point ladder.

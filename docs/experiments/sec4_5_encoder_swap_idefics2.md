# §4.5 — Cross-encoder swap (Idefics2 third point) (run log, 2026-04-25)

- **Configs**: `configs/encoder_swap_idefics2.py`, `configs/encoder_swap_idefics2_label_free.py`
- **Commands**:
  ```bash
  uv run python scripts/02_run_inference.py --config configs/encoder_swap_idefics2.py
  uv run python scripts/02_run_inference.py --config configs/encoder_swap_idefics2_label_free.py
  uv run python scripts/encoder_swap_analyze.py
  ```
- **Output dirs**: `outputs/encoder_swap_idefics2_*/predictions.{jsonl,parquet,csv}` + `outputs/encoder_swap_summary/encoder_swap_{pmr_nolabel,h7}.csv`
- **Wall clock**: 8 min on GPU 0 (1200 labeled + 400 label-free = 1600 inferences).
- **Stim**: M8a (5 shapes × 4 obj × 2 bg × 2 cue × 1 event × 5 seeds = 400 stimuli), reused from `inputs/m8a_qwen_20260425-091713_8af4836f/`.
- **Sampling**: T=0.7, top_p=0.95, max_new_tokens=96.
- **Deep dive**: `docs/insights/encoder_swap_idefics2.md`.

## Setup motivation

Idefics2-8b = SigLIP-SO400M + Mistral-7B-instruct. The encoder family matches Qwen (SigLIP), the LM differs (Mistral vs Qwen2). This is a near-clean encoder-swap counterfactual relative to LLaVA-1.5 (CLIP + Vicuna).

## 3-model PMR(\_nolabel) comparison on M8a

| Model | Encoder | LM | mean PMR(_nolabel) |
|---|---|---|---:|
| Qwen2.5-VL-7B | SigLIP | Qwen2-7B | 0.838 |
| Idefics2-8b | SigLIP-SO400M | Mistral-7B | **0.882** |
| LLaVA-1.5-7B | CLIP-ViT-L | Vicuna-7B | 0.175 |

H7 strict pre-registration: 1/5 (Qwen) and 1/5 (Idefics2) shapes pass. LLaVA 4/5 pass.

## Headline

**H-encoder-saturation causally confirmed at the encoder-family level.** Encoder type (SigLIP vs CLIP) drives the PMR ceiling regardless of LM family (Qwen2-7B vs Mistral-7B).

## Cross-stim follow-ups (same milestone)

- M8d labeled + label-free with Idefics2 (~8 min): mean PMR(_nolabel) **0.890** matches Qwen 0.869 (vs LLaVA 0.331).
- M8c labeled + label-free with Idefics2 (~3 min): PMR **0.417** vs Qwen 0.550 vs LLaVA 0.283 — all 3 models compress on photos.

(Combined wall clock: 11 min for 2160 additional inferences.)

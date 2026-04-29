# §4.8 — Qwen 7B vs 32B PMR scaling on M2 open prompt (run log, 2026-04-28)

- **Config**: `configs/m2_qwen_32b.py` (open-prompt M2 for Qwen 2.5-VL-32B).
- **Command**:
  ```bash
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/02_run_inference.py \
      --config configs/m2_qwen_32b.py \
      --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3
  uv run python scripts/03_score_and_summarize.py --run-dir outputs/m2_qwen_32b_<ts>
  ```
- **Output dir**: `outputs/m2_qwen_32b_20260427-212653_a167494f/`
  - `predictions.{jsonl,parquet,csv}`, `predictions_scored.csv`, `response_consistency.csv`, `summary_overall.csv`, `summary_by_*.csv`
- **Wall clock**: ~16 min on H200 (1440 inferences, max_new_tokens=96, single-GPU bf16).
- **Stim**: M2 mvp_full (1440 stim × 1 prompt = 1440 inferences).
- **Models compared**: Qwen2.5-VL-7B (existing M2 run) vs Qwen2.5-VL-32B (this run).
- **Deep dive**: `docs/insights/sec4_8_pmr_scaling.md`.

## Headline numbers

| Metric | 7B (existing) | 32B (this run) | Δ |
|---|---:|---:|---:|
| **Aggregate PMR (open)** | 0.931 | **0.926** | −0.005 (≈ noise) |
| `abstract_reject` rate (overall) | 0.002 | **0.065** | **+35 ×** (rare event amplified) |
| `abstract_reject` on `cue=none` | 0.012 | 0.110 | +9× |
| H2 label gap `ball − circle` | +0.071 | +0.010 | **halved** (label effect attenuated by scale) |
| Per-cell `cue=none` PMR | 0.797 | **0.711** | −8.6 pp |

## Headlines

1. **5× scaling does not move overall PMR** (MechBench-style "scale doesn't help grounding" supported).
2. The 32B model is **more cue-sensitive**: when the visual cue is weak (`cue=none`), 32B is more likely to refuse physics-mode (abstract_reject 35× higher than 7B). On the 5 % of cells where the visual prior is weakest, scaling helps grounding.
3. **H2 weakened, not eliminated**: label gap halves but stays positive. Language-prior dominance regime survives but the dependence on label is weaker.

## Reading for paper

Scale-doesn't-help-PMR is the headline; scaling-helps-edge-cases is the nuance. Useful as a baseline for §6.1 (Computational level) — the PMR threshold is *robust to scale*, which is part of why the threshold is real (not just a small-model artifact).

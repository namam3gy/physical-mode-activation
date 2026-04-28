# M-PSwap feasibility spike (2026-04-28)

> **Status**: ✅ spike complete — proceed with full LoRA training (advisor-estimated 1-2 weeks).
> **Track B reference**: `references/submission_plan.md` Pillar B / week 4-5; `references/paper_gaps.md` G3.
> **Goal**: 1-day check on whether Idefics2 perceiver-resampler can be swapped with an MLP projector cleanly. Per advisor 2026-04-28: "verify before committing to week 5 full training."

## Architecture (verified via direct loading)

```
Idefics2Model.connector (= "the connector / projector")
├── modality_projection: Idefics2MLP (1152 → 14336 → 4096)   ← MLP, already in LM-space
└── perceiver_resampler: Idefics2PerceiverResampler           ← cross-attn with learned latents
    └── latents: (64, 4096)                                    ← FIXED 64-token output budget
    └── layers: 3 × Idefics2PerceiverLayer
        ├── self_attn: cross-attention from latents into context
        └── mlp: Idefics2MLP (4096 → 16384 → 4096)
```

**Key insight**: Idefics2 already has an MLP projection (`modality_projection`) before the perceiver-resampler. The perceiver-resampler's role is **cross-attentional resampling** from variable-length context (vision encoder output, after modality_projection) to a fixed 64-token budget the LM consumes.

## Test 1 — Identity bypass (cheapest possible swap)

Replace `perceiver_resampler` with a `BypassResampler` that:
1. Takes context `(batch, seq_len, 4096)` — output of modality_projection
2. Pads or truncates to `(batch, 64, 4096)` to match the LM's expected input shape
3. Returns the result without any cross-attention

### Result: **FAILS — garbage output**

Sample output on `shaded/ground/both ball` stim:

> `, , , I. er, I., I. er, I., I., I., I., I., I., I., I., I., I., I., I., I`

The LM emits punctuation + repeated "I." tokens. **Identity bypass does not produce any meaningful language**, confirming that:

- The 64 fixed token slots the LM consumes are NOT just "any 64 vision-encoder tokens." The LM has been trained on cross-attentionally-aggregated structures.
- Post-modality_projection raw context (variable length, 4096-dim) is *not* in the same statistical distribution as the perceiver-resampler's output.

**Conclusion**: cannot bypass the perceiver via identity. **LoRA-style trainable replacement is required.**

## Implications for M-PSwap

The advisor's original concern stands:

> "Perceiver-resampler is non-trivially integrated into Idefics2's forward pass (cross-attention with learned queries). LoRA-style replacement may require structural surgery, not just LoRA adapters."

The architecture is non-trivial because:
1. **Cross-attention with learned latents** — `latents.shape = (64, 4096)` are trained, not produced from input. Replacing with MLP requires retraining the latents OR replacing the latents-cross-attn pattern entirely.
2. **3 stacked layers** — multi-layer dependency means swap can't be a single linear projection.
3. **Trained jointly with the LM** — LM's input distribution is cross-attentional output. A swapped MLP must produce same statistical pattern after training.

### Required swap design (week 5)

Replace `perceiver_resampler` with an MLP-style projector + learned token reduction:

```python
class MLPResampler(nn.Module):
    def __init__(self, in_dim=4096, out_dim=4096, n_out_tokens=64):
        super().__init__()
        # Token-mixing component: (variable, in_dim) → (n_out_tokens, in_dim)
        # Approach: learned attention pooling (similar to perceiver but 1 layer + MLP)
        self.query = nn.Parameter(torch.randn(n_out_tokens, in_dim) * 0.02)
        self.attn_pool = nn.MultiheadAttention(in_dim, num_heads=8, batch_first=True)
        # Per-token MLP refinement
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4),
            nn.GELU(),
            nn.Linear(in_dim * 4, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, context, attention_mask=None, **kwargs):
        b, s, d = context.shape
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        # Cross-attention pool: queries learn to aggregate context
        pooled, _ = self.attn_pool(q, context, context, key_padding_mask=~attention_mask if attention_mask is not None else None)
        return self.norm(pooled + self.mlp(pooled))
```

Note: this is **not strictly an MLP** — it has a single attention-pool layer. A pure MLP cannot compress variable-length sequences to fixed 64 tokens without some form of learned token mixing. The closest approximation to "LLaVA-style projector" is:
- `LLaVA-1.5`: 2-layer MLP applied PER-TOKEN (preserves token count from CLIP encoder, ~576 patches).
- `Idefics2-MLP variant`: 1-layer attention-pool → MLP → 64 tokens (forced same budget for fair compare).

### Training spec (per `references/paper_gaps.md` G3)

Already pinned in 2026-04-28 advisor-fix commit:
- Dataset: `liuhaotian/LLaVA-Instruct-150K` (10K subsample)
- LoRA rank-32 alpha-64 on `q_proj/v_proj/k_proj/o_proj` (LM-side)
- Replace MLPResampler module entirely (full finetune of the new module, not LoRA — too small to need it)
- AdamW lr=1e-4, batch 32 effective, 5K-10K steps, bf16
- Regression-eval gate: POPE F1 ≥ 0.70 + 50-sample VQA sanity

### Effort estimate (refined)

- Module design: 1 day (above MLPResampler is a starting point)
- Training: 2-5 GPU-hr (10K samples × 5K-10K steps × bf16)
- Regression-eval (POPE + VQA sanity): 1-2 GPU-hr + manual review
- §4.6 + M5b re-run on swapped variant: 4-6 GPU-hr
- **Total: 3-5 GPU days, 1-2 calendar weeks** (matches original advisor estimate; no shortcut available).

## Decision: **GO** for full LoRA training (week 5 of submission_plan)

The bypass shortcut doesn't work, but the architecture is now mapped cleanly:
- Component to swap is a self-contained `Idefics2PerceiverResampler` module (clean replacement boundary).
- Inputs/outputs are well-typed: `(batch, var_seq, 4096)` → `(batch, 64, 4096)`.
- Replacement module spec (above) is clear.
- Training data + regression-eval already pinned.

Recommendation: schedule M-PSwap implementation as planned for week 5 (post-Pillar A wrap). Don't expect a 1-day shortcut.

## Risk register update

| Risk | Pre-spike | Post-spike |
|---|---|---|
| Identity bypass works (1-day) | 30% | **0% (verified fail)** |
| LoRA swap technically feasible (1-2 weeks) | 70% | **90%** (architecture clean, replacement boundary clear) |
| LoRA swap fails to converge / passes regression | 20% | 20% (unchanged — depends on training) |
| Total fall-through to Pillar B B2-only | 50% | 30% (B1 path more concrete now) |

## Files

- This doc: `docs/m_pswap_feasibility.md`
- (No code changes — exploratory only.)
- `references/paper_gaps.md` G3 already has the LoRA training spec.

# M6 r5 — M8c photo encoder probe (4-model, cross-stim)

> **Recap of codes used in this doc** (one-line each; full definitions in `references/roadmap.md` §1.3 + §2)
>
> - **H-encoder-saturation** — Behavioral PMR(_nolabel) saturation on synthetic stim is determined at the architecture level (joint encoder + LM), not encoder representational capacity alone.
> - **M8a** — Stim diversification — non-circle synthetic shapes (square / triangle / hexagon / polygon / wedge × Qwen + LLaVA, labeled + label-free).
> - **M8c** — Stim diversification — real photographs (60 photos × 5 categories from COCO + WikiArt). Photos REDUCE Qwen PMR(_nolabel) 18-48 pp.
> - **M8d** — Stim diversification — non-ball physical-object categories (car / person / bird × abstraction × bg × cue × {fall, horizontal} × seeds).
> - **M9** — Generalization audit — paper Table 1 (3 models × 3 stim sources × bootstrap CIs, 5000 iters); replaces PASS/FAIL binarization with CI separation.
> - **M6 r3** — Idefics2 SigLIP-SO400M probe — vision encoder probe AUC 0.93 closes the encoder-AUC ↔ PMR chain (3-point).
> - **M6 r4** — InternVL3 InternViT probe — AUC 0.89 / PMR 0.92, extends the chain to 4 model points; H-encoder-saturation "non-CLIP-general".
> - **M6 r5** — M8c photo encoder probe (4 models, cross-stim) — behavioral-y AUC inverts but stim-y AUC stays at 1.0 → encoder discriminability is uniform; architecture-level reframe.

**Status**: Complete 2026-04-25.

## Motivation

M9 found that photographs compress the encoder gap behaviorally: PMR(_nolabel)
non-CLIP saturation drops from 0.84–0.92 (synthetic M8a) into 0.28–0.55 (M8c
photos), while CLIP-LLaVA stays low across both. The encoder-side counterpart
was untested. M6 r5 asks:

- Does encoder probe AUC also compress on photos, or stay high while behavior
  collapses?
- With the stim-y reframe from M6 r4 (encoder discriminability is uniform on
  M8a), do photos break that uniformity, or preserve it?

## Method

`scripts/02_run_inference.py --config configs/encoder_swap_internvl3_m8c{,_label_free}.py`
to fill in InternVL3's M8c behavioral run (Qwen/LLaVA/Idefics2 already done in
§4.5 ext).

`scripts/04_capture_vision.py` for all 4 models on the M8c photo stim
(60 photos × 4 layers each). Wall clock per model: ~10–40 s on GPU 0.
Total inference + capture: ~5 min.

`scripts/encoder_swap_probe.py` (behavioral-y mode, per-stim mean PMR across
3 labels) and `scripts/encoder_swap_probe_stim_y.py --target physical_shape_vs_abstract_shape`
(stim-y mode, ball/car/person/bird vs abstract).

## Results

### Behavioral-y probe AUC on M8c photos

| Model      | Encoder         | LM            | M8c PMR(_nolabel) | M8a behavioral-y AUC | M8c behavioral-y AUC (mean) | M8c behavioral-y AUC (deepest) |
|------------|-----------------|---------------|------------------:|---------------------:|----------------------------:|-------------------------------:|
| Qwen2.5-VL | SigLIP          | Qwen2-7B      | **0.550**         | 0.880                | **0.582**                   | 0.438                          |
| LLaVA-1.5  | CLIP-ViT-L      | Vicuna-7B     | **0.283**         | 0.771                | **0.785**                   | 0.856                          |
| Idefics2   | SigLIP-SO400M   | Mistral-7B    | **0.417**         | 0.926                | **0.745**                   | 0.771                          |
| InternVL3  | InternViT       | InternLM2-7B  | **0.533**         | 0.886                | **0.661**                   | 0.585                          |

**The behavioral-y AUC pattern inverts from synthetic to photos.** On M8a
synthetic stim, non-CLIP architectures had higher behavioral-y AUC (Qwen
0.88, Idefics2 0.93, InternVL3 0.89) than CLIP-LLaVA (0.77). On M8c photos:
**LLaVA hits the highest behavioral-y AUC (0.86)**, Qwen drops to **0.44**,
Idefics2 to **0.77**, InternVL3 to **0.59**.

### Stim-y probe AUC on M8c photos (physical_shape_vs_abstract_shape)

| Model      | Stim-y AUC (mean across layers) | Stim-y AUC (deepest layer) |
|------------|--------------------------------:|---------------------------:|
| Qwen2.5-VL | **1.000**                       | 1.000                      |
| LLaVA-1.5  | **0.988**                       | 1.000                      |
| Idefics2   | **0.992**                       | 1.000                      |
| InternVL3  | **0.996**                       | 1.000                      |

**All 4 encoders linearly separate physical-shape photos (ball / car / person /
bird) from abstract photos at AUC ≈ 1.0** — same uniform-encoder-discriminability
finding as M8a (where 3 different stim-y targets all gave AUC = 1.0).

## Interpretation

The two AUC views give a clean joint picture:

1. **Encoder representational capacity is stim-invariant and family-invariant.**
   On both M8a synthetic and M8c photos, every encoder we tested
   (SigLIP / CLIP-ViT-L / SigLIP-SO400M / InternViT) linearly separates
   physical-vs-abstract stim categories at AUC ~ 1.0 with stim-defined y.
2. **Behavioral PMR(_nolabel) is architecture-driven and stim-conditional.**
   Non-CLIP architectures saturate on synthetic stim (0.84–0.92) and
   collapse on photos (0.42–0.55). CLIP-LLaVA stays low on synth (0.18) and
   modestly higher on photos (0.28). Cross-architecture comparison only
   makes sense within a stim source.
3. **The behavioral-y AUC inverts cross-stim because it's a measure of
   "encoder ↔ behavior alignment", not encoder discriminability.**
   - On M8a, non-CLIP encoders' representations strongly co-vary with their
     own saturated behavioral PMR pattern → high behavioral-y AUC.
   - On M8c, non-CLIP behavioral PMR becomes more variable while encoder
     representations stay equally informative → lower behavioral-y AUC.
   - LLaVA's CLIP encoder co-varies with LLaVA's behavioral PMR equally well
     in both regimes (its behavior is variable in both, and CLIP was
     photo-trained → photo-side alignment is natural).

This locks in the M6 r4 reframe at the cross-stim level: the H-encoder-
saturation chain is at the **encoder–LM fusion** level (how the LM consumes
encoder output as physics-mode signal), not at encoder discriminability.

## Hypothesis updates

- **H-encoder-saturation** — *cross-stim confirmation of the M6 r4 reframe*.
  The pattern from synthetic stim (encoder-LM fusion drives behavioral
  saturation on minimal-context stim) holds on photos: encoder
  discriminability stays at AUC ≈ 1.0, while behavioral-y AUC reorganizes
  with each stim's per-model PMR distribution. The paper claim is now
  even sharper: encoders contain physics-relevant information uniformly;
  behavioral physics-mode reading is determined by LM-side fusion, and that
  fusion's strength varies by stim type.
- **Behavioral PMR cross-stim story** (consolidated): non-CLIP architectures
  show "synth-stim minimality saturation" (M8c finding), with both encoder
  contribution AND stim simplicity required for the saturation regime.
  Photos remove the stim-simplicity factor; behavior collapses uniformly.

## Limitations

1. **n=60 photos per model** is small for the behavioral-y probe (n_pos ≈ 20,
   n_neg ≈ 40). AUC variance is high; differences across models on photo
   behavioral-y should not be over-interpreted.
2. **Photos are a particular sample from COCO + WikiArt** — generalization
   to other photo distributions (natural scenes, web images, etc.) is open.
3. **Per-shape / per-category breakdown not done on photos** — only 12
   photos per category, too small for sub-probe AUC.

## Headline figure

(no new figure for this round; numbers feed into encoder_chain_4model.png
caveat: photo-side AUCs differ from M8a-side AUCs in a way that supports
the encoder-LM fusion reframe).

## Roadmap implications

- **§4.5 + M9 + M6 r3 + M6 r4 + M6 r5 = full encoder-saturation story.**
  4 model points × 2 stim types × 2 y modes = 16 AUC cells. Clean
  paper-ready narrative.
- **Optional next**: same-LM encoder swap (Bunny SigLIP+Phi-2 if chat
  template works) for the LM-controlled counterfactual. Round 6 candidate.
- **Optional next**: M8d photo probe (synth non-ball categories don't have
  M8c-style real-photo equivalents yet). Lower priority.

## Artifacts

- `configs/encoder_swap_internvl3_m8c{,_label_free}.py`.
- `outputs/encoder_swap_internvl3_m8c_*/predictions.{jsonl,parquet,csv}`.
- `outputs/encoder_swap_{qwen,llava,idefics2,internvl3}_m8c_vision_activations/*.safetensors`.
- `outputs/encoder_swap_{qwen,llava,idefics2,internvl3}_m8c_probe/{layer_sweep,by_object_level,by_shape}.csv`.
- `outputs/encoder_swap_{qwen,llava,idefics2,internvl3}_m8c_probe_stim_y/layer_sweep_stim_y_physical_shape_vs_abstract_shape.csv`.
- `docs/insights/m6_r5_m8c_photo_probe.md` (+ `_ko.md`).

# M-MP MCQ Phase 1 smoke summary

Pre-committed parse-rate gate for Phase 2: ≥85% per model (advisor 2026-04-28).


## Headline (per model)

| Model | n | PMR | Parse rate | circle | ball | planet | Δ ball−circle | Δ planet−circle | Phase 2 gate |
|---|---|---|---|---|---|---|---|---|---|
| Qwen | 1440 | 0.208 | 1.000 | 0.290 | 0.240 | 0.094 | -0.050 | -0.196 | ✅ PASS |
| LLaVA-1.5 | 1440 | 0.394 | 1.000 | 0.194 | 0.525 | 0.465 | +0.331 | +0.271 | ✅ PASS |
| LLaVA-Next | 1440 | 0.033 | 1.000 | 0.000 | 0.021 | 0.077 | +0.021 | +0.077 | ✅ PASS |
| Idefics2 | 1440 | 0.867 | 1.000 | 0.798 | 0.890 | 0.915 | +0.092 | +0.117 | ✅ PASS |
| InternVL3 | 1440 | 0.553 | 1.000 | 0.412 | 0.627 | 0.619 | +0.215 | +0.206 | ✅ PASS |

## Cell variation (representative cells)

| Cell | Qwen | LLaVA-1.5 | LLaVA-Next | Idefics2 | InternVL3 |
|---|---|---|---|---|---|
| line/blank/none | 0.00 | 0.13 | 0.00 | 0.00 | 0.00 |
| line/ground/cast_shadow | 0.00 | 0.17 | 0.00 | 0.17 | 0.10 |
| textured/blank/none | 0.00 | 0.33 | 0.00 | 0.37 | 0.50 |
| textured/ground/cast_shadow | 0.20 | 0.47 | 0.03 | 1.00 | 0.77 |
| shaded/ground/both | 0.57 | 0.40 | 0.03 | 1.00 | 1.00 |

**Decision**: 5/5 models pass the parse-rate gate. Proceed to Phase 2.

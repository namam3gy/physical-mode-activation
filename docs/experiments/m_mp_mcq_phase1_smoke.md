# M-MP MCQ Phase 1 smoke summary

Pre-committed parse-rate gate for Phase 2: ≥85% per model (advisor 2026-04-28).


## Headline (per model)

| Model | n | PMR | Parse rate | circle | ball | planet | Δ ball−circle | Δ planet−circle | Phase 2 gate |
|---|---|---|---|---|---|---|---|---|---|
| Qwen | 144 | 0.257 | 1.000 | 0.333 | 0.354 | 0.083 | +0.021 | -0.250 | ✅ PASS |
| LLaVA-1.5 | 144 | 0.389 | 1.000 | 0.125 | 0.583 | 0.458 | +0.458 | +0.333 | ✅ PASS |
| LLaVA-Next | 144 | 0.035 | 1.000 | 0.000 | 0.042 | 0.062 | +0.042 | +0.062 | ✅ PASS |
| Idefics2 | 144 | 0.875 | 1.000 | 0.792 | 0.896 | 0.938 | +0.104 | +0.146 | ✅ PASS |
| InternVL3 | 144 | 0.528 | 1.000 | 0.333 | 0.604 | 0.646 | +0.271 | +0.313 | ✅ PASS |

## Cell variation (representative cells)

| Cell | Qwen | LLaVA-1.5 | LLaVA-Next | Idefics2 | InternVL3 |
|---|---|---|---|---|---|
| line/blank/none | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| line/ground/cast_shadow | 0.00 | 0.67 | 0.00 | 0.33 | 0.00 |
| textured/blank/none | 0.00 | 0.67 | 0.00 | 0.33 | 0.33 |
| textured/ground/cast_shadow | 0.00 | 0.33 | 0.33 | 1.00 | 1.00 |
| shaded/ground/both | 0.33 | 0.33 | 0.00 | 1.00 | 1.00 |

**Decision**: 5/5 models pass the parse-rate gate. Proceed to Phase 2.

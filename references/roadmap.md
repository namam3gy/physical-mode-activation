# ROADMAP — Physical-Mode Activation

> **Role of this document.** Single source of truth for "where we are now and what's next." When starting a new session, **read this file first**. Update §3 every time a milestone completes. Detailed material is linked out to the relevant doc / code.
>
> - Research philosophy & hypotheses (canonical): `references/project.md`
> - Architecture: `docs/architecture.md`
> - Stimulus spec: `docs/stimulus_spec.md` / Scoring rubric: `docs/scoring_rubric.md`
> - Run history: `docs/experiments/m{1,2,...}_*.md`
> - Code-level next-step entry points: `docs/next_steps.md`
> - Per-milestone insight deep dives: `docs/insights/m{1,3,4,5}_*.md`

---

## 1. Research definition

### 1.1 Central question

**Above what visual-cue threshold does an open-source VLM stop processing an abstract shape (a circle) as geometry and start processing it as a physical object (a ball)?**

Measured at two layers:

- **Behavior**: PMR (physics-mode priming rate) / GAR (gravity-align rate) / RC (response consistency) of next-state-prediction prompt responses.
- **Mechanism**: linear probe AUC of vision-encoder activations, layer-wise logit-lens trajectories in the LM backbone, causal bottleneck layers / heads revealed by activation patching.

### 1.2 Sub-task structure (per `references/project.md` §2)

| # | Title | Content | Input | Output |
|---|---|---|---|---|
| ST1 | **PhysCue** behavioral thresholds | 4-5-axis factorial stimulus + next-state-prediction prompt | programmatic / photo stimuli | PMR/GAR/RC tables, per-factor curves |
| ST2 | Vision-encoder probing | linear probe + Gandelsman head decomposition + SAE on CLIP / SigLIP / InternViT activations | ST1 captured vision activations | layer×head AUC, monosemantic features |
| ST3 | LM backbone layer-wise emergence | logit lens + per-layer probes at visual-token positions (Neo et al. 2024 recipe) | ST1 captured LM activations | layer×token heatmap, switching-layer |
| ST4 | Causal localization | Semantic Image Pairs + activation patching + VTI steering + SAE intervention | pilot pairs | IE curve, steering vector, head ranking |
| ST5 | Cross-model + prompt-steering | same factorial on LLaVA-1.5 / LLaVA-Next / Qwen2-VL / InternVL2 + prompt steering (Gavrikov et al. 2024) | extended EvalConfig | model-comparison tables, prompt-bias curves |

### 1.3 Hypothesis scorecard

Original H1-H3 from `references/project.md` §2.2 plus H4-H7 derived during the pilot and MVP-full milestones. Pilot evidence in `docs/insights/m1_pilot.md`.

| ID | Hypothesis | Status (post-M5a-ext recheck) | Evidence / next test |
|---|---|---|---|
| **H1** | PMR rises S-shaped along the abstraction axis (line → textured); 3D shading and ground introduction produce the largest step-changes. | **supported** | M2: monotone across all 4 object_levels (0.744 → 0.790 → 0.822 → 0.832). T=0.7 + 10 seeds resolved the pilot's mid-tie. |
| **H2** | The "ball" label substantially raises PMR even on line drawings → independent contribution of the language prior. | **revised** | M2 +15 pp "ball vs circle" gap is `circle suppression`, not `ball enhancement`. Paired vs label-free baseline (M4b, 2026-04-25): `PMR(ball) − PMR(_nolabel) = +0.006`, `PMR(planet) − PMR(_nolabel) = +0.006`, `PMR(circle) − PMR(_nolabel) = −0.065`. Language prior is *asymmetric*: ball ≈ no-label (visual default), circle is an abstract override, planet adds orbit prior only on abstract images. |
| **H3** | Scene inconsistency degrades RC. | **untested** | Axis E was dropped from M2 (complexity); reserved for a focused mini-experiment. RC infrastructure was validated in M2 (103/288 cells with RC<1). |
| **H4** (pilot-derived) | The open vs forced-choice PMR gap is a stable signature of the **language-prior ↔ visual-evidence** conflict. | **supported — extended** | M2: gap present at every object_level (line 32 pp → textured 22 pp). Higher abstraction ⇒ larger gap — a structural prediction that abstraction weakens visual evidence so the language prior dominates more. Next test: ST5 cross-model. |
| **H5** (pilot-derived) | The single ground line causes a **larger** PMR shift than going from no-ground textured ball to with-ground textured ball. | **mixed** | M2: bg delta (blank 0.67 → scene 0.88 = +21 pp) > object delta (line 0.74 → textured 0.83 = +9 pp). Direction matches; however, scene also surpasses ground. |
| **H6** (pilot-derived) | The arrow+shadow cue saturation is driven entirely by **cast shadow alone**; the arrow is closer to annotation. | **supported (revised)** | M2 decomposition: cast_shadow alone = +17.5 pp above none (Kersten ground-attachment cue confirmed); **but the arrow alone also saturates at 0.96** — partially refutes the "arrow = annotation" sub-claim. Arrow is the dominant cue, shadow is secondary. |
| **H7** (M2-derived) | The label does not toggle PMR — it selects **which physics regime** to apply. | **supported but narrower** | M2 GAR: ball 0.79 / circle 0.70 / planet 0.48. M5a-ext Exp 2: label flip @ +α=40 swaps B vs A on `line/blank/none`. M5a-ext Exp 3 qualifier: on `textured/blank/none`, label-only flip fails (+α=40 → A regardless of label); regime is chosen by joint (image, label, α sign). |
| **H-boomerang** | Encoder knows, decoder gates: vision encoder linearly separates physics-mode classes even where behavior fails. | **supported + causal** | M3: encoder AUC = 1.00 on every factorial axis at every probed layer; behavior 0.28-0.95. M4: information preserved through LM (AUC 0.94-0.95). M5a: causal intervention at L10 flips behavior. |
| **H-locus** (M4-derived) | The bottleneck is at the LM final layers + decoding head, not earlier. | **supported (early-mid sweet spot)** | M5a: L10 α=40 flips 10/10 abstract → physical responses; later layers do not move. M5a-ext Exp 3: L10 regime-flip (A vs B by α sign) holds in all tested cells. Aligns with the Basu et al. 2024 early-layer constraint-storage finding. |
| **H-direction-bidirectional** (M5a-ext, 2026-04-24; revised 2026-04-25) | `v_L10` is a simple bidirectional concept axis where −α suppresses physics-mode back to abstract. | **revised — regime axis within physics-mode** | Exp 1 (textured/ground/both ceiling): −α has no effect → initially framed as "one-way activator". Exp 3 (textured/blank/none moderate baseline, 2026-04-25): −α=40 flips D → B ("stays still") uniformly across (line, textured) × (ball, circle). Both signs of α activate physics-mode; sign selects regime (+kinetic / −static). Baseline D sits *below* the \|α\| threshold, not at one end of the axis. |
| **H-regime** (M5a-derived) | The steering direction is binary "object-ness", not "which physics" — physics regime is label-driven. | **refuted in current form** | Replaced by H-direction-bidirectional's regime-axis interpretation (kinetic vs static is already a regime distinction that the steering sign selects, label-independent at |α|=40). Separately, label *does* select regime in the narrow `line/blank/none × +α=40` case (Exp 2), but not globally — this is now folded into the H7 qualifier. |

### 1.4 Target models & venue

- **Round 1 (pilot / MVP-full)**: Qwen/Qwen2.5-VL-7B-Instruct — proven loader, 15 GB, 1.0 it/s on H200.
- **ST5 extension**: LLaVA-1.5-7B, LLaVA-Next-7B, InternVL2-8B, (stretch) Qwen2-VL-7B (for layer-index alignment with the models named in `references/project.md` §2.4).
- **Venue**: EMNLP long (grounding-failure / language-prior-dominance angle) primary; NeurIPS main (ST3-4 mechanistic localization) stretch.

---

## 2. Milestone overview

| # | Milestone | Scope | Status | Completed |
|---|---|---|---|---|
| M0 | Infrastructure scaffold | Package layout, configs, scripts, tests, base docs set | ✅ | 2026-04-24 |
| M1 | **ST1 Pilot** (Qwen2.5-VL-7B) | 240 stim × 2 prompts = 480 inferences; first behavioral S-curve measurement | ✅ | 2026-04-24 |
| M2 | **ST1 MVP-full** (incorporating pilot lessons) | axis C decomposition, axis D expansion, T=0.7, LM hidden-state capture, 2880 inferences | ✅ | 2026-04-24 |
| M3 | **ST2 — Vision encoder probing** | Vision-block capture (8 layers, 12 GB) + layer-wise linear probes. **Boomerang confirmed**: encoder AUC = 1.0 on every axis; behavioral PMR 0.28-0.95. | ✅ | 2026-04-24 |
| M4 | **ST3 — LM logit lens / layer-wise probe** | LM hidden @ visual tokens AUC 0.94-0.95 across all probed layers; L20 peak. Label prior drives physics margin from L5; object_level effect is 7× smaller. | ✅ | 2026-04-24 |
| M5a | **ST4 Phase 1+2 — VTI steering** | Direction extraction + residual-stream injection. **L10 α=40 flips 10/10 D → B** — "physical object-ness" direction causally confirmed. | ✅ | 2026-04-24 |
| M5a-ext | **VTI follow-ups (neg α, label swap, bidirectionality recheck)** | Exp 1-2 (2026-04-24): neg α at ceiling + label=ball side-by-side. Exp 3 (2026-04-25): (α × label × obj) grid on moderate baseline. **Key result**: `v_L10` is a regime axis within physics-mode — +α → A (falls), −α → B (stays still), baseline D below threshold. | ✅ | 2026-04-25 |
| M4b | **Label-free prompt — H2 null test** | `open_no_label` variant on M2 stimuli. **Key result**: `ball` ≈ no-label; `circle` suppresses PMR by 6.5 pp. Original H2 reframed: language prior is asymmetric — circle override, not ball enhancement. M4 visual-token capture is prompt-independent (structural artefact). | ✅ | 2026-04-25 |
| **M5b** | **ST4 Phase 3 — SIP + patching + SAE** | Semantic Image Pairs + activation patching (needs attention re-capture) + SAE feature decomposition. | ▶ **next (optional)** | — |
| M6 | ST5 — Cross-model sweep | LLaVA-1.5/Next, InternVL2, (optional) Qwen2-VL | pending | — |
| M7 | Human baseline + paper writing | Prolific 20 raters × 50 stimuli + EMNLP/NeurIPS draft | optional | — |

---

## 3. Detailed status

### M0 — Infrastructure scaffold ✅ (2026-04-24)

Completed:
- `src/physical_mode/` modules (config, utils, stimuli, models, inference, metrics, probing scaffold).
- `scripts/0{1,2,3}_*.py` argparse runners.
- `configs/{pilot,mvp_full}.py` — config-as-code.
- `tests/` — 35 tests (stimulus determinism + PMR scoring regression).
- `docs/` (architecture / stimulus_spec / scoring_rubric / run-log / next-steps / insights).
- `notebooks/demo.ipynb` — 32-cell walkthrough with cached outputs.
- `CLAUDE.md`, `README.md`, `pyproject.toml` (cu130 index), `.gitignore`, `uv.lock`.
- Project repo: private https://github.com/namam3gy/physical-mode-activation.

Success criteria (all met):
- `uv sync` succeeds; `uv run python -m pytest` passes.
- `scripts/01_generate_stimuli.py --config configs/pilot.py --limit 10` succeeds.
- Pilot inference + score pipeline runs end-to-end.

### M1 — ST1 Pilot (Qwen2.5-VL-7B) ✅ (2026-04-24)

Run: `uv run python scripts/02_run_inference.py --config configs/pilot.py`.
Output: `outputs/pilot_20260424-072418_2c16efb6/` — 480 predictions, 8 min wall clock.

**Headline findings** (`docs/insights/m1_pilot.md` §2, §3):

| Observation | Number | Implication |
|---|---|---|
| Ground effect | blank 0.49 → ground 0.85 (+36 pp) | **Largest single-factor effect.** Ground is the cheapest, strongest physics trigger. |
| Abstraction endpoints | line 0.58 → textured 0.81 | H1 partially supported; middle two levels tied. |
| Arrow + shadow cue | 1.000 (saturated) | Unmeasurable cell — needs decomposition in MVP-full. |
| Wind cue | 0.513 ≈ none 0.500 | Invisible to the VLM — needs replacement. |
| Open vs forced-choice | PMR 0.80 vs 0.54, abstract_reject 0.00 vs 0.45 | **Language-prior dominance** — H2 strongly supported. |

**Hypothesis scoring**: H1 partial, H2 strong, H3 untested, H4-H6 candidates derived.

**Validated infrastructure properties**:
- `PhysModeVLM` works on Qwen2.5-VL with the generic `AutoModelForImageTextToText` loader. ST5 model swap is a config-only change.
- `predictions.jsonl` streaming flush is crash-safe.
- Among factorial axes, **event_template** has no behavioral output effect → downgrade in MVP-full.

### M2 — ST1 MVP-full ✅ (2026-04-24)

Run: `uv run python scripts/02_run_inference.py --config configs/mvp_full.py`.
Output: `outputs/mvp_full_20260424-094103_8ae1fa3d/` — 2880 predictions, 55 min wall clock, 5.2 GB of LM activations.

**Success-criteria results** (details in `docs/experiments/m2_mvp_full.md`):

| Criterion | Status |
|---|---|
| Monotone S-curve over object_level (forced-choice) | ✅ 0.583 < 0.647 < 0.711 < 0.714 |
| Open-vs-forced gap at every object_level | ✅ 22-32 pp, positive correlation with abstraction |
| cast_shadow alone > none + 20 pp | ✅ +17.5 pp on average (+23 in the blank-bg condition) |
| RC < 1 cells exist (T>0 verification) | ✅ 103/288 (36 %) cells with RC<1 |
| `outputs/*/activations/` populated | ✅ 480 safetensors, 5 layers, bf16 hidden states |

**New headlines**:
1. Abstraction-axis monotone S-curve cleanly confirmed (H1).
2. Label selects the physics regime — same image with `circle/ball/planet` → static / rolls / orbits the Sun (H7, new).
3. cast_shadow alone = +17.5 pp; arrow is the dominant cue (H6 revised).
4. Open-ended PMR 0.93, abstract_reject 0.002 (3/1440) — language-prior dominance reconfirmed and extended (H4).

**Blocking-issue resolution**:
- The motion-trail primitive ended up unused (the axis-C redesign replaced wind entirely).
- Axis E (scene consistency) was dropped from M2; deferred to a focused mini-experiment in `docs/next_steps.md`.
- The `capture_lm_attentions=False` flag was the right call — disk usage is ⅓ of the alternative (5.2 GB vs 15+ GB with attentions on).

### M3 — ST2 Vision encoder probing ✅ (2026-04-24)

Run: `uv run python scripts/04_capture_vision.py --stimulus-dir inputs/mvp_full_... --output-dir outputs/mvp_full_.../vision_activations --layers 3,7,11,15,19,23,27,31`
Output: `outputs/mvp_full_20260424-094103_8ae1fa3d/vision_activations/` (480 safetensors, 12 GB) + `probing_vision/*.csv`.
Deep dive: `docs/insights/m3_encoder_boomerang.md`.

**Key findings** (details in `docs/experiments/m3_encoder_probing.md`):

- **Encoder AUC = 1.00 on every factorial axis from layer 3 onward.** The vision encoder perfectly encodes bg / object / cue. No information bottleneck.
- **Behavioral forced-choice PMR ranges 0.28 (cue=none) to 0.95 (both).** The LM gating produces the gap.
- **Controlled no-cue subset (120 stimuli)**: encoder AUC 0.89 vs behavioral PMR 0.28 → the encoder knows which cells will trigger physics-mode but the LM only lets a fraction through.
- **Per-object-level encoder AUC ~0.95 constant while behavior 0.58-0.71**: gap is largest at line (most abstract) at +36 pp — internal-mechanism evidence for H4 (higher abstraction ⇒ stronger language prior).

**Hypothesis updates**:
- H-boomerang (originally §1.4 of the project doc, "encoder knows, decoder doesn't"): **supported (saturated evidence)**.
- H4 and H6 both gain mechanism-level evidence.

**Blocking-issue resolution / wins**:
- `PhysModeVLM.capture()` now has vision hooks (`_resolve_vision_blocks` helper covers Qwen / LLaVA / InternVL).
- Methodological caveat recorded: programmatic stimuli make encoder AUC 1.0 trivial. A photorealistic stimulus extension is needed for validation — see `docs/next_steps.md`.

### M4 — ST3 LM backbone logit lens / layer-wise probing ✅ (2026-04-24)

Run: `uv run python scripts/05_lm_probing.py --run-dir outputs/mvp_full_20260424-094103_8ae1fa3d`.
Output: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_lm/*.{csv,parquet}`.
Deep dive: `docs/insights/m4_logit_lens.md`.

**Key results**:
- LM per-layer probe AUC (forced-choice PMR) = **0.94-0.95 across all layers**, peak L20 = 0.953.
- Logit lens: physics logit > geometry logit from L5 onward because the "ball" label primes the LM.
- Object_level effect (L25 line 3.76 vs textured 4.35, margin +0.6) is **~14 % of the label effect (+4.0 flat shift)**.
- The switching-layer metric is degenerate under label-primed prompts (all = L5) → §4.9 "label-free prompt" promoted to a pre-M5 mini-experiment.

**Boomerang precise location**: vision encoder (0.94-1.0) → LM hidden (0.95) preserves information. ~29 pp accuracy loss at decoding. ST4 intervention priority: LM final layers + logit head.

### M5a — ST4 Phase 1+2 VTI steering ✅ (2026-04-24)

Run:
- Phase 1: inline Python driver, `compute_steering_vectors` from `src/physical_mode/probing/steering.py`.
- Phase 2: `uv run python scripts/06_vti_steering.py --run-dir outputs/mvp_full_... --test-subset line/blank/none --label circle --layers 10,15,20,25 --alphas 0,5,10,20,40`.

Output: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/` (vectors) + `steering_experiments/` (intervention predictions). Deep dive: `docs/insights/m5_vti_steering.md`.

**Key results**:
- Steering vectors `v_L = mean(h | PMR=1) − mean(h | PMR=0)`. Norm grows 5× through the LM (L5: 5.9 → L25: 31).
- Projection at L20 aligns with the factorial cue axis (none 22.3 → both 42.7).
- **Layer 10 α=40 injection flips 10/10 `line/blank/none` responses from "D: abstract" to "B: stays still"**. L15/L20/L25 do not flip at the same α.
- The intervention causes an "abstract → physical object" binary shift, not gravity. Goes to "B: stays" rather than "A: falls" → direction is object-ness, not gravity. Consistent with H7 / H-regime.

**Hypothesis updates**:
- H-boomerang: extended + **causally supported**.
- H-locus: **supported (early-mid layer L10)**.
- H-regime (new): **candidate** — steering direction is coarse "object-ness", regime selection is label-driven.

### M5a-ext — VTI follow-ups ✅ (2026-04-24, 2026-04-25)

Run: `uv run python scripts/06_vti_steering.py` with `--output-subdir` flag
to partition sub-experiments within the same M2 output tree.

Output: `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/{neg_alpha_textured_ground_both, ball_line_blank_none, bidirectional_recheck_*}/`.
Deep dive: `docs/insights/m5a_ext_bidirection_and_label.md`. Numbers: `docs/experiments/m5a_ext_neg_alpha_and_label.md`.

**Key results**:
- Exp 1 (ceiling): `-α · v_L10` at `textured/ground/both × circle` leaves the
  first-letter distribution at 10/10 A. Interpretation was initially "one-way
  activator"; recheck (Exp 3) revealed this was a ceiling artifact.
- Exp 2 (label swap): `+α=40 · v_L10` at `line/blank/none × ball` flips 10/10
  to A ("falls") vs 10/10 B ("stays still") under label=`circle` (M5a).
  Label-driven regime flip causally demonstrated.
- Exp 3 (bidirectional recheck, 2026-04-25): full (α × label × obj) grid on
  `{line, textured} × blank × none`. **New finding**: `-α=40` flips 10/10 to
  B ("stays still") **uniformly** across all four (obj × label) cells. `v_L10`
  is therefore a regime axis within physics-mode (+α kinetic, −α static), not
  a physics-vs-abstract activator. Baseline D sits below the |α| activation
  threshold, not at one axis endpoint.
- Exp 3 qualifier on H7: +α=40 on `textured/blank/none` gives A regardless of
  label; label-only regime flipping fails as soon as the image carries a
  physical-object signal. Regime is chosen by a joint (image, label, α sign)
  function.

**Hypothesis updates**:
- H-direction-bidirectional (new): **revised 2026-04-25** — regime axis
  interpretation replaces the earlier "one-way activator" framing.
- H-regime: **refuted in its original form** — the label-only regime flip
  does not generalize; subsumed by H-direction-bidirectional + the H7
  qualifier.
- H-locus: **unchanged (reinforced)** — L10 regime-flip holds across all
  four Exp 3 cells.

### M4b — Label-free prompt H2 null test ✅ (2026-04-25)

Run: `uv run python scripts/02_run_inference.py --config configs/label_free.py --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3` followed by `scripts/03_score_and_summarize.py` and `scripts/05_lm_probing.py --sources open_no_label`.

Output: `outputs/label_free_20260425-031430_315c5318/` — 480 predictions + 480 activation safetensors. Deep dive: `docs/insights/m4b_label_free.md`. Numbers: `docs/experiments/m4b_label_free.md`.

**Key results**:
- Paired PMR delta vs label-free baseline (480 matched seeds): ball +0.006,
  planet +0.006, **circle −0.065**. The "ball vs circle" gap reported in M2
  is actually circle suppression, not ball enhancement.
- Per-cell structure: circle suppresses more on abstract images
  (line: −9.2 pp; filled: −4.2 pp); `motion_arrow` cue overrides circle
  suppression entirely (+0.000); `none` cue gives the largest suppression
  (−15.0 pp).
- `line/blank/none` 4-label table cleanly separates label contributions:
  ball (regime shift, kinetic→static), circle (full suppression, PMR 0.40
  → 0.10), planet (+30 pp PMR — only label that genuinely *adds* physics
  over the visual default, due to orbital prior).
- M4 re-run on label-free activations reproduces M2's physics-margin table
  bit-for-bit, confirming visual-token captures are prompt-independent
  (image tokens precede question text under causal attention). The
  collapsed switching-layer at L5 is a structural artefact of the capture
  point, not evidence of label-independent LM commitment.

**Hypothesis updates**:
- H2: **revised** — ball ≈ no-label; circle is a suppressive override.
  Per-label contributions are asymmetric.
- H-boomerang: **reinforced** — visual-token hidden states are
  prompt-independent, so the L5 physics bias is image-only.
- H-locus: **unchanged** — label's behavioral effect localizes downstream
  of visual-token positions, consistent with M5a's L10 efficacy on the
  image-preceding trajectory.
- H4: **refined** — circle suppression strength scales with image
  abstraction, the image-side dual of the abstraction → language-prior-gap
  scaling.

### M5b — ST4 Phase 3 (SIP patching + SAE) — work plan

**Sub-tasks**:
1. Re-capture activations with `capture_lm_attentions=True` on a focused subset (~120 stimuli covering single-axis-differ pairs). Disk: ~15 GB additional.
2. Construct Semantic Image Pairs from the M2 factorial — single-axis-differ pairs (e.g. `shaded_ground_none` vs `line_ground_none` on the same seed). Emit `sip_manifest.parquet`.
3. **Activation patching** (TransformerLens or raw PyTorch hooks):
   - Capture clean and corrupted forward passes.
   - Layer sweep: replace each layer's visual-token activations from corrupted to clean; measure PMR-probability recovery (indirect effect).
4. **Attention knockout**: zero out specific heads' visual-to-last-token attention; measure PMR delta.
5. **VTI ablation**: `v_layer = mean(h_clean) − mean(h_corrupted)` (per-pair version of M5a's class-mean approach). Add at test time. Confirm or extend M5a's L10 finding.
6. **SAE** (stretch): train SAE on vision-encoder activations; identify monosemantic "shading" / "ground" features + intervention.

**Headline-claim candidate**: "Knockout of LLaVA layer 19 head 14 drops PMR by 50 pp; preserving only this head while removing all other visual attention preserves the effect" (project doc §3.2 sentence template).

### M6 — ST5 Cross-model sweep

Tasks:
1. `configs/cross_model.py` — list of `model_id`s + same factorial.
2. Modify `scripts/02_run_inference.py` to iterate over a model list.
3. LLaVA-1.5-7B (~13 GB), LLaVA-Next-7B (~14 GB), InternVL2-8B (~16 GB), (stretch) Qwen2-VL-7B (~15 GB). Total download ~60 GB.
4. Behavioral table per model + (where possible) reduced ST3/4 versions.
5. **Prompt steering**: use `system_prompt_override` with `"treat this as an abstract geometric shape"` vs `"treat this as a physical object subject to gravity"` → measure PMR shift.

**Hypothesis**: the ground effect (H5) replicates across all open-source VLMs; the open-vs-forced gap (H4) varies in magnitude across models.

### M7 — Human baseline (optional) + paper

- 20 Prolific raters × 50 stimuli (random subset) × open-ended prompt.
- Per-cell alignment between human PMR and VLM PMR.
- EMNLP long (primary) / NeurIPS main (stretch) draft.

---

## 4. Additional ideas not in the original project doc

Extensions that came up during the pilot, or that aren't in `references/project.md` §2. Optional — each ~1-2 weeks of work.

### 4.1 Block stack as a separate "abstract-physical" path

The code (`primitives.py::_draw_block_stack`) exists but the pilot never used it. Blocks are an "abstract geometry + clearly physical" combination, asking a **different question from the circle-ball axis**: "given an abstract shape but a physical configuration (stacking), which way does the VLM go?" → expected: high PMR + low abstract_reject. Useful as a control on the circle-ball axis.

### 4.2 Reverse prompting

What happens to PMR when an `"abstract diagram"` label is attached to a *real* photograph of a ball? A counterfactual for H4 (language-prior dominance). 1-hour experiment.

### 4.3 Label language switching

Does a Korean `"공"` vs an English `"ball"` on the same stimulus produce different PMR? Qwen2.5-VL is multilingual → measure per-language prior strength. Extends the language-grounding narrative from project doc §3.

### 4.4 Video frame pair → Michotte-style causality

Give a (t=0, t=1) frame pair where only the object position differs, then ask "launched by X?". Does the Michotte (1946) launching effect appear in VLMs? A 2-image prompt is a proxy that does not require a video model.

### 4.5 Cross-encoder swap (SigLIP vs CLIP vs DINOv2)

Hypothesis: "cues that are invisible when the encoder is CLIP can be seen by a DINOv2-based model". Continuation of the Eyes Wide Shut (Tong et al. 2024) MoF proposal. Note: a standalone-encoder comparison is implicitly part of M6 (LLaVA-1.5 with CLIP-ViT-L/14 vs Qwen2.5-VL with SigLIP).

### 4.6 Activation-based counterfactual stimulus generation

Use a SAE / VTI steering vector in reverse to gradient-ascent synthesize a stimulus that "maximizes physics-mode in the VLM's eyes". An **adversarial physics-mode prompt** → evidence for shortcut interpretation in open-source VLMs.

### 4.7 Decision-consistency boundary measurement

Pilot couldn't measure RC because T=0 made it degenerate. Reinterpret M2's RC (under T=0.7) as **per-axis decision stability**: which cue stabilizes decisions? Expected: ground + shaded + fall → high RC (uniform answer); line + blank + none → mid RC (consistently stationary); borderline cells have low RC.

### 4.8 PMR scaling

Per-model PMR for H-class (Qwen2.5-VL-7B/32B/72B) and LLaVA-1.5-7B/13B. Does MechBench (Zhang et al. 2024)'s "scale doesn't help" claim hold for PMR? Strong interpretability implications for H6.

### 4.9 Label-free prompt

`"What do you see? What might happen next?"` — ask the question **without** the word "ball". Measures H2's language-prior contribution as a null-hypothesis test. Easy addition — `prompts.py` `open_no_label` variant.

### 4.10 Attention visualization UI

Captured attentions → interactive heatmap (notebook-based). Per-stimulus, per-layer, per-head visual-token attention. For the paper appendix figure.

### 4.11 H7 follow-up — label-regime category annotation

Systematically validate the M2 finding that "label selects the physics regime" (circle → static / ball → rolls / planet → orbits). Hand-annotate or zero-shot-classify open-ended responses into 5 categories (gravity-fall / gravity-roll / orbital / inertial / static) → axis D × category distribution as a confusion matrix. Verify whether the per-label GAR difference (ball 0.79 / planet 0.48) is really a categorical split of "which physics is invoked".

---

## 5. Reference rules during work

**At the start of every new session** (future Claude or user):

1. Read this file (`references/roadmap.md`) first to identify "which milestone are we on?".
2. Check the current milestone's **success criteria** and **blocking issues**.
3. Detailed technical questions: drill down `docs/architecture.md` → `docs/next_steps.md` in order.
4. For the latest experimental results: `docs/experiments/m{N}_*.md` (numbers) + `docs/insights/m{N}_*.md` (per-milestone deep dive).

**Insights file convention**: every major milestone produces one `docs/insights/m{N}_<slug>.md` written in English (with a `_ko.md` Korean translation). Include: link to the raw numbers, interpretation, hypothesis-scorecard updates, and paper implications. Current tree:
- `docs/insights/m1_pilot.md` — M1 pilot (originally `docs/05_insights.md`)
- `docs/insights/m3_encoder_boomerang.md` — M3 encoder boomerang
- `docs/insights/m4_logit_lens.md` — M4 LM logit lens
- `docs/insights/m4b_label_free.md` — M4b label-free prompt H2 null test
- `docs/insights/m5_vti_steering.md` — M5a VTI steering causal intervention
- `docs/insights/m5a_ext_bidirection_and_label.md` — M5a extensions (negative α, label × steering, bidirectionality recheck)
- (M5b, M6 ... to be added)

**Bilingual file convention**: `references/project.md`, `references/roadmap.md`, every `docs/insights/*.md`, and the various `docs/*.md` reference docs all have an English canonical `*.md` and a Korean translation `*_ko.md`. English is authoritative; if translations drift, English wins. New content → write English first, then translate to Korean.

**Whenever a milestone completes**:

- Update §2 status column (▶ → ✅) and record the completion date.
- Add "verified facts / blocker resolution / new hypothesis" to the milestone's §3 section.
- If a new hypothesis emerges, add an `H*` row to the §1.3 scorecard.
- Add a per-run entry to `docs/experiments/m{N}_<slug>.md` (one file per run, in addition to this file).

**When a hypothesis is refuted or revised**:

- Change the status in §1.3 with a one-line reason.
- Major revisions go in this ROADMAP rather than `references/project.md` (the canonical project doc is read-only spec).

**When a new idea emerges**:

- Add it numbered to §4 immediately. Decide later which milestone to slot it into.

---

## 6. Change log

| Date | Change | Commit |
|---|---|---|
| 2026-04-24 | First draft — M0/M1 complete, M2 prepared | `23171b6` |
| 2026-04-24 | M2 complete: scorecard updates (H1 → supported, H2 → quantified, H4 → supported, H5 → mixed, H6 → supported-revised, H7 new); M3 set as next milestone; H7 follow-up added to §4. | `1d17252` |
| 2026-04-24 | M3 complete: vision encoder probing — boomerang confirmed (encoder AUC = 1.0 / behavioral 0.28-0.95); M4 set as next milestone. | `1205821` |
| 2026-04-24 | M4 complete: LM logit lens + per-layer probe. LM AUC 0.94-0.95 across all layers (peak L20 = 0.953); label drives physics margin from L5; M5 set as next milestone. | `2abdc32` |
| 2026-04-24 | M5a complete (VTI steering): L10 α=40 flips 10/10 of `line/blank/none` from D (abstract) to B (physical-static). "Object-ness" direction causally confirmed. M5b (SIP+SAE) and M6 still to do. | `61ffd29` |
| 2026-04-24 | Repository restructure: `references/`, `docs/{insights,experiments,figures}/` scheme; everything bilingual (English canonical + `_ko.md` translation). | `963e219` |
| 2026-04-24 | M5a-ext Exp 1+2 complete: negative α at ceiling (null result — later found to be a ceiling artifact) + label=ball swap on line/blank/none (clean B→A flip). H-direction-bidirectional added (initially as "one-way activator"), H-regime upgraded to supported. | `9a0ed86` (merge) |
| 2026-04-25 | M5a-ext Exp 3 (bidirectionality recheck on `textured/blank/none` moderate baseline): −α=40 → 10 B uniformly across (line/textured) × (ball/circle). H-direction-bidirectional revised to "regime axis within physics-mode" (+α kinetic, −α static, baseline D below threshold). H-regime refuted in original form and narrowed to an H7 qualifier. | `f8f0fdd` |
| 2026-04-25 | M4b complete: label-free prompt as H2 null test on M2 stimuli. Paired PMR(ball) − PMR(_nolabel) = +0.006 ≈ 0; PMR(circle) − PMR(_nolabel) = −0.065. **H2 revised** — language prior is asymmetric (circle override, not ball enhancement). M4 visual-token capture is prompt-independent (causal-attention artefact); switching-layer collapse is structural. | (this commit) |

# physical-mode-activation

When does an open-source VLM stop treating a circle as geometry and start
treating it as a physical object? Sub-task 1 of the research program in
`references/project.md`: controlled factorial stimuli + behavioral PMR/GAR/RC
metrics on Qwen2.5-VL-7B, plus mechanistic probes (M3-M5) of the encoder /
LM activations.

Documentation is bilingual — English canonical, Korean translation as `*_ko.md`.

- **Roadmap & milestone status**: `references/roadmap.md`
- **Research definition**: `references/project.md`
- **Architecture & module map**: `docs/architecture.md`
- **Factorial axes + stimulus rendering**: `docs/stimulus_spec.md`
- **Scoring rubric**: `docs/scoring_rubric.md`
- **Per-milestone insights** (deep dives): `docs/insights/m{1,3,4,5}_*.md`
- **Per-milestone run logs** (numbers only): `docs/experiments/m{1,2,3,4,5}_*.md`
- **Code-level next steps**: `docs/next_steps.md`
- **Walkthrough notebook**: `notebooks/demo.ipynb`

## Quick start

```bash
uv sync
uv run python -m pytest
uv run python scripts/01_generate_stimuli.py --config configs/pilot.py
uv run python scripts/02_run_inference.py --config configs/pilot.py --limit 5   # smoke
uv run python scripts/02_run_inference.py --config configs/pilot.py             # full
uv run python scripts/03_score_and_summarize.py --run-dir outputs/<run_id>

# Deeper analysis (after a run with capture_lm_layers set, e.g. configs/mvp_full.py).
uv run python scripts/04_capture_vision.py --stimulus-dir inputs/<run> --output-dir outputs/<run>/vision_activations --layers 3,7,11,15,19,23,27,31
uv run python scripts/05_lm_probing.py --run-dir outputs/<run>
uv run python scripts/06_vti_steering.py --run-dir outputs/<run> --stimulus-dir inputs/<run> --test-subset line/blank/none --label circle --layers 10,15,20,25 --alphas 0,5,10,20,40

# Open the demo notebook.
uv run jupyter lab notebooks/demo.ipynb
```

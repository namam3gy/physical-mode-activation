# physical-mode-activation

When does an open-source VLM stop treating a circle as geometry and start
treating it as a physical object? Sub-task 1 of the research program in
`research_plan.md`: controlled factorial stimuli + behavioral PMR/GAR/RC
metrics on Qwen2.5-VL-7B.

- Architecture & module map: `docs/00_architecture.md`
- Factorial axes + stimulus rendering: `docs/01_stimulus_spec.md`
- Scoring rubric: `docs/02_scoring_rubric.md`
- Run history: `docs/03_run_log.md`
- Follow-up plan (Sub-tasks 2-5): `docs/04_next_steps.md`
- **Walkthrough notebook** (stimuli → inference → scoring → pilot plots): `notebooks/demo.ipynb`

## Quick start

```bash
uv sync
uv run python -m pytest
uv run python scripts/01_generate_stimuli.py --config configs/pilot.py
uv run python scripts/02_run_inference.py --config configs/pilot.py --limit 5   # smoke
uv run python scripts/02_run_inference.py --config configs/pilot.py             # full
uv run python scripts/03_score_and_summarize.py --run-dir outputs/<run_id>

# Open the demo notebook.
uv run jupyter lab notebooks/demo.ipynb
```

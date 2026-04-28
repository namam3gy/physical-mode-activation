"""M-MP — multi-prompt PMR summary across 5 models × 3 prompts.

Consumes the multi_prompt_*_<timestamp>/ output dirs (or any explicit list
of paths). Produces:
  - Per-(model, prompt, label) PMR table.
  - Per-(model, prompt) headline PMR (label-averaged).
  - Cross-prompt H2 paired-deltas (per-model: PMR_open(ball) − PMR_open(circle), same for describe + yesno).
  - Cell-variation table (5 models × N representative cells × 3 prompts).
  - Markdown export to docs/experiments/m_mp_phase2.md (or specified path).

Usage:
  uv run python scripts/m_mp_summarize.py auto
    → auto-discovers latest multi_prompt_<model>_*/ for each of the 5 models.

  uv run python scripts/m_mp_summarize.py paths \\
    --qwen outputs/multi_prompt_qwen_<id> \\
    --llava outputs/multi_prompt_llava_<id> \\
    --llava-next outputs/multi_prompt_llava_next_<id> \\
    --idefics2 outputs/multi_prompt_idefics2_<id> \\
    --internvl3 outputs/multi_prompt_internvl3_<id> \\
    [--md-out docs/experiments/m_mp_phase2.md]

Scorers (per `references/paper_gaps.md` G1):
  - open prompt: score_pmr (existing kinetic-prediction scorer).
  - describe_scene: score_describe (M-MP lexicon, physics-tokens-win-over-abstract).
  - meta_phys_yesno: score_meta_yesno (yes/no parse; -1 → unparseable).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physical_mode.metrics.pmr import score_describe, score_meta_yesno, score_pmr  # noqa: E402


MODEL_NAMES = ("Qwen", "LLaVA-1.5", "LLaVA-Next", "Idefics2", "InternVL3")
MODEL_DIR_KEY = {
    "Qwen": "multi_prompt_qwen",
    "LLaVA-1.5": "multi_prompt_llava",
    "LLaVA-Next": "multi_prompt_llava_next",
    "Idefics2": "multi_prompt_idefics2",
    "InternVL3": "multi_prompt_internvl3",
}

REPRESENTATIVE_CELLS = [
    ("line", "blank", "none"),       # most abstract
    ("line", "ground", "cast_shadow"),
    ("filled", "blank", "none"),
    ("filled", "ground", "cast_shadow"),
    ("textured", "blank", "none"),
    ("textured", "ground", "cast_shadow"),
    ("shaded", "ground", "both"),    # most physics
]


def _score_all(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score_open"] = out["raw_text"].apply(score_pmr)
    out["score_describe"] = out["raw_text"].apply(score_describe)
    out["score_yesno"] = out["raw_text"].apply(score_meta_yesno)
    return out


def _resolve_pmr_for_prompt(row) -> int:
    if row["prompt_variant"] == "open":
        return int(row["score_open"])
    if row["prompt_variant"] == "describe_scene":
        return int(row["score_describe"])
    if row["prompt_variant"] == "meta_phys_yesno":
        return int(row["score_yesno"]) if row["score_yesno"] != -1 else 0
    return 0


def _load_model(model: str, path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path / "predictions.parquet")
    df["model"] = model
    df = _score_all(df)
    df["pmr"] = df.apply(_resolve_pmr_for_prompt, axis=1)
    return df


def auto_discover() -> dict[str, Path]:
    outputs = Path("outputs")
    if not outputs.is_dir():
        raise SystemExit("outputs/ not found")
    found = {}
    for model, key in MODEL_DIR_KEY.items():
        cands = sorted(outputs.glob(f"{key}_*"), key=lambda p: p.name, reverse=True)
        # Require Phase 2-sized output (at least 480 stim — could be 432 for Phase 1 smoke).
        for c in cands:
            try:
                meta = pd.read_parquet(c / "predictions.parquet")
                # Phase 2 = 4320 rows. Phase 1 smoke = 432 rows.
                if len(meta) >= 4320:
                    found[model] = c
                    break
            except Exception:
                continue
        if model not in found:
            print(f"WARNING: no Phase 2 output dir found for {model}; using latest available")
            for c in cands:
                if (c / "predictions.parquet").exists():
                    found[model] = c
                    break
    return found


def make_summary(model_paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    all_dfs = []
    for model in MODEL_NAMES:
        if model not in model_paths:
            print(f"SKIP {model}: no path provided / discovered")
            continue
        df = _load_model(model, model_paths[model])
        all_dfs.append(df)

    big = pd.concat(all_dfs, ignore_index=True)

    # Per-(model, prompt, label) PMR
    by_mpl = big.groupby(["model", "prompt_variant", "label"]).pmr.mean().reset_index()
    by_mpl["pmr"] = by_mpl["pmr"].round(3)

    # Per-(model, prompt) headline (label-averaged)
    by_mp = big.groupby(["model", "prompt_variant"]).pmr.mean().reset_index()
    by_mp["pmr"] = by_mp["pmr"].round(3)

    # Cross-prompt H2 paired-delta: PMR(ball) − PMR(circle) per (model, prompt)
    pivot = by_mpl.pivot_table(index=["model", "prompt_variant"], columns="label", values="pmr").reset_index()
    pivot["h2_ball_minus_circle"] = (pivot["ball"] - pivot["circle"]).round(3)
    pivot["h2_planet_minus_circle"] = (pivot["planet"] - pivot["circle"]).round(3)

    # Yes/No parse rate (only for meta_phys_yesno)
    yesno = big[big.prompt_variant == "meta_phys_yesno"].copy()
    yesno_unparse = yesno.groupby("model").apply(
        lambda g: (g["score_yesno"] == -1).sum()
    ).rename("unparseable").reset_index()

    # Cell-variation: representative cells × 5 models × 3 prompts
    cell_rows = []
    for cell in REPRESENTATIVE_CELLS:
        sub = big[(big.object_level == cell[0]) & (big.bg_level == cell[1]) & (big.cue_level == cell[2])]
        for model in big.model.unique():
            mod_sub = sub[sub.model == model]
            row = {
                "cell": f"{cell[0]}/{cell[1]}/{cell[2]}",
                "model": model,
            }
            for prompt in ("open", "describe_scene", "meta_phys_yesno"):
                p = mod_sub[mod_sub.prompt_variant == prompt]
                row[prompt] = round(p.pmr.mean(), 3) if len(p) > 0 else float("nan")
            cell_rows.append(row)
    cell_df = pd.DataFrame(cell_rows)

    return {
        "by_model_prompt_label": by_mpl,
        "by_model_prompt": by_mp,
        "h2_paired_delta": pivot[["model", "prompt_variant", "circle", "ball", "planet",
                                  "h2_ball_minus_circle", "h2_planet_minus_circle"]],
        "yesno_unparseable": yesno_unparse,
        "cell_variation": cell_df,
        "raw_combined": big,
    }


def render_markdown(summary: dict[str, pd.DataFrame], out_path: Path | None) -> str:
    lines = []
    lines.append("# M-MP — Phase 2 multi-prompt PMR summary\n")
    lines.append(f"_Auto-generated by `scripts/m_mp_summarize.py`._\n")

    lines.append("## Per-(model, prompt) PMR (label-averaged)\n")
    pivot = summary["by_model_prompt"].pivot(index="model", columns="prompt_variant", values="pmr").reset_index()
    pivot = pivot[["model", "open", "describe_scene", "meta_phys_yesno"]]
    lines.append(pivot.to_markdown(index=False))
    lines.append("")

    lines.append("## H2 paired-delta per (model, prompt)\n")
    lines.append("Δ ball−circle and Δ planet−circle. Reveals whether each prompt preserves the M2 H2 ordering.\n")
    h2 = summary["h2_paired_delta"]
    lines.append(h2.to_markdown(index=False))
    lines.append("")

    lines.append("## yes/no parser rate\n")
    lines.append(summary["yesno_unparseable"].to_markdown(index=False))
    lines.append("")

    lines.append("## Cell-variation (representative cells × 5 models × 3 prompts)\n")
    lines.append("PMR per cell, by model, for each prompt.\n")
    cell = summary["cell_variation"]
    lines.append(cell.to_markdown(index=False))
    lines.append("")

    text = "\n".join(lines)
    if out_path:
        out_path.write_text(text)
        print(f"Wrote markdown summary to {out_path}")
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    p_auto = sub.add_parser("auto", help="Auto-discover latest Phase 2 outputs.")
    p_auto.add_argument("--md-out", type=Path, default=None,
                        help="Markdown output path (default: print to stdout)")
    p_paths = sub.add_parser("paths", help="Explicit per-model paths.")
    for m in MODEL_NAMES:
        flag = "--" + m.lower().replace(".", "").replace("-", "-")
        p_paths.add_argument(flag, type=Path, default=None)
    p_paths.add_argument("--md-out", type=Path, default=None)
    args = p.parse_args()

    if args.cmd == "auto":
        paths = auto_discover()
    else:
        paths = {}
        for m in MODEL_NAMES:
            attr = m.lower().replace(".", "").replace("-", "_")
            v = getattr(args, attr, None)
            if v:
                paths[m] = v

    print(f"Found {len(paths)} model(s):", {k: str(v) for k, v in paths.items()})
    summary = make_summary(paths)
    md = render_markdown(summary, args.md_out)
    if not args.md_out:
        print(md)


if __name__ == "__main__":
    main()

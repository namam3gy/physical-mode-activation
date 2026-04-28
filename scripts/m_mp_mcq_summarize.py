"""M-MP MCQ — categorical-MCQ PMR summary across 5 models (audit follow-up).

Companion to `m_mp_summarize.py`. Loads `mcq_<model>_<timestamp>/` outputs
(produced from `configs/mcq_<model>.py`) and computes:

  - Per-(model, label) PMR + parse rate.
  - H2 paired-delta (ball − circle, planet − circle) per model.
  - MCQ-vs-yesno comparison (uses the most recent multi_prompt_<model>_*
    Phase 2 yesno run for the same model when available).
  - Pre-committed parse-rate threshold gate (≥85% per model).

Usage:
  uv run python scripts/m_mp_mcq_summarize.py auto

  uv run python scripts/m_mp_mcq_summarize.py paths \\
    --qwen outputs/mcq_qwen_<id> \\
    --llava outputs/mcq_llava_<id> \\
    --llava-next outputs/mcq_llava_next_<id> \\
    --idefics2 outputs/mcq_idefics2_<id> \\
    --internvl3 outputs/mcq_internvl3_<id>

Scorer: score_meta_phys_mcq (A/B/C/D parse → A=1, BCD=0, unparseable=-1).
The `parse_rate` column reports % rows where raw_score != -1.

Per audit (`docs/insights/review_audit_2026-04-28.md` follow-up #8): this
analysis is load-bearing for the Qwen "generative-vs-categorical"
dissociation claim. If MCQ behaves like yesno (low PMR, no flip on M5a),
format is irrelevant; if MCQ behaves like describe (high PMR, flips on
M5a), format is the boundary. Both outcomes paper-defensible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physical_mode.metrics.pmr import score_meta_phys_mcq  # noqa: E402


MODEL_NAMES = ("Qwen", "LLaVA-1.5", "LLaVA-Next", "Idefics2", "InternVL3")
MCQ_DIR_KEY = {
    "Qwen": "mcq_qwen",
    "LLaVA-1.5": "mcq_llava",
    "LLaVA-Next": "mcq_llava_next",
    "Idefics2": "mcq_idefics2",
    "InternVL3": "mcq_internvl3",
}

PARSE_RATE_THRESHOLD = 0.85  # advisor 2026-04-28: pre-committed Phase 2 gate


def _score_mcq(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score_mcq_raw"] = out["raw_text"].apply(score_meta_phys_mcq)
    out["pmr"] = out["score_mcq_raw"].apply(lambda s: int(s) if s != -1 else 0)
    out["parseable"] = (out["score_mcq_raw"] != -1).astype(int)
    return out


def _load_model(model: str, path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path / "predictions.parquet")
    df["model"] = model
    return _score_mcq(df)


def auto_discover() -> dict[str, Path]:
    outputs = Path("outputs")
    if not outputs.is_dir():
        raise SystemExit("outputs/ not found — run from project root")
    import re
    found = {}
    for model, key in MCQ_DIR_KEY.items():
        # Match `<key>_<timestamp_digits>...`. Forbids longer prefixes like
        # `mcq_llava_next_*` slipping through when key='mcq_llava' (the suffix
        # after `<key>_` must start with a digit, i.e. the timestamp).
        pattern = re.compile(rf"^{re.escape(key)}_\d")
        cands = sorted(
            (p for p in outputs.iterdir() if p.is_dir() and pattern.match(p.name)),
            key=lambda p: p.name,
            reverse=True,
        )
        for c in cands:
            if (c / "predictions.parquet").exists():
                found[model] = c
                break
        if model not in found:
            print(f"WARNING: no mcq output dir found for {model}")
    return found


def headline(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(model) headline: PMR + parse rate + per-label PMR + H2 paired-deltas."""
    rows = []
    for model in MODEL_NAMES:
        sub = df[df["model"] == model]
        if len(sub) == 0:
            continue
        row = {
            "model": model,
            "n": len(sub),
            "pmr": float(sub["pmr"].mean()),
            "parse_rate": float(sub["parseable"].mean()),
        }
        for label in ("circle", "ball", "planet"):
            l = sub[sub["label"] == label]
            row[f"pmr_{label}"] = float(l["pmr"].mean()) if len(l) else float("nan")
        row["delta_ball_circle"] = row["pmr_ball"] - row["pmr_circle"]
        row["delta_planet_circle"] = row["pmr_planet"] - row["pmr_circle"]
        row["passes_parse_gate"] = row["parse_rate"] >= PARSE_RATE_THRESHOLD
        rows.append(row)
    return pd.DataFrame(rows)


def cell_variation(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, cell) PMR for representative cells."""
    cells = [
        ("line", "blank", "none"),
        ("line", "ground", "cast_shadow"),
        ("textured", "blank", "none"),
        ("textured", "ground", "cast_shadow"),
        ("shaded", "ground", "both"),
    ]
    rows = []
    for model in MODEL_NAMES:
        sub = df[df["model"] == model]
        if len(sub) == 0:
            continue
        for obj, bg, cue in cells:
            cell = sub[
                (sub["object_level"] == obj)
                & (sub["bg_level"] == bg)
                & (sub["cue_level"] == cue)
            ]
            rows.append({
                "model": model,
                "cell": f"{obj}/{bg}/{cue}",
                "n": len(cell),
                "pmr": float(cell["pmr"].mean()) if len(cell) else float("nan"),
            })
    return pd.DataFrame(rows)


def format_md(head: pd.DataFrame, cells: pd.DataFrame) -> str:
    lines = ["# M-MP MCQ Phase 1 smoke summary\n"]
    lines.append(
        "Pre-committed parse-rate gate for Phase 2: "
        f"≥{PARSE_RATE_THRESHOLD:.0%} per model (advisor 2026-04-28).\n"
    )

    lines.append("\n## Headline (per model)\n")
    lines.append(
        "| Model | n | PMR | Parse rate | "
        "circle | ball | planet | Δ ball−circle | Δ planet−circle | Phase 2 gate |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in head.iterrows():
        gate = "✅ PASS" if r["passes_parse_gate"] else "❌ FAIL"
        lines.append(
            f"| {r['model']} | {r['n']} | {r['pmr']:.3f} | {r['parse_rate']:.3f} | "
            f"{r['pmr_circle']:.3f} | {r['pmr_ball']:.3f} | {r['pmr_planet']:.3f} | "
            f"{r['delta_ball_circle']:+.3f} | {r['delta_planet_circle']:+.3f} | {gate} |"
        )

    lines.append("\n## Cell variation (representative cells)\n")
    lines.append("| Cell | " + " | ".join(MODEL_NAMES) + " |")
    lines.append("|---|" + "|".join(["---"] * len(MODEL_NAMES)) + "|")
    for cell in cells["cell"].unique():
        sub = cells[cells["cell"] == cell].set_index("model")["pmr"]
        row = [cell]
        for m in MODEL_NAMES:
            v = sub.get(m, float("nan"))
            row.append(f"{v:.2f}" if pd.notna(v) else "—")
        lines.append("| " + " | ".join(row) + " |")

    n_pass = int(head["passes_parse_gate"].sum())
    n_total = len(head)
    decision = (
        f"\n**Decision**: {n_pass}/{n_total} models pass the parse-rate gate. "
        + ("Proceed to Phase 2." if n_pass == n_total else "Revise prompt or note caveat per model.")
    )
    lines.append(decision)
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    auto_p = sub.add_parser("auto", help="auto-discover latest mcq_<model> dirs")
    auto_p.add_argument("--md-out", type=Path, default=None)
    paths_p = sub.add_parser("paths", help="provide explicit paths")
    paths_p.add_argument("--qwen", type=Path, required=True)
    paths_p.add_argument("--llava", type=Path, required=True)
    paths_p.add_argument("--llava-next", type=Path, required=True)
    paths_p.add_argument("--idefics2", type=Path, required=True)
    paths_p.add_argument("--internvl3", type=Path, required=True)
    paths_p.add_argument("--md-out", type=Path, default=None)
    args = p.parse_args()

    if args.cmd == "auto":
        paths = auto_discover()
    else:
        paths = {
            "Qwen": args.qwen,
            "LLaVA-1.5": args.llava,
            "LLaVA-Next": args.llava_next,
            "Idefics2": args.idefics2,
            "InternVL3": args.internvl3,
        }

    dfs = []
    for model, path in paths.items():
        if not (path / "predictions.parquet").exists():
            print(f"SKIP {model}: predictions.parquet not found at {path}")
            continue
        dfs.append(_load_model(model, path))

    if not dfs:
        raise SystemExit("No model outputs found")

    df_all = pd.concat(dfs, ignore_index=True)
    head = headline(df_all)
    cells = cell_variation(df_all)

    print(format_md(head, cells))

    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(format_md(head, cells))
        print(f"\nWrote: {args.md_out}")


if __name__ == "__main__":
    main()

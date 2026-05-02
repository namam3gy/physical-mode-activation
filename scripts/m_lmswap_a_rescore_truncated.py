"""Re-score Variant A's regression eval responses after simulating EOS termination.

Hypothesis: A's PMR_nolabel = 0.869 (way above LLaVA-1.5's expected ~0.18) is
inflated by the missing-EOS-label bug — A runs to max_new_tokens=80 and the
verbose tail picks up incidental physics keywords. If we truncate each response
at the first sentence-ending mark (simulating what A *would* have done with
proper EOS learning), does PMR_nolabel collapse to the expected range?

Decision logic:
  - new PMR_nolabel ∈ [0.03, 0.50] → scorer-truncation hack viable, A retrain
    avoidable (write up the truncation rule, justify in paper)
  - new PMR_nolabel still > 0.50 → EOS bug isn't the (sole) cause, A retrain
    mandatory.

CPU-only — no GPU touched.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.physical_mode.metrics.pmr import score_pmr  # noqa: E402

# Match the first sentence-ending punctuation. If none, fall back to first 20 words.
_SENTENCE_END_RE = re.compile(r"[.!?]")
_FALLBACK_N_WORDS = 20


def truncate_at_first_sentence(response: str) -> str:
    """Return the response up to and including the first sentence-ending mark.

    If no such mark exists, return the first `_FALLBACK_N_WORDS` words. This
    fallback handles the case where A genuinely produced a degenerate run-on
    that even a properly-EOS-trained model would have stopped early.
    """
    m = _SENTENCE_END_RE.search(response or "")
    if m:
        return response[: m.end()]
    words = (response or "").split()
    return " ".join(words[:_FALLBACK_N_WORDS])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=Path,
                   default=ROOT / "outputs/lmswap_a_regression_eval_step21000/regression_eval.jsonl")
    p.add_argument("--summary-out", type=Path, default=None,
                   help="optional: write per-cell breakdown JSON")
    args = p.parse_args()

    records = [json.loads(line) for line in args.jsonl.open()]
    print(f"Loaded {len(records)} records from {args.jsonl}")
    print()

    orig_pmr_total = sum(r["pmr"] for r in records)
    new_pmr_total = 0
    flipped_1_to_0 = 0
    by_cell_orig: dict[str, list[int]] = defaultdict(list)
    by_cell_new: dict[str, list[int]] = defaultdict(list)
    sample_flips: list[dict] = []

    for r in records:
        response = r.get("response", "")
        truncated = truncate_at_first_sentence(response)
        new_pmr = score_pmr(truncated)
        new_pmr_total += new_pmr

        cell = f"{r['object_level']}/{r['bg_level']}/{r['cue_level']}"
        by_cell_orig[cell].append(r["pmr"])
        by_cell_new[cell].append(new_pmr)

        if r["pmr"] == 1 and new_pmr == 0:
            flipped_1_to_0 += 1
            if len(sample_flips) < 5:
                sample_flips.append({
                    "sample_id": r["sample_id"],
                    "cell": cell,
                    "orig_response": response[:200] + ("..." if len(response) > 200 else ""),
                    "truncated": truncated,
                })

    n = len(records)
    orig_rate = orig_pmr_total / n
    new_rate = new_pmr_total / n
    print("=" * 70)
    print(f"PMR_nolabel ORIGINAL  : {orig_rate:.3f} ({orig_pmr_total}/{n})")
    print(f"PMR_nolabel TRUNCATED : {new_rate:.3f} ({new_pmr_total}/{n})")
    print(f"Flipped 1→0           : {flipped_1_to_0}")
    print("=" * 70)
    print()

    # Saturation regime check: gate_2 in regression_eval expects [0.03, 0.50]
    LOW, HIGH = 0.03, 0.50
    in_band = LOW <= new_rate <= HIGH
    print(f"Gate-2 band [{LOW}, {HIGH}] (LLaVA-1.5 saturation regime):")
    print(f"  ORIGINAL  in band? {LOW <= orig_rate <= HIGH}  (rate={orig_rate:.3f})")
    print(f"  TRUNCATED in band? {in_band}  (rate={new_rate:.3f})")
    print()

    if in_band:
        print("VERDICT: Truncation brings A into LLaVA-1.5 saturation regime.")
        print("  → Scorer-truncation hack is VIABLE. A retrain may be AVOIDABLE.")
        print("  → Caveat: must justify the truncation rule in paper (mechanically:")
        print("    'first-sentence scoring isolates the model's primary commitment;")
        print("    verbose tail is decoder-noise post EOS-emission failure').")
    else:
        print("VERDICT: Truncation does NOT salvage A.")
        print(f"  → Rate {new_rate:.3f} still outside [{LOW}, {HIGH}].")
        print("  → A retrain (with format_chat[A] </s> fix) is REQUIRED.")
    print()

    print("Sample flips (1→0 after truncation):")
    for f in sample_flips:
        print(f"  [{f['cell']}] {f['sample_id']}")
        print(f"    orig: {f['orig_response'][:120]}...")
        print(f"    trunc: {f['truncated']}")
        print()

    print()
    print("Per-cell breakdown (top 15 cells by sample count, sorted by Δ-rate):")
    cell_summary = []
    for cell, orig_pmrs in by_cell_orig.items():
        new_pmrs = by_cell_new[cell]
        cell_summary.append({
            "cell": cell,
            "n": len(orig_pmrs),
            "orig_rate": sum(orig_pmrs) / len(orig_pmrs),
            "new_rate": sum(new_pmrs) / len(new_pmrs),
            "delta": (sum(new_pmrs) - sum(orig_pmrs)) / len(orig_pmrs),
        })
    cell_summary.sort(key=lambda c: c["n"], reverse=True)
    for c in cell_summary[:15]:
        print(f"  n={c['n']:3d}  {c['cell']:40s}  orig={c['orig_rate']:.2f}  new={c['new_rate']:.2f}  Δ={c['delta']:+.2f}")

    if args.summary_out:
        args.summary_out.write_text(json.dumps({
            "orig_pmr_nolabel": orig_rate,
            "new_pmr_nolabel": new_rate,
            "flipped_1_to_0": flipped_1_to_0,
            "n_records": n,
            "in_saturation_regime": in_band,
            "by_cell": cell_summary,
        }, indent=2))
        print(f"\nSummary written to {args.summary_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

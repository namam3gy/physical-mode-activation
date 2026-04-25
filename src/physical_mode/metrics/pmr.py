"""Physics-Mode Priming Rate (PMR), Gravity-Align Rate (GAR), Response Consistency (RC)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import pandas as pd

from .lexicons import (
    ABSTRACT_MARKERS,
    DOWN_DIRECTION_PHRASES,
    HOLD_STILL_STEMS,
    PHYSICS_VERB_STEMS,
)


_WORD_RE = re.compile(r"[A-Za-z]+")


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _any_stem_hit(words: Iterable[str], stems: frozenset[str]) -> bool:
    for w in words:
        for stem in stems:
            if w.startswith(stem):
                return True
    return False


def _any_phrase_hit(text: str, phrases: frozenset[str]) -> bool:
    t = (text or "").lower()
    return any(p in t for p in phrases)


def score_pmr(text: str) -> int:
    """1 if the response contains a physics verb and no explicit abstract-rejection; else 0.

    The abstract-rejection check prevents "this is just a circle — it won't move"
    from counting as physical because of "move".
    """
    if not text:
        return 0
    if _any_phrase_hit(text, ABSTRACT_MARKERS):
        return 0
    words = _words(text)
    if _any_stem_hit(words, PHYSICS_VERB_STEMS):
        return 1
    return 0


def score_hold_still(text: str) -> int:
    if not text:
        return 0
    words = _words(text)
    return int(_any_stem_hit(words, HOLD_STILL_STEMS))


def score_gar(text: str, event_template: str, bg_level: str) -> int | None:
    """Gravity-Align Rate: does the model predict *downward* motion?

    Defined only when a ground exists (bg_level in {ground, scene}) AND the
    event template expects gravity to be salient (fall). Returns None
    for cells where GAR is undefined.
    """
    if bg_level not in ("ground", "scene"):
        return None
    if event_template not in ("fall", "roll_slope"):
        return None
    if not text:
        return 0
    return int(_any_phrase_hit(text, DOWN_DIRECTION_PHRASES))


def score_abstract_reject(text: str) -> int:
    return int(_any_phrase_hit(text or "", ABSTRACT_MARKERS))


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def score_rows(df: pd.DataFrame, response_col: str = "raw_text") -> pd.DataFrame:
    """Add pmr / gar / hold_still / abstract_reject columns."""
    out = df.copy()
    out["pmr"] = out[response_col].map(score_pmr)
    out["hold_still"] = out[response_col].map(score_hold_still)
    out["abstract_reject"] = out[response_col].map(score_abstract_reject)
    out["gar"] = [
        score_gar(t, e, b)
        for t, e, b in zip(out[response_col], out["event_template"], out["bg_level"])
    ]
    return out


def summarize(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Grouped summaries — overall and by each factorial axis."""
    out: dict[str, pd.DataFrame] = {}
    out["overall"] = _agg(df, group_cols=[])
    for col in ("object_level", "bg_level", "cue_level", "event_template", "label", "prompt_variant", "shape"):
        if col in df.columns:
            out[f"by_{col}"] = _agg(df, group_cols=[col])
    return out


def _agg(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if not group_cols:
        g = df
        base = pd.DataFrame([{"n": len(g)}])
    else:
        g = df.groupby(group_cols, dropna=False)
        base = g.size().rename("n").reset_index()
    # PMR / hold_still / abstract_reject are binary; gar may be None.
    def _mean(col):
        if group_cols:
            return df.groupby(group_cols, dropna=False)[col].mean().reset_index()
        return pd.DataFrame([{col: df[col].mean() if col in df else float("nan")}])

    res = base
    for col in ("pmr", "hold_still", "abstract_reject"):
        if col not in df.columns:
            continue
        m = _mean(col)
        res = res.merge(m, on=group_cols) if group_cols else pd.concat([res, m], axis=1)
    # GAR: mean over non-null rows.
    if "gar" in df.columns:
        gar_df = df.dropna(subset=["gar"])
        if group_cols:
            gm = gar_df.groupby(group_cols, dropna=False)["gar"].mean().reset_index()
            res = res.merge(gm, on=group_cols, how="left")
        else:
            res["gar"] = gar_df["gar"].mean() if len(gar_df) else float("nan")
    return res


def response_consistency(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """RC = fraction of the majority PMR outcome within a factorial cell.

    Groups by the given columns (typically all factors except seed) and
    computes, for each cell, the most-common pmr value's share.
    """
    def _rc(s):
        if len(s) == 0:
            return float("nan")
        c = Counter(s.tolist())
        return max(c.values()) / len(s)

    return (
        df.groupby(group_cols, dropna=False)["pmr"]
        .apply(_rc)
        .rename("rc")
        .reset_index()
    )

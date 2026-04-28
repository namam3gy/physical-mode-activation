"""Physics-Mode Priming Rate (PMR), Gravity-Align Rate (GAR), Response Consistency (RC)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import pandas as pd

from .lexicons import (
    ABSTRACT_MARKERS,
    CATEGORY_REGIME_KEYWORDS,
    CHINESE_PHYSICS_VERB_STEMS,
    DOWN_DIRECTION_PHRASES,
    HOLD_STILL_STEMS,
    JAPANESE_ABSTRACT_MARKERS,
    JAPANESE_PHYSICS_VERB_STEMS,
    KOREAN_ABSTRACT_MARKERS,
    KOREAN_PHYSICS_VERB_STEMS,
    PHYSICS_VERB_STEMS,
    UNIVERSAL_KINETIC_STEMS,
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

    Korean / Japanese fallback: when a model emits non-ASCII responses
    (observed in §4.3 cross-model runs), the English `_words` tokenizer
    returns nothing. The Korean / Japanese substring matches catch those
    responses so they aren't silently scored 0.
    """
    if not text:
        return 0
    if _any_phrase_hit(text, ABSTRACT_MARKERS):
        return 0
    if _any_phrase_hit(text, KOREAN_ABSTRACT_MARKERS):
        return 0
    if _any_phrase_hit(text, JAPANESE_ABSTRACT_MARKERS):
        return 0
    words = _words(text)
    if _any_stem_hit(words, PHYSICS_VERB_STEMS):
        return 1
    if _any_phrase_hit(text, KOREAN_PHYSICS_VERB_STEMS):
        return 1
    if _any_phrase_hit(text, JAPANESE_PHYSICS_VERB_STEMS):
        return 1
    if _any_phrase_hit(text, CHINESE_PHYSICS_VERB_STEMS):
        return 1
    return 0


_YES_TOKENS: frozenset[str] = frozenset({"yes", "y", "yeah", "yep", "yup"})
_NO_TOKENS: frozenset[str] = frozenset({"no", "n", "nope"})
_LEADING_PREFIX_RE = re.compile(r"^\s*(?:answer|response|a)\s*[:\-]\s*", re.IGNORECASE)


def score_meta_yesno(text: str) -> int:
    """1 if the response begins with 'yes', 0 if 'no', -1 if unparseable.

    Used for the `meta_phys_yesno` prompt: "Is this a depiction of a real-world
    physical event? Answer 'yes' or 'no'." The model's first English token after
    any leading "Answer:" / "Response:" prefix decides.
    """
    if not text:
        return -1
    stripped = _LEADING_PREFIX_RE.sub("", text.strip())
    m = re.match(r"^([A-Za-z]+)", stripped)
    if not m:
        return -1
    word = m.group(1).lower()
    if word in _YES_TOKENS:
        return 1
    if word in _NO_TOKENS:
        return 0
    return -1


# MCQ uses its own prefix regex that does NOT accept bare "a" as a prefix
# (since "A" is a valid MCQ answer letter). Only "Answer:" / "Response:"
# prefixes are stripped before letter parsing.
_MCQ_PREFIX_RE = re.compile(r"^\s*(?:answer|response)\s*[:\-]\s*", re.IGNORECASE)
_MCQ_LETTER_RE = re.compile(r"^\s*\(?([A-Da-d])(?:[\)\.\:]|\s|$)")


def score_meta_phys_mcq(text: str) -> int:
    """1 if option A (physical event); 0 if B/C/D; -1 if unparseable.

    Used for the `meta_phys_mcq` prompt: "Which option best describes what
    this image depicts? A) physical event ... B) geometric ... C) symbol ...
    D) none." Companion to `meta_phys_yesno` — same categorical task, MCQ
    format instead of yes/no binary. Used to dissociate "task type" from
    "format" in the generative-vs-categorical finding (audit follow-up).

    Parsing tolerates leading "Answer:" / "Response:" prefixes and "(A)" /
    "A)" / "A." / "A:" forms before the letter.

    A → 1 (physical event = physics-mode commitment)
    B/C/D → 0 (geometric / symbol / none = no physics-mode commitment)
    other / no leading letter → -1 (unparseable)
    """
    if not text:
        return -1
    stripped = _MCQ_PREFIX_RE.sub("", text.strip())
    m = _MCQ_LETTER_RE.match(stripped)
    if not m:
        return -1
    letter = m.group(1).upper()
    if letter == "A":
        return 1
    if letter in ("B", "C", "D"):
        return 0
    return -1


_DESCRIBE_PHYSICS_STEMS: frozenset[str] = frozenset({
    # Action verbs implying motion / dynamics
    "fall", "fell", "fallen", "drop", "dropp", "tumbl", "rolling", "roll",
    "bounc", "collid", "land", "hit", "hits", "moving", "move", "moves",
    "moved", "slid", "slide", "swung", "swing", "swings", "pull", "push",
    "descend", "ascend", "impact",
    # State / pose verbs implying physical context
    "suspend", "hover", "float", "rest", "settl", "leans",
    # Physics nouns / phrases
    "gravit", "momentum", "mass", "weight", "weigh", "veloci",
    "trajector", "force", "motion", "kinetic",
    # Object-with-physical-context phrases (multi-word)
    "in mid-air", "midair", "in the air", "about to fall", "about to hit",
    "about to impact", "about to land", "about to drop", "about to bounce",
    "is about to", "going to fall", "going to hit",
})

# Abstract markers: phrases that explicitly *frame* the input as non-physical.
# These should be conservative — incidental scene descriptors like
# "white background" should NOT override a clear physics-mode statement.
_DESCRIBE_ABSTRACT_STEMS: frozenset[str] = frozenset({
    # Explicit geometric / diagram framing
    "outline", "sketch", "geometric", "diagram", "illustration",
    "line drawing", "minimalist", "abstract",
    # "depicts X" alone is descriptive but ambiguous; require it to lack physics tokens.
})


def score_describe(text: str) -> int:
    """1 if the description contains physics-mode language; 0 otherwise.

    Decision logic:
      - If the response has an explicit *abstract framing* marker (outline,
        sketch, geometric, diagram, line drawing, etc.) AND no physics token,
        return 0.
      - Otherwise, return 1 if any physics token present; 0 if not.

    Physics tokens fire on broader vocabulary than `score_pmr`'s
    kinetic-action stems (suspend, hover, gravitational, motion, kinetic,
    trajectory, weight, ...). Background phrases like "on white background"
    do *not* override physics tokens.

    Examples:
      "A bowling ball suspended above a bowling lane." → 1 (suspend)
      "Gravity pulls the ball down." → 1 (gravit + pull)
      "A simple outline of a circle." → 0 (outline + no physics)
      "Ball falling on a white background." → 1 (fall, even with background phrase)
    """
    if not text:
        return 0
    t = text.lower()
    words = _words(text)

    has_physics_word = any(w.startswith(s) for w in words for s in _DESCRIBE_PHYSICS_STEMS if " " not in s)
    has_physics_phrase = any(s in t for s in _DESCRIBE_PHYSICS_STEMS if " " in s)
    has_physics = has_physics_word or has_physics_phrase

    has_abstract_phrase = any(s in t for s in _DESCRIBE_ABSTRACT_STEMS if " " in s)
    has_abstract_word = any(w.startswith(s) for w in words for s in _DESCRIBE_ABSTRACT_STEMS if " " not in s)
    has_abstract = has_abstract_phrase or has_abstract_word

    if has_abstract and not has_physics:
        return 0
    return 1 if has_physics else 0


def score_for_variant(text: str, variant: str) -> int:
    """Dispatch to the right scorer based on prompt variant.

    Returns 1/0 in all cases. Used by Phase 3 cross-prompt M5a/M5b runs
    where a single run uses one prompt and we want a uniform pmr column.

    For `meta_phys_yesno` and `meta_phys_mcq`, unparseable (-1) is treated
    as 0 (no physics-mode commitment) so the pmr column stays binary.
    """
    if variant in ("open", "open_no_label"):
        return score_pmr(text)
    if variant == "describe_scene":
        return score_describe(text)
    if variant == "meta_phys_yesno":
        s = score_meta_yesno(text)
        return s if s != -1 else 0
    if variant == "meta_phys_mcq":
        s = score_meta_phys_mcq(text)
        return s if s != -1 else 0
    # forced_choice / forced_choice_no_label rely on letter parsing,
    # not raw_text scoring; fall back to score_pmr for raw_text cases.
    return score_pmr(text)


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


def classify_regime(category: str, text: str) -> str:
    """Classify a free-form response into one of {kinetic, static, abstract, ambiguous}.

    Order of checks:
      1. abstract markers override everything (e.g., "this is just a silhouette,
         the bird flies" → abstract because the explicit reject takes precedence
         over the kinetic keyword).
      2. category-specific kinetic / static stems decided by `_any_stem_hit`.
      3. fallback: ambiguous.

    Categories without an entry in CATEGORY_REGIME_KEYWORDS produce a ValueError
    so M8a categories don't accidentally fall through.
    """
    if category not in CATEGORY_REGIME_KEYWORDS:
        raise ValueError(f"classify_regime called for unsupported category {category!r}")
    if not text:
        return "ambiguous"
    if _any_phrase_hit(text, ABSTRACT_MARKERS):
        return "abstract"
    words = _words(text)
    table = CATEGORY_REGIME_KEYWORDS[category]
    # Category-specific kinetic stems are checked before universal-kinetic
    # so categorically-distinctive verbs (drives / walks / flies) get
    # precedence in case of stem overlap.
    if _any_stem_hit(words, table["kinetic"]):
        return "kinetic"
    if _any_stem_hit(words, UNIVERSAL_KINETIC_STEMS):
        return "kinetic"
    if _any_stem_hit(words, table["static"]):
        return "static"
    return "ambiguous"

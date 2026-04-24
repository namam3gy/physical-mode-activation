"""Extract the first A/B/C/D letter from a forced-choice response string.

M5a §3.4 noted that PMR is a noisy signal for steering interventions
(option-text quoting inflates hits). The first-letter distribution is the
cleaner causal signal. Use this helper from the steering-sweep script
and from analysis notebooks.
"""

from __future__ import annotations

import re

_FIRST_LETTER_RE = re.compile(r"^\s*([ABCDabcd])(?=[\s\)\.\:\-—,]|$)")


def extract_first_letter(text: str | None) -> str:
    """Return "A"/"B"/"C"/"D" for the leading choice token, else "other"."""
    if not text:
        return "other"
    m = _FIRST_LETTER_RE.match(text)
    if not m:
        return "other"
    return m.group(1).upper()

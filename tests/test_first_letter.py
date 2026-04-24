"""Tests for first-letter response parsing — canonical M5a response shapes."""

from __future__ import annotations

import pytest

from physical_mode.metrics.first_letter import extract_first_letter


@pytest.mark.parametrize(
    "text, expected",
    [
        # M5a insight doc §3.3 — baseline and L20 α=40
        ("D — This is an abstract shape and as such, it does not have "
         "physical properties that would allow it to fall, move, or change "
         "in any way.", "D"),
        # M5a insight doc §3.3 — L10 α=40 intervention
        ("B) It stays still. — Justification: The circle in the image "
         "appears to be floating or suspended in space without any external "
         "force acting upon it.", "B"),
        # Plain letter + period
        ("A. The ball falls.", "A"),
        # Letter + colon
        ("C: rolls to the right.", "C"),
        # Leading whitespace tolerated
        ("   B) stays still", "B"),
        # Letter + newline
        ("A\nThe ball falls due to gravity.", "A"),
    ],
)
def test_canonical_forms(text: str, expected: str) -> None:
    assert extract_first_letter(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "The ball falls because of gravity.",  # no leading letter
        "Option A: falls",  # leading "Option" word — strict form requires letter-first
        "ABCD — all options",  # ambiguous — must be followed by boundary
    ],
)
def test_non_matching_returns_other(text) -> None:
    assert extract_first_letter(text) == "other"


def test_lowercase_letter_matches_with_uppercase_result() -> None:
    assert extract_first_letter("b) stays still") == "B"

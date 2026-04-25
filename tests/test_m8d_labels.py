"""Sanity tests that M8d additions to type aliases and label tables exist."""

from __future__ import annotations

from physical_mode.config import FactorialSpec
from physical_mode.inference.prompts import LABELS_BY_SHAPE, labels_for_shape


def test_m8d_shapes_factorial_iter():
    """FactorialSpec accepts the new M8d shapes without error."""
    spec = FactorialSpec(
        shapes=("car", "person", "bird"),
        object_levels=("line",),
        bg_levels=("blank",),
        cue_levels=("none",),
        event_templates=("fall",),
        seeds_per_cell=1,
    )
    rows = list(spec.iter())
    assert len(rows) == 3
    assert {r.shape for r in rows} == {"car", "person", "bird"}


def test_m8d_labels_by_shape_present():
    """All three new categories registered in LABELS_BY_SHAPE."""
    for category in ("car", "person", "bird"):
        assert category in LABELS_BY_SHAPE, f"{category!r} missing from LABELS_BY_SHAPE"
        triplet = LABELS_BY_SHAPE[category]
        assert isinstance(triplet, tuple), f"{category!r} entry not a tuple"
        assert len(triplet) == 3, f"{category!r} entry not a 3-tuple: {triplet}"
        assert all(isinstance(x, str) for x in triplet), f"{category!r} entry has non-str: {triplet}"


def test_m8d_label_triplet_values():
    """Spec-pinned label triplets per category."""
    assert LABELS_BY_SHAPE["car"]    == ("car",    "silhouette",  "figurine")
    assert LABELS_BY_SHAPE["person"] == ("person", "stick figure", "statue")
    assert LABELS_BY_SHAPE["bird"]   == ("bird",   "silhouette",  "duck")


def test_m8d_labels_for_shape():
    """labels_for_shape() returns the configured triplet."""
    assert labels_for_shape("car")    == ("car",    "silhouette",  "figurine")
    assert labels_for_shape("person") == ("person", "stick figure", "statue")
    assert labels_for_shape("bird")   == ("bird",   "silhouette",  "duck")

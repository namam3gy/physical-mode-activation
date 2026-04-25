"""Prompt templates for PhysCue evaluation.

Variants:
  - `open`: free-form next-state prediction with a `{label}` referent.
  - `forced_choice`: 4-way MCQ over {falls, stays, moves sideways, abstract shape} with a `{label}` referent.
  - `open_no_label`: free-form, no label referent — H2 null-hypothesis probe (§4.9).

The `{label}` slot carries axis D of the factorial (circle / ball / planet /
shape). `open_no_label` is label-free by design; its `label` field should be the
`_nolabel` sentinel so output provenance stays unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass


SYSTEM_PROMPT_OPEN = (
    "You are a careful observer of images. When asked what will happen next, "
    "describe the most likely next state or motion in one short sentence."
)

SYSTEM_PROMPT_FC = (
    "You are a careful observer of images. Answer the multiple-choice question "
    "with a single letter A, B, C, or D followed by a brief justification."
)


OPEN_TEMPLATE = (
    "The image shows a {label}. Describe what will happen to the {label} in the next moment, "
    "in one short sentence."
)

OPEN_TEMPLATE_NO_LABEL = (
    "What do you see in the image? What might happen next? "
    "Answer in one short sentence."
)

FORCED_CHOICE_TEMPLATE = (
    "The image shows a {label}. Which option best describes what will happen next?\n"
    "A) It falls down.\n"
    "B) It stays still.\n"
    "C) It moves sideways.\n"
    "D) This is an abstract shape — nothing physical happens.\n"
    "Answer with a single letter (A, B, C, or D), then briefly justify."
)

FC_CHOICES: tuple[str, ...] = ("A", "B", "C", "D")


@dataclass(frozen=True)
class RenderedPrompt:
    variant: str
    label: str
    system: str
    user: str
    choice_letters: tuple[str, ...] | None  # for forced-choice logit capture


def render(variant: str, label: str) -> RenderedPrompt:
    if variant == "open":
        return RenderedPrompt(
            variant="open",
            label=label,
            system=SYSTEM_PROMPT_OPEN,
            user=OPEN_TEMPLATE.format(label=label),
            choice_letters=None,
        )
    if variant == "open_no_label":
        return RenderedPrompt(
            variant="open_no_label",
            label=label,
            system=SYSTEM_PROMPT_OPEN,
            user=OPEN_TEMPLATE_NO_LABEL,
            choice_letters=None,
        )
    if variant == "forced_choice":
        return RenderedPrompt(
            variant="forced_choice",
            label=label,
            system=SYSTEM_PROMPT_FC,
            user=FORCED_CHOICE_TEMPLATE.format(label=label),
            choice_letters=FC_CHOICES,
        )
    raise ValueError(f"unknown prompt variant: {variant}")

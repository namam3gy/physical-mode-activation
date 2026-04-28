"""Prompt templates for PhysCue evaluation.

Variants:
  - `open`: free-form next-state prediction with a `{label}` referent.
  - `forced_choice`: 4-way MCQ over {falls, stays, moves sideways, abstract shape} with a `{label}` referent.
  - `open_no_label`: free-form, no label referent — H2 null-hypothesis probe (§4.9).
  - `forced_choice_no_label`: 4-way MCQ with "the depicted object" antecedent, no label token. Companion to `open_no_label` for FC-side H2 null-hypothesis testing.
  - `describe_scene`: free-form scene description with a `{label}` referent (Track B Pillar A, M-MP). Tests whether physics-mode commitment fires under non-prediction prompts.
  - `meta_phys_yesno`: meta-categorization yes/no probe with a `{label}` referent (Track B Pillar A, M-MP). Direct probe of whether the model treats the input as a real-world physical event.
  - `meta_phys_mcq`: meta-categorization 4-way MCQ probe with a `{label}` referent (Track B Pillar A audit follow-up). Same categorical task as `meta_phys_yesno` but in MCQ-letter format — dissociates "categorical task" from "yes/no binary format" in the gen-vs-cat finding.

The `{label}` slot carries axis D of the factorial (circle / ball / planet /
shape). The two `_no_label` variants are label-free by design; their `label`
field should be the `_nolabel` sentinel so output provenance stays unambiguous.
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

FORCED_CHOICE_TEMPLATE_NO_LABEL = (
    "Look at the image. Which option best describes what will happen next?\n"
    "A) The depicted object falls down.\n"
    "B) The depicted object stays still.\n"
    "C) The depicted object moves sideways.\n"
    "D) This is an abstract shape — nothing physical happens.\n"
    "Answer with a single letter (A, B, C, or D), then briefly justify."
)

DESCRIBE_SCENE_TEMPLATE = (
    "The image shows a {label}. Describe what you see in the image, "
    "in one short sentence."
)

META_PHYS_YESNO_TEMPLATE = (
    "The image shows a {label}. Is this a depiction of a real-world physical event "
    "(an object subject to gravity, mass, momentum, or other physical forces)? "
    "Answer with 'yes' or 'no', followed by a brief justification."
)

META_PHYS_MCQ_TEMPLATE = (
    "The image shows a {label}. Which option best describes what this image depicts?\n"
    "A) A real-world physical event (an object subject to gravity, mass, or momentum).\n"
    "B) A geometric figure or abstract shape with no physical context.\n"
    "C) A symbol, icon, or schematic diagram.\n"
    "D) None of the above.\n"
    "Answer with a single letter (A, B, C, or D), then briefly justify."
)

FC_CHOICES: tuple[str, ...] = ("A", "B", "C", "D")


# M8a external-validity round: per-shape label triplets parallel to the
# circle pilot's (ball / circle / planet). Each tuple is ordered as
# (physical-mode label, abstract-shape label, exotic / context-flipping label).
#
#   physical : a label that strongly invites a physical-object reading
#   abstract : the geometric-class label (matches the shape name itself)
#   exotic   : a label whose physical reading is unusual / context-shifted —
#              the per-shape analogue of "planet" for circle. Tests whether
#              the H7 GAR-by-label ordering generalizes beyond circle.
LABELS_BY_SHAPE: dict[str, tuple[str, str, str]] = {
    "circle":   ("ball",   "circle",   "planet"),
    "square":   ("brick",  "square",   "tile"),
    "triangle": ("wedge",  "triangle", "sign"),
    "hexagon":  ("nut",    "hexagon",  "coin"),
    # All three roles must be physics-suggesting (or geometric, for `abstract`),
    # not "more abstract than the geometric label". `boulder` plays the
    # planet/tile/sign/coin role: an unusual physical reading that still
    # commits to mass/gravity. (Avoid using "shape" here — "shape" is more
    # abstract than "polygon" and would invert the role ordering.)
    "polygon":  ("rock",   "polygon",  "boulder"),
    # M8d non-ball categories. abstract role is depiction-style ("silhouette",
    # "stick figure") rather than a forced geometric class because non-ball
    # categories don't have natural geometric-class names.
    "car":      ("car",    "silhouette",  "figurine"),
    "person":   ("person", "stick figure", "statue"),
    "bird":     ("bird",   "silhouette",  "duck"),
    # M8c real photographs. `ball` reuses the circle triplet (a ball is a
    # ball regardless of whether it's drawn or photographed). `abstract`
    # gets a fresh triplet for unstructured / depiction-style photos.
    "ball":     ("ball",   "circle",      "planet"),
    "abstract": ("object", "drawing",     "diagram"),
}


def labels_for_shape(shape: str) -> tuple[str, str, str]:
    """Return the (physical, abstract, exotic) label triplet for a shape."""
    return LABELS_BY_SHAPE[shape]


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
    if variant == "forced_choice_no_label":
        return RenderedPrompt(
            variant="forced_choice_no_label",
            label=label,
            system=SYSTEM_PROMPT_FC,
            user=FORCED_CHOICE_TEMPLATE_NO_LABEL,
            choice_letters=FC_CHOICES,
        )
    if variant == "describe_scene":
        return RenderedPrompt(
            variant="describe_scene",
            label=label,
            system=SYSTEM_PROMPT_OPEN,
            user=DESCRIBE_SCENE_TEMPLATE.format(label=label),
            choice_letters=None,
        )
    if variant == "meta_phys_yesno":
        return RenderedPrompt(
            variant="meta_phys_yesno",
            label=label,
            system=SYSTEM_PROMPT_OPEN,
            user=META_PHYS_YESNO_TEMPLATE.format(label=label),
            choice_letters=None,
        )
    if variant == "meta_phys_mcq":
        return RenderedPrompt(
            variant="meta_phys_mcq",
            label=label,
            system=SYSTEM_PROMPT_FC,
            user=META_PHYS_MCQ_TEMPLATE.format(label=label),
            choice_letters=FC_CHOICES,
        )
    raise ValueError(f"unknown prompt variant: {variant}")

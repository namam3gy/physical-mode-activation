from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

from .utils import WORKSPACE

ObjectLevel = Literal["line", "filled", "shaded", "textured", "block_stack"]
BgLevel = Literal["blank", "ground", "scene"]
CueLevel = Literal[
    "none",
    "wind",          # legacy (pilot); invisible to Qwen2.5-VL — see docs/insights/m1_pilot.md §3.4
    "arrow_shadow",  # legacy (pilot); saturated at PMR=1.0 — split into the two below
    "cast_shadow",   # shadow only, no arrow (Kersten ground-attachment cue)
    "motion_arrow",  # red directional arrow only, no shadow
    "both",          # shadow + arrow (equivalent to legacy arrow_shadow)
]
EventTemplate = Literal["fall", "roll_slope", "wall_bounce", "hover", "horizontal"]
Label = Literal["circle", "ball", "planet", "shape", "object"]
PromptVariant = Literal["open", "forced_choice"]


@dataclass(frozen=True)
class StimulusRow:
    sample_id: str
    event_template: EventTemplate
    object_level: ObjectLevel
    bg_level: BgLevel
    cue_level: CueLevel
    seed: int


@dataclass
class FactorialSpec:
    object_levels: tuple[ObjectLevel, ...] = ("line", "filled", "shaded", "textured")
    bg_levels: tuple[BgLevel, ...] = ("blank", "ground", "scene")
    cue_levels: tuple[CueLevel, ...] = ("none", "wind", "arrow_shadow")
    event_templates: tuple[EventTemplate, ...] = ("fall", "horizontal")
    seeds_per_cell: int = 10
    base_seed: int = 1000

    def iter(self) -> Iterator[StimulusRow]:
        seed = self.base_seed
        for obj in self.object_levels:
            for bg in self.bg_levels:
                for cue in self.cue_levels:
                    for ev in self.event_templates:
                        for k in range(self.seeds_per_cell):
                            sid = f"{obj}_{bg}_{cue}_{ev}_{k:03d}"
                            yield StimulusRow(
                                sample_id=sid,
                                event_template=ev,
                                object_level=obj,
                                bg_level=bg,
                                cue_level=cue,
                                seed=seed,
                            )
                            seed += 1

    def total(self) -> int:
        return (
            len(self.object_levels)
            * len(self.bg_levels)
            * len(self.cue_levels)
            * len(self.event_templates)
            * self.seeds_per_cell
        )


@dataclass
class EvalConfig:
    run_name: str = "pilot"
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0

    # Stimulus spec.
    factorial: FactorialSpec = field(default_factory=FactorialSpec)
    labels: tuple[Label, ...] = ("ball",)  # prompt-time axis D
    prompt_variants: tuple[PromptVariant, ...] = ("open", "forced_choice")
    image_size: int = 512

    # Paths (resolved relative to WORKSPACE).
    inputs_root: Path = field(default_factory=lambda: WORKSPACE / "inputs")
    outputs_root: Path = field(default_factory=lambda: WORKSPACE / "outputs")

    # Activation capture (hidden states / attentions).
    # None = no capture. Otherwise a list of language-model layer indices.
    capture_lm_layers: tuple[int, ...] | None = None
    capture_vision_layers: tuple[int, ...] | None = None
    # Attention tensors are ~3-5x the size of hidden states at the same layer.
    # Keep False for Sub-task 3 (logit lens needs only hidden states); flip to
    # True for Sub-task 4 (activation patching / attention knockout).
    capture_lm_attentions: bool = False

    # Misc.
    random_seed: int = 42
    limit: int | None = None  # truncate the stimulus iteration for smokes

"""M-LMSwap LoRA training — controlled CLIP+LM pair (variant A or B).

Symmetric design (per `docs/m_lmswap_design.md` §7 locked decisions):
    - Variant A: CLIP-ViT-L-336 + 2-layer MLP + Vicuna-7B-v1.5
    - Variant B: CLIP-ViT-L-336 + 2-layer MLP + Mistral-7B-Instruct-v0.2

Held-fixed across A and B:
    - Same CLIP encoder (frozen), same MLP design, same training data, recipe.

Trains:
    - The fresh `multi_modal_projector` (linear_1 → GELU → linear_2),
      full-finetune (~17M params).
    - LoRA rank-32 alpha-64 adapters on the LM's q/k/v/o_proj.

Frozen:
    - CLIP vision encoder.
    - Base LM weights (LoRA wraps them).

Dataset: configurable. Default `HuggingFaceM4/the_cauldron` matches M-PSwap
plumbing (already validated for 1450+ clean steps in mpswap_fp32_20260429-053240).
Switch to LCS-558K (`liuhaotian/LLaVA-Pretrain`) via `--dataset llava_pretrain`
once D0a-found offending batch(es) are filtered (via `--skip-batch-ids`).

Reference: `references/submission_plan.md` Pillar B (B2);
`references/paper_gaps.md` G3; `docs/m_lmswap_design.md`.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import interleave_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import IterableDataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    CLIPImageProcessor,
    get_cosine_schedule_with_warmup,
)


VISION_MODEL = "openai/clip-vit-large-patch14-336"
VARIANTS = {
    "A": "lmsys/vicuna-7b-v1.5",
    "B": "mistralai/Mistral-7B-Instruct-v0.2",
}
DATASET_DEFAULTS = {
    "the_cauldron": {
        "id": "HuggingFaceM4/the_cauldron",
        "subsets": ["aokvqa", "vqav2", "ocrvqa", "tallyqa", "iconqa"],
    },
    "llava_pretrain": {
        "id": "liuhaotian/LLaVA-Pretrain",
        "subsets": [None],  # single corpus
    },
}


@dataclass
class TrainCfg:
    variant: str  # "A" (vicuna) or "B" (mistral)
    output_dir: Path
    dataset: str = "the_cauldron"
    n_subsets: int = 5
    max_samples_per_subset: int = 2000
    max_steps: int = 5000
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 1e-4
    warmup_steps: int = 100
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    log_every: int = 25
    sample_every: int = 500
    save_every: int = 1000
    abort_loss_at_1k: float = 5.0
    seed: int = 42
    device: str = "cuda:0"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999  # standard SFT (advisor flagged 0.95 as risky)
    grad_clip: float = 1.0
    skip_sample_ids: tuple[str, ...] = ()  # batches to filter (D0a output)


# ----------------------------- Build model -----------------------------


def build_variant_model(variant: str, device: str):
    """Construct CLIP+LM+fresh-MLP via LlavaForConditionalGeneration.

    Returns (model, processor). Vision tower is frozen, projector is full-FT,
    LM gets LoRA adapters in the next step.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be 'A' or 'B', got {variant!r}")
    lm_id = VARIANTS[variant]
    print(f"Building variant {variant}: CLIP-336 + fresh-MLP + {lm_id}")

    vc = AutoConfig.from_pretrained(VISION_MODEL).vision_config
    mc = AutoConfig.from_pretrained(lm_id)

    # Build tokenizer — add <image> if missing.
    tok = AutoTokenizer.from_pretrained(lm_id)
    if "<image>" not in tok.get_added_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<image>"]})
    image_token_id = tok.convert_tokens_to_ids("<image>")

    cfg = LlavaConfig(
        vision_config=vc,
        text_config=mc,
        image_token_id=image_token_id,
    )

    # Construct on meta first, then load real weights piecewise.
    print("  loading vision tower...")
    from transformers import CLIPVisionModel
    vision = CLIPVisionModel.from_pretrained(VISION_MODEL, dtype=torch.bfloat16)

    print(f"  loading LM ({lm_id})...")
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(lm_id, dtype=torch.bfloat16)
    # Resize embeddings to match new tokenizer
    lm.resize_token_embeddings(len(tok))

    # Build the LlavaForConditionalGeneration shell + inject pre-loaded components.
    print("  assembling Llava wrapper + fresh MLP projector...")
    model = LlavaForConditionalGeneration(cfg)
    model.model.vision_tower = vision
    model.model.language_model = lm.model  # transformer body only
    model.lm_head = lm.lm_head
    # multi_modal_projector keeps its random-init weights (full-FT target).

    # Move to device + bf16.
    model = model.to(device=device, dtype=torch.bfloat16)
    print(f"  ✓ assembled on {device} (bf16)")

    # Build processor (image processor + tokenizer).
    image_processor = CLIPImageProcessor.from_pretrained(VISION_MODEL)
    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tok)

    return model, processor


def apply_freeze_and_lora(model, lora_rank: int, lora_alpha: int, lora_dropout: float):
    """Freeze vision tower, full-FT projector, LoRA on LM q/k/v/o_proj."""
    # Freeze everything by default.
    for p in model.parameters():
        p.requires_grad_(False)

    # Vision tower stays frozen.
    # Projector: full FT.
    for p in model.model.multi_modal_projector.parameters():
        p.requires_grad_(True)

    # LM: LoRA on q/k/v/o_proj.
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj",
    )
    model = get_peft_model(model, lora_cfg)

    # Re-unfreeze projector after PEFT wrap.
    for p in model.base_model.model.model.multi_modal_projector.parameters():
        p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {n_trainable:,} / {n_total:,} ({n_trainable / n_total * 100:.4f}%)")
    return model


# ----------------------------- Data -----------------------------
# (the_cauldron streaming reused from M-PSwap; LCS-558K loader added separately
# once we know whether to use it from D0a result.)


def make_the_cauldron_stream(subsets: list[str], max_per_subset: int, seed: int = 42):
    """Stream the_cauldron — yields {'images', 'messages'} dicts."""

    def _format(ex: dict) -> dict | None:
        if not ex.get("images") or not ex.get("texts"):
            return None
        turn = ex["texts"][0]
        user = (turn.get("user") or "").strip()[:2048]
        asst = (turn.get("assistant") or "").strip()[:2048]
        if not user or not asst:
            return None
        user_content = [{"type": "image"} for _ in ex["images"]]
        user_content.append({"type": "text", "text": user})
        return {
            "images": ex["images"],
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": asst}]},
            ],
        }

    class _Stream(IterableDataset):
        def __iter__(self) -> Iterator[dict]:
            ds_list = []
            for s in subsets:
                sub = load_dataset(
                    DATASET_DEFAULTS["the_cauldron"]["id"], s, split="train", streaming=True
                )
                sub = sub.shuffle(seed=seed, buffer_size=2000).take(max_per_subset)
                ds_list.append(sub)
            merged = interleave_datasets(ds_list, seed=seed, stopping_strategy="all_exhausted")
            for ex in merged:
                fmt = _format(ex)
                if fmt is not None:
                    yield fmt

    return _Stream()


# ----------------------------- Train loop -----------------------------
# (Skeleton — full collate / step loop fork from m_pswap_train.py once D0a
# result locks the recipe parameters. Day 1 task.)


def train(cfg: TrainCfg) -> None:
    raise NotImplementedError(
        "M-LMSwap train loop is a Day-1 fork from m_pswap_train.py. "
        "Awaiting D0a determinism repro result before locking the data path."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["A", "B"], required=True,
                   help="A = CLIP+Vicuna-7B-v1.5; B = CLIP+Mistral-7B-Instruct-v0.2")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset", choices=list(DATASET_DEFAULTS), default="the_cauldron")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-samples-per-subset", type=int, default=2000)
    p.add_argument("--n-subsets", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--skip-sample-ids", nargs="*", default=[],
                   help="Sample IDs (D0a output) to filter from the data stream")
    args = p.parse_args()

    cfg = TrainCfg(
        variant=args.variant,
        output_dir=args.output_dir,
        dataset=args.dataset,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_samples_per_subset=args.max_samples_per_subset,
        n_subsets=args.n_subsets,
        device=args.device,
        seed=args.seed,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        grad_clip=args.grad_clip,
        skip_sample_ids=tuple(args.skip_sample_ids),
    )
    train(cfg)


if __name__ == "__main__":
    main()

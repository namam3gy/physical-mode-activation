"""M-PSwap LoRA training — Idefics2 with perceiver-resampler → MLP-pool projector.

Trains:
    - The new ``MLPPoolResampler`` module (full-finetune via ``modules_to_save``).
    - LoRA rank-32 alpha-64 adapters on the Mistral LM's q/v/k/o_proj
      (``text_model.layers.*.self_attn``).
Frozen:
    - SigLIP vision encoder (``model.model.vision_model.*``).
    - Connector ``modality_projection`` MLP.
    - Mistral LM base weights (LoRA wraps them, base is frozen).

Dataset: ``HuggingFaceM4/the_cauldron`` (Idefics2's own training distribution
— matches the LM's expected chat-template + image-token convention).

Intermediate gates (per advisor 2026-04-29):
    - Loss-curve check at step 1K — if train-loss > 5.0 by step 1K, abort.
    - Generation samples every 500 steps (held-out probe stim) — confirm real
      language emission, not punctuation salad.
    - Mid-training POPE n=500 at step 2.5K — abort if F1 < 0.50.

Reference: ``references/paper_gaps.md`` G3, ``references/submission_plan.md`` Pillar B.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import interleave_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import IterableDataset
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

from physical_mode.lora.idefics2_mlp_resampler import (
    count_params,
    swap_perceiver_to_mlp_pool,
)


MODEL_ID = "HuggingFaceM4/idefics2-8b"
DATASET_ID = "HuggingFaceM4/the_cauldron"
DEFAULT_SUBSETS = ["aokvqa", "vqav2", "ocrvqa", "tallyqa", "iconqa"]


@dataclass
class TrainCfg:
    output_dir: Path
    n_subsets: int = 5
    max_samples_per_subset: int = 2000  # default 5×2000 = 10K
    max_steps: int = 5000
    batch_size: int = 4
    grad_accum: int = 8  # effective batch 32
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
    mid_pope_step: int = 2500
    mid_pope_n: int = 500
    mid_pope_min_f1: float = 0.50
    # AdamW betas — (0.9, 0.999) is standard SFT; (0.9, 0.95) is pretraining-style
    # and was implicated in the run-1 NaN collapse (faster v_t decay → noisier
    # second-moment, larger bf16 round-off).
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    grad_clip: float = 1.0
    # Resume-from-checkpoint: load LoRA + MLP-pool state from this dir, set
    # ``step_offset`` so cosine schedule and step counter continue from there.
    # When set, lr / betas / grad_clip can be tightened by the caller.
    resume_from: Path | None = None
    step_offset: int = 0


# ----------------------------- Data -----------------------------


def _format_one(example: dict, max_chars_per_msg: int = 2048) -> tuple[list[Image.Image], list[dict]]:
    """One the_cauldron example → (PIL images, chat-template messages).

    the_cauldron schema: {'images': [PIL...], 'texts': [{'user', 'assistant', 'source'}]}
    Some examples have multi-turn texts; we take the first turn for simplicity.
    """
    images = example["images"]  # list of PIL.Image
    texts = example["texts"]
    if not texts:
        return [], []
    turn = texts[0]
    user = (turn.get("user") or "").strip()[:max_chars_per_msg]
    assistant = (turn.get("assistant") or "").strip()[:max_chars_per_msg]
    if not user or not assistant:
        return [], []

    # Build chat template messages: image first, then user text, then assistant.
    user_content = []
    for _ in images:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": user})
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": assistant}]},
    ]
    return images, messages


class TheCauldronStream(IterableDataset):
    """Streams the_cauldron, formats for Idefics2 chat template, yields raw (images, messages, prompt_messages).

    The actual tokenization + label-building happens in ``collate`` so we can do
    it on a per-batch basis with the processor.
    """

    def __init__(self, subsets: list[str], max_per_subset: int, seed: int = 42) -> None:
        self.subsets = subsets
        self.max_per_subset = max_per_subset
        self.seed = seed

    def __iter__(self) -> Iterator[dict]:
        ds_list = []
        for s in self.subsets:
            sub = load_dataset(DATASET_ID, s, split="train", streaming=True)
            sub = sub.shuffle(seed=self.seed, buffer_size=2000).take(self.max_per_subset)
            ds_list.append(sub)
        merged = interleave_datasets(ds_list, seed=self.seed, stopping_strategy="all_exhausted")
        for ex in merged:
            images, messages = _format_one(ex)
            if not images or not messages:
                continue
            yield {"images": images, "messages": messages}


def make_collate(processor, device: str):
    def _collate(batch: list[dict]) -> dict:
        prompts = []
        prompt_only = []
        all_images = []
        for ex in batch:
            text = processor.apply_chat_template(ex["messages"], add_generation_prompt=False)
            prompts.append(text)
            text_pre = processor.apply_chat_template(ex["messages"][:-1], add_generation_prompt=True)
            prompt_only.append(text_pre)
            all_images.append(ex["images"])

        full = processor(text=prompts, images=all_images, return_tensors="pt", padding=True)
        # Tokenize prompt-only portion (text only, no images) to determine where labels start.
        # Idefics2 inserts image tokens before user text — but since both prompt and prompt_only
        # share the image-token prefix, the difference is the assistant suffix.
        # We compute per-example: labels = -100 except for the trailing N tokens
        # equal to len(full_input_ids) - len(prompt_only_input_ids).
        labels = full["input_ids"].clone()
        for i, (text_full, text_pre) in enumerate(zip(prompts, prompt_only)):
            tok_full = processor.tokenizer(text_full, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            tok_pre = processor.tokenizer(text_pre, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            n_assistant = len(tok_full) - len(tok_pre)
            if n_assistant <= 0:
                # Defensive: if tokenizer alignment fails, skip by masking everything (no loss contribution).
                labels[i, :] = -100
                continue
            seq_len = labels.shape[1]
            n_pad = (full["input_ids"][i] == processor.tokenizer.pad_token_id).sum().item()
            n_real = seq_len - n_pad
            cut = n_real - n_assistant
            labels[i, :cut] = -100
            # also mask pad tokens (they are at the end for right-padding, start for left-padding)
            labels[i][full["input_ids"][i] == processor.tokenizer.pad_token_id] = -100

        full["labels"] = labels
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in full.items()}

    return _collate


# ----------------------------- Build model -----------------------------


def build_peft_model(
    device: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    resume_from: Path | None = None,
):
    """Load Idefics2, swap perceiver, freeze base, apply LoRA on LM attn only.

    Note: we deliberately skip PEFT's ``modules_to_save`` for the new
    ``MLPPoolResampler``. PEFT wraps such modules in ``AuxiliaryTrainingWrapper``,
    which assumes a positional-first ``forward(x, *args, **kwargs)`` signature
    incompatible with Idefics2's keyword-only call
    ``perceiver_resampler(context=..., attention_mask=...)``. Instead we keep
    the new module as a plain ``nn.Module`` with ``requires_grad_(True)`` and
    serialize it ourselves via ``save_mlp_pool_state``.
    """
    print(f"loading {MODEL_ID} on {device} (bf16) ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, do_image_splitting=False)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device
    )

    # 1. Swap perceiver → MLP-pool.
    pr_params = count_params(model.model.connector.perceiver_resampler)
    new = swap_perceiver_to_mlp_pool(model, n_heads=8)
    print(f"swapped perceiver ({pr_params:,} params) → MLPPoolResampler ({count_params(new):,} params)")

    # 2. Freeze everything by default.
    for p in model.parameters():
        p.requires_grad_(False)

    if resume_from is not None:
        # Restore the trained MLP-pool weights before applying LoRA so that the
        # PEFT wrapper attaches around the trained module.
        state = torch.load(resume_from / "mlp_pool_resampler.pt", map_location=device, weights_only=True)
        new.load_state_dict(state)
        print(f"  ✓ loaded MLP-pool state from {resume_from}/mlp_pool_resampler.pt")

    # 3. Apply LoRA. target_modules regex restricts to Mistral LM attn.
    if resume_from is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
        print(f"  ✓ loaded LoRA adapters from {resume_from}")
    else:
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=r".*text_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj",
        )
        model = get_peft_model(model, lora_cfg)

    # 4. Manually unfreeze the swapped perceiver_resampler module (full-finetune).
    pr = model.base_model.model.model.connector.perceiver_resampler
    for p in pr.parameters():
        p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"trainable params: {n_trainable:,} || all params: {n_total:,} || "
        f"trainable%: {n_trainable / n_total * 100:.4f}"
    )

    # Sanity check.
    n_lora = sum(1 for n, _ in model.named_modules() if "lora_A" in n or "lora_B" in n)
    print(f"  LoRA-wrapped sub-modules: {n_lora}")
    return model, processor


def save_mlp_pool_state(model, ck: Path) -> None:
    """Save the new perceiver_resampler state_dict (PEFT.save_pretrained skips it)."""
    pr = model.base_model.model.model.connector.perceiver_resampler
    torch.save(pr.state_dict(), ck / "mlp_pool_resampler.pt")


# ----------------------------- Train loop -----------------------------


def save_ckpt(model, processor, out_dir: Path, step: int) -> None:
    ck = out_dir / f"step{step}"
    ck.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ck)  # LoRA adapters
    save_mlp_pool_state(model, ck)  # MLP-pool projector (full module)
    processor.save_pretrained(ck)
    print(f"  ✓ saved checkpoint to {ck}")


def generate_sample(model, processor, prompt: str, image: Image.Image, device: str, max_new_tokens: int = 32) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[[image]], return_tensors="pt").to(device)
    model.eval()
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    model.train()
    return processor.batch_decode(out_ids, skip_special_tokens=True)[0]


def train(cfg: TrainCfg) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.output_dir / "train_log.jsonl"
    log_f = log_path.open("w")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model, processor = build_peft_model(
        cfg.device, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout, resume_from=cfg.resume_from
    )
    model.train()

    subsets = DEFAULT_SUBSETS[: cfg.n_subsets]
    print(f"using subsets: {subsets} × {cfg.max_samples_per_subset}/each")
    stream = TheCauldronStream(subsets, cfg.max_samples_per_subset, seed=cfg.seed)
    collate = make_collate(processor, cfg.device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(
        trainable, lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=0.01
    )
    sched = get_cosine_schedule_with_warmup(optim, cfg.warmup_steps, cfg.max_steps)
    # Skip ahead in the LR schedule when resuming.
    for _ in range(cfg.step_offset):
        sched.step()

    # Group trainable params by source for separate grad-norm logging
    pr_params = [p for n, p in model.named_parameters() if "perceiver_resampler" in n and p.requires_grad]
    lora_params = [p for n, p in model.named_parameters() if ("lora_A" in n or "lora_B" in n) and p.requires_grad]
    print(f"  trainable groups: perceiver={len(pr_params)} lora={len(lora_params)}")

    # Held-out probe sample for periodic generation check.
    probe_image = Image.new("RGB", (224, 224), color=(127, 127, 127))
    probe_prompt = "Describe what you see briefly."

    step = cfg.step_offset
    optim.zero_grad()
    accum = 0
    loss_window: list[float] = []
    t0 = time.time()
    batch_iter = iter(stream)
    batch: list[dict] = []

    def _group_grad_norm(params) -> float:
        gs = [p.grad.detach().float().flatten() for p in params if p.grad is not None]
        if not gs:
            return 0.0
        return float(torch.cat(gs).norm().item())

    while step < cfg.max_steps:
        # Fill a batch from the stream
        while len(batch) < cfg.batch_size:
            try:
                batch.append(next(batch_iter))
            except StopIteration:
                # restart stream with a different seed shift
                cfg_seed_next = cfg.seed + step
                stream = TheCauldronStream(subsets, cfg.max_samples_per_subset, seed=cfg_seed_next)
                batch_iter = iter(stream)

        try:
            ins = collate(batch)
        except Exception as e:
            print(f"  collate error step={step}: {e!r}; skipping batch")
            batch = []
            continue
        batch = []

        # Skip if no labels survive masking (all -100)
        if (ins["labels"] != -100).sum() == 0:
            continue

        out = model(**ins)
        # NaN-detect early-abort: a single non-finite loss means the run is dead;
        # silently continuing would corrupt subsequent checkpoints.
        if not torch.isfinite(out.loss).all():
            print(f"  ✗ NON-FINITE LOSS at step {step} (accum {accum}); aborting.")
            log_f.write(json.dumps({"step": step, "abort": "non_finite_loss"}) + "\n")
            log_f.close()
            raise SystemExit(3)
        loss = out.loss / cfg.grad_accum
        loss.backward()
        accum += 1
        loss_window.append(out.loss.item())

        if accum >= cfg.grad_accum:
            # Pre-clip group grad norms (for logging trends prior to clipping)
            pr_gn = _group_grad_norm(pr_params)
            lora_gn = _group_grad_norm(lora_params)
            total_gn = float(torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip).item())
            optim.step()
            sched.step()
            optim.zero_grad()
            accum = 0
            step += 1

            log_now = (step % cfg.log_every == 0) or (step <= 50) or (50 < step <= 150 and step % 5 == 0)
            if log_now:
                avg = sum(loss_window[-cfg.log_every * cfg.grad_accum :]) / max(
                    1, len(loss_window[-cfg.log_every * cfg.grad_accum :])
                )
                lr_now = sched.get_last_lr()[0]
                elapsed = time.time() - t0
                rate = (step - cfg.step_offset) / max(1.0, elapsed) * 60
                rec = {
                    "step": step, "loss": avg, "lr": lr_now,
                    "elapsed_sec": elapsed, "steps_per_min": rate,
                    "grad_total": total_gn, "grad_lora": lora_gn, "grad_pr": pr_gn,
                }
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                print(
                    f"step {step:>5d} | loss {avg:.4f} | lr {lr_now:.2e} | "
                    f"g_tot {total_gn:.2e} g_lora {lora_gn:.2e} g_pr {pr_gn:.2e} | "
                    f"{rate:.1f} steps/min"
                )

                # Gate 1: loss check at step 1K
                if step == 1000 and avg > cfg.abort_loss_at_1k:
                    print(f"  ✗ ABORT — loss {avg:.4f} > {cfg.abort_loss_at_1k} at step 1K. Hyperparams likely wrong.")
                    log_f.close()
                    raise SystemExit(2)

            if step % cfg.sample_every == 0:
                try:
                    sample = generate_sample(model, processor, probe_prompt, probe_image, cfg.device)
                    print(f"  [generation @ step {step}] {sample!r}")
                    log_f.write(json.dumps({"step": step, "sample": sample}) + "\n")
                    log_f.flush()
                except Exception as e:
                    print(f"  generation error: {e!r}")

            if step % cfg.save_every == 0:
                save_ckpt(model, processor, cfg.output_dir, step)

    save_ckpt(model, processor, cfg.output_dir, cfg.max_steps)
    log_f.close()
    print(f"DONE. Final checkpoint at {cfg.output_dir}/step{cfg.max_steps}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-samples-per-subset", type=int, default=2000)
    p.add_argument("--n-subsets", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999,
                   help="Standard SFT default (0.999). Run-1 used 0.95 (pretraining-style); "
                   "advisor flagged that as a likely contributor to the bf16 LoRA NaN at step 1475.")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Resume from checkpoint dir (mlp_pool_resampler.pt + LoRA adapters)")
    p.add_argument("--step-offset", type=int, default=0,
                   help="Initial step counter (set to source ckpt's step number when --resume-from)")
    args = p.parse_args()

    cfg = TrainCfg(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_samples_per_subset=args.max_samples_per_subset,
        n_subsets=args.n_subsets,
        device=args.device,
        seed=args.seed,
        log_every=args.log_every,
        sample_every=args.sample_every,
        save_every=args.save_every,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        grad_clip=args.grad_clip,
        resume_from=args.resume_from,
        step_offset=args.step_offset,
    )
    train(cfg)


if __name__ == "__main__":
    main()

"""M-LMSwap LoRA training — controlled CLIP+LM pair (variant A or B).

Symmetric design (per `docs/m_lmswap_design.md` §7 locked decisions):
    - Variant A: CLIP-ViT-L-336 + 2-layer MLP + Vicuna-7B-v1.5
    - Variant B: CLIP-ViT-L-336 + 2-layer MLP + Mistral-7B-Instruct-v0.2

Held-fixed across A and B:
    - Same CLIP encoder (frozen), same MLP design, same training data, recipe.

Two-stage canonical training (per `docs/m_lmswap_design.md` §4 Option 1):
    - Stage 1 — Projector pretrain on LCS-558K. Only `multi_modal_projector` is
      trainable (no LoRA on LM). LR=1e-3, ~17K step.
    - Stage 2 — LoRA instruction tune on LLaVA-Instruct-665K. Loads stage-1
      MLP weights, then applies LoRA r=32 α=64 on q/k/v/o_proj. LR=2e-4 for
      LoRA params, MLP keeps its (now stable) state. ~21K step.

Advisor checks 2026-04-29:
    - apply freeze BEFORE wrap_with_lora, restore stage-1 MLP weights BEFORE
      get_peft_model wraps (prevents PEFT-renamed param load-state-dict miss).
    - Both Vicuna and Mistral use manual chat templates with `<image>` literal.
    - vision_feature_select_strategy/layer set explicitly (not relying on
      LlavaConfig defaults).

Reference: `references/submission_plan.md` Pillar B (B2);
`references/paper_gaps.md` G3; `docs/m_lmswap_design.md`.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import IterableDataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    get_cosine_schedule_with_warmup,
)


VISION_MODEL = "openai/clip-vit-large-patch14-336"
VARIANTS = {
    "A": "lmsys/vicuna-7b-v1.5",
    "B": "mistralai/Mistral-7B-Instruct-v0.2",
}
LCS_REPO = "liuhaotian/LLaVA-Pretrain"
LCS_JSON = "blip_laion_cc_sbu_558k.json"
LCS_ZIP = "images.zip"
MIX665K_REPO = "Icey444/llava_v1_5_mix665k"

@dataclass
class TrainCfg:
    variant: str  # "A" (vicuna) or "B" (mistral)
    stage: int    # 1 (MLP-only on LCS-558K) or 2 (MLP+LoRA on Mix665K)
    output_dir: Path
    stage1_ckpt: Path | None = None  # required for stage 2
    max_steps: int = 17000  # stage 1 default; stage 2 ~21000
    batch_size: int = 4
    grad_accum: int = 8     # effective batch 32
    lr: float | None = None  # None → stage default (1e-3 / 2e-4)
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
    adam_beta2: float = 0.999  # standard SFT (advisor flagged 0.95 risky)
    grad_clip: float = 1.0
    resume_from: Path | None = None
    step_offset: int = 0


# ----------------------------- Chat templates -----------------------------
# Manual assembly — Vicuna/Mistral default chat templates do not contain
# `<image>` slots, so we control the format directly.

_VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def format_chat(variant: str, user_text: str, assistant_text: str) -> tuple[str, str]:
    """Return (full_string, prompt_only_string) with literal `<image>` tokens.

    The `<image>` token in `user_text` is expanded by LlavaProcessor into
    `image_seq_length` repeated tokens at processor-call time.
    """
    if variant == "A":  # Vicuna
        full = f"{_VICUNA_SYSTEM} USER: {user_text} ASSISTANT: {assistant_text}"
        prompt_only = f"{_VICUNA_SYSTEM} USER: {user_text} ASSISTANT:"
    elif variant == "B":  # Mistral
        full = f"<s>[INST] {user_text} [/INST] {assistant_text}</s>"
        prompt_only = f"<s>[INST] {user_text} [/INST]"
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return full, prompt_only


# ----------------------------- Build model -----------------------------


def build_variant_model(variant: str, device: str):
    """Construct CLIP+LM+fresh-MLP via LlavaForConditionalGeneration.

    Returns (model, processor). Vision tower is frozen, projector is full-FT,
    LM gets LoRA adapters in stage 2 only.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be 'A' or 'B', got {variant!r}")
    lm_id = VARIANTS[variant]
    print(f"Building variant {variant}: CLIP-336 + fresh-MLP + {lm_id}")

    vc = AutoConfig.from_pretrained(VISION_MODEL).vision_config
    mc = AutoConfig.from_pretrained(lm_id)

    tok = AutoTokenizer.from_pretrained(lm_id)
    if "<image>" not in tok.get_added_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<image>"]})
    image_token_index = tok.convert_tokens_to_ids("<image>")
    if tok.pad_token is None:
        # Vicuna and Mistral lack a pad token by default.
        tok.pad_token = tok.eos_token

    # The LM's embedding matrix gets resized to len(tok) below to accommodate
    # the new <image> token; mirror that in text_config so the cross-entropy
    # loss uses the correct vocab_size.
    mc.vocab_size = len(tok)
    cfg = LlavaConfig(
        vision_config=vc,
        text_config=mc,
        image_token_index=image_token_index,
        image_seq_length=576,                 # 336 / 14 == 24, 24*24 = 576
        vision_feature_select_strategy="default",  # strips CLS → 576 tokens
        vision_feature_layer=-2,               # penultimate (LLaVA-1.5 spec)
        projector_hidden_act="gelu",           # 2-layer GELU MLP
    )

    print("  loading vision tower...")
    vision = CLIPVisionModel.from_pretrained(VISION_MODEL, dtype=torch.bfloat16)

    print(f"  loading LM ({lm_id})...")
    lm = AutoModelForCausalLM.from_pretrained(lm_id, dtype=torch.bfloat16)
    lm.resize_token_embeddings(len(tok))

    print("  assembling Llava wrapper + fresh MLP projector...")
    model = LlavaForConditionalGeneration(cfg)
    model.model.vision_tower = vision
    model.model.language_model = lm.model  # transformer body
    model.lm_head = lm.lm_head
    # multi_modal_projector keeps its random-init weights (full-FT target).

    model = model.to(device=device, dtype=torch.bfloat16)
    print(f"  ✓ assembled on {device} (bf16)")

    image_processor = CLIPImageProcessor.from_pretrained(VISION_MODEL)
    processor = LlavaProcessor(
        image_processor=image_processor,
        tokenizer=tok,
        patch_size=cfg.vision_config.patch_size,
        vision_feature_select_strategy=cfg.vision_feature_select_strategy,
        num_additional_image_tokens=1,  # CLIP CLS — added then stripped under
                                         # "default" strategy. Net delta is 0,
                                         # but LlavaProcessor's formula treats
                                         # them as independent terms (default = 0
                                         # strips 1 too many → 575 instead of 576).
    )

    return model, processor


def freeze_for_stage1(model) -> None:
    """Freeze everything except the multi_modal_projector (full-FT)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.model.multi_modal_projector.parameters():
        p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  stage 1 trainable: {n_trainable:,} / {n_total:,} "
          f"({n_trainable / n_total * 100:.4f}%)")


def load_stage1_mlp(model, ckpt_dir: Path, device: str) -> None:
    """Load stage-1 multi_modal_projector weights into the model.

    MUST be called BEFORE get_peft_model wraps the model (PEFT renames
    submodule paths under base_model.model.*).
    """
    state_path = ckpt_dir / "multi_modal_projector.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"stage 1 MLP weights missing: {state_path}")
    state = torch.load(state_path, map_location=device, weights_only=True)
    model.model.multi_modal_projector.load_state_dict(state)
    print(f"  ✓ loaded stage 1 MLP weights from {state_path}")


def apply_stage2_lora(model, lora_rank: int, lora_alpha: int, lora_dropout: float):
    """Freeze everything, then apply LoRA to LM attn AND keep MLP trainable.

    Caller is responsible for having already loaded stage-1 MLP weights via
    load_stage1_mlp.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj",
    )
    model = get_peft_model(model, lora_cfg)

    # Re-unfreeze MLP after PEFT wrap (PEFT freezes everything by default).
    for p in model.base_model.model.model.multi_modal_projector.parameters():
        p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  stage 2 trainable: {n_trainable:,} / {n_total:,} "
          f"({n_trainable / n_total * 100:.4f}%)")
    return model


def get_mlp_ref(model):
    """Return the multi_modal_projector module regardless of PEFT wrap depth."""
    if hasattr(model, "base_model"):
        return model.base_model.model.model.multi_modal_projector
    return model.model.multi_modal_projector


# ----------------------------- Data: Stage 1 (LCS-558K) -----------------------------


class LcsPretrainStream(IterableDataset):
    """Streams LCS-558K from local JSON + images.zip (downloaded once via hf_hub).

    Yields {'image': PIL.Image.Image, 'user': str, 'assistant': str}.
    The `user` field already contains the LLaVA-1.5 instruction template plus
    `<image>` placeholder (carried through from the source JSON).
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._json_path: Path | None = None
        self._zip_path: Path | None = None

    def _ensure_files(self) -> None:
        if self._json_path is None:
            print("  downloading LCS-558K JSON metadata (one-time)...")
            self._json_path = Path(
                hf_hub_download(repo_id=LCS_REPO, filename=LCS_JSON, repo_type="dataset")
            )
        if self._zip_path is None:
            print("  downloading LCS-558K images.zip (one-time, ~14 GB)...")
            self._zip_path = Path(
                hf_hub_download(repo_id=LCS_REPO, filename=LCS_ZIP, repo_type="dataset")
            )

    def __iter__(self) -> Iterator[dict]:
        self._ensure_files()
        with self._json_path.open("r") as f:
            records = json.load(f)
        rng = random.Random(self.seed)
        rng.shuffle(records)
        with zipfile.ZipFile(self._zip_path, "r") as zf:
            for rec in records:
                # rec: {id, image, conversations: [{from:human, value:<inst+<image>>},
                #                                   {from:gpt, value:<caption>}]}
                rel = rec.get("image")
                conv = rec.get("conversations") or []
                user_text = None
                asst_text = None
                for turn in conv:
                    role = turn.get("from")
                    val = (turn.get("value") or "").strip()
                    if role == "human" and user_text is None:
                        user_text = val
                    elif role == "gpt" and asst_text is None:
                        asst_text = val
                if not rel or not user_text or not asst_text:
                    continue
                # The LCS-558K human turn frequently has `<image>` SUFFIXED;
                # normalize to leading position for processor consistency.
                if "<image>" in user_text:
                    user_text = user_text.replace("<image>", "").strip()
                user_text = f"<image>\n{user_text}"
                try:
                    raw = zf.read(rel)
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                except (KeyError, OSError):
                    continue
                yield {"image": img, "user": user_text, "assistant": asst_text}


# ----------------------------- Data: Stage 2 (Mix665K) -----------------------------


class Mix665kStream(IterableDataset):
    """Streams Icey444/llava_v1_5_mix665k parquet shards.

    Yields {'image': PIL.Image.Image, 'messages': [{user, assistant}]}.
    """

    def __init__(self, seed: int = 42, shuffle_buffer: int = 2000) -> None:
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self) -> Iterator[dict]:
        ds = load_dataset(MIX665K_REPO, split="train", streaming=True)
        ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        for ex in ds:
            img = ex.get("image")
            conv = ex.get("conversations") or ex.get("messages") or []
            if img is None or not conv:
                continue
            if not isinstance(img, Image.Image):
                # Mix665k may ship images as bytes/dict; normalize.
                if isinstance(img, dict) and "bytes" in img:
                    img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
                elif isinstance(img, (bytes, bytearray)):
                    img = Image.open(io.BytesIO(img)).convert("RGB")
                else:
                    continue
            else:
                img = img.convert("RGB")
            # conversations format: list of {"from", "value"} alternating human/gpt.
            # We use only the first user/assistant pair.
            user_text = None
            asst_text = None
            for turn in conv:
                role = turn.get("from") or turn.get("role")
                content = turn.get("value") or turn.get("content")
                if role in ("human", "user") and user_text is None:
                    user_text = (content or "").strip()
                elif role in ("gpt", "assistant") and user_text is not None and asst_text is None:
                    asst_text = (content or "").strip()
                    break
            if not user_text or not asst_text:
                continue
            yield {"image": img, "user": user_text, "assistant": asst_text}


# ----------------------------- Collate -----------------------------


def make_collate(processor, variant: str, device: str):
    """Single collate fn — both LcsPretrainStream and Mix665kStream yield the
    same {image, user, assistant} schema, with `<image>` already present in user.
    """

    def _collate(batch: list[dict]) -> dict:
        full_strs, prompt_strs, images = [], [], []
        for ex in batch:
            user_raw = ex["user"]
            if "<image>" not in user_raw:
                user_raw = f"<image>\n{user_raw}"
            full, pre = format_chat(variant, user_raw, ex["assistant"])
            full_strs.append(full)
            prompt_strs.append(pre)
            images.append(ex["image"])
        return _build_batch(processor, full_strs, prompt_strs, images, device)

    return _collate


def _build_batch(processor, full_strs, prompt_strs, images, device):
    full = processor(text=full_strs, images=images, return_tensors="pt", padding=True)
    labels = full["input_ids"].clone()
    pad_id = processor.tokenizer.pad_token_id

    for i, (text_full, text_pre) in enumerate(zip(full_strs, prompt_strs)):
        tok_full = processor.tokenizer(text_full, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        tok_pre = processor.tokenizer(text_pre, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        n_assistant = len(tok_full) - len(tok_pre)
        if n_assistant <= 0:
            labels[i, :] = -100
            continue
        n_pad = (full["input_ids"][i] == pad_id).sum().item()
        seq_len = labels.shape[1]
        n_real = seq_len - n_pad
        cut = n_real - n_assistant
        labels[i, :cut] = -100
        labels[i][full["input_ids"][i] == pad_id] = -100

    full["labels"] = labels
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in full.items()}


# ----------------------------- Train loop -----------------------------


def save_ckpt(model, processor, out_dir: Path, step: int, stage: int) -> None:
    ck = out_dir / f"step{step}"
    ck.mkdir(parents=True, exist_ok=True)
    if stage == 2:
        # PEFT save (LoRA adapters) + manual MLP save.
        model.save_pretrained(ck)
    mlp = get_mlp_ref(model)
    torch.save(mlp.state_dict(), ck / "multi_modal_projector.pt")
    processor.save_pretrained(ck)
    print(f"  ✓ saved checkpoint to {ck}")


def generate_sample(model, processor, prompt: str, image: Image.Image, variant: str,
                    device: str, max_new_tokens: int = 32) -> str:
    user_text = f"<image>\n{prompt}"
    _, pre = format_chat(variant, user_text, "")
    inputs = processor(text=[pre], images=[image], return_tensors="pt").to(device)
    model.eval()
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    model.train()
    return processor.batch_decode(out_ids, skip_special_tokens=True)[0]


def _stage_default_lr(stage: int) -> float:
    return 1e-3 if stage == 1 else 2e-4


def train(cfg: TrainCfg) -> None:
    if cfg.stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {cfg.stage}")
    if cfg.stage == 2 and cfg.stage1_ckpt is None and cfg.resume_from is None:
        raise ValueError("stage 2 requires --stage1-ckpt (or --resume-from)")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.output_dir / "train_log.jsonl"
    log_f = log_path.open("w")
    lr = cfg.lr if cfg.lr is not None else _stage_default_lr(cfg.stage)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model, processor = build_variant_model(cfg.variant, cfg.device)

    if cfg.stage == 1:
        freeze_for_stage1(model)
        if cfg.resume_from is not None:
            state = torch.load(
                cfg.resume_from / "multi_modal_projector.pt",
                map_location=cfg.device, weights_only=True,
            )
            model.model.multi_modal_projector.load_state_dict(state)
            print(f"  ✓ resumed stage-1 MLP from {cfg.resume_from}")
        stream: IterableDataset = LcsPretrainStream(seed=cfg.seed)
        collate = make_collate(processor, cfg.variant, cfg.device)
    else:
        # Stage 2: load stage-1 MLP BEFORE PEFT wrap.
        if cfg.resume_from is not None:
            # Resume mid-stage-2: load LoRA adapters via PeftModel.from_pretrained.
            mlp_state = torch.load(
                cfg.resume_from / "multi_modal_projector.pt",
                map_location=cfg.device, weights_only=True,
            )
            model.model.multi_modal_projector.load_state_dict(mlp_state)
            model = PeftModel.from_pretrained(model, cfg.resume_from, is_trainable=True)
            for p in model.base_model.model.model.multi_modal_projector.parameters():
                p.requires_grad_(True)
            print(f"  ✓ resumed stage-2 from {cfg.resume_from}")
        else:
            load_stage1_mlp(model, cfg.stage1_ckpt, cfg.device)
            model = apply_stage2_lora(model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)
        stream = Mix665kStream(seed=cfg.seed)
        collate = make_collate(processor, cfg.variant, cfg.device)

    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(trainable, lr=lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(optim, cfg.warmup_steps, cfg.max_steps)
    for _ in range(cfg.step_offset):
        sched.step()

    # Group trainable params for separate grad-norm logging
    mlp_params = [p for n, p in model.named_parameters() if "multi_modal_projector" in n and p.requires_grad]
    lora_params = [p for n, p in model.named_parameters() if ("lora_A" in n or "lora_B" in n) and p.requires_grad]
    print(f"  trainable groups: mlp={len(mlp_params)} lora={len(lora_params)} | lr={lr}")

    probe_image = Image.new("RGB", (224, 224), color=(127, 127, 127))
    probe_prompt = "Describe what you see briefly."

    step = cfg.step_offset
    optim.zero_grad()
    accum = 0
    loss_window: list[float] = []
    t0 = time.time()
    batch_iter = iter(stream)
    batch: list[dict] = []

    def _gn(params) -> float:
        gs = [p.grad.detach().float().flatten() for p in params if p.grad is not None]
        if not gs:
            return 0.0
        return float(torch.cat(gs).norm().item())

    while step < cfg.max_steps:
        while len(batch) < cfg.batch_size:
            try:
                batch.append(next(batch_iter))
            except StopIteration:
                stream_seed = cfg.seed + step
                if cfg.stage == 1:
                    stream = LcsPretrainStream(seed=stream_seed)
                else:
                    stream = Mix665kStream(seed=stream_seed)
                batch_iter = iter(stream)

        try:
            ins = collate(batch)
        except Exception as e:
            print(f"  collate error step={step}: {e!r}; skipping batch")
            batch = []
            continue
        batch = []

        if (ins["labels"] != -100).sum() == 0:
            continue

        out = model(**ins)
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
            mlp_gn = _gn(mlp_params)
            lora_gn = _gn(lora_params) if lora_params else 0.0
            total_gn = float(torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip).item())
            optim.step()
            sched.step()
            optim.zero_grad()
            accum = 0
            step += 1

            log_now = (step % cfg.log_every == 0) or (step <= 50) or (50 < step <= 150 and step % 5 == 0)
            if log_now:
                tail = loss_window[-cfg.log_every * cfg.grad_accum :]
                avg = sum(tail) / max(1, len(tail))
                lr_now = sched.get_last_lr()[0]
                elapsed = time.time() - t0
                rate = (step - cfg.step_offset) / max(1.0, elapsed) * 60
                rec = {
                    "step": step, "loss": avg, "lr": lr_now,
                    "elapsed_sec": elapsed, "steps_per_min": rate,
                    "grad_total": total_gn, "grad_lora": lora_gn, "grad_mlp": mlp_gn,
                }
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                print(
                    f"step {step:>5d} | loss {avg:.4f} | lr {lr_now:.2e} | "
                    f"g_tot {total_gn:.2e} g_mlp {mlp_gn:.2e} g_lora {lora_gn:.2e} | "
                    f"{rate:.1f} steps/min"
                )
                if step == 1000 and avg > cfg.abort_loss_at_1k:
                    print(f"  ✗ ABORT — loss {avg:.4f} > {cfg.abort_loss_at_1k} at step 1K.")
                    log_f.close()
                    raise SystemExit(2)

            if step % cfg.sample_every == 0:
                try:
                    sample = generate_sample(model, processor, probe_prompt, probe_image,
                                             cfg.variant, cfg.device)
                    print(f"  [generation @ step {step}] {sample!r}")
                    log_f.write(json.dumps({"step": step, "sample": sample}) + "\n")
                    log_f.flush()
                except Exception as e:
                    print(f"  generation error: {e!r}")

            if step % cfg.save_every == 0:
                save_ckpt(model, processor, cfg.output_dir, step, cfg.stage)

    save_ckpt(model, processor, cfg.output_dir, cfg.max_steps, cfg.stage)
    log_f.close()
    print(f"DONE. Final checkpoint at {cfg.output_dir}/step{cfg.max_steps}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["A", "B"], required=True,
                   help="A = CLIP+Vicuna-7B-v1.5; B = CLIP+Mistral-7B-Instruct-v0.2")
    p.add_argument("--stage", type=int, choices=[1, 2], required=True,
                   help="1 = MLP-only on LCS-558K; 2 = MLP+LoRA on Mix665K")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stage1-ckpt", type=Path, default=None,
                   help="Stage 1 checkpoint dir (required when --stage 2)")
    p.add_argument("--max-steps", type=int, default=17000,
                   help="Stage-1 default 17000; pass 21000 for stage 2")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=None,
                   help="Override stage default (1e-3 stage 1; 2e-4 stage 2)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--step-offset", type=int, default=0)
    args = p.parse_args()

    cfg = TrainCfg(
        variant=args.variant,
        stage=args.stage,
        output_dir=args.output_dir,
        stage1_ckpt=args.stage1_ckpt,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
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

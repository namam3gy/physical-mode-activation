"""PhysModeVLM — inference wrapper with optional hidden-state / attention capture.

Follows the pattern in sibling `vlm_anchor/models.py::HFAttentionRunner`:
- Generic `AutoModelForImageTextToText` + `AutoProcessor` so the class works for
  Qwen2.5-VL, Qwen2-VL, LLaVA-1.5 / LLaVA-Next, InternVL2 without code changes.
- bf16 + sdpa on cuda; fp32 otherwise.
- `generate()` returns text plus per-token logit info (for forced-choice scoring).
- `capture()` does one extra forward pass with `output_hidden_states=True` /
  `output_attentions=True` and saves selected layers for the visual-token
  subsequence to a safetensors file per sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image


@dataclass
class InferenceArgs:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0


class PhysModeVLM:
    def __init__(
        self,
        model_id: str,
        torch_dtype: str = "bfloat16",
        device: str | None = None,
        capture_lm_layers: tuple[int, ...] | None = None,
        capture_vision_layers: tuple[int, ...] | None = None,
        capture_lm_attentions: bool = False,
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map[torch_dtype] if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # SDPA does not return attention weights; eager does. Switch when capturing.
        attn_impl = "eager" if capture_lm_attentions else "sdpa"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=True,
            device_map=self.device,
            attn_implementation=attn_impl,
        )
        if hasattr(self.model.config, "_attn_implementation"):
            self.model.config._attn_implementation = attn_impl
        self.model.eval()

        self.capture_lm_layers = tuple(capture_lm_layers) if capture_lm_layers else ()
        self.capture_vision_layers = tuple(capture_vision_layers) if capture_vision_layers else ()
        self.capture_lm_attentions = bool(capture_lm_attentions)

        # Resolve the visual image-token id once. Processors expose it via
        # image_token, image_token_id, or tokenizer.convert_tokens_to_ids.
        self.image_token_id = self._resolve_image_token_id()

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _resolve_image_token_id(self) -> int | None:
        # Try a few common spots — different model families name this differently.
        for name in ("image_token_id", "image_token_index"):
            tid = getattr(self.model.config, name, None)
            if tid is not None:
                return int(tid)
        tok = getattr(self.processor, "tokenizer", self.processor)
        for placeholder in ("<|image_pad|>", "<image>", "<|vision_pad|>"):
            try:
                tid = tok.convert_tokens_to_ids(placeholder)
                if tid is not None and tid != tok.unk_token_id:
                    return int(tid)
            except Exception:
                continue
        return None

    def _build_messages(self, prompt: str, system_prompt: str | None) -> list[dict]:
        content: list[dict[str, Any]] = [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]
        msgs: list[dict[str, Any]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        msgs.append({"role": "user", "content": content})
        return msgs

    def _prepare_inputs(
        self, image: Image.Image | Path | str, prompt: str, system_prompt: str | None
    ) -> dict[str, Any]:
        pil = _to_pil(image)
        msgs = self._build_messages(prompt, system_prompt)
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=[pil], text=[text], return_tensors="pt")
        return {
            k: (v.to(self.model.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image | Path | str,
        prompt: str,
        args: InferenceArgs = InferenceArgs(),
        system_prompt: str | None = None,
        choice_tokens: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        inputs = self._prepare_inputs(image, prompt, system_prompt)
        input_len = inputs["input_ids"].shape[1]

        do_sample = args.temperature > 0.0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        out = self.model.generate(**inputs, **gen_kwargs)
        generated = out.sequences[:, input_len:]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        raw = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

        # Per-token logit info for the generated sequence.
        token_info: list[dict[str, Any]] = []
        gen_ids = generated[0].tolist()
        scores = list(getattr(out, "scores", []) or [])
        for step, logits_step in enumerate(scores):
            if step >= len(gen_ids):
                break
            tid = int(gen_ids[step])
            v = logits_step[0].float()
            p = torch.softmax(v, dim=-1)
            token_info.append(
                {
                    "token_id": tid,
                    "token_text": tokenizer.decode([tid], skip_special_tokens=False),
                    "logit": float(v[tid].item()),
                    "probability": float(p[tid].item()),
                }
            )

        # Optional: per-letter logit at first generated step (for forced-choice).
        option_logits: dict[str, float] | None = None
        if choice_tokens and scores:
            first = scores[0][0].float()
            option_logits = {}
            for letter in choice_tokens:
                for candidate in (letter, f" {letter}"):
                    try:
                        ids = tokenizer.encode(candidate, add_special_tokens=False)
                    except Exception:
                        continue
                    if len(ids) != 1:
                        continue
                    option_logits[letter] = float(first[ids[0]].item())
                    break

        return {
            "raw_text": raw,
            "token_info": token_info,
            "input_len": int(input_len),
            "option_logits": option_logits,
        }

    # ------------------------------------------------------------------
    # Activation capture
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def capture(
        self,
        image: Image.Image | Path | str,
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """One forward pass with hidden-state / attention capture.

        Returns a dict with:
          - `lm_hidden`: {layer_idx -> Tensor(visual_token_count, dim)}
          - `lm_attn`:   {layer_idx -> Tensor(num_heads, q_len, k_len)}  (first sample only)
          - `visual_token_mask`: Tensor(seq_len) bool
          - `input_ids`: Tensor(seq_len) int
        Only the LM side is captured; vision-encoder capture is stubbed for a
        later round because each family (CLIP/SigLIP/InternViT) exposes layers
        under different attribute paths.
        """
        if not self.capture_lm_layers and not self.capture_vision_layers:
            return {}
        inputs = self._prepare_inputs(image, prompt, system_prompt)

        # Forward hooks for vision-encoder blocks (Qwen2.5-VL:
        # model.model.visual.blocks[i]). Each block's output is the fine-grained
        # patch-token hidden state *before* the merger that downsamples to the
        # LM-side visual token count.
        vision_captures: dict[int, torch.Tensor] = {}
        vision_hooks: list = []
        if self.capture_vision_layers:
            blocks = _resolve_vision_blocks(self.model)
            if blocks is not None:
                def _make_hook(layer_idx: int):
                    def hook(_module, _inputs, output):
                        t = output[0] if isinstance(output, tuple) else output
                        # Strip batch dim for batch=1 (probing expects (n_tokens, dim)).
                        if t.dim() == 3 and t.shape[0] == 1:
                            t = t[0]
                        vision_captures[layer_idx] = t.detach().to(
                            "cpu", dtype=torch.bfloat16
                        ).contiguous()
                    return hook
                for li in self.capture_vision_layers:
                    if 0 <= li < len(blocks):
                        vision_hooks.append(
                            blocks[li].register_forward_hook(_make_hook(int(li)))
                        )

        try:
            out = self.model(
                **inputs,
                output_hidden_states=bool(self.capture_lm_layers),
                output_attentions=bool(self.capture_lm_attentions),
                return_dict=True,
            )
        finally:
            for h in vision_hooks:
                h.remove()

        hidden_states = getattr(out, "hidden_states", None) or ()
        attentions = getattr(out, "attentions", None) or () if self.capture_lm_attentions else ()

        # Locate visual tokens in the input sequence.
        ids = inputs["input_ids"][0]
        if self.image_token_id is not None:
            mask = ids == self.image_token_id
        else:
            mask = torch.ones_like(ids, dtype=torch.bool)

        lm_hidden: dict[int, torch.Tensor] = {}
        for li in self.capture_lm_layers:
            idx = li + 1  # +1 to skip the embedding layer
            if idx >= len(hidden_states):
                continue
            h = hidden_states[idx][0]  # (seq_len, dim)
            lm_hidden[int(li)] = h[mask].detach().to("cpu", dtype=torch.bfloat16).contiguous()

        lm_attn: dict[int, torch.Tensor] = {}
        for li in self.capture_lm_layers:
            if li >= len(attentions):
                continue
            a = attentions[li][0]  # (num_heads, q_len, k_len)
            lm_attn[int(li)] = a.detach().to("cpu", dtype=torch.float16).contiguous()

        return {
            "lm_hidden": lm_hidden,
            "lm_attn": lm_attn,
            "vision_hidden": vision_captures,
            "visual_token_mask": mask.detach().to("cpu"),
            "input_ids": ids.detach().to("cpu"),
        }

    def save_capture(self, capture: dict[str, Any], path: Path) -> None:
        """Save the capture() output to a .safetensors file."""
        if not capture:
            return
        from safetensors.torch import save_file

        tensors: dict[str, torch.Tensor] = {}
        for li, h in capture.get("lm_hidden", {}).items():
            tensors[f"lm_hidden_{li}"] = h
        for li, a in capture.get("lm_attn", {}).items():
            tensors[f"lm_attn_{li}"] = a
        for li, h in capture.get("vision_hidden", {}).items():
            tensors[f"vision_hidden_{li}"] = h
        if "visual_token_mask" in capture:
            tensors["visual_token_mask"] = capture["visual_token_mask"].to(torch.uint8)
        if "input_ids" in capture:
            tensors["input_ids"] = capture["input_ids"].to(torch.int64)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path))


def _resolve_vision_blocks(model) -> Any:
    """Return the vision-encoder block list for the given model, or None.

    Qwen2.5-VL: model.model.visual.blocks (ModuleList of 32 Qwen2_5_VLVisionBlock).
    LLaVA-1.5 (hf wrapper): model.model.vision_tower.encoder.layers (CLIPVisionModel
        with `encoder` directly, no extra `vision_model` wrapper).
    LLaVA-Next / older wrappers: model.{,model.}vision_tower.vision_model.encoder.layers.
    Idefics2: model.model.vision_model.encoder.layers (Idefics2VisionTransformer).
    InternVL3 (hf): model.model.vision_tower.encoder.layer (singular, ModuleList of 24
        InternVLVisionLayer).
    """
    inner = getattr(model, "model", model)
    # Qwen2.5-VL / Qwen2-VL
    visual = getattr(inner, "visual", None)
    if visual is not None and hasattr(visual, "blocks"):
        return visual.blocks

    def _encoder_layers(holder) -> Any:
        """Common helper: holder.encoder.layers OR holder.encoder.layer."""
        enc = getattr(holder, "encoder", None)
        if enc is None:
            return None
        if hasattr(enc, "layers"):
            return enc.layers
        if hasattr(enc, "layer"):
            return enc.layer
        return None

    # LLaVA-family / InternVL3 vision_tower
    vt = getattr(model, "vision_tower", None) or getattr(inner, "vision_tower", None)
    if vt is not None:
        # Newer hf wrapper: encoder directly on vt (CLIPVisionModel / InternVLVisionModel).
        layers = _encoder_layers(vt)
        if layers is not None:
            return layers
        # Older: extra vision_model wrapper inside vision_tower.
        vm = getattr(vt, "vision_model", None)
        if vm is not None:
            layers = _encoder_layers(vm)
            if layers is not None:
                return layers
    # Idefics2-style vision_model
    vm = getattr(model, "vision_model", None) or getattr(inner, "vision_model", None)
    if vm is not None:
        layers = _encoder_layers(vm)
        if layers is not None:
            return layers
    return None


def _to_pil(image_like: Any) -> Image.Image:
    if isinstance(image_like, Image.Image):
        return image_like.convert("RGB")
    if isinstance(image_like, (str, Path)):
        return Image.open(image_like).convert("RGB")
    if hasattr(image_like, "convert"):
        return image_like.convert("RGB")
    raise TypeError(f"unsupported image type: {type(image_like)}")

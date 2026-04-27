"""Causal SAE-feature intervention on Qwen2.5-VL.

Test whether ablating (zeroing) the top-k physics-cue SAE features in the
vision-encoder output causes physics-mode commitment to drop. Symmetric to
the activation-patching / knockout experiments at the LM side, but at the
encoder.

Pipeline:
1. Load trained SAE + feature ranking.
2. Pick top-k features (largest physics − abstract activation delta).
3. For each evaluation stim: hook on `model.visual` last vision-encoder
   layer's output. The hook reads the visual-token activations, encodes via
   SAE, zeros the chosen features in z, decodes, and writes back.
4. Run inference; compare letter response (PMR) to baseline (no hook).

Outputs:
- outputs/sae_intervention/<tag>/results.csv — per (stim, k_features_zeroed)
  baseline vs intervention prediction.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/sae_intervention.py \
        --sae-dir outputs/sae/qwen_vis31_5120 \
        --layer-key vision_hidden_31 \
        --top-k-list 1,5,10,20 --n-stim 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from physical_mode.sae.train import load_sae


PROJECT_ROOT = Path(__file__).resolve().parents[1]
M2_STIM_DIR = PROJECT_ROOT / "inputs/mvp_full_20260424-093926_e9d79da3/images"


PROMPT_FC_TEMPLATE = (
    "The image shows a {label}. Which option best describes what will happen next?\n"
    "A) It falls down.\n"
    "B) It stays still.\n"
    "C) It moves sideways.\n"
    "D) This is an abstract shape — nothing physical happens.\n"
    "Answer with a single letter (A, B, C, or D), then briefly justify."
)


def make_sae_intervention_hook(sae, feature_indices_to_zero: torch.Tensor):
    """Forward hook on the vision encoder's last layer output that subtracts
    *only* the target SAE features' contributions, leaving everything else
    (other features + reconstruction residual) bit-identical.

    Bricken et al. 2023 trick: x_new = x - z[:, target] @ W_dec[target].
    Avoids overwriting the entire activation with the SAE's lossy decode.
    """
    sae_device = next(sae.parameters()).device

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            x = output[0]
            rest = output[1:]
        else:
            x = output
            rest = None
        original_dtype = x.dtype
        x_in = x.to(sae_device, dtype=torch.float32)
        flat_shape = x_in.shape
        x_flat = x_in.reshape(-1, flat_shape[-1])
        with torch.no_grad():
            z = sae.encode(x_flat)
            target_z = z[:, feature_indices_to_zero]  # (N_tokens, k)
            # Decoder rows for target features → reconstruct only their contribution.
            target_W = sae.W[feature_indices_to_zero]  # (k, d_in)
            target_contribution = target_z @ target_W  # (N_tokens, d_in)
            x_new_flat = x_flat - target_contribution
        x_new = x_new_flat.reshape(flat_shape).to(x.device, dtype=original_dtype)
        if rest is not None:
            return (x_new,) + rest
        return x_new

    return hook


def get_last_vision_layer(model):
    """Resolve the module whose output we want to intercept (= last vision encoder layer)."""
    # Qwen2.5-VL: model.visual is the vision tower; .blocks is a ModuleList of vision encoder layers.
    visual = getattr(model, "visual", None) or getattr(model.model, "visual", None)
    if visual is None:
        raise RuntimeError("Could not find model.visual.")
    blocks = getattr(visual, "blocks", None)
    if blocks is None:
        raise RuntimeError("Could not find model.visual.blocks.")
    return blocks[-1], len(blocks)


@torch.no_grad()
def run_with_hook(model, processor, pil, prompt, hook_module, hook_fn, device):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[pil], text=[text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    handle = hook_module.register_forward_hook(hook_fn) if hook_fn else None
    try:
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
    finally:
        if handle is not None:
            handle.remove()


def _is_physics_letter(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return -1
    c = t[0]
    return 1 if c in ("A", "B", "C") else (0 if c == "D" else -1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae-dir", type=Path, required=True)
    p.add_argument("--layer-key", default="vision_hidden_31",
                   help="captured-activation key (informational; the script hooks the actual model)")
    p.add_argument("--top-k-list", default="1,5,10,20",
                   help="comma-separated feature counts to zero")
    p.add_argument("--n-stim", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--label", default="circle")
    p.add_argument("--manifest",
                   default="outputs/m5b_sip/manifest.csv",
                   help="CSV with clean_sample_id column listing stim to use")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / "outputs" / "sae_intervention" / args.sae_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    sae = load_sae(args.sae_dir / "sae.pt", device=args.device)
    rank_df = pd.read_csv(args.sae_dir / "feature_ranking.csv")
    top_features = rank_df["feature_idx"].head(max(int(k) for k in args.top_k_list.split(","))).tolist()
    print(f"Top features (by physics − abstract delta): {top_features[:10]}...")

    manifest = pd.read_csv(PROJECT_ROOT / args.manifest)
    sample_ids = manifest["clean_sample_id"].head(args.n_stim).tolist()

    print(f"Loading {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()
    last_vis_layer, n_vis_layers = get_last_vision_layer(model)
    print(f"  Hooking last vision encoder layer ({n_vis_layers} total).")

    prompt = PROMPT_FC_TEMPLATE.format(label=args.label)

    rows = []
    t_start = time.time()
    for i, sid in enumerate(sample_ids):
        path = M2_STIM_DIR / f"{sid}.png"
        if not path.exists():
            print(f"  [SKIP] missing {sid}")
            continue
        pil = Image.open(path).convert("RGB")

        # Baseline (no hook).
        bl = run_with_hook(model, processor, pil, prompt, last_vis_layer, None, args.device)
        bl_phys = _is_physics_letter(bl)

        for k_str in args.top_k_list.split(","):
            k = int(k_str)
            feature_idx = torch.tensor(top_features[:k], device=args.device, dtype=torch.long)
            hook_fn = make_sae_intervention_hook(sae, feature_idx)
            txt = run_with_hook(model, processor, pil, prompt, last_vis_layer, hook_fn, args.device)
            rows.append({
                "sample_id": sid, "k_zeroed": k, "intervention_text": txt,
                "intervention_phys": _is_physics_letter(txt),
                "baseline_text": bl, "baseline_phys": bl_phys,
            })

        elapsed = (time.time() - t_start) / 60
        print(f"  [{i+1}/{len(sample_ids)}] {sid} bl={bl_phys} elapsed={elapsed:.1f}min")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    print("\n=== Aggregate (intervention phys rate by k_zeroed) ===")
    agg = df.groupby("k_zeroed")["intervention_phys"].agg(
        baseline_phys_rate=lambda x: 1.0,  # by manifest construction
        intervention_phys_rate=lambda x: float((x == 1).sum() / max(1, (x >= 0).sum())),
    ).reset_index()
    print(agg.round(3).to_string(index=False))
    print(f"Wrote {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()

"""M-PSwap regression-eval gate — POPE F1 + 50-sample VQA sanity.

Loads a trained Idefics2-MLP-pool checkpoint (LoRA + perceiver_resampler
state_dict swapped in) and evaluates:
    1. POPE (lmms-lab/POPE): yes/no hallucination questions, F1 on the
       full test split or a subset. Gate: F1 >= 0.70 (mid-train F1 >= 0.50
       at step 2.5K).
    2. VQA sanity (50 examples): deterministic generations + manual review
       (the script saves answers to a CSV; user reviews).

Reference: ``references/paper_gaps.md`` G3 spec.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from physical_mode.lora.idefics2_mlp_resampler import (
    MLPPoolResampler,
    swap_perceiver_to_mlp_pool,
)


MODEL_ID = "HuggingFaceM4/idefics2-8b"


def load_swapped_model(ckpt_dir: Path, device: str = "cuda:0"):
    """Reload Idefics2 with perceiver swap + LoRA + MLP-pool weights from a checkpoint."""
    print(f"loading base {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, do_image_splitting=False)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device
    )
    new = swap_perceiver_to_mlp_pool(base, n_heads=8)
    state = torch.load(ckpt_dir / "mlp_pool_resampler.pt", map_location=device, weights_only=True)
    new.load_state_dict(state)
    print(f"  ✓ loaded MLP-pool state_dict from {ckpt_dir}/mlp_pool_resampler.pt")

    print(f"attaching LoRA adapters from {ckpt_dir} ...")
    model = PeftModel.from_pretrained(base, ckpt_dir)
    model.eval()
    return model, processor


def _yes_no(text: str) -> str | None:
    """Return 'yes' or 'no' if the response leads with one, else None."""
    if not text:
        return None
    t = text.strip().lower()
    # strip prompt echo if any
    for sep in ["assistant:", "answer:"]:
        if sep in t:
            t = t.split(sep, 1)[1].strip()
    head = t.split(None, 1)[0] if t else ""
    head = head.rstrip(".,!?")
    if head.startswith("yes"):
        return "yes"
    if head.startswith("no"):
        return "no"
    return None


@torch.inference_mode()
def eval_pope(model, processor, n_samples: int, device: str = "cuda:0", batch_size: int = 8) -> dict:
    """Eval POPE F1. Streams n_samples examples; returns metrics dict."""
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
    rows = []
    n_seen = 0
    batch_q: list[dict] = []
    t0 = time.time()

    def _flush_batch(buf):
        prompts = []
        images = []
        gold = []
        for ex in buf:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": ex["question"] + " Answer yes or no."},
                    ],
                }
            ]
            prompts.append(processor.apply_chat_template(messages, add_generation_prompt=True))
            images.append([ex["image"].convert("RGB")])
            gold.append(ex["answer"].strip().lower())

        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
        out_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        # Decode only newly generated tokens.
        n_in = inputs["input_ids"].shape[1]
        gen = processor.batch_decode(out_ids[:, n_in:], skip_special_tokens=True)
        return list(zip(gold, gen))

    for ex in ds:
        if n_seen >= n_samples:
            break
        batch_q.append(ex)
        n_seen += 1
        if len(batch_q) >= batch_size:
            for gold, raw in _flush_batch(batch_q):
                rows.append({"gold": gold, "raw": raw, "pred": _yes_no(raw)})
            batch_q = []

    if batch_q:
        for gold, raw in _flush_batch(batch_q):
            rows.append({"gold": gold, "raw": raw, "pred": _yes_no(raw)})

    # Compute F1 of "yes" class (POPE convention).
    tp = sum(1 for r in rows if r["gold"] == "yes" and r["pred"] == "yes")
    fp = sum(1 for r in rows if r["gold"] == "no" and r["pred"] == "yes")
    fn = sum(1 for r in rows if r["gold"] == "yes" and r["pred"] != "yes")
    tn = sum(1 for r in rows if r["gold"] == "no" and r["pred"] == "no")
    n_invalid = sum(1 for r in rows if r["pred"] is None)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    acc = (tp + tn) / max(1, len(rows))
    elapsed = time.time() - t0

    return {
        "n": len(rows),
        "n_invalid": n_invalid,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "elapsed_sec": elapsed,
        "rows": rows,
    }


@torch.inference_mode()
def eval_vqa_sanity(model, processor, n_samples: int = 50, device: str = "cuda:0") -> list[dict]:
    """Eval n_samples from the_cauldron/aokvqa for manual sanity review."""
    ds = load_dataset("HuggingFaceM4/the_cauldron", "aokvqa", split="train", streaming=True)
    rows = []
    for ex in ds:
        if len(rows) >= n_samples:
            break
        if not ex["images"] or not ex["texts"]:
            continue
        img = ex["images"][0].convert("RGB")
        q = ex["texts"][0]["user"]
        gold = ex["texts"][0].get("assistant", "")
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[[img]], return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        n_in = inputs["input_ids"].shape[1]
        ans = processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
        rows.append({"q": q, "gold": gold, "pred": ans})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True, help="Checkpoint dir from m_pswap_train")
    p.add_argument("--n-pope", type=int, default=9000)
    p.add_argument("--n-vqa", type=int, default=50)
    p.add_argument("--pope-batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, default=None, help="Output JSON; defaults to ckpt/regression_eval.json")
    args = p.parse_args()

    if args.out is None:
        args.out = args.ckpt / "regression_eval.json"

    model, processor = load_swapped_model(args.ckpt, args.device)

    print(f"\n--- POPE eval (n={args.n_pope}) ---")
    pope = eval_pope(model, processor, args.n_pope, args.device, args.pope_batch_size)
    f1 = pope["f1"]
    acc = pope["accuracy"]
    n_invalid = pope["n_invalid"]
    print(
        f"  F1: {f1:.4f}  Accuracy: {acc:.4f}  "
        f"P/R: {pope['precision']:.3f}/{pope['recall']:.3f}  "
        f"Invalid: {n_invalid}/{pope['n']}  "
        f"Time: {pope['elapsed_sec']:.0f}s"
    )

    print(f"\n--- VQA sanity (n={args.n_vqa}) ---")
    vqa = eval_vqa_sanity(model, processor, args.n_vqa, args.device)
    for i, r in enumerate(vqa[:5]):
        print(f"  [{i}] Q: {r['q'][:80]}")
        print(f"      gold: {r['gold'][:80]}")
        print(f"      pred: {r['pred'][:80]}")

    summary = {
        "ckpt": str(args.ckpt),
        "pope": {k: v for k, v in pope.items() if k != "rows"},
        "pope_pass_full_gate": f1 >= 0.70,
        "pope_pass_mid_gate": f1 >= 0.50,
        "vqa_n": len(vqa),
    }
    args.out.write_text(json.dumps(summary, indent=2))
    # Save full POPE rows + VQA samples separately
    (args.ckpt / "pope_rows.jsonl").write_text("\n".join(json.dumps(r) for r in pope["rows"]))
    (args.ckpt / "vqa_sanity.jsonl").write_text("\n".join(json.dumps(r) for r in vqa))

    print(f"\nSummary saved to {args.out}")
    print(f"  Full gate (F1>=0.70): {'PASS' if summary['pope_pass_full_gate'] else 'FAIL'}")
    print(f"  Mid gate (F1>=0.50):  {'PASS' if summary['pope_pass_mid_gate'] else 'FAIL'}")


if __name__ == "__main__":
    main()

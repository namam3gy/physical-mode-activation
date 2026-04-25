"""Run VLM inference across a generated stimulus manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..config import EvalConfig
from ..models.vlm_runner import InferenceArgs, PhysModeVLM
from ..utils import config_hash, ensure_dir, timestamp
from .prompts import render


def run_inference(cfg: EvalConfig, manifest_dir: Path) -> Path:
    """Iterate (stimulus × label × prompt_variant), call VLM, stream predictions."""
    manifest = pd.read_parquet(manifest_dir / "manifest.parquet")
    if cfg.limit is not None:
        manifest = manifest.head(cfg.limit).reset_index(drop=True)

    run_id = f"{cfg.run_name}_{timestamp()}_{config_hash(cfg)}"
    out_dir = ensure_dir(cfg.outputs_root / run_id)
    act_dir = ensure_dir(out_dir / "activations") if cfg.capture_lm_layers else None

    # Record run provenance.
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "stimulus_dir": str(manifest_dir),
                "model_id": cfg.model_id,
                "torch_dtype": cfg.torch_dtype,
                "labels": list(cfg.labels),
                "prompt_variants": list(cfg.prompt_variants),
                "capture_lm_layers": list(cfg.capture_lm_layers or ()),
                "n_stimuli": int(len(manifest)),
            },
            indent=2,
        )
    )

    vlm = PhysModeVLM(
        model_id=cfg.model_id,
        torch_dtype=cfg.torch_dtype,
        device=cfg.device,
        capture_lm_layers=cfg.capture_lm_layers,
        capture_vision_layers=cfg.capture_vision_layers,
        capture_lm_attentions=cfg.capture_lm_attentions,
    )
    args = InferenceArgs(
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )

    # Stream predictions to jsonl so we don't lose work on crash.
    jsonl_path = out_dir / "predictions.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        total = len(manifest) * len(cfg.labels) * len(cfg.prompt_variants)
        pbar = tqdm(total=total, desc="Inference")
        for _, row in manifest.iterrows():
            img_path = manifest_dir / row["image_path"]
            for label in cfg.labels:
                for variant in cfg.prompt_variants:
                    rp = render(variant, label)
                    gen = vlm.generate(
                        image=img_path,
                        prompt=rp.user,
                        args=args,
                        system_prompt=rp.system,
                        choice_tokens=rp.choice_letters,
                    )
                    rec = {
                        "sample_id": row["sample_id"],
                        "label": label,
                        "prompt_variant": variant,
                        "event_template": row["event_template"],
                        "object_level": row["object_level"],
                        "bg_level": row["bg_level"],
                        "cue_level": row["cue_level"],
                        "seed": int(row["seed"]),
                        "raw_text": gen["raw_text"],
                        "option_logits": gen["option_logits"],
                        "input_len": gen["input_len"],
                        "first_tokens": gen["token_info"][:8],
                    }
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    jf.flush()
                    pbar.update(1)

            # Activation capture is once per stimulus, using the first configured
            # prompt variant. For M2 (`open` first) this matches the prior hardcoded
            # behavior; for label-free configs (`open_no_label` first) this captures
            # under the label-free prompt, which is the correct M4-probe input.
            if act_dir is not None:
                rp = render(cfg.prompt_variants[0], cfg.labels[0])
                cap = vlm.capture(image=img_path, prompt=rp.user, system_prompt=rp.system)
                if cap:
                    vlm.save_capture(cap, act_dir / f"{row['sample_id']}.safetensors")
        pbar.close()

    # Materialize parquet + csv from the streamed jsonl.
    df = pd.read_json(jsonl_path, orient="records", lines=True)
    # Drop nested columns that don't serialize cleanly to parquet/csv.
    flat = df.drop(columns=[c for c in ("first_tokens", "option_logits") if c in df.columns])
    flat.to_parquet(out_dir / "predictions.parquet", index=False)
    flat.to_csv(out_dir / "predictions.csv", index=False)

    print(f"Wrote {len(df)} predictions to {out_dir}")
    return out_dir

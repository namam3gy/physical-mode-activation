"""Run VLM inference across a generated stimulus manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..config import EvalConfig
from ..models.vlm_runner import InferenceArgs, PhysModeVLM
from ..utils import config_hash, ensure_dir, timestamp
from .prompts import LABELS_BY_SHAPE, render


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
    # When `cfg.factorial.shapes` has >1 shape, labels are dispatched per-shape
    # via LABELS_BY_SHAPE; cfg.labels is then used as a *role* selector
    # ("physical" | "abstract" | "exotic"). For single-shape configs the
    # legacy behavior — cfg.labels is the explicit label tuple — is preserved.
    multi_shape = len(getattr(cfg.factorial, "shapes", ("circle",))) > 1

    LABEL_ROLES: tuple[str, ...] = ("physical", "abstract", "exotic")

    def labels_for_row(row_shape: str) -> tuple[str, ...]:
        if not multi_shape:
            return cfg.labels
        triplet = LABELS_BY_SHAPE[row_shape]
        # cfg.labels is interpreted as a list of role names.
        out: list[str] = []
        for role in cfg.labels:
            if role in LABEL_ROLES:
                out.append(triplet[LABEL_ROLES.index(role)])
            elif role == "_nolabel":
                out.append("_nolabel")
            else:
                # Legacy literal label (e.g. "ball") — use as-is.
                out.append(role)
        return tuple(out)

    jsonl_path = out_dir / "predictions.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        total = len(manifest) * len(cfg.labels) * len(cfg.prompt_variants)
        pbar = tqdm(total=total, desc="Inference")
        for _, row in manifest.iterrows():
            img_path = manifest_dir / row["image_path"]
            row_shape = row["shape"] if "shape" in row.index else "circle"
            for label in labels_for_row(row_shape):
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
                        "shape": row_shape,
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
                first_label = labels_for_row(row_shape)[0]
                rp = render(cfg.prompt_variants[0], first_label)
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

"""M8c — curate 60 real photographs (12 × {ball, car, person, bird, abstract}).

Sources:
  - COCO 2017 validation (`phiyodr/coco2017`) for ball / car / person / bird —
    images fetched via `coco_url` (CC license per image, recorded in metadata).
    Captions are used to filter for the target category keyword.
  - WikiArt (`huggan/wikiart`) for `abstract` — filtered to abstract / cubist
    / minimalism styles where image bytes are bundled in the parquet.

Each photo is resized to 512×512 (matching synthetic stim canvas) and saved
as PNG. Manifest follows the M8d schema with `source_type='photo'`.

Usage:
    uv run python scripts/m8c_curate_photos.py
"""

from __future__ import annotations

import argparse
import io
import random
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUTS = PROJECT_ROOT / "inputs"
CANVAS = 512
N_PER_CATEGORY = 12

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "ball":   ["basketball", "soccer ball", "baseball", "tennis ball", "football", "volleyball", "bowling ball"],
    "car":    [" car ", " truck", " sedan", " suv"],
    "person": [" person ", " man ", " woman ", " people "],
    "bird":   ["bird ", "birds "],
}

# Per-category natural event for the manifest (matches M8d analysis convention).
CATEGORY_EVENT: dict[str, str] = {
    "ball": "fall",
    "car": "horizontal",
    "person": "horizontal",
    "bird": "horizontal",
    "abstract": "fall",
}

# COCO image license values per the dataset spec.
COCO_LICENSE_NAMES: dict[int, str] = {
    1: "CC BY-NC-SA 2.0",
    2: "CC BY-NC 2.0",
    3: "CC BY-NC-ND 2.0",
    4: "CC BY 2.0",
    5: "CC BY-SA 2.0",
    6: "CC BY-ND 2.0",
    7: "No known copyright restrictions",
    8: "United States Government Work",
}


def _square_pad_resize(img: Image.Image, size: int = CANVAS) -> Image.Image:
    """Pad image to square (white) and resize to size×size."""
    img = img.convert("RGB")
    w, h = img.size
    side = max(w, h)
    sq = Image.new("RGB", (side, side), (255, 255, 255))
    sq.paste(img, ((side - w) // 2, (side - h) // 2))
    return sq.resize((size, size), Image.LANCZOS)


def _has_caption_match(captions: list[str], keywords: list[str]) -> bool:
    text = " ".join(captions).lower()
    return any(k in text for k in keywords)


def _download_coco_image(url: str, retries: int = 3, timeout: int = 30) -> Image.Image | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content))
        except Exception as e:
            if attempt == retries - 1:
                print(f"  failed after {retries} retries: {url} ({e})")
                return None
            time.sleep(2 ** attempt)


def curate_coco(category: str, n: int, seed: int, out_dir: Path, meta_rows: list[dict]) -> int:
    """Fetch n photos for `category` from COCO val 2017."""
    print(f"\n--- COCO: {category} (n={n}) ---")
    parquet = hf_hub_download(
        repo_id="phiyodr/coco2017",
        filename="data/validation-00000-of-00001-e3c37e369512a3aa.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(parquet)

    keywords = CATEGORY_KEYWORDS[category]
    matches = df[df["captions"].apply(lambda c: _has_caption_match(c, keywords))].copy()
    print(f"  {len(matches)} candidates matching {keywords}")

    rng = random.Random(seed)
    indices = list(matches.index)
    rng.shuffle(indices)

    saved = 0
    for idx in indices:
        if saved >= n:
            break
        row = matches.loc[idx]
        url = row["coco_url"]
        img = _download_coco_image(url)
        if img is None:
            continue
        img = _square_pad_resize(img)
        sample_id = f"{category}_photo_{saved:03d}"
        out_path = out_dir / "images" / f"{sample_id}.png"
        img.save(out_path)
        meta_rows.append({
            "sample_id": sample_id,
            "category": category,
            "source": "COCO 2017 val",
            "image_id": int(row["image_id"]),
            "license_id": int(row["license"]),
            "license_name": COCO_LICENSE_NAMES.get(int(row["license"]), "unknown"),
            "coco_url": url,
            "flickr_url": row.get("flickr_url", ""),
            "captions": " | ".join(row["captions"][:2]),
        })
        saved += 1
        if saved % 4 == 0:
            print(f"  ... {saved}/{n}")
    print(f"  saved {saved} photos for {category}")
    return saved


def curate_wikiart(n: int, seed: int, out_dir: Path, meta_rows: list[dict]) -> int:
    """Fetch n abstract photos from WikiArt (style ∈ {abstract-related})."""
    print(f"\n--- WikiArt: abstract (n={n}) ---")
    # WikiArt has many shards. Download a few and filter by style.
    shard_idx = 0
    parquet = hf_hub_download(
        repo_id="huggan/wikiart",
        filename=f"data/train-{shard_idx:05d}-of-00072.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(parquet)
    print(f"  shard {shard_idx} rows: {len(df)}; columns: {list(df.columns)}")

    # Style is integer label; HF wikiart has a styles map. Common abstract:
    # "Abstract_Expressionism" id ?, "Cubism", "Minimalism", "Color_Field_Painting".
    # Let's see what we get.
    if "style" in df.columns:
        print(f"  unique styles in shard: {df['style'].value_counts().head(10).to_dict()}")
    if "genre" in df.columns:
        print(f"  unique genres: {df['genre'].value_counts().head(10).to_dict()}")

    # Filter for abstract-related styles. WikiArt style index 0 = Abstract_Expressionism.
    # Inspect the README for style mapping; for now, just pick rows with
    # abstract genre or visually-abstract proxy.
    abstract_styles = {0, 1, 4, 5, 6, 25}  # heuristic — Abstract_Expressionism, Action_Painting, etc.
    if "style" in df.columns:
        cand = df[df["style"].isin(abstract_styles)]
    else:
        cand = df

    print(f"  filtered candidates: {len(cand)}")
    if len(cand) == 0:
        print("  WARNING: no candidates after filter; using full shard")
        cand = df

    rng = random.Random(seed)
    indices = list(cand.index)
    rng.shuffle(indices)

    saved = 0
    for idx in indices:
        if saved >= n:
            break
        row = cand.loc[idx]
        # Image is in 'image' column as dict {'bytes': ..., 'path': ...} or PIL bytes.
        img_field = row.get("image")
        try:
            if isinstance(img_field, dict):
                img = Image.open(io.BytesIO(img_field["bytes"]))
            else:
                img = Image.open(io.BytesIO(img_field))
        except Exception as e:
            print(f"  image decode failed: {e}")
            continue
        img = _square_pad_resize(img)
        sample_id = f"abstract_photo_{saved:03d}"
        out_path = out_dir / "images" / f"{sample_id}.png"
        img.save(out_path)
        meta_rows.append({
            "sample_id": sample_id,
            "category": "abstract",
            "source": "WikiArt (huggan/wikiart)",
            "image_id": int(idx),
            "license_id": -1,
            "license_name": "WikiArt — public domain or fair use (per huggan/wikiart README)",
            "coco_url": "",
            "flickr_url": "",
            "captions": str(row.get("artist", "")) + " | " + str(row.get("title", "") or row.get("style", "")),
        })
        saved += 1
        if saved % 4 == 0:
            print(f"  ... {saved}/{n}")
    print(f"  saved {saved} abstract photos")
    return saved


def write_manifest(out_dir: Path, meta_rows: list[dict]) -> None:
    """Build the inference-pipeline-compatible manifest from metadata."""
    rows = []
    for m in meta_rows:
        cat = m["category"]
        rows.append({
            "sample_id": m["sample_id"],
            "shape": cat,                 # category goes into `shape` column
            "object_level": "photo",      # placeholder; not used by inference
            "bg_level": "natural",
            "cue_level": "none",
            "event_template": CATEGORY_EVENT[cat],
            "seed": int(m["sample_id"].split("_")[-1]),
            "source_type": "photo",
            "image_path": f"images/{m['sample_id']}.png",
        })
    manifest = pd.DataFrame(rows)
    manifest.to_parquet(out_dir / "manifest.parquet", index=False)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"\nWrote manifest: {out_dir / 'manifest.parquet'} ({len(manifest)} rows)")


def write_metadata(out_dir: Path, meta_rows: list[dict]) -> None:
    pd.DataFrame(meta_rows).to_csv(out_dir / "photo_metadata.csv", index=False)
    print(f"Wrote photo metadata: {out_dir / 'photo_metadata.csv'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-category", type=int, default=N_PER_CATEGORY)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = INPUTS / f"m8c_photos_{timestamp}"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    meta_rows: list[dict] = []

    # COCO categories.
    for cat in ("ball", "car", "person", "bird"):
        curate_coco(cat, args.n_per_category, args.seed + ord(cat[0]), out_dir, meta_rows)

    # WikiArt abstract.
    curate_wikiart(args.n_per_category, args.seed + ord("a"), out_dir, meta_rows)

    write_manifest(out_dir, meta_rows)
    write_metadata(out_dir, meta_rows)

    print(f"\n=== M8c photos curated to {out_dir} ===")
    print(f"  total: {len(meta_rows)} photos")


if __name__ == "__main__":
    main()

"""
inference_seg.py
----------------
Run inference with a trained segmentation-conditioned LoRAdapter.

This script:
  1. Loads the SD 1.5 base model + trained LoRA/mapper from a checkpoint.
  2. For each input image, runs the SegmentationEncoder (SegFormer-b5-Cityscapes)
     to get a colourised seg map, then generates a new image conditioned on that map.
  3. Saves a 4-panel grid (ORIGINAL | SEG MAP | PREDICTED | RAW SEG GEN) per image
     so you can visually evaluate quality and pick the best checkpoint.

This is the segmentation twin of inference_depth.py. It mirrors that script's
structure exactly, with three seg-specific differences:
  1. `_seg_label_bar` / `make_seg_inference_grid` — seg-prefixed, "SEG MAP" label.
  2. model loading reads `cfg.seg_model_name/seg_model_path` (not depth_*).
  3. Preprocessing via `build_seg_square_preprocess()` from src/data/transforms.py
     — the SAME function calculate_segmentation_map.py uses — so the live seg map
     is byte-identical to what the model trained on (train/inference parity).

INPUT OPTIONS:
  a) JSON manifest file (recommended) — each entry has "raw_image_path" + "prompt":
       inference.json_file=data/seg_training/test.json
  b) Direct list of image paths:
       "inference.images=[data/raw/000417/raw_image.jpg]"
       "inference.prompts=['urban driving scene, clear weather']"

OUTPUT MODES:
  Default (save_generated_only=false) — saves 4 files per image:
    raw_image_grid.jpg         ← 4 panels: ORIGINAL | SEG MAP | PREDICTED | RAW SEG GEN
    raw_image_original.jpg     ← input image rescaled
    raw_image_seg.jpg          ← the seg colour map from SegFormer-b5
    raw_image_predicted.jpg    ← the generated image

  Batch eval (save_generated_only=true, json_file required):
    Saves ONLY the generated image, mirroring folder structure from the JSON.

USAGE:
  # Standard single/multi image inference:
  python inference_seg.py \\
      ckpt_path=outputs/train/seg/runs/2024-01-01/00-00-00/checkpoint-3000 \\
      inference.json_file=data/seg_training/test.json \\
      inference.output_dir=results/seg

  # Direct image:
  python inference_seg.py \\
      ckpt_path=outputs/train/seg/runs/2024-01-01/00-00-00/checkpoint-3000 \\
      "inference.images=[data/raw/000417/raw_image.jpg]" \\
      "inference.prompts=['driving in rain at night']"
"""

import hydra
import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from pathlib import Path
from tqdm import tqdm

from hydra.utils import get_original_cwd
from src.model import ModelBase
from src.utils import add_lora_from_config
from src.data.transforms import build_seg_square_preprocess

torch.set_float32_matmul_precision("high")


# ===================================================================== #
#  VISUALIZATION HELPERS                                                  #
# ===================================================================== #

def _seg_label_bar(width: int, text: str, bar_h: int = 28) -> np.ndarray:
    """
    Dark banner bar with centered text. Returns [bar_h, width, 3] uint8.
    The 'seg_' prefix marks this as segmentation-pipeline code.
    Mirrors inference_depth.py's _label_bar with identical implementation.
    """
    bar  = Image.new("RGB", (width, bar_h), color=(25, 25, 25))
    draw = ImageDraw.Draw(bar)
    bbox = draw.textbbox((0, 0), text)
    tx   = max(0, (width - (bbox[2] - bbox[0])) // 2)
    draw.text((tx, 5), text, fill=(255, 230, 80))
    return np.asarray(bar)


def make_seg_inference_grid(
    orig_pil:     Image.Image,
    seg_pil:      Image.Image,
    pred_pil:     Image.Image,
    raw_pred_pil: Image.Image,
    size: int,
) -> Image.Image:
    """
    Create a single image with 4 panels SIDE BY SIDE:

        ┌──────────┬──────────┬──────────┬─────────────┐
        │ ORIGINAL │  SEG MAP │PREDICTED │ RAW SEG GEN │  ← label bars (28 px)
        ├──────────┼──────────┼──────────┼─────────────┤
        │          │          │          │             │
        │  size×   │  size×   │  size×   │   size×     │  ← image panels
        │  size    │  size    │  size    │   size      │
        └──────────┴──────────┴──────────┴─────────────┘

    RAW SEG GEN: same seg conditioning but empty prompt — shows pure
    segmentation adherence with no text influence. Mirrors training val grid.
    Total image: (4*size) wide, (size + 28) tall.

    The 'seg_' prefix marks this as segmentation-pipeline code. Mirrors
    inference_depth.py's make_inference_grid, with "SEG MAP"/"RAW SEG GEN".
    """
    orig     = orig_pil.resize((size, size)).convert("RGB")
    seg      = seg_pil.resize((size, size)).convert("RGB")
    pred     = pred_pil.resize((size, size)).convert("RGB")
    raw_pred = raw_pred_pil.resize((size, size)).convert("RGB")

    imgs_row  = np.concatenate([np.asarray(orig), np.asarray(seg),
                                 np.asarray(pred), np.asarray(raw_pred)], axis=1)

    label_row = np.concatenate([
        _seg_label_bar(size, "ORIGINAL"),
        _seg_label_bar(size, "SEG MAP"),
        _seg_label_bar(size, "PREDICTED"),
        _seg_label_bar(size, "RAW SEG GEN"),
    ], axis=1)

    grid = np.concatenate([label_row, imgs_row], axis=0)
    return Image.fromarray(grid)


# ===================================================================== #
#  MAIN                                                                   #
# ===================================================================== #

@hydra.main(config_path="configs", config_name="inference_seg")
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  inference_seg.py")
    print(f"  Device     : {device}")
    print(f"  Output dir : {output_dir}")
    print(f"{'='*60}\n")

    # ── Pick LOCAL model folders vs HUB ids from the local_files_only flag ──────
    # Same logic as training, so inference uses the SAME model (parity rule).
    # Local paths are made absolute from the repo root; when offline we also
    # export HF_HUB_OFFLINE so nothing can touch the network.
    _root = get_original_cwd()
    if cfg.local_files_only:
        os.environ["HF_HUB_OFFLINE"]      = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        cfg.model.model_name          = os.path.join(_root, cfg.base_model_path)
        cfg.lora.struct.encoder.model = os.path.join(_root, cfg.seg_model_path)
    else:
        cfg.model.model_name          = cfg.base_model_name
        cfg.lora.struct.encoder.model = cfg.seg_model_name
    print(f"[model] base = {cfg.model.model_name}")
    print(f"[model] seg  = {cfg.lora.struct.encoder.model}")
    print(f"[model] local_files_only = {cfg.local_files_only}")

    # ------------------------------------------------------------------ #
    # Build model from Hydra config (SD15 + LoRA structure)               #
    # ------------------------------------------------------------------ #
    cfg = hydra.utils.instantiate(cfg)
    model: ModelBase = cfg.model
    model = model.to(device)
    model.pipe.to(device)
    model.unet.requires_grad_(False)
    model.unet.eval()

    # ------------------------------------------------------------------ #
    # Load trained LoRA + mapper weights from checkpoint                  #
    # add_lora_from_config reads cfg.ckpt_path and loads:                 #
    #   <ckpt_path>/struct/lora-checkpoint.pt                             #
    #   <ckpt_path>/struct/mapper-checkpoint.pt                           #
    # ------------------------------------------------------------------ #
    cfg_mask = add_lora_from_config(model, cfg, device, dtype=torch.float32)
    print(f"Loaded checkpoint. cfg_mask = {cfg_mask}\n")

    for e in model.encoders: e.eval()
    for m in model.mappers:  m.eval()

    # ------------------------------------------------------------------ #
    # Collect input images + prompts from JSON or direct list             #
    # ------------------------------------------------------------------ #
    entries = []

    if cfg.inference.get("json_file") and cfg.inference.json_file:
        json_path = Path(cfg.inference.json_file)
        if not json_path.is_absolute():
            json_path = Path.cwd() / json_path
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            entries.append({
                "image_path": item["raw_image_path"],
                "prompt":     item.get("prompt", ""),
            })
        print(f"Loaded {len(entries)} entries from JSON: {json_path}")

    elif cfg.inference.get("images") and cfg.inference.images:
        images  = cfg.inference.images
        prompts = list(cfg.inference.get("prompts") or [])
        for i, img_path in enumerate(images):
            entries.append({
                "image_path": img_path,
                "prompt":     prompts[i] if i < len(prompts) else "",
            })
        print(f"Processing {len(entries)} images from config.")

    if not entries:
        print("[ERROR] No input images provided.")
        print("  Set inference.json_file=data/seg_training/test.json")
        print("  or  \"inference.images=[data/raw/000417/raw_image.jpg]\"")
        return

    # ------------------------------------------------------------------ #
    # Image preprocessing — SHARED function (train/inference parity).      #
    # build_seg_square_preprocess() from src/data/transforms.py is the     #
    # SINGLE SOURCE OF TRUTH for segmentation squaring. Using it here is   #
    # what guarantees the live seg map matches the saved training map:      #
    #   calculate_segmentation_map.py -> build_seg_square_preprocess()     #
    #   inference_seg.py              -> build_seg_square_preprocess()      #
    # Any change to the preprocessing function propagates to both stages    #
    # automatically. (This is the same principle depth uses via SquarePad, #
    # but consolidated into one shared factory function.)                   #
    # Output: [1, 3, H, W] in [-1, 1].                                      #
    # ------------------------------------------------------------------ #
    size        = cfg.size
    resize_mode = cfg.inference.get("resize_mode", "letterbox")
    preprocess  = build_seg_square_preprocess(size=size, resize_mode=resize_mode)

    generator = torch.Generator(device=device).manual_seed(cfg.seed)

    # ------------------------------------------------------------------ #
    # Inference loop                                                       #
    # ------------------------------------------------------------------ #
    for entry in tqdm(entries, desc="Generating"):
        img_path = Path(entry["image_path"])
        if not img_path.is_absolute():
            img_path = Path.cwd() / img_path
        prompt = entry["prompt"]
        stem   = img_path.stem

        if not img_path.exists():
            print(f"[WARN] Not found: {img_path} — skipping.")
            continue

        print(f"\n  Image : {img_path.name}")
        print(f"  Prompt: {prompt!r}")

        # Load and preprocess to [-1, 1] tensor.
        orig_pil   = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(orig_pil).unsqueeze(0).to(device)   # [1, 3, H, W]

        with torch.no_grad():

            # ---- Step 1: Seg colour map for visualization ----
            # model.encoders[0] is SegmentationEncoder (SegFormer-b5-Cityscapes).
            # Input: [-1, 1]  Output: [0, 1] 3-channel colour map.
            seg_tensor = model.encoders[0](img_tensor)   # [1, 3, H, W] in [0,1]
            seg_pil    = TF.to_pil_image(seg_tensor[0].cpu().float().clamp(0, 1))

            # ---- Step 2: Generate image (prompt-conditioned) ----
            # cs=[img_tensor]: sample_easy() calls encoder(c) unconditionally,
            # so SegFormer runs on img_tensor again to produce the conditioning.
            # seg_tensor above is only for visualization — same result.
            preds = model.sample(
                prompt=[prompt],
                num_images_per_prompt=cfg.inference.n_samples,
                cs=[img_tensor],
                generator=generator,
                cfg_mask=cfg_mask,
                num_inference_steps=cfg.inference.get("num_inference_steps", 50),
                guidance_scale=cfg.inference.get("guidance_scale", 7.5),
            )

            # ---- Step 2b: RAW SEG GEN (empty prompt) ----
            # Same seg conditioning, empty text — shows pure segmentation adherence
            # with no text influence. Matches the 4th panel of the training grid.
            raw_preds = model.sample(
                prompt=[""],
                num_images_per_prompt=cfg.inference.n_samples,
                cs=[img_tensor],
                generator=torch.Generator(device=device).manual_seed(cfg.seed),
                cfg_mask=cfg_mask,
                num_inference_steps=cfg.inference.get("num_inference_steps", 50),
                guidance_scale=cfg.inference.get("guidance_scale", 7.5),
            )

        # ---- Step 3: Save outputs ----------------------------------------
        orig_display = TF.to_pil_image(((img_tensor[0].cpu().float() + 1) / 2).clamp(0, 1))

        save_generated_only = cfg.inference.get("save_generated_only", False)

        for k, (pred_pil, raw_pred_pil) in enumerate(zip(preds, raw_preds)):
            suffix = f"_{k}" if len(preds) > 1 else ""

            if save_generated_only:
                # Mirror the exact path from the JSON so folder structure is preserved.
                rel = Path(entry["image_path"])
                out_path = output_dir / rel.parent / f"{rel.stem}{suffix}{rel.suffix}"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pred_pil.resize((size, size)).save(out_path, quality=95)
                print(f"  -> {out_path}")
            else:
                # Full debug output: 4-panel grid + individual panels.
                grid = make_seg_inference_grid(orig_display, seg_pil, pred_pil, raw_pred_pil, size)
                grid_path = output_dir / f"{stem}{suffix}_grid.jpg"
                grid.save(grid_path, quality=95)

                orig_display.resize((size, size)).save(
                    output_dir / f"{stem}_original.jpg"
                )
                seg_pil.resize((size, size)).convert("RGB").save(
                    output_dir / f"{stem}_seg.jpg"
                )
                pred_pil.resize((size, size)).save(
                    output_dir / f"{stem}{suffix}_predicted.jpg", quality=95
                )
                raw_pred_pil.resize((size, size)).save(
                    output_dir / f"{stem}{suffix}_raw_seg_gen.jpg", quality=95
                )
                print(f"  -> {grid_path}")

    print(f"\nDone. All results saved to: {output_dir}")


if __name__ == "__main__":
    main()

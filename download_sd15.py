"""
download_sd15.py
----------------
FIRST-TIME SETUP: download every model this project needs and save it to
checkpoints/local_models/ for fully offline training and inference.

Run this ONCE on a machine with internet access, then copy
checkpoints/local_models/ to your training machine and set
local_files_only: true in all configs.

Models downloaded:
  1. stable-diffusion-v1-5      -- base diffusion model (backbone, always frozen)
  2. dpt-hybrid-midas            -- depth encoder for Stage A + inference
  3. taesd                       -- Tiny AutoEncoder (fast VAE preview, optional)
  4. segformer-b5-cityscapes     -- segmentation encoder for Stage C + inference
                                    (b5 = highest accuracy on driving-scene classes)

IMPORTANT — SegformerImageProcessor:
  We save the SegformerImageProcessor alongside the model even though the repo
  does NOT use it at runtime (SegmentationEncoder does manual preprocessing).
  Reason: keeping it in the local folder ensures from_pretrained() never fails
  with a "preprocessor_config.json not found" error on any code path, including
  third-party tools that inspect the folder. It costs nothing to have it there.

  Same note applies to DPTImageProcessor for the MiDaS model — saved but not
  used at runtime (DepthEstimator also does manual preprocessing; the processor
  call is commented out in src/annotators/midas.py).

Usage:
  python download_sd15.py

  With a HF token (required for gated models, optional here since all models
  below are public):
    Set HF_TOKEN below, or export HF_TOKEN=your_token before running.
"""

import os
from diffusers import StableDiffusionPipeline, AutoencoderTiny
from transformers import (
    DPTForDepthEstimation,
    DPTImageProcessor,
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

# --- Hugging Face authentication -------------------------------------------- #
# Paste your HF READ token here, or leave empty for public models.
#
# SECURITY: never commit this file to a public repo with a token in it.
# Safer: export HF_TOKEN=hf_xxx in your shell, then use token=os.environ.get("HF_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

LOCAL_MODEL_DIR = "checkpoints/local_models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# Helper: print a section banner
def banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── 1. Stable Diffusion 1.5 (base model, always frozen) ──────────────────── #
banner("1/4  Stable Diffusion 1.5")
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    token=HF_TOKEN or None,
)
sd_path = os.path.join(LOCAL_MODEL_DIR, "stable-diffusion-v1-5")
sd_pipe.save_pretrained(sd_path)
print(f"  Saved -> {sd_path}")


# ── 2. MiDaS / DPT-Hybrid (depth encoder for Stage A + depth_inference.py) ─ #
#
# NOTE on DPTImageProcessor:
#   DepthEstimator (src/annotators/midas.py) does NOT use DPTImageProcessor at
#   runtime — the processor call is commented out and replaced with manual
#   preprocessing ((x+1)/2 -> better_resize -> direct model call).
#   We still save the processor here so the checkpoint folder is complete and
#   no tool ever complains about a missing preprocessor_config.json.
banner("2/4  MiDaS DPT-Hybrid (depth encoder)")
midas_model = DPTForDepthEstimation.from_pretrained(
    "Intel/dpt-hybrid-midas",
    token=HF_TOKEN or None,
)
midas_processor = DPTImageProcessor.from_pretrained(
    "Intel/dpt-hybrid-midas",
    token=HF_TOKEN or None,
)
midas_path = os.path.join(LOCAL_MODEL_DIR, "dpt-hybrid-midas")
midas_model.save_pretrained(midas_path)
midas_processor.save_pretrained(midas_path)
print(f"  Saved -> {midas_path}")
print(f"  (DPTImageProcessor saved for completeness — not used at runtime)")


# ── 3. Tiny VAE / TAESD (fast VAE preview — optional) ────────────────────── #
banner("3/4  Tiny VAE (TAESD)")
tiny_vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesd",
    token=HF_TOKEN or None,
)
vae_path = os.path.join(LOCAL_MODEL_DIR, "taesd")
tiny_vae.save_pretrained(vae_path)
print(f"  Saved -> {vae_path}")


# ── 4. SegFormer-b5-Cityscapes (seg encoder for Stage C + seg_inference.py) ─ #
#
# LOCKED MODEL: nvidia/segformer-b5-finetuned-cityscapes-1024-1024
#   b5 = MiT-B5 backbone (82M params), highest boundary accuracy on Cityscapes.
#   b0 (3.7M) is on disk but NOT used — it misclassifies thin structures
#   (pedestrians, poles, traffic lights) that matter for structural conditioning.
#   References.md §9 documents this choice.
#
# SAVED TO: checkpoints/local_models/segformer-b5-cityscapes/
#   *** CRITICAL: must be "segformer-b5-cityscapes", NOT "segformer-b0-cityscapes" ***
#   The config at configs/experiment/train_seg.yaml and
#   configs/lora/encoder/segformer.yaml both point to this exact folder name.
#
# NOTE on SegformerImageProcessor:
#   SegmentationEncoder (src/encoders/seg_encoder.py) does NOT use
#   SegformerImageProcessor at runtime. It manually applies:
#     (x+1)/2  ->  F.interpolate(512x512)  ->  ImageNet normalize
#   This is numerically equivalent to the processor to 2.4e-7 but runs at our
#   fixed 512x512 size instead of the processor's 1024x1024.
#   We still save the processor here so the folder is complete.
banner("4/4  SegFormer-b5-Cityscapes (seg encoder)")
segformer_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    token=HF_TOKEN or None,
)
segformer_processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    token=HF_TOKEN or None,
)
# *** CORRECT FOLDER NAME = segformer-b5-cityscapes (not b0) ***
segformer_path = os.path.join(LOCAL_MODEL_DIR, "segformer-b5-cityscapes")
segformer_model.save_pretrained(segformer_path)
segformer_processor.save_pretrained(segformer_path)
print(f"  Saved -> {segformer_path}")
print(f"  (SegformerImageProcessor saved for completeness — not used at runtime)")


# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  All models downloaded.")
print(f"  Location: {os.path.abspath(LOCAL_MODEL_DIR)}/")
print()
print("  Folder structure:")
for name in [
    "stable-diffusion-v1-5",
    "dpt-hybrid-midas",
    "taesd",
    "segformer-b5-cityscapes",
]:
    path = os.path.join(LOCAL_MODEL_DIR, name)
    status = "OK" if os.path.isdir(path) else "MISSING"
    print(f"    [{status}]  {name}")
print()
print("  Next step: set local_files_only: true in all configs.")
print(f"{'='*60}\n")

# models_pre_downloading.py

import os
from diffusers import StableDiffusionPipeline, AutoencoderTiny
from transformers import DPTForDepthEstimation, DPTImageProcessor, SegformerImageProcessor, SegformerForSemanticSegmentation

# --- Hugging Face authentication ---------------------------------------- #
# Paste your Hugging Face READ token below. A read token is sufficient for
# downloading public/gated-read models.
#
# SECURITY NOTE: hardcoding a token directly in a file is risky if this file
# is ever committed to git or shared/pushed to a public/shared repo — anyone
# with the file gets your token. If you commit this repo (even to a private
# remote), add this file to .gitignore, or at minimum confirm the remote
# stays private, before committing this version of the file.
HF_TOKEN = ""

# Define an absolute path inside your home directory workspace to bypass root restrictions
LOCAL_MODEL_DIR = "checkpoints/local_models"

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

print("Downloading Stable Diffusion 1.5 weights...")
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    token=HF_TOKEN,
)
sd_path = os.path.join(LOCAL_MODEL_DIR, "stable-diffusion-v1-5")
sd_pipe.save_pretrained(sd_path)
print(f"Saved SD1.5 to: {sd_path}")

print("\nDownloading MiDaS Structural Encoder weights...")
midas_model = DPTForDepthEstimation.from_pretrained(
    "Intel/dpt-hybrid-midas",
    token=HF_TOKEN,
)
midas_processor = DPTImageProcessor.from_pretrained(
    "Intel/dpt-hybrid-midas",
    token=HF_TOKEN,
)
midas_path = os.path.join(LOCAL_MODEL_DIR, "dpt-hybrid-midas")
midas_model.save_pretrained(midas_path)
midas_processor.save_pretrained(midas_path)
print(f"Saved MiDaS to: {midas_path}")

print("\nDownloading Tiny VAE (TAESD) weights...")
tiny_vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesd",
    token=HF_TOKEN,
)
vae_path = os.path.join(LOCAL_MODEL_DIR, "taesd")
tiny_vae.save_pretrained(vae_path)
print(f"Saved Tiny VAE to: {vae_path}")

# --- Segmentation encoder: SegFormer fine-tuned on Cityscapes ----------- #
# NOTE: this is intentionally Cityscapes, not the ADE20K b5 model shown
# commented-out in the original screenshot. references.md locks the
# Cityscapes-finetuned model for this project (driver's-perspective /
# autonomous-driving images), since ADE20K is a general scene-segmentation
# dataset, not a driving-scene one. Confirm against references.md section 7
# before changing this if you need a different SegFormer variant.
print("\nDownloading SegFormer (Cityscapes-finetuned) weights...")
segformer_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    token=HF_TOKEN,
)
segformer_processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    token=HF_TOKEN,
)
segformer_path = os.path.join(LOCAL_MODEL_DIR, "segformer-b0-cityscapes")
segformer_model.save_pretrained(segformer_path)
segformer_processor.save_pretrained(segformer_path)
print(f"Saved SegFormer (Cityscapes) to: {segformer_path}")
# Segmentation Pipeline — Zero to Hero Guide

**Pipeline**: CTRLorALTer segmentation-conditioning arm (ECCV 2024, arXiv:2405.07913)  
**What it does**: Fine-tunes LoRA blocks inside a Stable Diffusion 1.5 UNet so the model generates images that respect a user-supplied semantic segmentation map. The seg signal (19-class Cityscapes colour palette) is injected via the same mapper-network architecture used by the depth pipeline, keeping the base model frozen.

---

## 1 · What & Why

The segmentation pipeline is the structural twin of the depth pipeline. Where depth injects monocular depth maps, seg injects 19-class Cityscapes colour maps. Both use identical LoRA architecture (`NewStructLoRAConv`, rank=128, `only_res_conv`), the same mapper network, and the same training loop — only the encoder and offline preprocessing differ.

Why segmentation as a conditioning signal?
- Depth is scale-ambiguous (near and far objects look similar if they happen to have similar relative depths). Segmentation identifies semantic categories (road, sky, pedestrian, vehicle), giving the model sharper spatial structure for complex outdoor scenes.
- The two pipelines can be evaluated under identical conditions for a fair comparison (same dataset, same architecture, same hyperparameters).

---

## 2 · Files in Order

| Stage | File | When |
|-------|------|------|
| C — offline preprocessing | `seg_map_calculations.py` | **Once**, before any training |
| D — training | `seg_training.py` | Iterative; resumes from checkpoint |
| D — inference | `seg_inference.py` | After training; per-image generation |

Supporting files:

| File | Role |
|------|------|
| `src/encoders/seg_encoder.py` | `SegmentationEncoder`, `SEG_CITYSCAPES_PALETTE` (SSOT), `seg_colorize_ids()` |
| `src/data/local_seg.py` | `SegJsonDataset`, `SegJsonDataModule` |
| `src/data/transforms.py` | `build_seg_square_preprocess()` — single shared factory for seg preprocessing |
| `configs/train_seg.yaml` | Base training config |
| `configs/experiment/train_seg.yaml` | Unified experiment overrides (use this one) |
| `configs/inference_seg.yaml` | Inference config |
| `configs/data/local_seg.yaml` | Seg data module config |
| `configs/lora/encoder/segformer.yaml` | SegFormer-b5 encoder config |

---

## 3 · Data Flow Diagram

```
RAW IMAGE (arbitrary aspect ratio)
        |
        v
  [Stage C -- seg_map_calculations.py]  (run ONCE offline)
        |
        +-- build_seg_square_preprocess(size=512, resize_mode="letterbox")
        |     (same factory used by seg_inference.py -- parity guaranteed)
        +-- SegmentationEncoder.label_ids(tensor)
        |     (SegFormer-b5 prediction --> raw class IDs [0..18])
        +-- Save as 8-bit grayscale PNG (mode "L", values 0..18)
        |       --> data/raw_seg/.../img.png
        +-- Write data/seg_training/{train,val,test}.json
               (keys: raw_image_path, seg_path, prompt)

        |
        v
  [Stage D -- seg_training.py]  (per training step)
        |
        +-- SegJsonDataset.__getitem__:
        |     read seg PNG (L mode) --> NEAREST resize --> seg_colorize_ids() with palette
        |     --> 3-channel colour map Tensor [0,1]
        |     (NEAREST interpolation is REQUIRED -- bilinear would corrupt class IDs)
        |
        +-- batch["seg"]  --->  StructMapper  --->  LoRA delta-weights
        |   (skip_encode=True: SegFormer NOT called during training)
        |
        +-- model.forward_easy(imgs, prompts, [seg_maps], skip_encode=True)
        |
        +-- val/loss logged every val_steps; checkpoint+grid every ckpt_steps

        |
        v
  [Stage D -- seg_inference.py]
        |
        +-- build_seg_square_preprocess(size=512, resize_mode="letterbox")
        |   (SAME factory function as Stage C -- parity by construction)
        +-- LIVE encoder call: SegmentationEncoder.forward(image) --> colour map
        |   (skip_encode=False: SegFormer runs in model.sample() path)
        |
        +-- Generates 4-panel grid: ORIGINAL | SEG MAP | PREDICTED | RAW SEG GEN
```

---

## 4 · SegFormer-b5 Encoder

**Model**: `nvidia/segformer-b5-finetuned-cityscapes-1024-1024`  
**Class**: `src.encoders.seg_encoder.SegmentationEncoder`

Key behaviours:

- **Locked model (b5 only)**: SegFormer-b5 was chosen over b0–b4 for best boundary precision on driving-scene classes (pedestrians, vehicles, traffic lights). The model is frozen (`requires_grad=False`). b0 gives worse boundary accuracy and must not be used — see the bug note in §7.
- **Manual ImageNet normalisation**: The encoder applies ImageNet mean/std normalisation manually in `_predict_ids()` rather than using `SegformerImageProcessor`. This is intentional: the Hugging Face image processor resizes internally to its own resolution, breaking compatibility with our fixed preprocessing chain.
- **19 Cityscapes classes**: IDs 0–18. Any pixel with an unknown class gets ID 0 (road) — no out-of-range values.
- **Output of `label_ids()`**: raw class IDs `[B, H, W]` int64 — used ONLY by the offline calc script.
- **Output of `forward()`**: colour map `[B, 3, H, W]` float `[0, 1]` — used at live inference.
- **Parity guarantee**: both `label_ids()` and `forward()` share the same `_predict_ids()` internals. Only the final step differs (IDs vs colour lookup). Running `label_ids()` then `seg_colorize_ids()` on the saved PNG produces bit-identical output to calling `forward()` live.

### 4.1 Q&A — Can MiDaS be reused for segmentation instead of SegFormer?

**Q: We already have a MiDaS encoder for depth. Can we swap it into the segmentation pipeline instead of building/using SegFormer? Would it work?**

**A: No — not "works worse," it cannot work at all. This is an architecture mismatch, not a quality tradeoff.**

MiDaS is a **regression** model: its DPT decoder predicts one continuous depth value per pixel ("how far away is this point"), which `src/annotators/midas.py` min-max normalises to `[0,1]` and replicates across 3 channels. There is no class information anywhere in its weights or output — it was never trained to distinguish "this pixel is a pedestrian" from "this pixel is a building," only "near" from "far."

Segmentation needs a **classifier**: a per-pixel probability distribution over the 19 Cityscapes classes, with the highest-probability class picked as the label. SegFormer's decode head does exactly this; MiDaS's does not and cannot, regardless of which checkpoint is loaded.

Concretely, if MiDaS were forced into the seg encoder slot:
- Output would still be `[B, 1, H, W]` continuous depth in `[0,1]`, replicated to 3 channels — not class IDs.
- Colourising that with `SEG_CITYSCAPES_PALETTE` would produce "near = shade A, far = shade B" gradients, not road/car/person/sky regions.
- The LoRA mapper would learn a depth-shaped conditioning signal mislabelled as segmentation — zero semantic content, not noisy semantic content.

This is why `SegmentationEncoder` (`src/encoders/seg_encoder.py`) had to be built as a brand-new class rather than just pointing the existing `midas` encoder slot at a different model file — see §4 above and the encoder-slot-contract comment at the top of that file for how it satisfies the same input/output shape contract while doing fundamentally different (classification, not regression) work internally.

---

### 4.2 Q&A — Do we use `SegformerFeatureExtractor`? Did we write our own?

**Q: The HuggingFace example uses two objects — `SegformerFeatureExtractor` and `SegformerForSemanticSegmentation`. Do we use both?**

**A: We use `SegformerForSemanticSegmentation` (the model). We do NOT use `SegformerFeatureExtractor`. We wrote our own preprocessing that replaces it — three lines instead of one function call, with two deliberate differences.**

#### What the HuggingFace example does

```python
# Step 1 — feature extractor = preprocessing wrapper
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-...")
# Internally: resize to 1024x1024, convert [0,255] -> [0,1], apply ImageNet mean/std

# Step 2 — the actual model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-...")
# Takes pixel_values -> logits [B, 19, H/4, W/4]
```

#### What our encoder does (from `src/encoders/seg_encoder.py:213-298`)

```python
# __init__: we load the MODEL only — no SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation   # imported
# SegformerFeatureExtractor / SegformerImageProcessor       # NOT imported

self.seg_model = SegformerForSemanticSegmentation.from_pretrained(
    model, local_files_only=local_files_only
)

# _predict_ids(): we do the feature extractor's job manually
x = (imgs + 1.0) / 2.0                               # our [-1,1] -> [0,1]
x = F.interpolate(x, size=(512, 512), mode="bilinear")  # resize to OUR size, not 1024
x = (x - self._seg_mean) / self._seg_std             # ImageNet normalize (buffers, on GPU)
logits = self.seg_model(pixel_values=x).logits        # same as the HF example from here
logits = F.interpolate(logits, size=(512, 512))       # upsample logits
ids    = logits.argmax(dim=1)                         # class IDs [B, 512, 512]
```

So yes — `_predict_ids()` IS our own implementation of `SegformerFeatureExtractor`. It does the same three operations (range convert, resize, normalize) with two intentional differences:

| | `SegformerFeatureExtractor` | Our `_predict_ids()` |
|---|---|---|
| Input range | `[0, 255]` uint8 or `[0, 1]` float | `[-1, 1]` (our pipeline's training format) |
| Resize target | `1024 × 1024` (checkpoint's training size) | `512 × 512` (our pipeline's size) |
| Normalization | ImageNet mean/std | Same values, as GPU buffers |
| Output | `{"pixel_values": tensor}` dict | tensor directly |
| Numerical match | — | **2.4e-7** absolute diff (verified, documented in file header) |

#### Why resize to 512 instead of 1024

SegFormer's encoder uses overlapping patch embeddings with a /4 stride — it can accept any spatial size that is a multiple of 4. 512 is valid. The accuracy drop vs 1024 is small for driving scenes; the benefit is that Stage C (`seg_map_calculations.py`) and live inference use the **same code path at the same resolution**, so the training maps and inference maps are pixel-identical rather than differing by an extra resize step.

---

### 4.3 Proof — MiDaS encoder also skips its feature extractor (verified against original repo)

**Verified by fetching `https://github.com/CompVis/LoRAdapter` directly (2026-07-01).**

Three facts confirmed from the original CompVis/LoRAdapter repository and paper:

1. **`DPTImageProcessor` is commented out in the original repo's `midas.py`** — this is not something we added. The skip-processor pattern was already present in the code as shipped by the original authors.
2. **`src/encoders/seg_encoder.py` does not exist in the original repo (HTTP 404)** — segmentation conditioning is entirely our custom addition. The original repo has no SegFormer encoder, no Cityscapes pipeline, nothing seg-related.
3. **The paper (CTRLorALTer, arXiv 2405.07913) only implements depth and style conditioning** — segmentation is not mentioned in the abstract, contributions, experiments, or demos. The project page confirms only depth-based structure conditioning and style conditioning are demonstrated.

#### Code evidence from the original repo (`src/annotators/midas.py`)

```python
# Lines 6-8: DPTImageProcessor IS imported at the top...
from transformers import (
    DPTImageProcessor,         # imported
    DPTForDepthEstimation,
)

# Line 28 in __init__: processor instantiation is COMMENTED OUT
#   self.feature_extractor = DPTImageProcessor.from_pretrained(...)

# Lines 44-48 in forward: processor call is COMMENTED OUT
#   depth_dict = self.feature_extractor(imgs, do_rescale=False, return_tensors="pt")
#   for k, v in depth_dict.items():
#       if isinstance(v, torch.Tensor):
#           depth_dict[k] = v.to(device=imgs.device)

# Instead — manual preprocessing, direct model call:
imgs = (imgs + 1.0) / 2.0
imgs = better_resize(imgs, self.model_size)   # 384
depth_map = self.depth_estimator(pixel_values=imgs).predicted_depth
```

This is the **original CompVis code as published**. The authors chose to skip `DPTImageProcessor` and do manual preprocessing. Our `SegmentationEncoder` mirrors this exact decision for the same reason.

#### Side by side — both encoders skip their processor

| | `DepthEstimator` (original repo, midas.py) | `SegmentationEncoder` (our addition, seg_encoder.py) |
|---|---|---|
| Source | Original CompVis/LoRAdapter | Custom — does not exist in original repo |
| Range convert | `(imgs + 1.0) / 2.0` | `(imgs + 1.0) / 2.0` (identical) |
| Resize | `better_resize(imgs, 384)` | `F.interpolate(imgs, 512)` |
| Normalize | none (DPT handles it internally) | ImageNet mean/std (SegFormer requires it) |
| Feature extractor | `DPTImageProcessor` — imported, **commented out** | `SegformerImageProcessor` — not imported at all |
| Model call | `.predicted_depth` | `.logits` |
| Exists in paper | Yes — depth conditioning is the paper's core | No — segmentation is our custom extension |

#### Why we followed this pattern

When we built `SegmentationEncoder`, the MiDaS encoder in the repo was already doing manual preprocessing with the processor commented out. Matching that pattern ensures:
- Both encoders accept the same `[-1, 1]` input contract
- Both are audited the same way when the preprocessing changes
- No special cases: "encoder 1 uses the HF processor, encoder 2 doesn't"

---

## 5 · Key Design Concepts

### 5.1 Fixed Colour Palette — Why Not Per-Image Normalisation

**The critical difference from depth**: depth uses per-image min-max normalisation because the DPT output is unbounded and varies per scene. Segmentation must NOT do this.

Seg maps are discrete class assignments. Applying per-image normalisation would:
- Make the colour of class 0 (road) depend on which other classes appear in the image.
- The mapper network would see different colours for the same class in different images.
- Training would fail to learn a stable class-colour-to-latent mapping.

**Solution**: a fixed 19-entry colour palette (`SEG_CITYSCAPES_PALETTE` in `src/encoders/seg_encoder.py`) maps each class ID to a fixed RGB colour. This palette is the **single source of truth (SSOT)** — used identically in `SegmentationEncoder.forward()` (live inference) and `SegJsonDataset._load_seg_colormap()` (training data loading).

### 5.2 `SEG_CITYSCAPES_PALETTE` — The SSOT

Defined in `src/encoders/seg_encoder.py` as a `list[tuple]` with 19 entries. Helper functions:

```python
seg_palette_tensor(palette)     # --> [19, 3] float tensor in [0, 1]
seg_colorize_ids(ids, palette)  # [B,H,W] long --> [B,3,H,W] float [0,1]
```

Both the offline calc script (via `SegJsonDataset`) and the live encoder (`SegmentationEncoder.forward()`) call `seg_colorize_ids()` with this same palette. The palette is NOT stored in a JSON or YAML file — code is the SSOT.

### 5.3 Image Discovery and Path Control — how the pipeline finds your images

Same mechanics as DEPTH.md §5.2 — read that for the complete explanation. This section states the seg-specific values and repeats the critical `--image_root` rule.

#### The JSONL file types and their key names

```
data/train.jsonl                          ← source split
  {"target": "data/raw/000417/raw_image.jpg", "prompt": "..."}
   ▲ key name set by --image_path (default "source", commonly "target")

data/seg_training/train.jsonl             ← seg training manifest (output of Stage C)
  {"raw_image_path": "data/raw/000417/raw_image.jpg",
   "seg_path":       "data/raw_seg/000417/raw_image.png",
   "prompt": "..."}
```

The seg-map is always `.png` (lossless, preserves integer class-ID values). Other files in the same folder — including other `.jpg` and `.png` files — are completely ignored; the script processes only paths listed in the JSONL, never scans directories.

#### `--image_path` — which key holds the image path

```bash
python seg_map_calculations.py --data_dir data/ --image_path target
```

Pass whatever key your JSONL uses. Default is `"source"`.

#### `--image_root` — REQUIRED when images are outside the repo

`--image_root` serves two purposes: (1) resolving relative paths to absolute, and (2) computing the correct nested output path under `data/raw_seg/`. **Without it, all seg maps write to the same file and overwrite each other.**

```
# Without --image_root — ALL images → data/raw_seg/raw_image.png  ← COLLISION
# With --image_root /workstation:
/workstation/dataset/sceneA/lvl1/raw_image.jpg  → data/raw_seg/dataset/sceneA/lvl1/raw_image.png
/workstation/dataset/sceneB/lvl1/raw_image.jpg  → data/raw_seg/dataset/sceneB/lvl1/raw_image.png
/workstation/dataset/run_001/cam_left/raw_image.jpg → data/raw_seg/dataset/run_001/cam_left/raw_image.png
```

Set `--image_root` to the COMMON ROOT of all image paths in your JSONL. Folder depth can be arbitrary.

#### The full seg discovery → output flow (VERIFIED by execution)

```
data/train.jsonl                               ← --data_dir
  {"target": "data/raw/000417/raw_image.jpg", "prompt": "..."}
       │
       │ --image_path target
       ▼
  _get_image_path(entry, "target")
       │  → "data/raw/000417/raw_image.jpg"
       ▼
  image_root / raw_str    (repo_root / raw_str when --image_root not set)
       │  → /repo/data/raw/000417/raw_image.jpg
       │
       │ other files in same folder (other.jpg, mask.png, etc.) NOT touched
       ▼
  SegmentationEncoder.label_ids(image)   → class-ID map [H, W] integer
       │
       ▼
  seg_colorize_ids(ids, palette) → RGB colour map [3, 512, 512]   saved as PNG
       │  → data/raw_seg/000417/raw_image.png   (nested structure preserved)
       ▼
  data/seg_training/train.jsonl  (output — VERIFIED mapping)
  {
    "raw_image_path": "data/raw/000417/raw_image.jpg",   ← original path, unchanged
    "seg_path":       "data/raw_seg/000417/raw_image.png",
    "prompt":         "..."                               ← verbatim
  }
```

#### Quick reference

| Situation | Command |
|---|---|
| Images inside repo, JSONL key = `"source"` | `python seg_map_calculations.py --data_dir data/` |
| Images inside repo, JSONL key = `"target"` | `... --image_path target` |
| Images outside repo | `... --image_path target --image_root /path/to/common/root` |
| Quick smoke-test | `... --dry_run_n 5` |

**Never omit `--image_root` when images are outside the repo.**

### 5.3b Dataset-scan mode (`--dataset_dir`) — save maps in a SIBLING, mirrored tree [VERIFIED]

Identical to DEPTH.md §5.2b, with `_seg_map` instead of `_depth_map`. For a dataset outside the repo, the script **scans** for `raw_image.jpg` and saves each seg map into a **new sibling folder** next to the dataset root, mirroring its internal structure. **The source dataset folder is never written into.**

**Command:**
```bash
python seg_map_calculations.py \
  --dataset_dir /path/to/custome_dataset \
  --data_dir data/ \
  --image_path target
```

**Where the sibling folder is created:** automatically derived from `--dataset_dir`.
```
--dataset_dir  = /path/to/custome_dataset
sibling output = /path/to/custome_dataset_seg_map     (same parent, name + "_seg_map")
```

**What it produces (VERIFIED on a 914-folder dataset):**
```
/path/custome_dataset/000417/raw_image.jpg          ← input (found by scan, untouched)

/path/custome_dataset_seg_map/000417/000417_seg_map.png
                                                      ← NEW sibling tree, mirrors internal structure

data/seg_training/train.jsonl   (+ val, test)       ← rebuilt from the originals:
  {
    "raw_image_path": "/path/custome_dataset/000417/raw_image.jpg",           ← absolute, unchanged
    "seg_path":       "/path/custome_dataset_seg_map/000417/000417_seg_map.png",
    "prompt":         "..."                                                    ← from the split JSONL, verbatim
  }
```

**Arguments** are identical to depth's scan mode: `--dataset_dir` (scan root), `--data_dir` (split + prompt source, required), `--image_path` (JSONL key), `--image_name` (default `raw_image.jpg`).

**Verified guarantees (execution + two deliberate negative tests):**
- The source dataset folder is never modified — verified: `custome_dataset/000417/` contains only `raw_image.jpg` after a scan run, both depth and seg maps land in their own sibling trees.
- Running seg scan and depth scan back-to-back on the same dataset **does not interfere** — each writes to its own sibling folder (`custome_dataset_depth_map/` vs `custome_dataset_seg_map/`), and each scan still only picks up `raw_image.jpg`, ignoring the other pipeline's sibling folder entirely (verified: 6 seg maps from 6 folders that already had depth maps, not 12).
- The verifier (`_verify_scan_seg_training_jsonl`) was fed known-bad cases (map in-folder instead of sibling; wrong mirrored leaf-folder name) and correctly FAILed both while passing the valid entry.

**Dry run first:**
```bash
python seg_map_calculations.py --dataset_dir /path/custome_dataset --data_dir data/ --image_path target --dry_run_n 6
```

### 5.4 NEAREST Interpolation for Class IDs

When `SegJsonDataset` loads a saved seg PNG and resizes it, it uses `NEAREST` interpolation, not bilinear. This is critical:

- Bilinear interpolation would average adjacent class IDs (e.g., class 3 and class 7 blended → float 5.0, which rounds to class 5, a third wrong class).
- NEAREST interpolation selects the nearest pixel's class ID without mixing.

The resize happens in `_load_seg_colormap()` before `seg_colorize_ids()` is called.

### 5.5 `build_seg_square_preprocess()` — Single Source of Truth

The depth pipeline triplicates its preprocessing across three files. The seg pipeline fixes this with a single factory function:

```python
# src/data/transforms.py
def build_seg_square_preprocess(size, resize_mode="letterbox"):
    ...
```

This function is imported by both:
- `seg_map_calculations.py` (Stage C offline)
- `seg_inference.py` (Stage D live inference)

Parity is guaranteed by construction — there is only one definition.

### 5.6 skip_encode — Same Pattern as Depth

```python
model.forward_easy(..., skip_encode=True)   # training:  uses pre-saved colour PNG
model.sample(...)                            # inference: calls SegFormer live
```

During training `batch["seg"]` contains the colour map loaded from the saved PNG via `SegJsonDataset`. `skip_encode=True` bypasses the SegFormer entirely. During inference, `SegmentationEncoder.forward()` runs live inside `model.sample()`.

### 5.7 Checkpoint Grid, val_steps/ckpt_steps, test.json

Identical to the depth pipeline — see DEPTH.md §5.4–5.6. The seg trainer (`seg_training.py`) is a direct mirror of `train_depth.py` with `batch["depth"]` replaced by `batch["seg"]` and depth-specific helpers renamed to their `_seg_*` equivalents.

### 5.8 SegFormer-b5 vs b0 — Why b0 is Wrong

SegFormer-b0 is a small model designed for speed, not accuracy. On Cityscapes driving scenes:
- b0 misclassifies thin structures (pedestrian poles, traffic lights) that matter for structural conditioning.
- b5 achieves significantly higher mIoU on Cityscapes, especially on boundary-sensitive classes.

Two old experiment configs (`train_seg_12gb.yaml`, `train_seg_cluster.yaml`) incorrectly referenced b0 and had wrong JSON paths (missing `data/` prefix). **Deleted on 2026-06-30** — confirmed no other file referenced them (grepped the repo), confirmed `train_seg.yaml` fully supersedes them. Use `configs/experiment/train_seg.yaml` only.

---

## 6 · YAML Parameters Explained

### `configs/experiment/train_seg.yaml` (use this for all runs)

```yaml
size: 512
learning_rate: 2.0e-4
lr_warmup_steps: 500
lr_scheduler: cosine
epochs: 5
val_steps: 500
ckpt_steps: 1000
val_batches: 64
n_grid_images: 10
grid_include_empty_prompt: true   # seg includes empty-prompt panel (unlike depth)
bf16: true
gradient_checkpointing: true
gradient_accumulation_steps: 4
tag: seg
local_files_only: true
ignore_check: true
data:
  json_file:     data/seg_training/train.jsonl
  val_json_file: data/seg_training/val.jsonl   # NEVER test.json here
lora:
  struct:
    encoder:
      model: nvidia/segformer-b5-finetuned-cityscapes-1024-1024   # b5, not b0
```

### `configs/lora/encoder/segformer.yaml`

```yaml
_target_: src.encoders.seg_encoder.SegmentationEncoder
model: nvidia/segformer-b5-finetuned-cityscapes-1024-1024
size: ${size}
local_files_only: ${local_files_only}
```

### Dead keys (same as depth pipeline)

`use_empty_prompt_eval`, `n_samples`, `save_grid`, `log_cond` — present in base config, not read by `seg_training.py`.

### Broken configs (deleted)

`train_seg_12gb.yaml` and `train_seg_cluster.yaml` (b0 model, wrong JSON paths) were deleted on 2026-06-30 — `train_seg.yaml` is now the only seg experiment config and supersedes both use cases (12GB single-GPU and multi-GPU launch instructions are both documented inline in `train_seg.yaml`'s footer).

---

## 7 · Run Commands & Success Criteria

### Prerequisites

```
data/
  raw/             # original images
  seg_training/    # does NOT exist yet; Stage C creates it
checkpoints/local_models/
  stable-diffusion-v1-5/
  segformer-b5-cityscapes/   # required for local_files_only=true; see note below
```

**Model download note**: The b5 model is NOT bundled in the repo. On first run with `local_files_only: false` it auto-downloads from Hugging Face to the HF cache (`~/.cache/huggingface/`). To make it available offline, copy the HF cache to `checkpoints/local_models/segformer-b5-cityscapes/` and set `local_files_only: true`.

Activate the conda environment before any Python command:
```powershell
. "D:\MyWorkplace\installedSW\miniforge3\shell\condabin\conda-hook.ps1"
conda activate loradapter
```

### Stage C — Precompute Segmentation Maps (run once)

Dry run on 1 image first:
```powershell
python seg_map_calculations.py --data_dir data/ --dry_run_n 1 --local_files_only False
```
Success: no errors; seg PNG written to `data/raw_seg/`; class IDs in `[0, 18]`.

Full run (after b5 model is cached):
```powershell
python seg_map_calculations.py --data_dir data/
```
Success:
- `data/raw_seg/` populated (one 8-bit PNG per image)
- `data/seg_training/train.jsonl`, `val.json`, `test.json` written
- Verification output shows `0 failures`
- Dataset sizes: 639 train / 137 val / 137 test

### Stage D — Training

```powershell
python seg_training.py experiment=train_seg
```

Expected startup log:
```
[model] base = .../stable-diffusion-v1-5
Number params Mapper Network(s) 1,245,072
Number params all LoRAs(s) 29,999,104
Grid: 10 val scenes = 5 fixed (random per run) [...] + 5 re-randomized
start training
```

Per-checkpoint log pattern (every 1000 steps):
```
[val] step1000: val/loss = 0.XXXXXX
[grid] checkpoint-epoch1/step1000: 10 scene images -> .../checkpoint-epoch1/step1000
```

TensorBoard: `tensorboard --logdir outputs/train/seg/runs/`

Expected tags (same structure as depth, by design):
- Scalars: `train/loss`, `train/lr`, `val/loss`
- Images: `val/sample_00` … `val/sample_09`
- Tensors: `val/prompts/text_summary`

The tfevents hostname field will be `seg` (from `cfg.tag = "seg"`), not the machine hostname.

### Stage D — Inference

```powershell
python seg_inference.py \
  ckpt_path=outputs/train/seg/runs/YYYY-MM-DD/HH-MM-SS/best_model \
  inference.input_dir=data/raw
```

Success: 4-panel JPG grids written to `outputs/inference/seg/`.

---

## 8 · Known Limitations

1. **b5 model now in `checkpoints/local_models/segformer-b5-cityscapes/`**: Copied from HF cache on 2026-06-30. `local_files_only: true` is safe. If re-cloning to a new machine, copy the three files (`config.json`, `preprocessor_config.json`, `pytorch_model.bin`) from the HF cache (`~/.cache/huggingface/hub/models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/<latest>/`) to `checkpoints/local_models/segformer-b5-cityscapes/`.

2. **`train_seg_12gb.yaml` and `train_seg_cluster.yaml` are stale and buggy**: Both reference b0 and have wrong JSON paths. Unlike the equivalent depth configs, they have not been deleted. Do not use them. The correct config is `configs/experiment/train_seg.yaml`.

3. **`max_train_steps` does not stop training**: Same limitation as the depth pipeline — see DEPTH.md §8, item 1.

4. **19-class Cityscapes only**: The palette and class count are hardcoded to Cityscapes. Adapting to a different segmentation taxonomy requires changing `SEG_CITYSCAPES_PALETTE` in `seg_encoder.py` (the SSOT) and rerunning Stage C.

5. **Seg training never verified by execution**: As of this guide's writing, a full seg training run has not been completed. Stage C (offline seg map generation) and Stage D (parity check) were both verified by execution. Stage D training itself — step-level logs, actual val/loss convergence curve, TensorBoard tags — remains UNVERIFIED. The code mirrors `train_depth.py` exactly (which was verified), so structural correctness is expected, but a real run is needed to confirm.

6. **NEAREST-resize is slow for large batches**: `SegJsonDataset` applies NEAREST resize per sample in the dataloader. For large datasets this can be a bottleneck. Pre-resizing the seg PNGs to 512x512 during Stage C (currently not done) would eliminate this.

7. **No test-time evaluation script**: Same limitation as the depth pipeline — see DEPTH.md §8, item 7.

---

## 9 · Audit Report — Segmentation Pipeline

**EXECUTION-OVER-ASSERTION**: items below are PASS only if the step was run with real data and output inspected. "Code reads correctly" is a separate claim.

**Audit date**: 2026-06-30  
**Machine**: Windows 11 (dev); final training will run on Ubuntu.

| Item | Description | Status | Evidence |
|------|-------------|--------|----------|
| S1 | Config files — correct experiment YAML | **FIXED** | `configs/experiment/train_seg.yaml` confirmed correct (b5 model, correct `data/seg_training/*.json` paths). The two stale b0 configs (`train_seg_12gb.yaml`, `train_seg_cluster.yaml`) were deleted on 2026-06-30 after grepping the repo to confirm nothing else referenced them. |
| S2 | b5 model availability | FIXED | Model was in HF cache (`~/.cache/huggingface/hub/models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/`). Copied to `checkpoints/local_models/segformer-b5-cityscapes/` on 2026-06-30. Dry run with `local_files_only=True` confirmed load succeeded (19 classes, 1172 weights loaded). |
| S3 | Stage C — offline seg map generation | **PASS** | `python seg_map_calculations.py --data_dir data/` completed with 0 errors. 913 PNG files written to `data/raw_seg/`. All 3 JSON splits verified by built-in checker: `train.json 639/639`, `val.json 137/137`, `test.json 137/137`. PNG inspection: shape `(512, 512)`, dtype `uint8`, values in `[0, 18]` — correct. |
| S4 | Stage D — training startup | PENDING FIRST RUN | `seg_training.py` mirrors `train_depth.py` exactly; training not yet executed. Next step: `python seg_training.py experiment=train_seg`. Success criterion: startup log shows `19 classes`, loss begins decreasing within first 100 steps. |
| S5 | Train/inference preprocessing parity | CODE READS CORRECT — NOT YET VERIFIED BY EXECUTION | `build_seg_square_preprocess()` SSOT confirmed imported by both `seg_map_calculations.py` and `seg_inference.py`. Parity is guaranteed by construction. Cannot be verified by execution until inference is run with a trained checkpoint. |
| S6 | Val loss + checkpoint grid | PENDING FIRST RUN | Blocked on S4 (training). Code mirrors train_depth.py's verified grid logic. |
| S7 | TensorBoard tags | PENDING FIRST RUN | Blocked on S4. Expected tags: `train/loss`, `train/lr`, `val/loss`, `val/sample_00`…`val/sample_09`. |
| S8 | Inference | PENDING FIRST RUN | `seg_inference.py` untested — no checkpoint available yet. Blocked on S4. |
| S9 | Known issues documented | DOCUMENTED | See §8 above. NEAREST-resize slow on large batches, no test-eval script. Stale configs (formerly S1) are now deleted, not just documented. |

### What is now unblocked

Stage C is verified. The single remaining blocker before training can start is **running `seg_training.py`** — there are no more missing models, no empty data directories, no broken JSON paths. All prerequisites are met.

```powershell
# Run on Ubuntu (final training machine):
python seg_training.py experiment=train_seg
```

Watch for these in the first 50 steps to confirm training is working:
- `[model] base = .../stable-diffusion-v1-5` in startup log
- `Number params Mapper Network(s) 1,245,072` (same as depth — encoder frozen)
- `val/loss` decreasing (not stuck at a constant)
- Checkpoint grid at step 1000 shows recognisable colour blobs per Cityscapes class

---

## 10 · Full Parameter Control

Everything you can tune, where it lives, and what changing it does. Mirrors DEPTH.md §9 with seg-specific values. Read §9 first for concepts, then use this section for seg-specific differences.

---

### 10.1 How Hydra overrides work

See DEPTH.md §9.1 — identical for seg. The key difference: the experiment config file is `configs/experiment/train_seg.yaml`. Resolved config is written to `outputs/train/seg/runs/YYYY-MM-DD/HH-MM-SS/.hydra/config.yaml` after each run.

```powershell
# Override any key without editing a file:
python seg_training.py experiment=train_seg epochs=3 val_steps=100 data.batch_size=2
```

---

### 10.2 Stage C — `seg_map_calculations.py` CLI flags

Run once before training to precompute segmentation-ID PNGs from raw images.

```powershell
python seg_map_calculations.py --data_dir data/ [flags]
```

| Flag | Default | What it does | When to change |
|------|---------|--------------|----------------|
| `--data_dir` | *(required)* | Folder containing `train.jsonl`, `val.jsonl`, `test.jsonl`. Non-default names like `my_train.jsonl` are found automatically if the stem contains "train"/"val"/"test". | Always set. |
| `--dry_run_n N` | off | Process only the first N images per split. Always run `--dry_run_n 2` first to check model loading and output format. | Use before every full run on a new machine. |
| `--size` | `512` | Square side for saved seg-ID PNGs. Must match `size` in training config. Changing this requires rerunning Stage C. | Keep 512 unless you change training resolution. |
| `--batch_size` | `4` | Images fed to SegFormer at once. | Lower if you get OOM. SegFormer-b5 is heavier than DPT so you may need `--batch_size 2`. |
| `--model` | `checkpoints/local_models/segformer-b5-cityscapes` | Local path to SegFormer-b5. **Locked — do not change to b0 or any other variant.** See §4.1 for why b5 is non-negotiable. | Only if you moved the model files. |
| `--local_files_only` | `True` | Offline-only loading from the local model path. | Keep `True` now that b5 is in `checkpoints/local_models/`. |
| `--device` | `cuda` if available | `cuda` or `cpu`. | `cpu` is very slow for SegFormer-b5 (~10x slower). |
| `--resize_mode` | `letterbox` | How to make the image square before SegFormer. `letterbox` = edge-replication padding (locked default). `stretch` = squash to square (silently distorts geometry). **Never change this.** See DEPTH.md §5.1 for the full argument. | Do not change. |
| `--no_skip` | off | Recompute seg PNGs even if they already exist. | Add if you changed `--model` or `--size` and need to regenerate. |
| `--raw_dir` | `<data_dir>/raw` | Root of raw image tree. | Only if images are not under `<data_dir>/raw/`. |
| `--seg_dir` | `<data_dir>/raw_seg` | Where seg-ID PNGs are saved. | Only to redirect output. |
| `--output_dir` | `<data_dir>/seg_training` | Where the output `train.jsonl`, `val.jsonl`, `test.jsonl` manifests are written. | Only to redirect manifest location. |

---

### 10.3 Stage D Training — `configs/experiment/train_seg.yaml`

Identical structure to depth (DEPTH.md §9.3). Differences are noted below; everything else is the same.

#### Resolution and hardware — identical to depth

Same keys (`size`, `bf16`, `gradient_checkpointing`, `gradient_accumulation_steps`, `data.batch_size`, `data.workers`) with the same defaults. See DEPTH.md §9.3.

#### Learning rate and schedule — identical to depth

Same keys and defaults. The learning rate is kept identical to depth so val/loss curves are directly comparable between the two pipelines.

#### When to save / validate — identical to depth

Same keys (`val_steps=500`, `ckpt_steps=1000`, `val_batches=64`). See DEPTH.md §9.3.

#### Checkpoint monitoring grid — one difference from depth

| Key | Default (seg) | Difference from depth |
|-----|--------------|----------------------|
| `n_grid_images` | `10` | Identical: 5 fixed + 5 fresh per checkpoint. Fixed scenes chosen once at startup with OS entropy; same scenes appear in every checkpoint grid. Files: `sample_00_fixed.jpg`…`sample_04_fixed.jpg`, `sample_05_new.jpg`…`sample_09_new.jpg`. |
| `grid_include_empty_prompt` | `false` | Depth default is also `false`. For seg the empty-prompt panel is called "RAW SEG GEN" (generation with empty text, pure seg conditioning). |

**Reading the seg checkpoint grid:** Each panel row is `ORIGINAL | SEG MAP | PREDICTED`. The SEG MAP column shows the 19-class Cityscapes colour palette — you should see distinct road (purple), sky (steel blue), vegetation (green), and car (deep blue) regions. If the SEG MAP looks like a uniform colour smear, the seg encoder is producing bad predictions; check the SegFormer model files.

#### Dataset and model paths — seg-specific

| Key | Default | What it does | When to change |
|-----|---------|--------------|----------------|
| `data.json_file` | `data/seg_training/train.jsonl` | Training manifest. Each line: `{raw_image_path, seg_path, prompt}`. | Change if your dataset is elsewhere. |
| `data.val_json_file` | `data/seg_training/val.jsonl` | Val manifest. **Never `test.jsonl` here.** | Only change to use a different val set. |
| `data.image_root` | `null` (= repo root) | Prepended to relative `raw_image_path` values. Set to `/mnt/dataset` when images are on a different drive. | **Required when training on Ubuntu with images at a different path.** Same rule as depth. |
| `seg_model_path` | `checkpoints/local_models/segformer-b5-cityscapes` | Local SegFormer-b5. Used at **inference only** (skip_encode=True means it's not loaded during training). | Only if you moved the model. |
| `lora.struct.ckpt_path` | `null` | Resume from checkpoint. | Same usage as depth. |

---

### 10.4 Stage D Inference — `configs/inference_seg.yaml`

```powershell
# From a JSONL manifest:
python seg_inference.py \
  ckpt_path=outputs/train/seg/runs/YYYY-MM-DD/HH-MM-SS/best_model \
  inference.json_file=data/seg_training/test.jsonl

# Single image:
python seg_inference.py \
  ckpt_path=outputs/train/seg/runs/YYYY-MM-DD/HH-MM-SS/best_model \
  "inference.images=[data/raw/000417/raw_image.jpg]" \
  "inference.prompts=['a driving scene at night in the rain']"
```

| Key | Default | What it does | When to change |
|-----|---------|--------------|----------------|
| `ckpt_path` | *(required)* | Path to a checkpoint folder. Usually `best_model/` or a specific `checkpoint-epoch1/step1000/`. | Always set. |
| `inference.json_file` | `null` | JSONL to run inference on. | Use `test.jsonl` for final evaluation. Never `val.jsonl` or `train.jsonl`. |
| `inference.images` | `[]` | Direct list of image paths. | Quick single-image tests. |
| `inference.prompts` | `[]` | Prompts matching `inference.images`. | Required when using `inference.images`. |
| `inference.output_dir` | `outputs/inference/seg/results` | Where generated images are saved. Resolved from repo root. | Change per experiment. |
| `inference.save_generated_only` | `false` | `true` = save only the predicted image (no 4-panel grid). | Set `true` for clean batch evaluation. |
| `inference.resize_mode` | `letterbox` | Squaring method before SegFormer. **Must match `--resize_mode` used in Stage C (default: letterbox).** Changing this breaks train/inference parity. | Do not change. |
| `inference.n_samples` | `1` | Images generated per input. | `2`–`4` for diversity. |
| `inference.num_inference_steps` | `50` | Diffusion denoising steps. | `20` preview, `50` quality, `80`+ max. |
| `inference.guidance_scale` | `7.5` | CFG scale. Higher = more prompt-driven. | `3`–`5` creative, `7.5` standard, `12`+ tight. |
| `local_files_only` | `true` | Offline mode. | Keep `true`. |

---

### 10.5 Common scenarios — exact commands

#### Smoke test (verify pipeline works, ~5 minutes)
```powershell
python seg_map_calculations.py --data_dir data/ --dry_run_n 3

python seg_training.py experiment=train_seg `
  epochs=1 val_steps=10 ckpt_steps=20 val_batches=4 n_grid_images=2 `
  "data.workers=0" ignore_check=true
```

#### Full training run (Ubuntu, 5 epochs)
```bash
python seg_training.py experiment=train_seg
```

#### Resume interrupted training
```bash
python seg_training.py experiment=train_seg \
  "lora.struct.ckpt_path=outputs/train/seg/runs/2026-07-01/00-47-30/checkpoint-epoch2/step4000"
```

#### Reduce memory (OOM on a 12 GB GPU)
```powershell
python seg_training.py experiment=train_seg data.batch_size=1 gradient_accumulation_steps=4
```

#### Images on a different drive (Ubuntu training with images at /mnt/data)
```bash
python seg_training.py experiment=train_seg data.image_root=/mnt/data
```

#### Watch 20 fixed scenes per checkpoint (strong convergence signal)
```powershell
python seg_training.py experiment=train_seg n_grid_images=40
# 20 fixed + 20 fresh — each checkpoint grid shows same 20 scenes for comparison
```

#### Add empty-prompt panel (see pure seg conditioning)
```powershell
python seg_training.py experiment=train_seg grid_include_empty_prompt=true
# 4th panel: "RAW SEG GEN" — no text, only the seg map drives generation
```

#### Evaluate on test split after training
```powershell
python seg_inference.py `
  ckpt_path=outputs/train/seg/runs/YYYY-MM-DD/HH-MM-SS/best_model `
  inference.json_file=data/seg_training/test.jsonl `
  inference.save_generated_only=true `
  inference.output_dir=outputs/inference/seg/test_eval
```

#### Generate comparison report (depth vs seg)
```powershell
python training_report.py
python training_report.py --markdown   # GitHub-flavoured Markdown
```


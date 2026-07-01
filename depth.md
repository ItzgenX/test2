# Depth Pipeline — Zero to Hero Guide

**Pipeline**: CTRLorALTer depth-conditioning arm (ECCV 2024, arXiv:2405.07913)  
**What it does**: Fine-tunes LoRA blocks inside a Stable Diffusion 1.5 UNet so the model generates images that respect a user-supplied depth map. The depth signal is injected via a learned mapper network (not a ControlNet adapter), keeping the base model frozen.

---

## 1 · What & Why

Standard T2I diffusion ignores structure — the same prompt generates different spatial layouts each run. This pipeline injects a depth map as a structural conditioning signal through LoRA rank-128 adapters that sit inside the UNet's residual-conv layers. A lightweight mapper network translates the depth embedding into LoRA delta-weights; the base UNet weights are never updated.

The result: the model generates images that follow the supplied depth map's spatial structure while still following the text prompt.

---

## 2 · Files in Order

| Stage | File | When |
|-------|------|------|
| A — offline preprocessing | `depth_map_calculations.py` | **Once**, before any training |
| B — training | `depth_training.py` | Iterative; resumes from checkpoint |
| B — inference | `depth_inference.py` | After training; per-image generation |

Supporting files:

| File | Role |
|------|------|
| `src/annotators/midas.py` | `DepthEstimator` class wrapping Intel/dpt-hybrid-midas |
| `src/data/local.py` | `DepthJsonDataset`, `DepthJsonDataModule` |
| `src/data/transforms.py` | `SquarePad` (edge-replication padding) |
| `configs/train_depth.yaml` | Base training config (required keys + defaults) |
| `configs/experiment/train_depth.yaml` | Unified experiment overrides (use this one) |
| `configs/inference_depth.yaml` | Inference config |
| `configs/data/local_depth.yaml` | Data module + transform chain config |

---

## 3 · Data Flow Diagram

```
RAW IMAGE (arbitrary aspect ratio)
        |
        v
  [Stage A -- depth_map_calculations.py]  (run ONCE offline)
        |
        +-- SquarePad (edge-replication) --> square image (no distortion)
        +-- Resize to 512x512
        +-- ToTensor + Normalize [-1,1]
        +-- DepthEstimator (DPT-hybrid-midas)  -->  per-image min-max --> [0,1] depth
        +-- Scale x255, save as 8-bit grayscale PNG   --> data/raw_depth/.../img.png
        +-- Write data/depth_training/{train,val,test}.json
               (keys: raw_image_path, depth_path, prompt)

        |
        v
  [Stage B -- depth_training.py]  (per training step)
        |
        +-- DepthJsonDataset.__getitem__:
        |     read depth PNG (L mode) --> replicate to 3 channels --> Tensor [0,1]
        |     read raw image --> SquarePad-->Resize-->ToTensor-->Normalize [-1,1]
        |     (NO SquarePad on depth PNG -- it's already square from Stage A)
        |
        +-- batch["depth"]  --->  StructMapper  --->  LoRA delta-weights
        |   (skip_encode=True: DepthEstimator NOT called during training)
        |
        +-- model.forward_easy(imgs, prompts, [depth_maps], skip_encode=True)
        |
        +-- val/loss logged every val_steps; checkpoint+grid every ckpt_steps

        |
        v
  [Stage B -- depth_inference.py]
        |
        +-- Raw image --> SquarePad-->Resize-->ToTensor-->Normalize [-1,1]
        |   (same chain, applied LIVE here -- no pre-saved depth PNG needed)
        |
        +-- LIVE encoder call: DepthEstimator(image) --> depth map
        |   (skip_encode=False: encoder runs in the model.sample() path)
        |
        +-- Generates 4-panel grid: ORIGINAL | DEPTH MAP | PREDICTED | RAW DEPTH GEN
```

---

## 4 · MiDaS / DepthEstimator

**Model**: `Intel/dpt-hybrid-midas` (DPT-Hybrid architecture, `model_size=384`)  
**Class**: `src.annotators.midas.DepthEstimator`

Key behaviours:

- **Per-image min-max normalisation**: output is scaled to `[0, 1]` based on each image's own min/max depth. There is no cross-image calibration. This means depth values from different images are not on the same scale — only spatial structure is meaningful.
- **`better_resize()` internal crop**: MiDaS internally centre-crops non-square inputs before running. If you feed a 16:9 image directly you lose the left and right edges. **`SquarePad` must be applied before any encoder call to prevent this.**
- **Output shape**: `[B, 1, H, W]` float32 in `[0, 1]` (after normalisation).
- **3-channel replication**: `DepthJsonDataset` replicates the single-channel PNG to 3 channels so the mapper network sees the same tensor shape it would see from a colour-image encoder.

### 4.1 The 384 question — why `better_resize(384)` and not 512

You will see `self.model_size = 384` in `src/annotators/midas.py` and wonder: our whole pipeline uses 512×512, so why is there a 384 here?

**Answer: 384 is DPT-Hybrid-MiDaS's native training resolution. The depth model expects 384×384 input. 512 is the LoRAdapter pipeline canvas size. They serve different purposes and the two sizes correctly coexist.**

Here is the full image journey through the depth pipeline:

```
RAW IMAGE (e.g. 1280×800)
    │
    ▼ [Stage A preprocessing — depth_map_calculations.py]
    SquarePad()             → 1280×1280  (edge-replication, no distortion)
    Resize(512, 512)        → 512×512    (our pipeline canvas)
    ToTensor + Normalize    → [-1, 1]
    │
    ▼ [Inside DepthEstimator.forward() — midas.py]
    (imgs + 1.0) / 2.0      → [0, 1]    (convert from pipeline format)
    better_resize(384)      → 384×384   ← MiDaS's REQUIRED input size
    │   (center_crop to square → already square, so no-op)
    │   (avg_pool2d if needed → downscale factor = 512//384 = 1, so no-op)
    │   (bilinear interpolate to 384 → actual resize 512→384)
    depth_estimator(pixel_values)  →  depth logits 384×384
    F.interpolate(self.size=512)   →  depth map 512×512  (back to our canvas)
    min-max normalize per image    →  depth in [0, 1]
    cat([depth]*3)                 →  [B, 3, 512, 512]   (3-channel for mapper)
```

**The two 512s bookend the 384:**
- The image ENTERS DepthEstimator at 512×512 (our canvas, after SquarePad+Resize)
- `better_resize(384)` shrinks it to 384×384 for the model to run on
- `F.interpolate(512)` expands the depth map BACK to 512×512 (our canvas)

**Why 384 specifically?** DPT-Hybrid-MiDaS was fine-tuned on MiX-6 at 384×384 (that is the checkpoint's training size). Running it at exactly 384 gives the sharpest depth predictions. Running it larger or smaller changes the patch embedding stride resolution and degrades results.

**The `center_crop` in `better_resize` — why it is not dangerous here:**

```python
# src/annotators/util.py — better_resize() full implementation
def better_resize(imgs, image_size):
    H, W = imgs.shape[-2:]
    side  = min(H, W)             # e.g. min(512, 512) = 512
    imgs  = center_crop(imgs, [side, side])  # 512×512 → 512×512: NO-OP (already square)
    factor = side // image_size   # 512 // 384 = 1 → NO avg_pool
    if factor > 1:
        imgs = avg_pool2d(imgs, factor)
    imgs = interpolate(imgs, [image_size, image_size], mode="bilinear")
    return imgs
```

Because SquarePad already made the image square BEFORE it reaches `better_resize`, the `center_crop` step is a geometric no-op — cropping a 512×512 square to min(512,512)=512 changes nothing. Without SquarePad, a 1280×800 landscape image would be center-cropped to 800×800, silently discarding the left and right 240 px of content.

---

### 4.2 Q&A — Do we use `DPTImageProcessor`? Is this the original code?

**Q: The HuggingFace DPT example uses `DPTImageProcessor` + `DPTForDepthEstimation`. Do we use both?**

**A: `DPTForDepthEstimation` yes. `DPTImageProcessor` no — it is imported but commented out. This is the ORIGINAL CompVis code, not something we changed.**

**Verified by:**
- `git diff HEAD -- src/annotators/midas.py` returns empty — zero changes since the initial commit
- Fetching `https://raw.githubusercontent.com/CompVis/LoRAdapter/main/src/annotators/midas.py` directly and comparing line by line — files are byte-for-byte identical

The original authors shipped `midas.py` with the processor commented out and manual preprocessing in its place. We inherited this code unchanged.

#### What the original code does (lines 6–50 of `src/annotators/midas.py`)

```python
from transformers import (
    DPTImageProcessor,        # imported but NEVER used
    DPTForDepthEstimation,    # this IS used
)

class DepthEstimator(nn.Module):
    def __init__(self, size, model, local_files_only):
        self.depth_estimator = DPTForDepthEstimation.from_pretrained(model, ...)
        # self.feature_extractor = DPTImageProcessor.from_pretrained(...)  # COMMENTED OUT

    def forward(self, imgs):
        imgs = (imgs + 1.0) / 2.0          # [-1,1] -> [0,1]  (manual, no processor)
        imgs = better_resize(imgs, 384)     # resize to 384     (manual, no processor)
        # depth_dict = self.feature_extractor(...)              # COMMENTED OUT
        depth_map = self.depth_estimator(pixel_values=imgs).predicted_depth
        # ... min-max normalise, replicate to 3 channels
```

#### Why the original authors skipped the processor

`DPTImageProcessor` resizes the image and normalises pixel values. But `DepthEstimator` already does both steps manually in two lines before calling the model:
1. `(imgs + 1.0) / 2.0` — convert from the pipeline's `[-1, 1]` to `[0, 1]`
2. `better_resize(imgs, 384)` — resize to the model's input size

Using the processor on top of this would double-process the image and break the input contract. The manual path is also faster (no CPU round-trip, no dict wrapping) and keeps the preprocessing visible in the code rather than hidden inside a HuggingFace object.

Our `SegmentationEncoder` in `src/encoders/seg_encoder.py` was deliberately designed to mirror this exact pattern — see SEGMENTATION.md §4.2 and §4.3 for the full comparison.

---

## 5 · Key Design Concepts

### 5.1 SquarePad (edge-replication padding)

`src.data.transforms.SquarePad` pads the shorter dimension to make the image square using edge-pixel replication (not zero-padding, not centre-crop). This preserves spatial content at borders.

**Why it's required**: MiDaS's `better_resize()` internally crops to a square before running. Without `SquarePad`, a landscape image would lose its left and right portions inside the DPT encoder. The padding is applied **in all three preprocessing sites** to ensure consistency:

1. `depth_map_calculations.py` — Stage A offline preprocess
2. `depth_inference.py` — Stage B live inference preprocess
3. `configs/data/local_depth.yaml` — Stage B training transform chain (for raw images)

**Triplication risk**: These three sites must stay byte-identical. If you change the preprocessing (e.g., add a normalisation step), update all three.

### 5.2 Image Discovery and Path Control — how the pipeline finds your images

This section answers: where does the pipeline look for images, what key in the JSONL tells it where the image is, and what do you change if your images live somewhere else?

#### The three JSONL file types and their key names

```
data/train.jsonl                          ← source split (created by dataset prep)
  {"source": "data/raw/000417/raw_image.jpg", "prompt": "..."}
   ▲ default key = "source"

data/depth_training/train.jsonl           ← depth training manifest (output of Stage A)
  {"raw_image_path": "data/raw/000417/raw_image.jpg", "depth_path": "...", "prompt": "..."}
   ▲ key = "raw_image_path"
```

Image filename is always `raw_image.jpg` — the folder name (`000417/`) is the scene identifier.

#### `--image_path` — tell the script which key holds the image path

The CLI arg `--image_path` sets which key the script reads from each JSONL entry. Default is `"source"`. If your JSONL uses `"target"` (or any other name), pass it explicitly:

```bash
# Your JSONL has {"source": "...", "prompt": "..."}  → default, no flag needed
python depth_map_calculations.py --data_dir data/

# Your JSONL has {"target": "...", "prompt": "..."}
python depth_map_calculations.py --data_dir data/ --image_path target
```

The key name flows from CLI → `_get_image_path(entry, image_path)` → file open. Nothing else in the script hard-codes a key name.

#### `--image_root` — required whenever images live OUTSIDE the repo

`--image_root` does two things, both needed for external datasets:

1. **Path resolution**: prepend to relative paths in JSONL to build the absolute path to each image.
2. **Folder mirroring**: used as the base to compute where the depth PNG goes inside `data/raw_depth/`, preserving the full nested structure.

**Without `--image_root` — collision disaster (VERIFIED):**

If your images are external (e.g. `/workstation/dataset/sceneA/subdir/raw_image.jpg`) and you don't set `--image_root`, ALL depth maps land at the same path `data/raw_depth/raw_image.png` and every image overwrites the previous one. This is silent data loss.

**With `--image_root /workstation` — correct nested output (VERIFIED):**

```
/workstation/dataset/sceneA/lvl1/raw_image.jpg      → data/raw_depth/dataset/sceneA/lvl1/raw_image.png
/workstation/dataset/sceneB/lvl1/raw_image.jpg      → data/raw_depth/dataset/sceneB/lvl1/raw_image.png
/workstation/dataset/run_001/cam_left/raw_image.jpg → data/raw_depth/dataset/run_001/cam_left/raw_image.png
```

The folder depth can be arbitrary — any nesting is preserved. Folders can have any names. Other images in the same end-folder (e.g. `thumbnail.jpg`, `mask.png`) are naturally ignored because the script only processes paths listed in the JSONL — it never scans directories.

**Rule: set `--image_root` to the COMMON ROOT of all image paths in your JSONL.**

```bash
# Images at /workstation/custom_dataset/... — relative paths in JSONL
python depth_map_calculations.py --data_dir data/ --image_path target \
  --image_root /workstation

# Images at /workstation/custom_dataset/... — absolute paths in JSONL
# (image_root is NOT used for resolution here, but still needed for folder mirroring)
python depth_map_calculations.py --data_dir data/ --image_path target \
  --image_root /workstation
```

#### The full discovery → depth → output flow (VERIFIED by execution)

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
       │  → /repo/data/raw/000417/raw_image.jpg   (resolved absolute path)
       │
       │ other files in same folder (thumbnail.jpg, mask.png, etc.)
       │ are NOT touched — only JSONL-listed paths are processed
       ▼
  DepthEstimator.forward(image)   → depth tensor [1, 512, 512] in [0,1]
       │
       ▼
  _depth_out_path: mirrors nested folder structure under --image_root
       │  → data/raw_depth/000417/raw_image.png
       ▼
  data/depth_training/train.jsonl  (output — VERIFIED mapping)
  {
    "raw_image_path": "data/raw/000417/raw_image.jpg",   ← original path, unchanged
    "depth_path":     "data/raw_depth/000417/raw_image.png",
    "prompt":         "this is a close up of a person holding a map..."  ← verbatim
  }
```

#### Quick reference

| Situation | Command |
|---|---|
| Images inside repo, JSONL key = `"source"` (default) | `python depth_map_calculations.py --data_dir data/` |
| Images inside repo, JSONL key = `"target"` | `... --image_path target` |
| Images outside repo, any key | `... --image_path target --image_root /path/to/common/root` |
| Quick smoke-test | `... --dry_run_n 5` |

**Never omit `--image_root` when images are outside the repo.** Without it, all depth maps write to the same file and your data is silently corrupted.

### 5.2b Dataset-scan mode (`--dataset_dir`) — save maps in a SIBLING, mirrored tree [VERIFIED]

This is the mode for a dataset that lives OUTSIDE the repo, where each scene folder contains `raw_image.jpg` (plus possibly other files). Instead of trusting JSONL paths to *find* images, the script **scans the folder** and finds `raw_image.jpg` directly. The depth map is saved into a **new sibling folder** next to the dataset root, mirroring the dataset's internal structure — **the source dataset folder is never written into.**

**Command:**
```bash
python depth_map_calculations.py \
  --dataset_dir /path/to/custome_dataset \
  --data_dir data/ \
  --image_path target
```

**What each argument does:**
- `--dataset_dir` — the folder to scan recursively for `raw_image.jpg`. Activates scan mode.
- `--data_dir` — folder holding `train/val/test.jsonl`. Used ONLY to recover each image's **prompt** and which **split** it belongs to (matched by absolute image path). Required.
- `--image_path` — the JSONL key holding the image path (`target` here).
- `--image_name` — the exact filename to process (default `raw_image.jpg`). Every other file in the folder is ignored, so the source dataset can freely contain other images per folder.

**Where the sibling folder is created:** automatically derived from `--dataset_dir` — no separate flag needed.
```
--dataset_dir  = /path/to/custome_dataset
sibling output = /path/to/custome_dataset_depth_map     (same parent, name + "_depth_map")
```

**What it produces (VERIFIED on a 914-folder dataset):**
```
/path/custome_dataset/000417/raw_image.jpg              ← input (found by scan, untouched)

/path/custome_dataset_depth_map/000417/000417_depth_map.png
                                                          ← NEW sibling tree, mirrors internal structure,
                                                            file named after the leaf folder

data/depth_training/train.jsonl   (+ val, test)          ← rebuilt from the originals:
  {
    "raw_image_path": "/path/custome_dataset/000417/raw_image.jpg",               ← absolute, unchanged
    "depth_path":     "/path/custome_dataset_depth_map/000417/000417_depth_map.png",
    "prompt":         "..."                                                        ← from the split JSONL, verbatim
  }
```

**Key guarantees (all verified by execution, including two deliberate negative tests):**
- The source dataset folder is **never modified** — verified: after a scan run, `custome_dataset/000417/` contains only `raw_image.jpg`, nothing else.
- Only `raw_image.jpg` is processed — other files/images in the folder are ignored.
- The prompt + split for each image are preserved exactly (matched by absolute path against the original `data/*.jsonl`).
- Output JSONL paths are absolute (the data lives outside the repo).
- Re-runs skip already-computed maps unless you pass `--no_skip`.
- The verifier (`_verify_scan_training_jsonl`) was fed known-bad cases (map written in-folder instead of sibling; map under the wrong mirrored leaf-folder name) and correctly FAILed both, while passing the valid entry — so a FAIL from this checker is a real signal, not an unproven check.

**Dry run first:**
```bash
python depth_map_calculations.py --dataset_dir /path/custome_dataset --data_dir data/ --image_path target --dry_run_n 6
```
In a dry run the scan is capped to the first N images; split JSONLs contain only the entries whose images were in that capped set (the rest are reported as "not in the scanned subset — expected").

### 5.3 skip_encode — Training vs Inference Path

```python
model.forward_easy(..., skip_encode=True)   # training:   uses pre-saved depth PNG
model.sample(...)                            # inference:  calls DepthEstimator live
```

During training the depth map is loaded from `batch["depth"]` (the pre-computed PNG from Stage A). `skip_encode=True` tells `SD15.forward()` to skip the encoder entirely and feed the depth tensor directly to the mapper. This is fast and avoids running the GPU-heavy DPT model on every training step.

During inference the raw image is fed to the live `DepthEstimator` inside `model.sample()`. The preprocessing chain must be identical to Stage A to ensure the mapper sees the same distribution of depth maps.

### 5.3 Conditional LoRA (StructLoRA / NewStructLoRAConv)

LoRA adapters are inserted into the UNet's residual-conv layers only (`adaption_mode: only_res_conv`). Rank = 128. The mapper network takes a 128-dim depth embedding and outputs per-layer delta-weights. The base UNet is frozen throughout.

Parameter counts (verified by execution):
- Mapper Network: 1,245,072
- Encoder Network: 0 (DepthEstimator is frozen)
- LoRA adapters: 29,999,104

### 5.4 Checkpoint Grid (50/50 Fixed/Fresh Split)

Every `ckpt_steps` steps the trainer saves weights **and** monitoring images together in the same folder (`checkpoint-epochN/stepM/`). Images are:

- **Fixed half**: `n_grid_images // 2` val scenes chosen randomly ONCE per run (using OS entropy, not the training seed). These exact scenes recur at every checkpoint so you can watch the same scene improve over time. The chosen indices are logged at startup.
- **Fresh half**: re-drawn randomly from the remaining val indices at each checkpoint — a quick generalization peek.

Source is **validation set only** — never train or test. This is enforced by `dm.val_dataset`.

### 5.5 val_steps vs ckpt_steps (Decoupled)

```yaml
val_steps:  500    # cheap: compute val/loss, possibly update best_model
ckpt_steps: 1000   # heavy: save weights + N monitoring images
```

These are **independent**. You can validate every 50 steps and checkpoint every 500. Both can fire at the same step — the code handles this correctly by calling `do_validation` then `save_ckpt_and_grid` sequentially.

`best_model/` is written by `do_validation` (when val/loss improves), not by `save_ckpt_and_grid`. `best_model/info.txt` records the step and val/loss it came from.

### 5.6 Why test.jsonl Is Never Touched During Training

`data/depth_training/test.jsonll` is written by `depth_map_calculations.py` alongside `train.jsonl` and `val.jsonl`, but **no training or validation config ever reads it**. It exists so you can run a final held-out evaluation after training is complete, using `depth_inference.py`.

Training configs (`configs/experiment/train_depth.yaml`) reference only:
```yaml
json_file:     data/depth_training/train.jsonll
val_json_file: data/depth_training/val.jsonll
```

The test split stays clean.

### 5.7 Hostname Override in TensorBoard

TensorBoard embeds `socket.gethostname()` in the tfevents filename. Without intervention this would produce `events.out.tfevents.<ts>.aditya.<pid>.0`, leaking the machine name.

The code overrides it before calling `accelerator.init_trackers`:
```python
import socket as _socket
_socket.gethostname = lambda: str(cfg.get("tag", "loradapter"))
```

Verified result: `events.out.tfevents.1782770112.depth.13012.0` — the field is `depth` (from `cfg.tag = "depth"`).

---

## 6 · YAML Parameters Explained

### `configs/train_depth.yaml` (base — required keys and defaults)

| Key | Default | Notes |
|-----|---------|-------|
| `size` | `???` | **Required.** Image size; experiment sets 512 |
| `learning_rate` | `1e-4` | AdamW lr; experiment overrides to `2e-4` |
| `lr_scheduler` | `constant` | Experiment overrides to `cosine` |
| `lr_warmup_steps` | `0` | Experiment overrides to `500` |
| `epochs` | `10` | Experiment overrides to `5` |
| `val_steps` | `1000` | How often to run validation |
| `ckpt_steps` | `1000` | How often to save checkpoint + images |
| `val_batches` | `4` | Number of val batches per validation pass |
| `seed` | `42` | Training seed (not the fixed-image selection seed) |
| `bf16` | `false` | Experiment overrides to `true` |
| `gradient_checkpointing` | `false` | Experiment overrides to `true` |
| `gradient_accumulation_steps` | `1` | Experiment overrides to `4` |
| `tag` | `''` | Experiment sets `'depth'`; used as hostname in tfevents |
| `local_files_only` | `false` | Set `true` once models are downloaded |
| `ignore_check` | `false` | Suppress data-integrity pre-check |
| `prompt` | `null` | If set, overrides all per-sample captions |

**Dead keys** (present in config but NOT read by `depth_training.py`):

| Key | Why it exists |
|-----|---------------|
| `use_empty_prompt_eval` | Used by original `train.py`; depth trainer uses `grid_include_empty_prompt` |
| `n_samples` | Read by original `train.py`; not by depth/seg trainers |
| `save_grid` | Same — original trainer flag |
| `log_cond` | Same |

### `configs/experiment/train_depth.yaml` (use this for all runs)

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
grid_include_empty_prompt: false
bf16: true
gradient_checkpointing: true
gradient_accumulation_steps: 4
tag: depth
local_files_only: true
ignore_check: true
data:
  json_file:     data/depth_training/train.jsonll
  val_json_file: data/depth_training/val.jsonll
```

### `configs/inference_depth.yaml`

| Key | Value | Notes |
|-----|-------|-------|
| `ckpt_path` | `???` | **Required** — path to a checkpoint folder |
| `size` | `512` | Must match training size |
| `seed` | `42` | Reproducible generation |
| `local_files_only` | `true` | Offline mode |
| `inference.n_samples` | `1` | Images generated per input |
| `inference.num_inference_steps` | `50` | Denoising steps |
| `inference.guidance_scale` | `7.5` | CFG scale |

---

## 7 · Run Commands & Success Criteria

### Prerequisites

```
data/
  raw/             # original images
  depth_training/  # does NOT exist yet; Stage A creates it
checkpoints/local_models/
  stable-diffusion-v1-5/
  dpt-hybrid-midas/
```

Activate the conda environment before any Python command:
```powershell
. "D:\MyWorkplace\installedSW\miniforge3\shell\condabin\conda-hook.ps1"
conda activate loradapter
```

### Stage A — Precompute Depth Maps (run once)

Dry run on 3 images first:
```powershell
python depth_map_calculations.py --data_dir data/ --dry_run_n 3
```
Success: no errors; depth PNGs in `data/raw_depth/`.

Full run:
```powershell
python depth_map_calculations.py --data_dir data/
```
Success:
- `data/raw_depth/` populated (one PNG per image)
- `data/depth_training/train.jsonll`, `val.jsonl`, `test.jsonl` written
- Verification output shows `0 failures`
- Dataset sizes: 639 train / 137 val / 137 test

### Stage B — Training

```powershell
python depth_training.py experiment=train_depth
```

Expected startup log:
```
[model] base  = .../stable-diffusion-v1-5
[model] depth = .../dpt-hybrid-midas
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

TensorBoard: `tensorboard --logdir outputs/train/depth/runs/`

Expected tags (confirmed by execution):
- Scalars: `train/loss`, `train/lr`, `val/loss`
- Images: `val/sample_00` … `val/sample_09`
- Tensors: `val/prompts/text_summary`

### Stage B — Inference

```powershell
python depth_inference.py \
  ckpt_path=outputs/train/depth/runs/YYYY-MM-DD/HH-MM-SS/best_model \
  inference.input_dir=data/raw
```

Success: 4-panel JPG grids written to `outputs/inference/depth/`.

---

## 8 · Known Limitations

1. **`max_train_steps` does not stop training**: Setting `+max_train_steps=N` via Hydra affects the progress-bar total and LR scheduler warmup calculation only. The batch loop runs for the full number of epochs. To stop early, send SIGINT (Ctrl+C); the signal handler finishes the current step and saves a final checkpoint.

2. **Triplication risk in preprocessing**: The depth preprocessing transform chain (SquarePad → Resize → ToTensor → Normalize) is defined in three separate places (`depth_map_calculations.py`, `depth_inference.py`, `configs/data/local_depth.yaml`). Any future change must be applied to all three simultaneously or parity will break.

3. **Per-image min-max depth**: Depth values are normalised per image. You cannot meaningfully compare depth magnitudes across images. The pipeline learns a structural prior, not a metric depth prior.

4. **Float→uint8 round-trip quantisation**: Stage A saves depth as 8-bit PNG, introducing up to 1/255 ≈ 0.004 error per pixel. The observed max diff between freshly-computed depth and saved PNG was 0.00837 (~2 uint8 levels), caused by floating-point non-determinism between sessions. This is not a preprocessing bug — it is inherent to the round-trip.

5. **`references.md §6` incorrect file reference**: Lists `src/data/local_depth.py` as a separate file. This file does not exist. `DepthJsonDataset` and `DepthJsonDataModule` live in `src/data/local.py`.

6. **OOM with experiment defaults on consumer GPU**: Experiment config uses `batch_size=4`, `gradient_accumulation_steps=4`, `gradient_checkpointing=True`, `bf16=True`. On a GPU with less than 16 GB VRAM this may OOM during backward. Use `data.batch_size=1 gradient_accumulation_steps=1` for single-GPU runs.

7. **No test-time evaluation script**: `test.jsonl` exists but there is no built-in script to run batch inference on the test split and compute quantitative metrics. `depth_inference.py` with `save_generated_only=true` produces images but not scores.

---

## 9 · Full Parameter Control

Everything you can tune, where it lives, and what changing it does. One place to look up any flag.

---

### 9.1 How Hydra overrides work

`depth_training.py` and `depth_inference.py` use [Hydra](https://hydra.cc) for configuration. The base config is `configs/train_depth.yaml`; the experiment overrides live in `configs/experiment/train_depth.yaml`. You can override ANY key at the command line without editing a file:

```powershell
# Single override
python depth_training.py experiment=train_depth epochs=3

# Multiple overrides
python depth_training.py experiment=train_depth epochs=3 val_steps=100 data.batch_size=2

# Nested keys use dot notation
python depth_training.py experiment=train_depth data.batch_size=1 lora.struct.ckpt_path=outputs/train/.../step1000

# Adding a new key that is not in any config (use + prefix)
python depth_training.py experiment=train_depth +max_train_steps=500
```

Hydra writes its own log + a copy of the resolved config to:
- `outputs/train/depth/runs/YYYY-MM-DD/HH-MM-SS/.hydra/config.yaml` — what actually ran
- `outputs/train/depth/runs/YYYY-MM-DD/HH-MM-SS/.hydra/overrides.yaml` — what you passed on the CLI

If something behaves unexpectedly, read `config.yaml` — it shows every key's final resolved value.

---

### 9.2 Stage A — `depth_map_calculations.py` CLI flags

Run once before training to precompute depth PNGs from raw images.

```powershell
python depth_map_calculations.py --data_dir data/ [flags]
```

| Flag | Default | What it does | When to change |
|------|---------|--------------|----------------|
| `--data_dir` | *(required)* | Folder containing `train.jsonl`, `val.jsonl`, `test.jsonl`. The script scans for files whose names contain "train"/"val"/"test" — non-default names like `my_train.jsonl` are found automatically. | Always set this. |
| `--dry_run_n N` | off | Process only the first N images per split. Run with `--dry_run_n 3` first, check output, then run without it. | Always use before first full run. |
| `--size` | `512` | Square side for saved depth PNGs. Must match `size` in your training config. If you change this you must rerun Stage A. | Keep 512 unless you change training resolution. |
| `--batch_size` | `4` | Images fed to the depth model at once. Higher = faster but uses more GPU memory. | Lower if you get OOM during Stage A. |
| `--model` | `checkpoints/local_models/dpt-hybrid-midas` | Path to the local DPT model folder OR a HuggingFace repo ID. Default is the local offline copy. | Only change if you switch depth models (breaks parity with existing depth PNGs — delete `data/raw_depth/` and rerun). |
| `--local_files_only` | `True` | `True` = load model from local disk only (offline). `False` = allow HF download if model is not cached. | Set to `False` if you need to download the model for the first time. |
| `--device` | `cuda` if available | `cuda` or `cpu`. | Set to `cpu` if no GPU available (very slow). |
| `--no_skip` | off | Re-compute depth maps even if PNG already exists. By default existing PNGs are reused (fast). | Add this flag if you changed `--model` or `--size` and need to regenerate. |
| `--raw_dir` | `<data_dir>/raw` | Root of raw image tree. Mirror structure is preserved: `raw/000417/img.jpg` → `raw_depth/000417/img.png`. | Only if your images are not under `<data_dir>/raw/`. |
| `--depth_dir` | `<data_dir>/raw_depth` | Where depth PNGs are saved. | Only to redirect output location. |
| `--output_dir` | `<data_dir>/depth_training` | Where the output `train.jsonl`, `val.jsonl`, `test.jsonl` are written. | Only to redirect manifest location. |

---

### 9.3 Stage B Training — `configs/experiment/train_depth.yaml`

All training behaviour is controlled from here. Override any key on the command line (see §9.1).

#### Resolution and hardware

| Key | Default (experiment) | What it does | Impact of changing |
|-----|---------------------|--------------|-------------------|
| `size` | `512` | Square canvas size. Must match the size used in Stage A. | Changing requires rerunning Stage A AND deletes training parity. Do not change mid-training. |
| `bf16` | `true` | Brain-float16 mixed precision. Cuts VRAM by ~40%, minimal quality loss on modern GPUs. | Set `false` if your GPU does not support bf16 (older cards). Training becomes slower and heavier. |
| `gradient_checkpointing` | `true` | Trade compute for memory: recomputes activations during backward instead of storing them. Saves ~1.5 GB VRAM, slows backward ~20%. | Set `false` if you have plenty of VRAM and want faster training. Required for 12 GB GPUs. |
| `gradient_accumulation_steps` | `4` | How many micro-batches to accumulate before one optimizer step. Effective batch = `data.batch_size x gradient_accumulation_steps`. | Reduce if training is too slow and you have VRAM to spare. Increase if you want a larger effective batch without more VRAM. |
| `data.batch_size` | `4` | Per-GPU micro-batch size. | Reduce to `1` or `2` if you OOM. On a 24 GB GPU you can use `8`. |
| `data.workers` | `4` | DataLoader worker processes. | Set `0` on Windows if you get multiprocessing errors. |

#### Learning rate and schedule

| Key | Default | What it does | Impact of changing |
|-----|---------|--------------|-------------------|
| `learning_rate` | `1e-4` | Peak AdamW learning rate. | Too high (>5e-4): loss spikes and diverges. Too low (<1e-5): very slow convergence. |
| `lr_scheduler` | `cosine` | LR decay schedule after warmup. `cosine` decays smoothly to ~0; `constant` holds the peak LR throughout. | Use `constant` for a quick test; `cosine` for production runs (better final quality). |
| `lr_warmup_steps` | `500` | Steps where LR ramps from 0 up to `learning_rate`. Prevents early instability. | Reduce if your dataset is small and 500 steps is a large fraction of total training. Set `0` to disable. |

#### When to save / validate

| Key | Default | What it does | Impact of changing |
|-----|---------|--------------|-------------------|
| `epochs` | `5` | Total training passes over the dataset. | More epochs = more training time and potentially better quality, but also risk of overfitting. Watch val/loss — if it rises while train/loss falls, stop earlier. |
| `val_steps` | `500` | Every N optimizer steps: compute val/loss on held-out data + update `best_model/` if improved. Cheap (no image generation). | Lower = more frequent val/loss updates in TensorBoard. Higher = faster training throughput. |
| `ckpt_steps` | `1000` | Every N optimizer steps: save weights + generate N monitoring images. Heavy (runs the diffusion model). | Lower = more disk usage + slower overall training. Reasonable range: 500–2000 for a 5-epoch run. |
| `val_batches` | `64` | How many val batches to average for val/loss. More = more accurate estimate but slower. | Reduce to `4`–`8` for smoke tests. Keep `64` for real runs. |

#### Checkpoint monitoring grid (the 50/50 split)

| Key | Default | What it does | Impact of changing |
|-----|---------|--------------|-------------------|
| `n_grid_images` | `10` | Total images in each checkpoint's monitoring grid. Split 50/50: `n_grid_images // 2` are **fixed** (same scenes every checkpoint), the remaining half are **fresh** (re-randomized each checkpoint). | More = more disk use and slower checkpoint saves. Even number recommended (clean 50/50 split). Minimum 2. |
| `grid_include_empty_prompt` | `false` | When `true`, each monitoring image gets a 4th panel: generation with **empty prompt** (pure depth conditioning, no text). Useful for seeing how much the model leans on the depth map vs the prompt. | `true` doubles image generation time per checkpoint. Set `true` for diagnostic runs; keep `false` for speed. |

The fixed half: chosen **once at training start** using OS entropy (different each run). Stored in `_fixed_val_idxs`. Every checkpoint at `step1000`, `step2000`, `step3000` shows these **same scenes** so you can directly compare how the model improves. Files are named `sample_00_fixed.jpg` … `sample_04_fixed.jpg`.

The fresh half: re-drawn at each checkpoint from the remaining val indices (never overlaps the fixed half). Files are named `sample_05_new.jpg` … `sample_09_new.jpg`. Quickly checks generalization to unseen scenes.

#### Dataset and model paths

| Key | Default | What it does | When to change |
|-----|---------|--------------|----------------|
| `data.json_file` | `data/depth_training/train.jsonl` | Training manifest. Each line: `{raw_image_path, depth_path, prompt}`. | Change to point at a different JSONL if your dataset is elsewhere. |
| `data.val_json_file` | `data/depth_training/val.jsonl` | Validation manifest. **Never use `test.jsonl` here** — test split must stay uncontaminated. | Only change to use a different val set. |
| `data.image_root` | `null` (= repo root) | Base path prepended to relative `raw_image_path` values in the JSONL. Set to `/mnt/dataset` when images are on a different drive or machine. | **Required when training on Ubuntu with images at a different path from where you generated the JSONL.** |
| `local_files_only` | `true` | `true` = load all models from local disk (fully offline). `false` = allow HF downloads. | Keep `true` for training. |
| `base_model_path` | `checkpoints/local_models/stable-diffusion-v1-5` | Local path to SD1.5. | Only if you moved the model. |
| `depth_model_path` | `checkpoints/local_models/dpt-hybrid-midas` | Local path to DPT. | Only if you moved the model. |
| `lora.struct.ckpt_path` | `null` | Resume from a previous checkpoint. Set to e.g. `outputs/train/depth/runs/.../step2000`. | Use when continuing an interrupted training run. |
| `seed` | `42` | Random seed for training noise. Does **not** affect the fixed monitoring image selection (that uses OS entropy). | Change if you want to run multiple experiments with different initialization. |
| `prompt` | `null` | If set, overrides every image's caption with this single string. | Use for single-concept fine-tuning where all images share one prompt. |
| `tag` | `depth` | Written into the TensorBoard event filename and used as the output subfolder name. | Change if running multiple experiments to keep outputs separated. |
| `ignore_check` | `true` | Skip the startup data-integrity pre-check (verifies every JSONL entry exists on disk). Skipping saves ~30 seconds. | Set `false` if you suspect your JSONL has stale paths. |

---

### 9.4 Stage B Inference — `configs/inference_depth.yaml`

```powershell
# From a JSONL manifest (recommended for batch runs):
python depth_inference.py \
  ckpt_path=outputs/train/depth/runs/YYYY-MM-DD/HH-MM-SS/best_model \
  inference.json_file=data/depth_training/test.jsonl

# Single image:
python depth_inference.py \
  ckpt_path=outputs/train/depth/runs/YYYY-MM-DD/HH-MM-SS/best_model \
  "inference.images=[data/raw/000417/raw_image.jpg]" \
  "inference.prompts=['a driving scene at night in the rain']"
```

| Key | Default | What it does | When to change |
|-----|---------|--------------|----------------|
| `ckpt_path` | *(required)* | Path to a checkpoint folder containing `struct/lora-checkpoint.pt` + `struct/mapper-checkpoint.pt`. Usually `best_model/` or a specific `checkpoint-epoch1/step1000/`. | Always set. Use `best_model/` for final evaluation. Use step checkpoints to compare across training. |
| `inference.json_file` | `null` | JSONL file to run inference on. Each line needs `raw_image_path` and `prompt`. | Use `test.jsonl` for final held-out evaluation. Never use `val.jsonl` or `train.jsonl` here (contamination). |
| `inference.images` | `[]` | Direct list of image paths (alternative to json_file). | For quick single-image tests. |
| `inference.prompts` | `[]` | Prompts for the images list. Must match length of `inference.images`. | Required when using `inference.images`. |
| `inference.output_dir` | `outputs/inference/depth/results` | Where to save generated images. Resolved from repo root (absolute paths also accepted). | Change to organize results per experiment. |
| `inference.save_generated_only` | `false` | `false` = save 4-panel grid + individual panels (original, depth, predicted, raw-depth-gen). `true` = save only the predicted image, preserving the folder structure from the JSONL. | Set `true` for clean batch evaluation where you only need the generated images. |
| `inference.n_samples` | `1` | How many images to generate per input. | `2`–`4` for diversity comparison. Multiplies inference time. |
| `inference.num_inference_steps` | `50` | Diffusion denoising steps. More = slower but sharper. | `20` for fast preview, `50` for quality, `100` for maximum quality (diminishing returns above 80). |
| `inference.guidance_scale` | `7.5` | CFG scale: how strictly the model follows the text prompt. Higher = more prompt-driven, less variation. | `3`–`5` for creative / more variation. `7.5` standard. `12`–`15` for very tight prompt adherence. |
| `size` | `512` | Must match the size used during training. | Do not change. |
| `local_files_only` | `true` | Offline model loading. | Keep `true` after first download. |

---

### 9.5 Common scenarios — exact commands

#### Smoke test (verify pipeline works, ~5 minutes)
```powershell
python depth_map_calculations.py --data_dir data/ --dry_run_n 3

python depth_training.py experiment=train_depth `
  epochs=1 val_steps=10 ckpt_steps=20 val_batches=4 n_grid_images=2 `
  "data.workers=0" ignore_check=true
```

#### Full training run (Ubuntu, 5 epochs)
```bash
python depth_training.py experiment=train_depth
```

#### Resume interrupted training
```powershell
python depth_training.py experiment=train_depth `
  "lora.struct.ckpt_path=outputs/train/depth/runs/2026-07-01/00-41-13/checkpoint-epoch2/step4000"
```

#### Reduce memory (OOM on a 12 GB GPU)
```powershell
python depth_training.py experiment=train_depth data.batch_size=1 gradient_accumulation_steps=4
```

#### Images on a different drive (Ubuntu training with images at /mnt/data)
```bash
python depth_training.py experiment=train_depth data.image_root=/mnt/data
```
The JSONL has paths like `data/raw/000417/raw_image.jpg`. With `image_root=/mnt/data`, the dataset resolves each path to `/mnt/data/data/raw/000417/raw_image.jpg`. So keep the JSONL relative paths as-is and just point `image_root` at the drive root that makes them correct.

#### More monitoring images per checkpoint (watch 10 fixed scenes)
```powershell
python depth_training.py experiment=train_depth n_grid_images=20
# → 10 fixed + 10 fresh scenes per checkpoint grid
```

#### Add empty-prompt panel to see pure depth conditioning
```powershell
python depth_training.py experiment=train_depth grid_include_empty_prompt=true
# Each monitoring image gets a 4th panel: generated with empty prompt
```

#### Evaluate on test split after training
```powershell
python depth_inference.py `
  ckpt_path=outputs/train/depth/runs/YYYY-MM-DD/HH-MM-SS/best_model `
  inference.json_file=data/depth_training/test.jsonl `
  inference.save_generated_only=true `
  inference.output_dir=outputs/inference/depth/test_eval
```

#### Generate comparison report
```powershell
python training_report.py
python training_report.py --markdown   # GitHub-flavoured Markdown
```


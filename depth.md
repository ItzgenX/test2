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
| A — offline preprocessing | `pre_depth_calculations.py` | **Once**, before any training |
| B — training | `train_depth.py` | Iterative; resumes from checkpoint |
| B — inference | `inference_depth.py` | After training; per-image generation |

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
  [Stage A -- pre_depth_calculations.py]  (run ONCE offline)
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
  [Stage B -- train_depth.py]  (per training step)
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
  [Stage B -- inference_depth.py]
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

---

## 5 · Key Design Concepts

### 5.1 SquarePad (edge-replication padding)

`src.data.transforms.SquarePad` pads the shorter dimension to make the image square using edge-pixel replication (not zero-padding, not centre-crop). This preserves spatial content at borders.

**Why it's required**: MiDaS's `better_resize()` internally crops to a square before running. Without `SquarePad`, a landscape image would lose its left and right portions inside the DPT encoder. The padding is applied **in all three preprocessing sites** to ensure consistency:

1. `pre_depth_calculations.py` — Stage A offline preprocess
2. `inference_depth.py` — Stage B live inference preprocess
3. `configs/data/local_depth.yaml` — Stage B training transform chain (for raw images)

**Triplication risk**: These three sites must stay byte-identical. If you change the preprocessing (e.g., add a normalisation step), update all three.

### 5.2 skip_encode — Training vs Inference Path

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

### 5.6 Why test.json Is Never Touched During Training

`data/depth_training/test.json` is written by `pre_depth_calculations.py` alongside `train.json` and `val.json`, but **no training or validation config ever reads it**. It exists so you can run a final held-out evaluation after training is complete, using `inference_depth.py`.

Training configs (`configs/experiment/train_depth.yaml`) reference only:
```yaml
json_file:     data/depth_training/train.json
val_json_file: data/depth_training/val.json
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

**Dead keys** (present in config but NOT read by `train_depth.py`):

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
  json_file:     data/depth_training/train.json
  val_json_file: data/depth_training/val.json
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
python pre_depth_calculations.py --data_dir data/ --dry_run_n 3
```
Success: no errors; depth PNGs in `data/raw_depth/`.

Full run:
```powershell
python pre_depth_calculations.py --data_dir data/
```
Success:
- `data/raw_depth/` populated (one PNG per image)
- `data/depth_training/train.json`, `val.json`, `test.json` written
- Verification output shows `0 failures`
- Dataset sizes: 639 train / 137 val / 137 test

### Stage B — Training

```powershell
python train_depth.py experiment=train_depth
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
python inference_depth.py \
  ckpt_path=outputs/train/depth/runs/YYYY-MM-DD/HH-MM-SS/best_model \
  inference.input_dir=data/raw
```

Success: 4-panel JPG grids written to `outputs/inference/depth/`.

---

## 8 · Known Limitations

1. **`max_train_steps` does not stop training**: Setting `+max_train_steps=N` via Hydra affects the progress-bar total and LR scheduler warmup calculation only. The batch loop runs for the full number of epochs. To stop early, send SIGINT (Ctrl+C); the signal handler finishes the current step and saves a final checkpoint.

2. **Triplication risk in preprocessing**: The depth preprocessing transform chain (SquarePad → Resize → ToTensor → Normalize) is defined in three separate places (`pre_depth_calculations.py`, `inference_depth.py`, `configs/data/local_depth.yaml`). Any future change must be applied to all three simultaneously or parity will break.

3. **Per-image min-max depth**: Depth values are normalised per image. You cannot meaningfully compare depth magnitudes across images. The pipeline learns a structural prior, not a metric depth prior.

4. **Float→uint8 round-trip quantisation**: Stage A saves depth as 8-bit PNG, introducing up to 1/255 ≈ 0.004 error per pixel. The observed max diff between freshly-computed depth and saved PNG was 0.00837 (~2 uint8 levels), caused by floating-point non-determinism between sessions. This is not a preprocessing bug — it is inherent to the round-trip.

5. **`references.md §6` incorrect file reference**: Lists `src/data/local_depth.py` as a separate file. This file does not exist. `DepthJsonDataset` and `DepthJsonDataModule` live in `src/data/local.py`.

6. **OOM with experiment defaults on consumer GPU**: Experiment config uses `batch_size=4`, `gradient_accumulation_steps=4`, `gradient_checkpointing=True`, `bf16=True`. On a GPU with less than 16 GB VRAM this may OOM during backward. Use `data.batch_size=1 gradient_accumulation_steps=1` for single-GPU runs.

7. **No test-time evaluation script**: `test.json` exists but there is no built-in script to run batch inference on the test split and compute quantitative metrics. `inference_depth.py` with `save_generated_only=true` produces images but not scores.

# Segmentation Pipeline — Zero-to-Hero Training Parameter Guide

**Who this file is for:** Someone who has never trained a diffusion model and wants to
understand every parameter in `configs/experiment/train_seg.yaml` from first principles.

**Read first:** `DEPTH_TRAINING_GUIDE.md` — concepts like loss function, dataset math,
gradient accumulation, bf16, and TensorBoard reading are explained there in full.
This file documents only what is **different** for segmentation, plus a complete
parameter reference with seg-specific values.

**Companion files:**
- `SEGMENTATION.md` — architecture and pipeline explanation
- `configs/experiment/train_seg.yaml` — the config you edit
- `seg_training.py` — the training script

---

## 0. Key difference from depth — what changes in the conditioning signal

Both pipelines use the same LoRA + mapper architecture. The only differences are:

| | Depth | Segmentation |
|---|---|---|
| Conditioning signal | Grayscale depth map `[0,1]` per-image normalized | 19-class Cityscapes colour map `[0,1]` fixed palette |
| Offline calc script | `depth_map_calculations.py` → `data/raw_depth/` | `seg_map_calculations.py` → `data/raw_seg/` |
| Training manifest | `data/depth_training/*.jsonl` | `data/seg_training/*.jsonl` |
| Encoder (Stage C/live) | DPT-Hybrid-MiDaS (regression) | SegFormer-b5-Cityscapes (classifier) |
| Encoder during training | NOT called (`skip_encode=True`) | NOT called (`skip_encode=True`) |
| Mapper input | depth map → mapper | colour seg map → mapper |
| TensorBoard tag | `depth` | `seg` |
| Output panels in grid | `ORIGINAL \| DEPTH MAP \| PREDICTED` | `ORIGINAL \| SEG MAP \| PREDICTED` |

**Why `skip_encode=True` during training for both:**
The conditioning encoder (MiDaS or SegFormer) runs only at inference. During training
we use pre-saved maps from disk. This makes training ~3× faster than running the encoder
every step, and it lets the encoder stay frozen without ever touching GPU memory during training.

---

## 1. Dataset math — identical to depth, using the same numbers

```python
# 60K train / 4K val / 4K test — same as depth for fair comparison
N_TRAIN = 60_000
N_VAL   =  4_000

# Same hardware config as depth:
BATCH_SIZE          = 4     # data.batch_size
GRAD_ACCUM          = 4     # gradient_accumulation_steps
EFFECTIVE_BATCH     = 16

STEPS_PER_EPOCH     = 3_750     # ceil(60000 / 16)
EPOCHS              = 5
TOTAL_STEPS         = 18_750    # 5 × 3750

# val/loss checks: 3750 / 500 = 7–8 per epoch
# step checkpoints: 3750 / 1000 = 3–4 per epoch
```

**Why we keep the math identical to depth:**
The whole purpose of this project is to **compare** depth conditioning vs segmentation
conditioning under identical conditions. If the datasets, batch sizes, or training lengths
differ, the comparison becomes unfair. Keep these numbers the same as depth and only
change what segmentation genuinely requires (encoder, data paths).

---

## 2. Every parameter in `configs/experiment/train_seg.yaml`

Parameters identical to depth are not repeated here — see `DEPTH_TRAINING_GUIDE.md §2`
for full explanations of `size`, `learning_rate`, `lr_scheduler`, `lr_warmup_steps`,
`epochs`, `gradient_accumulation_steps`, `data.batch_size`, `bf16`,
`gradient_checkpointing`, `val_steps`, `ckpt_steps`, `val_batches`, `seed`, `prompt`,
`local_files_only`, `ignore_check`, `lora.struct.ckpt_path`.

Below are seg-specific parameters and values that differ.

---

### 2.1 `seg_model_path` and `seg_model_name` — the locked SegFormer-b5 model

```yaml
seg_model_name: nvidia/segformer-b5-finetuned-cityscapes-1024-1024
seg_model_path: checkpoints/local_models/segformer-b5-cityscapes
```

**What it is:** the frozen SegFormer-b5 encoder that produces segmentation maps.
Used at inference (`seg_inference.py`) and in Stage C (`seg_map_calculations.py`).
NOT used during training (skip_encode=True).

**Why b5 is locked (never use b0):**
SegFormer-b5 has 82M parameters in its MiT-B5 backbone. SegFormer-b0 has 3.7M.
For driving scenes (pedestrians, poles, traffic lights), b0 misclassifies thin structures
that matter for structural conditioning. See `SEGMENTATION.md §4` for full evidence.

**This model is NOT loaded during training.** Only at Stage C and inference.
The config key exists so `seg_inference.py` knows where to find it.

---

### 2.2 `data.json_file` and `data.val_json_file` — seg manifests

```yaml
data:
  json_file:     data/seg_training/train.jsonl
  val_json_file: data/seg_training/val.jsonl   # NEVER test.jsonl here
```

**What it is:** paths to the training and validation manifests built by `seg_map_calculations.py`.
Each line has three fields:
```json
{"raw_image_path": "data/raw/000417/raw_image.jpg",
 "seg_path":       "data/raw_seg/000417/raw_image.png",
 "prompt":         "a driving scene with pedestrians"}
```

**`seg_path` points to the raw class-ID PNG** (8-bit grayscale, values 0–18).
The dataset class (`SegJsonDataset`) reads this PNG, applies NEAREST resize, then
colourises with `SEG_CITYSCAPES_PALETTE` at load time. It does NOT call SegFormer.

**Warning — NEAREST resize is required for class-ID maps:**
```python
# src/data/local_seg.py — _load_seg_colormap()
ids_pil = ids_pil.resize((self.size, self.size), Image.NEAREST)
# NOT Image.BILINEAR — bilinear would blend class IDs and invent new classes
```

**Why test.jsonl must never appear here:**
`test.jsonl` is the final held-out evaluation set. Reading it during training would
let the val/loss metric see test data → contamination → your final evaluation is worthless.
The config enforces this: only `train.jsonl` and `val.jsonl` are ever listed here.

---

### 2.3 `data.image_root` — images on a different drive (seg version)

```yaml
data:
  image_root: null   # null = repo root
```

Same as depth. For Ubuntu training with images at `/mnt/dataset`:
```yaml
data:
  image_root: /mnt/dataset
  json_file:     data/seg_training/train.jsonl   # JSONL stays in repo
  val_json_file: data/seg_training/val.jsonl
```

The dataset class resolves `data/raw/000417/raw_image.jpg` as
`/mnt/dataset/data/raw/000417/raw_image.jpg`. The `seg_path` field in the JSONL
(`data/raw_seg/000417/raw_image.png`) is resolved the same way — so seg PNGs
must also be at `/mnt/dataset/data/raw_seg/`.

---

### 2.4 `n_grid_images` and `grid_include_empty_prompt` — seg grid settings

```yaml
n_grid_images: 10                # 5 fixed + 5 fresh (same as depth)
grid_include_empty_prompt: false # default off; enable to see "RAW SEG GEN" panel
```

**The 4th panel for segmentation — "RAW SEG GEN":**
When `grid_include_empty_prompt: true`, each monitoring image gets a 4th panel:
```
[ORIGINAL | SEG MAP | PREDICTED (with prompt) | RAW SEG GEN (empty prompt)]
```

`RAW SEG GEN` shows what the model generates with an empty text prompt — pure
segmentation conditioning with no text influence. This is useful for answering:
"is the model actually following the segmentation map, or is it relying on the text?"

If `RAW SEG GEN` shows similar spatial structure to `PREDICTED`, the segmentation
conditioning is working — text is enhancing but not overriding the structural signal.

**Reading the SEG MAP panel:**
The segmentation colour map uses the Cityscapes palette:
```
Purple  (128, 64, 128)  = road          ← should cover most of the lower frame
Cyan    (70, 130, 180)  = sky           ← should cover most of the upper frame  
Green   (107, 142, 35)  = vegetation
Deep blue (0, 0, 142)  = car
Red     (220, 20, 60)   = person
```
If the SEG MAP looks like a uniform blob instead of distinct coloured regions,
the SegFormer encoder is not producing correct segmentation — check the model files.

---

### 2.5 `tag` — separates seg outputs from depth outputs

```yaml
tag: seg
```

**What it does:**
- TensorBoard events go to `outputs/train/seg/runs/` (separate from depth's `outputs/train/depth/`)
- The tfevents hostname is set to `"seg"` (not the machine hostname)
- You can run `tensorboard --logdir outputs/train/` and see BOTH pipelines labeled

**Never change this** when running seg training. If you run multiple seg experiments,
use CLI overrides to add a sub-tag:
```powershell
python seg_training.py experiment=train_seg tag=seg_lr3e4
# outputs go to outputs/train/seg_lr3e4/runs/...
```

---

### 2.6 `lora.struct.ckpt_path` — resume a seg training run

```yaml
lora:
  struct:
    ckpt_path: null
```

Identical to depth. CLI usage:
```powershell
python seg_training.py experiment=train_seg \
  "lora.struct.ckpt_path=outputs/train/seg/runs/2026-07-01/00-47-30/checkpoint-epoch2/step7500"
```

---

## 3. The seg-specific preprocessing chain — what makes it different from depth

During training, the raw image goes through this exact chain
(`configs/data/local_seg.yaml` + `src/data/local_seg.py`):

```python
# ── RGB IMAGE (same as depth) ──────────────────────────────────────────────── #
# configs/data/local_seg.yaml — transform list:
SquarePad()                           # pad to square (edge-replication, no crop)
Resize((512, 512))                    # square → 512×512
ToTensor()                            # uint8 [0,255] → float [0,1]
Normalize(mean=[0.5]*3, std=[0.5]*3)  # [0,1] → [-1,1]
# → batch["jpg"]: [B, 3, 512, 512] in [-1,1]

# ── SEG MAP (unique to segmentation) ──────────────────────────────────────── #
# src/data/local_seg.py — SegJsonDataset._load_seg_colormap():
Image.open(seg_path).convert("L")         # 8-bit class-ID PNG (values 0–18)
ids_pil.resize((512, 512), Image.NEAREST) # NEAREST resize — never bilinear
ids = torch.from_numpy(np.asarray(ids_pil, dtype=np.int64))  # [512, 512] long
colour = seg_colorize_ids(ids, palette)   # palette lookup → [3, 512, 512] in [0,1]
# → batch["seg"]: [B, 3, 512, 512] in [0,1]
```

**Key difference from depth:**
Depth preprocessing applies a `Resize` that can use bilinear (fine for continuous values).
Seg preprocessing MUST use `NEAREST` — bilinear on class IDs produces non-existent classes.
This is enforced in `_load_seg_colormap()` and cannot be overridden via config.

---

## 4. Recommended configs for different GPU sizes — seg edition

### 4.1 12 GB GPU (default — matches depth exactly for fair comparison)

```yaml
# This IS configs/experiment/train_seg.yaml — shown here for clarity
gradient_checkpointing: true
gradient_accumulation_steps: 4   # effective batch 16
size: 512
data:
  batch_size: 4
bf16: true
learning_rate: 1.0e-4
lr_warmup_steps: 500
epochs: 5
val_steps: 500
ckpt_steps: 1000
val_batches: 64
n_grid_images: 10
grid_include_empty_prompt: false
tag: seg
```

### 4.2 8 GB GPU

```yaml
gradient_checkpointing: true
gradient_accumulation_steps: 8
data:
  batch_size: 1
bf16: true
learning_rate: 7.07e-5   # sqrt(8/16) × 1e-4
lr_warmup_steps: 500
epochs: 5
val_steps: 500
ckpt_steps: 1000
tag: seg
```

### 4.3 24 GB GPU

```yaml
gradient_checkpointing: false
gradient_accumulation_steps: 2
data:
  batch_size: 8
bf16: true
learning_rate: 1.0e-4   # effective batch still 16
lr_warmup_steps: 500
epochs: 5
val_steps: 500
ckpt_steps: 1000
tag: seg
```

---

## 5. Reading TensorBoard for segmentation

```powershell
tensorboard --logdir outputs/train/seg/runs/
# or both pipelines together:
tensorboard --logdir outputs/train/
```

Expected scalars (same tags as depth, different values):

| Scalar | What to watch |
|---|---|
| `train/loss` | Should decrease from ~0.15 to ~0.12–0.13 over 5 epochs |
| `val/loss` | True quality signal — should track train/loss without diverging |
| `train/grad_norm` | Should stay near 0.002, occasionally spikes to 0.003 |
| `train/epoch` | Fractional epoch number for context |
| `train/lr` | Should show cosine decay from 1e-4 to ~0 |

**Expected val/loss for segmentation vs depth:**
Both pipelines should reach similar val/loss ranges (~0.13–0.15 after 5 epochs)
because the loss function is the same (epsilon-prediction MSE). The comparison between
the two pipelines is done via:
1. Final val/loss (lower = better conditioning)
2. `python training_report.py` — generates the side-by-side comparison table
3. Visual inspection of checkpoint grid images

---

## 6. Reading the segmentation checkpoint monitoring grid

Each checkpoint-grid image shows three panels side by side:

```
┌──────────────┬──────────────┬──────────────┐
│  ORIGINAL    │   SEG MAP    │  PREDICTED   │
│              │  (palette    │  (generated  │
│  raw input   │   colours)   │   image)     │
└──────────────┴──────────────┴──────────────┘
```

**What to look for at different training stages:**

**Step 500–1000 (early):**
- PREDICTED will look like a rough driving scene but without specific structure
- The segmentation map may have limited influence — road/sky broad regions may start aligning
- The colour palette regions in SEG MAP should be distinct and recognisable

**Step 3000–7000 (mid):**
- PREDICTED should show road-sky boundary alignment with SEG MAP
- Large classes (road = purple, sky = blue) should be reproduced in PREDICTED
- Person/car colours in SEG MAP should appear in correct spatial locations

**Step 10000+ (late):**
- PREDICTED should closely follow the SEG MAP class regions
- Scene category (city street, highway, etc.) should match the original
- Fine details (individual pedestrian shapes) may still be blurry — this is acceptable

**Comparing to depth grids:**
At the same training step, compare depth's grid vs seg's grid on the same scene.
Depth conditioning preserves 3D spatial structure (foreground/background).
Segmentation conditioning preserves semantic layout (which class is where).
Neither is "better" — they condition on different aspects of the scene.

---

## 7. Segmentation-specific problems and fixes

| Problem | Symptom | Fix |
|---|---|---|
| SEG MAP is uniform colour | All pixels same class in grid | Check segformer-b5 model files in `checkpoints/local_models/segformer-b5-cityscapes/` |
| NEAREST interpolation error | `AttributeError: NEAREST` | Pillow version issue; run `pip install -U Pillow` |
| OOM on Stage C (seg calc) | CUDA OOM during `seg_map_calculations.py` | Reduce `--batch_size 2` or `--batch_size 1` during calc |
| Wrong class IDs in PNG | `ValueError: class id 255` in verifier | Wrong seg model (b5 vs b0 mismatch); rerun Stage C with correct model |
| Seg path missing from manifest | `KeyError: seg_path` on training start | Rerun `seg_map_calculations.py --data_dir data/` to rebuild manifests |
| Colour palette mismatch | Generated image has wrong class colours | `SEG_CITYSCAPES_PALETTE` in `seg_encoder.py` was modified; restore from git |
| Val/loss worse than depth | Seg val/loss stuck above depth's | Normal at early steps; both should converge to similar range by epoch 3 |
| PREDICTED ignores seg map | Generated image looks random | `skip_encode` may be False; check `seg_training.py` batch["seg"] path |

---

## 8. Comparing depth vs segmentation results

After training both pipelines to completion, run:

```powershell
python training_report.py
```

This generates a side-by-side table from both runs' TensorBoard logs:

```
  Metric                                  DEPTH           SEGMENTATION
  ────────────────────────────────────    ──────────────  ──────────────
  Best val/loss (best_model/)             0.132           0.134
  Train loss at epoch 5                   0.089           0.093
  Runtime                                 11h 24m         11h 31m
  Checkpoints saved                       22              22
  Grid images generated                   220             220
  Inference status                        1 grid generated  1 grid generated
```

**Interpreting the comparison:**
- Lower val/loss = the conditioning signal is being used more effectively
- Similar val/loss = both conditionings are roughly equally useful
- Look at the grid images: depth preserves 3D structure; segmentation preserves semantic layout

---

## 9. Quick reference — config values for 60K / 4K / 4K

```yaml
# Copy-paste ready config for 60,000 train / 4,000 val / 4,000 test
# Single 12 GB GPU (effective batch = 16)
# Run: python seg_training.py experiment=train_seg
# IMPORTANT: run seg_map_calculations.py --data_dir data/ FIRST

size: 512
learning_rate: 1.0e-4
lr_scheduler: cosine
lr_warmup_steps: 500               # 13% of first epoch (500 / 3750)
epochs: 5                          # 5 × 3750 = 18750 total optimizer steps
gradient_checkpointing: true       # required for 12 GB GPU
gradient_accumulation_steps: 4     # 4 bsz × 4 accum = 16 effective batch
data:
  batch_size: 4                    # max for 12 GB with checkpointing + bf16
  val_batch_size: 4
  workers: 4                       # set 0 on Windows if multiprocessing crashes
  json_file:     data/seg_training/train.jsonl   # seg manifests (not depth)
  val_json_file: data/seg_training/val.jsonl     # NEVER test.jsonl here
val_steps: 500                     # 7-8 val/loss checks per epoch
ckpt_steps: 1000                   # 3-4 checkpoints per epoch
val_batches: 64                    # 64 × 4 = 256 val images per check
n_grid_images: 10                  # 5 fixed + 5 fresh scenes per checkpoint
grid_include_empty_prompt: false   # true → add "RAW SEG GEN" panel (doubles gen time)
bf16: true
seed: 42
prompt: null                       # use per-image captions from JSONL
local_files_only: true
ignore_check: true                 # false → verify all seg_path PNGs exist on disk
tag: seg                           # DO NOT change — separates seg from depth in TensorBoard

# Model paths (locked — do not change):
seg_model_name: nvidia/segformer-b5-finetuned-cityscapes-1024-1024
seg_model_path: checkpoints/local_models/segformer-b5-cityscapes   # b5, NOT b0

# GPU scaling:
# 8 GB:  batch_size=1, gradient_accumulation_steps=16, learning_rate=7e-5
# 24 GB: batch_size=8, gradient_accumulation_steps=2, gradient_checkpointing=false
# 4×GPU: batch_size=4, gradient_accumulation_steps=1, learning_rate=2e-4
```

---

## 10. The full segmentation training checklist

Before running `python seg_training.py experiment=train_seg`:

```
[ ] Stage C complete: seg_map_calculations.py ran without errors
      → data/raw_seg/ exists with 913 PNG files (or your N files)
      → data/seg_training/train.jsonl, val.jsonl, test.jsonl all show [PASS]

[ ] Model present: checkpoints/local_models/segformer-b5-cityscapes/ exists
      → contains config.json, preprocessor_config.json, pytorch_model.bin
      → NOT segformer-b0-cityscapes

[ ] Base model present: checkpoints/local_models/stable-diffusion-v1-5/ exists

[ ] Source JSONL present: data/train.jsonl, data/val.jsonl exist
      → these are the split manifests (key = "source")

[ ] Config correct: configs/experiment/train_seg.yaml has
      → json_file: data/seg_training/train.jsonl   (not depth_training)
      → val_json_file: data/seg_training/val.jsonl  (not test.jsonl)
      → tag: seg

[ ] (Optional) Smoke test passes:
    python seg_training.py experiment=train_seg \
      epochs=1 val_steps=10 ckpt_steps=20 n_grid_images=2 "data.workers=0"
```

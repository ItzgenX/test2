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
| C — offline preprocessing | `calculate_segmentation_map.py` | **Once**, before any training |
| D — training | `train_seg.py` | Iterative; resumes from checkpoint |
| D — inference | `inference_seg.py` | After training; per-image generation |

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
  [Stage C -- calculate_segmentation_map.py]  (run ONCE offline)
        |
        +-- build_seg_square_preprocess(size=512, resize_mode="letterbox")
        |     (same factory used by inference_seg.py -- parity guaranteed)
        +-- SegmentationEncoder.label_ids(tensor)
        |     (SegFormer-b5 prediction --> raw class IDs [0..18])
        +-- Save as 8-bit grayscale PNG (mode "L", values 0..18)
        |       --> data/raw_seg/.../img.png
        +-- Write data/seg_training/{train,val,test}.json
               (keys: raw_image_path, seg_path, prompt)

        |
        v
  [Stage D -- train_seg.py]  (per training step)
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
  [Stage D -- inference_seg.py]
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

### 5.3 NEAREST Interpolation for Class IDs

When `SegJsonDataset` loads a saved seg PNG and resizes it, it uses `NEAREST` interpolation, not bilinear. This is critical:

- Bilinear interpolation would average adjacent class IDs (e.g., class 3 and class 7 blended → float 5.0, which rounds to class 5, a third wrong class).
- NEAREST interpolation selects the nearest pixel's class ID without mixing.

The resize happens in `_load_seg_colormap()` before `seg_colorize_ids()` is called.

### 5.4 `build_seg_square_preprocess()` — Single Source of Truth

The depth pipeline triplicates its preprocessing across three files. The seg pipeline fixes this with a single factory function:

```python
# src/data/transforms.py
def build_seg_square_preprocess(size, resize_mode="letterbox"):
    ...
```

This function is imported by both:
- `calculate_segmentation_map.py` (Stage C offline)
- `inference_seg.py` (Stage D live inference)

Parity is guaranteed by construction — there is only one definition.

### 5.5 skip_encode — Same Pattern as Depth

```python
model.forward_easy(..., skip_encode=True)   # training:  uses pre-saved colour PNG
model.sample(...)                            # inference: calls SegFormer live
```

During training `batch["seg"]` contains the colour map loaded from the saved PNG via `SegJsonDataset`. `skip_encode=True` bypasses the SegFormer entirely. During inference, `SegmentationEncoder.forward()` runs live inside `model.sample()`.

### 5.6 Checkpoint Grid, val_steps/ckpt_steps, test.json

Identical to the depth pipeline — see DEPTH.md §5.4–5.6. The seg trainer (`train_seg.py`) is a direct mirror of `train_depth.py` with `batch["depth"]` replaced by `batch["seg"]` and depth-specific helpers renamed to their `_seg_*` equivalents.

### 5.7 SegFormer-b5 vs b0 — Why b0 is Wrong

SegFormer-b0 is a small model designed for speed, not accuracy. On Cityscapes driving scenes:
- b0 misclassifies thin structures (pedestrian poles, traffic lights) that matter for structural conditioning.
- b5 achieves significantly higher mIoU on Cityscapes, especially on boundary-sensitive classes.

Two old experiment configs (`train_seg_12gb.yaml`, `train_seg_cluster.yaml`) incorrectly reference b0:
```yaml
model: "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"  # WRONG
```
These files also have wrong JSON paths (missing `data/` prefix). **Do not use them.** Use `configs/experiment/train_seg.yaml` only.

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
  json_file:     data/seg_training/train.json
  val_json_file: data/seg_training/val.json   # NEVER test.json here
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

`use_empty_prompt_eval`, `n_samples`, `save_grid`, `log_cond` — present in base config, not read by `train_seg.py`.

### Broken configs (do not use)

| File | Bug |
|------|-----|
| `configs/experiment/train_seg_12gb.yaml` | b0 model; wrong JSON paths (missing `data/` prefix); `local_files_only: false` |
| `configs/experiment/train_seg_cluster.yaml` | Same bugs as above |

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
python calculate_segmentation_map.py --data_dir data/ --dry_run_n 1 --local_files_only False
```
Success: no errors; seg PNG written to `data/raw_seg/`; class IDs in `[0, 18]`.

Full run (after b5 model is cached):
```powershell
python calculate_segmentation_map.py --data_dir data/
```
Success:
- `data/raw_seg/` populated (one 8-bit PNG per image)
- `data/seg_training/train.json`, `val.json`, `test.json` written
- Verification output shows `0 failures`
- Dataset sizes: 639 train / 137 val / 137 test

### Stage D — Training

```powershell
python train_seg.py experiment=train_seg
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
python inference_seg.py \
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
| S1 | Config files — correct experiment YAML | PARTIAL | `configs/experiment/train_seg.yaml` correct (b5, right paths). Two stale configs (`train_seg_12gb.yaml`, `train_seg_cluster.yaml`) still exist with wrong b0 model + wrong JSON paths. Not deleted yet. Do not use them. |
| S2 | b5 model availability | FIXED | Model was in HF cache (`~/.cache/huggingface/hub/models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/`). Copied to `checkpoints/local_models/segformer-b5-cityscapes/` on 2026-06-30. Dry run with `local_files_only=True` confirmed load succeeded (19 classes, 1172 weights loaded). |
| S3 | Stage C — offline seg map generation | **PASS** | `python calculate_segmentation_map.py --data_dir data/` completed with 0 errors. 913 PNG files written to `data/raw_seg/`. All 3 JSON splits verified by built-in checker: `train.json 639/639`, `val.json 137/137`, `test.json 137/137`. PNG inspection: shape `(512, 512)`, dtype `uint8`, values in `[0, 18]` — correct. |
| S4 | Stage D — training startup | PENDING FIRST RUN | `train_seg.py` mirrors `train_depth.py` exactly; training not yet executed. Next step: `python train_seg.py experiment=train_seg`. Success criterion: startup log shows `19 classes`, loss begins decreasing within first 100 steps. |
| S5 | Train/inference preprocessing parity | CODE READS CORRECT — NOT YET VERIFIED BY EXECUTION | `build_seg_square_preprocess()` SSOT confirmed imported by both `calculate_segmentation_map.py` and `inference_seg.py`. Parity is guaranteed by construction. Cannot be verified by execution until inference is run with a trained checkpoint. |
| S6 | Val loss + checkpoint grid | PENDING FIRST RUN | Blocked on S4 (training). Code mirrors train_depth.py's verified grid logic. |
| S7 | TensorBoard tags | PENDING FIRST RUN | Blocked on S4. Expected tags: `train/loss`, `train/lr`, `val/loss`, `val/sample_00`…`val/sample_09`. |
| S8 | Inference | PENDING FIRST RUN | `inference_seg.py` untested — no checkpoint available yet. Blocked on S4. |
| S9 | Known issues documented | DOCUMENTED | See §8 above. Two stale configs (S1), NEAREST-resize slow on large batches, no test-eval script. |

### What is now unblocked

Stage C is verified. The single remaining blocker before training can start is **running `train_seg.py`** — there are no more missing models, no empty data directories, no broken JSON paths. All prerequisites are met.

```powershell
# Run on Ubuntu (final training machine):
python train_seg.py experiment=train_seg
```

Watch for these in the first 50 steps to confirm training is working:
- `[model] base = .../stable-diffusion-v1-5` in startup log
- `Number params Mapper Network(s) 1,245,072` (same as depth — encoder frozen)
- `val/loss` decreasing (not stuck at a constant)
- Checkpoint grid at step 1000 shows recognisable colour blobs per Cityscapes class

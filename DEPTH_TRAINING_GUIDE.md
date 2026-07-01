# Depth Pipeline — Zero-to-Hero Training Parameter Guide

**Who this file is for:** Someone who has never trained a diffusion model before and wants to understand
every single knob in `configs/experiment/train_depth.yaml` from first principles.

**Companion files:**
- `DEPTH.md` — architecture and pipeline explanation (read that first)
- `configs/experiment/train_depth.yaml` — the actual config you edit (numbers match this guide)
- `depth_training.py` — the training script (you run this)

---

## 0. What are we actually training? (concepts first)

### 0.1 The three frozen things and the two trainable things

Stable Diffusion 1.5 has several components. During our training:

```
FROZEN (weights never change):
  ┌─────────────────────────────────────────┐
  │  Text encoder (CLIP)                    │  reads your text prompt
  │  VAE encoder/decoder                    │  compresses images to/from latent space
  │  SD1.5 UNet base weights                │  the denoising model
  │  DPT-Hybrid-MiDaS encoder              │  produces depth maps (only at inference)
  └─────────────────────────────────────────┘

TRAINABLE (what we update):
  ┌─────────────────────────────────────────┐
  │  LoRA adapters (rank=128)               │  29,999,104 parameters
  │    injected into UNet residual-conv      │  learn "depth-aware delta-weights"
  │  Mapper network                         │  1,245,072 parameters
  │    translates depth embedding → LoRA    │
  └─────────────────────────────────────────┘

Total trainable: ~31.2M out of ~865M total (≈3.6% of the model)
```

### 0.2 What the loss function measures

Every training step does this:

```python
# Simplified — actual code in src/model.py forward()

# 1. Load a training image + its pre-saved depth map + text prompt
image   = batch["jpg"]       # [B, 3, 512, 512] in [-1,1]
depth   = batch["depth"]     # [B, 3, 512, 512] in [0,1]  (pre-computed, skip_encode=True)
prompt  = batch["caption"]   # list of strings

# 2. Encode image → latent space (VAE encoder, frozen)
latent  = vae.encode(image)  # [B, 4, 64, 64]

# 3. Sample a random noise level t, add noise to latent
t       = random_timestep()               # e.g. t=700 out of 1000
noise   = torch.randn_like(latent)
noisy   = add_noise(latent, noise, t)     # noisy latent

# 4. Ask the UNet: "what noise was added?"
#    The depth map goes through mapper → LoRA delta-weights → modify the UNet
#    The text prompt goes through CLIP → cross-attention in the UNet
predicted_noise = unet(noisy, t, depth_via_mapper, text_via_clip)

# 5. Loss = MSE between predicted noise and actual noise
loss = MSE(predicted_noise, noise)   ← THIS is what val/loss shows in TensorBoard
```

**What "val/loss going down" actually means:**
The model is getting better at predicting the noise that was added to images.
Indirectly, this means it is learning to generate images that match the supplied depth map.
You will NOT see great images at step 100 — the model needs thousands of steps to produce
recognisable structure. The checkpoint grids (monitoring images) are your visual signal.

**What val/loss of ~0.15 means:**
At initialization, the LoRA adapters are near-zero, so the model ignores depth entirely.
Loss ≈ 0.15 is the baseline (SD1.5 predicting noise without depth conditioning).
After good training, val/loss should settle around 0.12–0.14 (roughly 5–10% improvement).
A 10% drop in val/loss is meaningful. A 1% drop might be noise.

---

## 1. Dataset math — what your 60K / 4K / 4K numbers mean

Replace 60,000 with your actual train count to scale everything below.

```python
# ── DATASET ────────────────────────────────────────────────────────────────── #
N_TRAIN = 60_000        # number of training images
N_VAL   =  4_000        # number of validation images (NEVER touched for training)
N_TEST  =  4_000        # number of test images (NEVER touched until final eval)

# ── HARDWARE (configs/experiment/train_depth.yaml) ───────────────────────────  #
BATCH_SIZE          = 4     # data.batch_size          — images per GPU per step
GRAD_ACCUM          = 4     # gradient_accumulation_steps — steps before one optimizer update
EFFECTIVE_BATCH     = BATCH_SIZE * GRAD_ACCUM   # = 16

# ── DERIVED NUMBERS ───────────────────────────────────────────────────────────  #
import math
STEPS_PER_EPOCH = math.ceil(N_TRAIN / EFFECTIVE_BATCH)   # = 3,750
EPOCHS          = 5
TOTAL_STEPS     = EPOCHS * STEPS_PER_EPOCH                # = 18,750

VAL_STEPS  = 500    # val_steps  — validate every N optimizer steps
CKPT_STEPS = 1000   # ckpt_steps — save checkpoint every N optimizer steps

VALS_PER_EPOCH   = STEPS_PER_EPOCH // VAL_STEPS    # ≈ 7–8 val/loss checks per epoch
CKPTS_PER_EPOCH  = STEPS_PER_EPOCH // CKPT_STEPS   # ≈ 3–4 step-level checkpoints per epoch

print(f"Steps per epoch  : {STEPS_PER_EPOCH}")
print(f"Total steps      : {TOTAL_STEPS}")
print(f"Val checks/epoch : {VALS_PER_EPOCH}")
print(f"Checkpoints/epoch: {CKPTS_PER_EPOCH}")
# Steps per epoch  : 3750
# Total steps      : 18750
# Val checks/epoch : 7
# Checkpoints/epoch: 3
```

**Why this matters:** every parameter below ultimately controls one of these numbers.
If you change `batch_size`, `grad_accum`, or `epochs`, re-run this calculation.

---

## 2. Every parameter in `configs/experiment/train_depth.yaml` — explained

### 2.1 `size` — canvas resolution

```yaml
size: 512
```

**What it is:** the spatial size (height = width) of every tensor in the pipeline.
Images are resized to `size × size` by SquarePad + Resize before entering the UNet.
Depth maps are computed and saved at `size × size`.

**What happens if you change it:**
- Increase to 768 or 1024 → higher resolution output, but VRAM usage scales quadratically
  (2× size = 4× VRAM for the UNet activations) and you MUST rerun Stage A to regenerate
  depth maps at the new resolution
- Decrease to 256 → faster but blurry output (SD 1.5 was trained at 512)

**Recommendation for 60K dataset:** keep 512. This is SD 1.5's native resolution.
Any other size requires rerunning `depth_map_calculations.py --size NEW_SIZE`.

---

### 2.2 `learning_rate` — the most important single number

```yaml
learning_rate: 1.0e-4   # = 0.0001
```

**What it is:** how large a step the AdamW optimizer takes each update.
Too large → loss spikes / diverges. Too small → training is very slow.

**Mental model:** learning rate is the size of each correction the model makes.
If you estimate you need to walk 10 metres north, a learning rate of 1 m/step is reasonable.
A learning rate of 100 m/step will overshoot. A learning rate of 1 mm/step will take forever.

**What the loss curve looks like at different learning rates:**

```
lr = 5e-4 (too high, likely to diverge):
  loss: 0.15 → 0.12 → 0.18 → 0.25 → spikes / diverges
  grad_norm: spikes to >1.0 repeatedly

lr = 1e-4 (good):
  loss: 0.15 → 0.145 → 0.138 → 0.132 → stabilizes
  grad_norm: starts ~0.002, slowly decreases

lr = 1e-5 (too low):
  loss: 0.15 → 0.149 → 0.148 → barely moves after 5000 steps
  grad_norm: consistently very low, near zero
```

**For 60K dataset, 5 epochs, effective batch 16:**
`1e-4` is the proven value (validated by running actual training). With cosine decay,
this LR gently decreases to near-zero by step 18,750.

**Formula for scaling when you change effective batch:**
If you multiply effective batch by N, scale LR by sqrt(N):
```python
# Example: moving from single GPU (batch 16) to 4-GPU cluster (batch 64)
new_lr = 1e-4 * math.sqrt(64 / 16)   # = 1e-4 * 2 = 2e-4
```

---

### 2.3 `lr_scheduler` — how LR changes over training

```yaml
lr_scheduler: cosine
```

**What it is:** the rule for how `learning_rate` changes over the total training run.

**Options:**

```
cosine (recommended):
  step:     0    500   3750   7500   11250   18750
  lr:     0.0  1e-4   9e-5   6e-5    3e-5   ~0.0
  shape: ramp up, then smooth cosine decay to ~zero

constant:
  step:     0    500   3750   ...   18750
  lr:     0.0  1e-4   1e-4   ...   1e-4
  shape: ramp up, then flat — no decay

linear:
  step:     0    500   3750   ...   18750
  lr:     0.0  1e-4   9e-5   ...   ~0.0
  shape: ramp up, then linear decay
```

**Why cosine is better for this task:**
- Early training: higher LR explores the loss landscape fast
- Late training: lower LR fine-tunes into a sharper minimum without overshooting
- The model will continue improving toward the end of training even when train/loss
  appears to plateau — the cosine tail is doing subtle fine-tuning

**When to use `constant`:** quick smoke tests (fewer than 100 steps)
where you just want to verify the pipeline works.

---

### 2.4 `lr_warmup_steps` — protecting against early instability

```yaml
lr_warmup_steps: 500
```

**What it is:** for the first `lr_warmup_steps` steps, the LR ramps linearly from 0
up to `learning_rate`. Only after warmup does the cosine decay start.

**Why it exists:** At the very start of training, the LoRA adapters are randomly initialized
near zero. If the LR is immediately large, the first few batches produce large gradient
updates that can push the weights to bad regions they may never recover from.
Warming up gives the optimizer a few hundred steps to "feel out" the landscape
at small step sizes before committing to the full learning rate.

**Visual:**
```
step:  0   100   200   300   400   500    600    1000   3750
lr:    0  2e-5  4e-5  6e-5  8e-5  1e-4   9.9e-5  9.7e-5   7e-5
        ←─────warmup──────→←──────── cosine decay ──────────────→
```

**For 60K dataset (3750 steps/epoch):**
500 warmup steps = 13% of epoch 1. This is a reasonable fraction — you warm up for
the first 1/7th of the first epoch, then decay.

**How to set warmup for different dataset sizes:**
```python
STEPS_PER_EPOCH   = math.ceil(N_TRAIN / EFFECTIVE_BATCH)
WARMUP_FRACTION   = 0.13   # warm up for ~13% of epoch 1
lr_warmup_steps   = int(STEPS_PER_EPOCH * WARMUP_FRACTION)
# 60K: 3750 * 0.13 = 487 ≈ 500
# 10K: 625  * 0.13 = 81  → use 100
# 200K: 12500 * 0.13 = 1625 → use 1500
```

---

### 2.5 `epochs` — how many times you see the whole dataset

```yaml
epochs: 5
```

**What it is:** one epoch = one complete pass through all 60,000 training images.
5 epochs = 18,750 total optimizer steps for our 60K dataset.

**What happens at epoch boundaries:**
The training script always runs a validation + saves a checkpoint at the end of every epoch,
regardless of `val_steps`/`ckpt_steps`. This guarantees you have one checkpoint per epoch.

**How many epochs do you need?**
```
Rule of thumb for LoRA fine-tuning on SD 1.5:

Dataset size     | Suggested epochs | Rationale
─────────────────|──────────────────|────────────────────────────────────────────
< 1,000 images   |  20–50           | Small dataset — model needs many passes
1K–10K images    |  10–20           | Medium dataset
10K–100K images  |  3–10            | Large dataset — each image seen enough times
> 100K images    |  1–5             | Very large — even 1 epoch = many updates
```

**For 60K:** 5 epochs gives 18,750 updates with effective batch 16.
Watch val/loss — if it stops decreasing by epoch 3, you can stop early.

**Early stopping:** press Ctrl+C. The signal handler in `depth_training.py` catches it,
finishes the current step, saves a final checkpoint, and exits cleanly.

---

### 2.6 `gradient_accumulation_steps` — virtual batch size without extra memory

```yaml
gradient_accumulation_steps: 4
```

**What it is:** instead of one optimizer update per forward+backward, accumulate
gradients across N micro-batches before updating. Effective batch = `batch_size × grad_accum`.

**The memory trick:** running 4 micro-batches of size 4 uses the SAME peak memory
as one micro-batch of size 4, but the gradients from all 4 are summed before the
optimizer step. The result is mathematically equivalent to batch size 16 — except
you did 4 small forward passes instead of 1 large one.

**Why we use 4 for 60K dataset:**
- Effective batch 16 is large enough for stable gradient estimates
- Effective batch 4 (accum=1) would have noisier gradient estimates → slower convergence
- Effective batch 64+ would be overkill at this dataset size and would require LR scaling

**Multi-GPU:** on a 4-GPU cluster, the effective batch becomes `4 × 4 × 4 = 64`.
In that case you may reduce grad_accum to 1 (each GPU already contributes its batch):
```yaml
# 4-GPU cluster config
gradient_accumulation_steps: 1   # 4 GPUs × batch 4 = 16, same as single-GPU
learning_rate: 2.0e-4            # scale LR by sqrt(4) = 2× for the 4× larger batch
```

---

### 2.7 `data.batch_size` — images per GPU per step

```yaml
data:
  batch_size: 4
```

**What it is:** how many images are processed simultaneously on each GPU in one forward pass.

**Memory usage:** batch_size is the primary VRAM knob.
With gradient_checkpointing and bf16 on a 12 GB GPU:

```
batch_size: 1  →  ~7 GB VRAM  (safe on 8 GB GPU)
batch_size: 2  →  ~9 GB VRAM
batch_size: 4  →  ~11 GB VRAM  (max for 12 GB with checkpointing+bf16)
batch_size: 8  →  ~14 GB VRAM  (requires 16 GB+)
batch_size: 16 →  ~20 GB VRAM  (requires 24 GB+)
```

**If you hit OOM (out of memory):**
```yaml
# Step 1: halve batch size
data:
  batch_size: 2

# Step 2: if still OOM
data:
  batch_size: 1

# NEVER turn off gradient_checkpointing to compensate — that uses more memory, not less
```

To keep the same effective batch (16) when reducing batch_size:
```python
# batch_size=2, want effective_batch=16
gradient_accumulation_steps = 16 // 2 = 8

# batch_size=1, want effective_batch=16
gradient_accumulation_steps = 16 // 1 = 16
```

---

### 2.8 `bf16` — brain-float16 precision

```yaml
bf16: true
```

**What it is:** uses 16-bit brain-floating-point numbers instead of 32-bit floats
for most model tensors during the forward and backward pass.

**Memory impact:** roughly halves VRAM usage for activations and gradients.
On an NVIDIA GPU with Ampere or newer architecture (RTX 3000/4000 series, A100, etc.)
this comes with negligible quality loss because bf16 preserves the exponent range of float32.

**When to set false:**
- Older GPUs (Volta/Turing — RTX 2000 series or older) do not support bf16 efficiently.
  On those, set `bf16: false`. Training will be slower and use more memory.
- You can verify support: `python -c "import torch; print(torch.cuda.is_bf16_supported())"`

**bf16 vs fp16:** bf16 has a wider exponent range than fp16, making it much less likely
to produce NaN/Inf during mixed precision training. Always prefer bf16 when available.

---

### 2.9 `gradient_checkpointing` — trade compute for memory

```yaml
gradient_checkpointing: true
```

**What it is:** during the backward pass, PyTorch normally saves all intermediate
activations from the forward pass so it can compute gradients for each layer.
With gradient checkpointing, only a subset of checkpoints are saved; the rest
are recomputed on-the-fly during backward.

**Memory vs speed trade-off:**
- Saves ~1.5 GB VRAM on 12 GB GPU
- Increases backward pass time by ~20% (the recomputation cost)
- On 24 GB+ GPUs you can turn this off for faster training

**For 12 GB GPU:** always keep `true`. Without it you will OOM.

**Code that enables it** (in `depth_training.py`):
```python
if cfg.get("gradient_checkpointing", False):
    model.unet.enable_gradient_checkpointing()
```

---

### 2.10 `val_steps` and `ckpt_steps` — the two independent save triggers

```yaml
val_steps:  500    # cheap: compute val/loss, maybe update best_model
ckpt_steps: 1000   # heavy: write weights + N monitoring images to disk
```

**These are completely independent.** The code in `depth_training.py` checks:
```python
if global_step % cfg.val_steps == 0:
    do_validation(f"step{global_step}")      # cheap: no file writes

if global_step % cfg.ckpt_steps == 0:
    save_ckpt_and_grid(f"checkpoint-epoch{epoch+1}/step{global_step}")   # heavy: disk I/O
```

**`val_steps` controls:**
- How often val/loss appears in TensorBoard (lower = denser curve)
- How often `best_model/` can be updated (it updates whenever val/loss improves)

**`ckpt_steps` controls:**
- How many weight files are saved to disk (each ~120 MB for LoRA + mapper)
- How many checkpoint-grid image folders are created

**For 60K dataset (3750 steps/epoch):**
```
val_steps=500  → val/loss logged at steps 500, 1000, 1500, 2000, ... (7-8 per epoch)
ckpt_steps=1000 → weights saved at steps 1000, 2000, 3000 (3-4 per epoch)

Total checkpoints in 5 epochs = ~19 step-checkpoints + 5 epoch-end checkpoints + 1 best_model
Disk usage ≈ 25 checkpoints × 120 MB = ~3 GB for weights
Plus grid images: 25 × 10 images × ~200 KB = ~500 MB
Total estimated: ~3.5 GB
```

**How to tune for your needs:**
```yaml
# Save more checkpoints (more disk, more granular comparison):
ckpt_steps: 500

# Save fewer checkpoints (less disk):
ckpt_steps: 2000

# Dense val/loss curve for TensorBoard (slightly slower training):
val_steps: 100

# Sparse val/loss (fastest training, least logging):
val_steps: 1000
```

---

### 2.11 `val_batches` — how much of the validation set to use per check

```yaml
val_batches: 64    # paired with val_batch_size=4 → 64×4 = 256 images per val check
```

**What it is:** the number of batches (not images) to average for each val/loss computation.
With `val_batch_size=4`, this gives `64 × 4 = 256` val images sampled per check.

**Why not use all 4000 val images?**
Running all 4000 would take ~4 seconds per validation check. With `val_steps=500`
and 18,750 total steps = 37 val checks → 37 × 4 seconds = 2.5 minutes wasted on val.
256 images is enough to estimate val/loss reliably (6% of 4000). Statistical noise
is small enough that true improvements are visible.

**For a small dataset (<1000 val images):** lower `val_batches` so you don't exceed
the number of actual val samples:
```python
max_val_batches = len(val_dataset) // val_batch_size
val_batches = min(64, max_val_batches)
```

---

### 2.12 `n_grid_images` — monitoring images per checkpoint

```yaml
n_grid_images: 10   # 5 fixed scenes + 5 fresh scenes per checkpoint
```

**What it is:** each time a checkpoint is saved, the model generates `n_grid_images`
labeled images (ORIGINAL | DEPTH MAP | PREDICTED) and saves them inside that checkpoint's folder.

**The 50/50 fixed/fresh split:**
```python
n_fixed  = n_grid_images // 2    # = 5  — SAME scenes at every checkpoint
n_random = n_grid_images - n_fixed  # = 5  — re-randomized each checkpoint

# Fixed scenes: chosen ONCE at training start with OS entropy (different each run).
# They NEVER change within a run, so checkpoint-step1000 and checkpoint-step10000
# show the SAME 5 scenes — you can directly compare how the model has improved.
# Files: sample_00_fixed.jpg ... sample_04_fixed.jpg

# Fresh scenes: re-drawn each checkpoint from the remaining val pool.
# Files: sample_05_new.jpg ... sample_09_new.jpg
```

**How to choose n_grid_images:**
- 2 : minimal — for smoke tests (saves fastest)
- 10 : good balance — 5 fixed scenes you can watch improve + 5 fresh diversity checks
- 20 : thorough — 10 fixed + 10 fresh; doubles checkpoint generation time
- 40 : for final evaluation runs

**Disk cost per checkpoint:** `n_grid_images × ~200 KB = 10 × 200 KB = 2 MB per checkpoint`

---

### 2.13 `grid_include_empty_prompt` — add a 4th "no-text" panel

```yaml
grid_include_empty_prompt: false   # depth training default
```

**What it is:** when `true`, each monitoring image gets a 4th panel showing what
the model generates with an **empty text prompt** — pure depth conditioning, no text influence.

```
false (default):
  [ORIGINAL | DEPTH MAP | PREDICTED WITH PROMPT]    3 panels

true:
  [ORIGINAL | DEPTH MAP | PREDICTED WITH PROMPT | RAW DEPTH GEN]    4 panels
```

**When to enable `true`:**
You want to know: "is the model actually following the depth map, or is it mostly
following the text prompt?"
If the 4th panel (no prompt) looks structurally similar to the 3rd panel (with prompt),
the depth conditioning is working — the model generates similar spatial structure
regardless of whether text is provided.

**Performance cost:** doubles the image generation time at each checkpoint
(one extra model.sample() call per image). Keep `false` for normal training.

---

### 2.14 `seed` — reproducibility

```yaml
seed: 42
```

**What it is:** random seed for PyTorch's RNG during training (noise sampling, timestep
sampling, dropout). Setting a fixed seed makes the training loss curve reproducible if
you run the same config twice.

**What it does NOT control:**
- The fixed monitoring scene selection (that uses OS entropy — different each run)
- DataLoader shuffling order (handled by a separate seed in PyTorch DataLoader)

**When to change seed:**
Run 3 experiments with seed=42, seed=7, seed=123 and compare val/loss curves.
If results are consistent across seeds, your training is stable. If they differ greatly,
the training is sensitive to initialization and you may need more data or lower LR.

---

### 2.15 `prompt` — per-image captions vs a single fixed prompt

```yaml
prompt: null    # null = each image uses its own "prompt" field from the JSONL
# prompt: "a photo"  # override ALL prompts with this single string
```

**What it is:** by default (`null`), every training image uses the text caption stored
in its JSONL entry (`"prompt"` key). If you set `prompt: "a photo"`, EVERY image uses
that exact string regardless of what is in the JSONL.

**When to use a fixed prompt:**
- Your dataset has no meaningful captions (all prompts would be "a photo" anyway)
- You are doing concept-specific fine-tuning where all images share one concept
- You want to isolate the depth conditioning completely from text conditioning

**When to keep null:**
- You have diverse images with rich, accurate captions
- You want the model to learn the interaction between depth structure AND text description
- Your captions are auto-generated (e.g., from BLIP or LLaVA) and vary per image

---

### 2.16 `data.image_root` — images on a different drive or machine

```yaml
data:
  image_root: null    # null = repo root is used to resolve relative paths
```

**What it is:** the base path prepended to every relative `raw_image_path` in your
training JSONL. When null, paths like `data/raw/000417/raw_image.jpg` are resolved
relative to the repo root (where you cloned LoRAdapter).

**When to set this:**
You generated the depth JSONL on Windows (paths like `data/raw/...`) but your final
training runs on Ubuntu where images are mounted at `/mnt/dataset/`. The JSONL still
has the original relative paths. Setting `image_root=/mnt/dataset` makes the dataset
class resolve `data/raw/000417/raw_image.jpg` as `/mnt/dataset/data/raw/000417/raw_image.jpg`.

```yaml
# Training on Ubuntu with images at /mnt/dataset/
data:
  image_root: /mnt/dataset
  json_file: data/depth_training/train.jsonl     # JSONL stays in the repo
  val_json_file: data/depth_training/val.jsonl
```

---

### 2.17 `lora.struct.ckpt_path` — resume from a checkpoint

```yaml
lora:
  struct:
    ckpt_path: null   # null = start fresh
```

**What it is:** path to a checkpoint folder to load before training begins.
Structure expected:
```
<ckpt_path>/
  struct/
    lora-checkpoint.pt    ← LoRA adapter weights
    mapper-checkpoint.pt  ← mapper network weights
```

**How to resume:**
```yaml
# In configs/experiment/train_depth.yaml, or override on CLI:
lora:
  struct:
    ckpt_path: outputs/train/depth/runs/2026-07-01/00-41-13/checkpoint-epoch2/step7500
```

Or via CLI without editing the file:
```powershell
python depth_training.py experiment=train_depth \
  "lora.struct.ckpt_path=outputs/train/depth/runs/2026-07-01/00-41-13/checkpoint-epoch2/step7500"
```

**Important:** when resuming, all other hyperparameters apply from the config as if training
started fresh — steps count from 0 again and val/loss history resets. The LR scheduler
restarts from step 0. This means the warmup happens again, which is actually fine for
resuming after an interruption.

---

### 2.18 `local_files_only` — offline vs online model loading

```yaml
local_files_only: true
```

**What it is:** when `true`, all models (SD 1.5, DPT-MiDaS) must already exist in
`checkpoints/local_models/`. No internet access is attempted. The code also sets
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` environment variables.

**Set to `false` only when:** downloading models for the first time (run `download_sd15.py`
instead — it handles this properly and saves to the right folder names).

---

### 2.19 `ignore_check` — skip startup file verification

```yaml
ignore_check: true
```

**What it is:** on startup, `depth_training.py` can verify that every entry in the
training JSONL has its `depth_path` PNG actually present on disk. This takes ~30 seconds
for 60K entries. `ignore_check: true` skips this check.

**When to set `false`:** if you suspect stale JSONL entries pointing to missing depth PNGs,
run with `false` once. It will print every missing file and count failures.

---

## 3. Recommended configs for different GPU sizes

### 3.1 8 GB GPU (minimum)

```yaml
# Save into configs/experiment/train_depth_8gb.yaml
# Then run: python depth_training.py experiment=train_depth_8gb

gradient_checkpointing: true
gradient_accumulation_steps: 8    # effective batch 1×8 = 8
size: 512
data:
  batch_size: 1
bf16: true
learning_rate: 7.07e-5            # 1e-4 × sqrt(8/16) ≈ 7e-5 (smaller batch → scale down LR)
lr_warmup_steps: 500
epochs: 5
val_steps: 500
ckpt_steps: 1000
n_grid_images: 4
```

Wall clock for 60K/5 epochs: ~2–3× slower than 12 GB (more grad accum steps).

---

### 3.2 12 GB GPU (default config — this is what train_depth.yaml already has)

```yaml
gradient_checkpointing: true
gradient_accumulation_steps: 4   # effective batch 4×4 = 16
size: 512
data:
  batch_size: 4
bf16: true
learning_rate: 1.0e-4
lr_warmup_steps: 500
epochs: 5
val_steps: 500
ckpt_steps: 1000
n_grid_images: 10
```

Estimated time for 60K/5 epochs: ~8–12 hours on a single RTX 3080/3090/4080.

---

### 3.3 24 GB GPU (RTX 3090/4090, A10)

```yaml
gradient_checkpointing: false     # off — you have the VRAM
gradient_accumulation_steps: 2   # effective batch 8×2 = 16 (keep same effective batch)
size: 512
data:
  batch_size: 8                  # 2× bigger batch → ~2× faster per epoch
bf16: true
learning_rate: 1.0e-4            # effective batch unchanged (same LR)
lr_warmup_steps: 500
epochs: 5
val_steps: 500
ckpt_steps: 1000
n_grid_images: 10
```

Estimated time for 60K/5 epochs: ~4–6 hours.

---

### 3.4 4× GPU cluster (e.g., 4× A100 80 GB)

```yaml
gradient_checkpointing: false
gradient_accumulation_steps: 1   # 4 GPUs × batch 4 = 16 effective batch
size: 512
data:
  batch_size: 4                  # per-GPU batch
bf16: true
learning_rate: 2.0e-4            # scale by sqrt(4) = 2 (4× more GPUs, same LR-to-batch ratio)
lr_warmup_steps: 150             # shorter warmup for higher LR
epochs: 5
val_steps: 100                   # steps/epoch drops to 938 → denser val logging
ckpt_steps: 250
n_grid_images: 10
```

Launch command:
```bash
accelerate launch --num_processes=4 depth_training.py experiment=train_depth
```

Estimated time for 60K/5 epochs: ~1–2 hours.

---

## 4. Reading TensorBoard — what to look for

```powershell
# Launch TensorBoard for your depth run:
tensorboard --logdir outputs/train/depth/runs/

# Then open http://localhost:6006 in your browser
```

### 4.1 `train/loss` — the training loss per step

**What it should look like:**
```
Step:     0    500   1000   2000   5000   10000   18750
Loss:  0.15   0.14   0.13   0.12   0.11    0.10    0.09
                                                 (ideal smooth descent)
```

**Noisy is normal:** each step uses a different random noise level (timestep t) sampled
from 1–1000. A step with t=900 (heavy noise) has a much larger loss than t=100 (light noise).
This is expected — look at the smoothed curve (TensorBoard's smoothing slider), not raw values.

**Red flags:**
- Loss suddenly jumps to 0.25 or higher → LR is too large, reduce by 5×
- Loss is flat for 2000+ steps → LR too small, or model is stuck
- Loss goes to NaN → almost always a numerical issue; check for corrupt images, try `bf16: false`

### 4.2 `val/loss` — the true quality signal

**What it should look like:**
```
Step:    500   1000   2000   5000   10000   18750
Val:    0.147  0.146  0.144  0.141   0.138   0.135
```

**Key insight:** val/loss tells you how well the model generalizes to images it has NEVER
seen during training. If `train/loss` keeps going down but `val/loss` plateaus or rises,
the model is memorizing training data (overfitting) — stop training.

**When to stop early:**
```python
# Rule of thumb: stop if val/loss has not improved for:
# - Small dataset (< 10K): 3–5 epochs
# - Large dataset (60K+):  2–3 epochs of stagnation
```

### 4.3 `train/grad_norm` — gradient health check

**What it should look like:**
```
Early training:  ~0.002  (small because LoRA starts near zero)
Mid training:    ~0.002  (steady)
Late training:   ~0.001  (decreasing as the model converges)
```

**Red flags:**
- `grad_norm` repeatedly hitting 1.0 → the gradient clipper is firing constantly → LR too high
- `grad_norm` spikes to 1.0 then back down → occasional instability, monitor closely
- `grad_norm` near 0 throughout → LR too small or model is not learning

### 4.4 `train/lr` — confirming the schedule

The LR curve should match your `lr_scheduler` exactly:
```
cosine: ramp 0→1e-4 (warmup), then smooth decay 1e-4→0
constant: ramp 0→1e-4 (warmup), then flat
```

---

## 5. Reading the checkpoint monitoring grid

After every `ckpt_steps` steps, the model generates N images inside that checkpoint's folder:
```
outputs/train/depth/runs/2026-07-01/00-41-13/
  checkpoint-epoch1/
    step1000/
      sample_00_fixed.jpg   ← ORIGINAL | DEPTH MAP | PREDICTED
      sample_01_fixed.jpg
      ...
      sample_05_new.jpg
      prompts.txt            ← the text prompts used
```

### What to look for in `ORIGINAL | DEPTH MAP | PREDICTED`:

**Step 500–1000 (early training):**
- PREDICTED should look somewhat like a plausible image but with wrong colours/textures
- Depth map structure (near/far) may not yet be respected
- This is NORMAL — the model is just starting to learn

**Step 3000–5000 (mid training):**
- PREDICTED should start following the coarse depth structure
- Foreground objects (close = bright depth) should appear in front of background

**Step 10000+ (late training):**
- PREDICTED should closely match the spatial layout of ORIGINAL
- Edges between near/far regions should be sharp
- Fine details (textures, small objects) may still differ — that is expected

**Signs of good conditioning:**
The same 5 FIXED scenes at step 1000 vs step 10000 should show visible improvement.
Compare `sample_00_fixed.jpg` from two different checkpoint folders side by side.

---

## 6. Common problems and fixes

| Problem | Symptom | Fix |
|---|---|---|
| Out of memory (OOM) | CUDA OOM error on startup | Reduce `data.batch_size` from 4→2→1 |
| Loss diverges | train/loss spikes to 0.3+ | Reduce `learning_rate` by 5× (e.g., 2e-5) |
| Loss not moving | train/loss flat for 1000+ steps | Increase `learning_rate` by 5×, or check data loading |
| val/loss rising | val/loss up while train/loss down | Overfitting — stop training or reduce epochs |
| NaN loss | `loss = nan` in terminal | Try `bf16: false`; check for corrupt images |
| Windows multiprocessing crash | DataLoader crash on startup | Set `data.workers: 0` |
| No checkpoint generated | No `checkpoint-epoch1/` folder | Check `ckpt_steps` is < steps_per_epoch |
| Grid images are black | PREDICTED is all black | LR too high — model collapsed; reduce 10× and retrain |
| JSONL key not found | KeyError on startup | Check `SOURCE_KEY` in `depth_map_calculations.py` |
| Images not found | FileNotFoundError in dataloader | Set `data.image_root` to the correct base path |

---

## 7. Quick reference — config values for 60K / 4K / 4K

```yaml
# Copy-paste ready config for 60,000 train / 4,000 val / 4,000 test
# Single 12 GB GPU (effective batch = 16)
# Run: python depth_training.py experiment=train_depth

size: 512
learning_rate: 1.0e-4
lr_scheduler: cosine
lr_warmup_steps: 500               # 13% of first epoch (step 500/3750)
epochs: 5                          # 5 × 3750 = 18750 total optimizer steps
gradient_checkpointing: true       # required for 12 GB
gradient_accumulation_steps: 4     # 4 bsz × 4 accum = 16 effective batch
data:
  batch_size: 4                    # max safe for 12 GB with checkpointing + bf16
  val_batch_size: 4
  workers: 4                       # set 0 on Windows if multiprocessing crashes
val_steps: 500                     # 7-8 val/loss checks per epoch
ckpt_steps: 1000                   # 3-4 checkpoints per epoch
val_batches: 64                    # 64 × 4 = 256 val images per check (6% of 4K val set)
n_grid_images: 10                  # 5 fixed + 5 fresh scenes per checkpoint
grid_include_empty_prompt: false   # enable for diagnostic runs only
bf16: true
seed: 42
prompt: null                       # use per-image captions from JSONL
local_files_only: true
ignore_check: true                 # set false if you suspect corrupt JSONL

# Adjust for your GPU:
# 8 GB:  batch_size=1, gradient_accumulation_steps=16, learning_rate=7e-5
# 24 GB: batch_size=8, gradient_accumulation_steps=2, gradient_checkpointing=false
# 4×GPU: batch_size=4, gradient_accumulation_steps=1, learning_rate=2e-4 (sqrt(4)×)
```

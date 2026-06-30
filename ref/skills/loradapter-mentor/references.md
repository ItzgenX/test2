# LoRAdapter Mentor — Pinned References

Source of truth. Claude reads this before answering. Facts are either
VERIFIED (read from the actual repo source / official docs) or OPEN QUESTION
(must be confirmed from the local code — never guessed).

================================================================
## 1. Project identity
- Repo: https://github.com/CompVis/LoRAdapter
- Paper: "CTRLorALTer: Conditional LoRAdapter for Efficient 0-Shot Control &
  Altering of T2I Models", arXiv:2405.07913 (ECCV 2024).
- Core idea: a conditional LoRA block unifying style + structure conditioning
  for zero-shot control of T2I diffusion models. Depth and segmentation are
  both STRUCTURE conditioning.
- Env (verified from environment.yaml): python 3.11, pytorch 2.1.2,
  pytorch-cuda 12.1, diffusers==0.25.0, accelerate, transformers, tensorboard,
  hydra-core, einops, open-clip-torch, torch-fidelity, basicsr. Env name:
  loradapter.

================================================================
## 2. VERIFIED repo mechanics (read from source on 2025; re-confirm on disk)

### train.py (verified)
- Hydra entry: @hydra.main(config_path="configs", config_name="train").
- Uses Accelerate, mixed_precision="bf16", gradient_accumulation.
- Per training step:
    imgs = batch["jpg"]; imgs = imgs.clip(-1,1)
    cs = [imgs] * n_loras          # conditioning input = RAW image, reused
    prompts = [cfg.prompt]*B if cfg.prompt is not None else batch["caption"]
    model_pred, loss, x0, _ = model.forward_easy(imgs, prompts, cs,
                                  cfg_mask=[True...], batch=batch)
- Validation block runs every cfg.val_steps; saves a checkpoint each time via
  save_checkpoint(); logs validation IMAGES to tensorboard already (a
  condition image row + a prediction row are concatenated). THIS is the block
  to EXTEND for the user's 4-image grid (do not rewrite it).
- Checkpoint cadence: tied to val_steps (the loop does validation+checkpoint
  together). NOTE the struct config also has ckpt_steps — confirm interaction.

### src/data/local.py (verified)
- ImageFolderDataset.__getitem__ returns ONLY {"jpg": image, "caption": label}.
  NO depth/seg field exists in the stock dataset.
- Caption source: if caption_from_name -> derived from filename stem
  (prefix + stem.split("_")[0] with '-'->' '); else read sibling <stem>.txt;
  else "".
- ZipDataset zips several ImageFolderDatasets (must be equal length). With one
  dataset it returns the single dict; with several it returns a tuple.
- ImageDataModule builds train/val dataloaders from `directories` /
  `val_directories` relative to project root.
  -> To train on SAVED maps, the user must extend this so each sample also
     carries its precomputed map (e.g. a parallel folder or a key in the dict),
     and the forward path must consume it instead of encoder(img).

### configs/experiment/train_struct_sd15.yaml (verified)
    defaults:
      - /lora@lora.struct: struct
      - override /lora/encoder@lora.struct.encoder: midas   # <- live MiDaS encoder
      - override /model: sd15
      - override /data: local
    data: { batch_size: 8, caption_from_name: true,
            caption_prefix: "a picture of ", directories: [data] }
    lora.struct.optimize: true
    size: 512            # training/sample size is 512x512
    log_c: true          # log the conditioning ("c") image in validation
    val_batches: 4
    learning_rate: 1e-4
    ckpt_steps: 3000
    val_steps: 3000
    epochs: 10
    prompt: null         # null => use dataset captions

### src/model.py (verified)
- ModelBase holds encoders[], mappers[], dps[] (DataProviders), lora_transforms.
- SD15.get_input: VAE-encode imgs to latents (scaled by vae scaling_factor);
  PROMPT DROPOUT: each prompt replaced by "" with prob c_dropout (default 0.05)
  for classifier-free guidance; tokenize + text_encoder -> prompt_embeds.
- SD15.forward (THE KEY PATH): add noise to latents; then for each
  (encoder, dp, mapper, lora_c) in zip(...):
      lora_c dropout (zeroed with prob c_dropout)
      if skip_encode: cond = lora_c        # <-- feeds the conditioning AS-IS
      else:           cond = encoder(lora_c)  # <-- LIVE MiDaS here (stock path)
      mapped_cond = mapper(cond); dp.set_batch(mapped_cond)
  Then UNet predicts noise; loss = MSE(model_pred, target) with target=noise
  for epsilon prediction (or velocity for v_prediction).
  -> `skip_encode=True` is the hook the user can use to feed a PRE-SAVED map
     directly (bypassing the live encoder). CONFIRM exact wiring on disk.
- Sampling (sample_easy / sample_custom): builds CFG by concatenating a neg
  (zeros) condition and the real condition; classifier-free guidance with
  guidance_scale; 50 inference steps; encoder(c) runs LIVE at inference.

### src/utils.py (verified)
- DataProvider: a tiny holder; set_batch/get_batch pass the mapped conditioning
  into the LoRA layers. Asserts shapes don't change across steps.
- add_lora_from_config: builds each LoRA from config; can load checkpoints
  (lora-checkpoint.pt, mapper-checkpoint.pt, optional encoder-checkpoint.pt).
- save_checkpoint: writes per-LoRA folders with lora/mapper(/encoder) .pt files.
- In add_lora_to_unet (model.py): the frozen original weight is `W`; only non-W
  params are saved -> that's why LoRA checkpoints are small.

================================================================
## 3. THE USER'S ARCHITECTURE DECISION (respect this)
- TRAIN on PRE-SAVED maps; INFER with LIVE map computation.
- HARD CONSTRAINT: saved-for-training maps and live-at-inference maps MUST come
  from the SAME model + SAME preprocessing + SAME value range. Otherwise
  train/test mismatch wrecks results.
- Mechanism to feed saved maps in training: extend local.py to load the map +
  use the `skip_encode=True` path (or equivalent) in forward so the saved map
  is used directly instead of encoder(img). CONFIRM exact wiring by reading the
  local code before implementing.

================================================================
## 4. OPEN QUESTIONS — resolve by READING LOCAL CODE, never guess
  Q-MIDAS [RESOLVED] What EXACT MiDaS variant + resize + value range does the
    repo's `midas` encoder use?
    VERIFIED ANSWER (read from configs/lora/encoder/midas.yaml,
    src/annotators/midas.py, src/annotators/util.py):
      • Checkpoint: Intel/dpt-hybrid-midas (DPTForDepthEstimation from HF).
        DPTImageProcessor is imported but fully commented out — NOT used.
      • Internal model_size = 384 (hardcoded in DepthEstimator.__init__).
      • Transform pipeline inside forward():
          1. Input must be [B,3,H,W] in [-1,1]  (asserted)
          2. (imgs + 1.0) / 2.0  →  [0,1]
          3. better_resize(imgs, 384):
               side   = min(H, W)                 # square-crop dimension
               factor = side // 384
               center_crop(imgs, [side, side])     # crop to square
               if factor > 1: avg_pool2d(factor)   # anti-alias
               interpolate([384,384], bilinear)    # final model input size
          4. DPT model receives 384×384 → predicted_depth (arbitrary units)
          5. F.interpolate(depth, size=(self.size, self.size),
                           mode="bicubic") → (512,512)  [self.size from config]
          6. Per-image min-max: (d - min) / (max - min + 1e-6)  → [0,1]
          7. torch.cat([depth]*3, dim=1) → [B,3,512,512] identical channels
      • Output: [B,3,512,512] in [0.0, 1.0], per-image normalised.
      • CRITICAL: better_resize ALWAYS center-crops to square first,
        unconditionally, inside the encoder, regardless of what's fed to it.
        See section 5 for what this means for the letterbox/stretch decision.
        A 512×512 input hits factor=1 → crop is a no-op, no avg_pool, just
        bilinear 512→384. A raw 1280×800 input hits center_crop(800×800) →
        cutting the left/right edges → then factor=2 avg_pool → bilinear
        400→384. These two paths differ, which is exactly why what you feed
        the encoder matters.
  Q-SKIPENC How exactly is skip_encode threaded from train.py -> forward()? Is
    there an existing config flag, or must one be added to feed saved maps?
    VERIFIED ANSWER (read 2026-06-26 from src/model.py):
      • skip_encode is a plain Python bool parameter — NOT a config key.
      • SD15.forward() line 415: skip_encode: bool = False.
        Line 460: if skip_encode: cond = lora_c  else: cond = encoder(lora_c).
      • SD15.forward_easy() line 511: accepts skip_encode, passes to self().
      • SD15.sample_custom() line 548: skip_encode: bool = False.
        Line 587: if skip_encode: cond = c  else: cond = encoder(c).
      • SD15.sample_easy() (called by model.sample()): NO skip_encode param.
        Always calls encoder(c). This is the correct live-inference path.
      • Usage: train_depth.py forward_easy(skip_encode=True) for training;
        sample_custom(skip_encode=True) for validation grids.
        inference_depth.py uses model.sample() → sample_easy() → encoder(c)
        — live encoder, no skip needed.
  Q-SEGENC For Stage C/D: there is NO stock live-seg encoder. To mirror depth's
    "live at inference" behaviour, a SegFormer encoder module must be written to
    plug into the same encoder slot. Confirm the encoder interface (input range,
    output shape) by reading the midas encoder + how forward() calls encoder().
    VERIFIED ANSWER (resolved 2026-06-29):
      • ENCODER SLOT CONTRACT (read from src/model.py SD15.forward line 460):
        INPUT  : [B, 3, H, W] float in [-1, 1]  (same as midas)
        OUTPUT : [B, 3, size, size] float in [0, 1]  (mapper is Conv2d(3,...))
      • ENCODER CLASS: SegmentationEncoder (src/encoders/seg_encoder.py).
        Wraps SegformerForSemanticSegmentation; frozen + eval, never trained.
        Registered as an nn.Module so accelerate's .prepare()/.to()/.eval()
        treat it identically to DepthEstimator.
      • TWO ENTRY POINTS sharing _predict_ids():
        - forward(imgs): [-1,1] -> colour seg map [B,3,size,size] in [0,1]
          (what model.py calls as encoder(c) at live inference)
        - label_ids(imgs): [-1,1] -> raw class IDs [B,size,size] long
          (used ONLY by calculate_segmentation_map.py to save ID PNGs offline)
      • CONDITIONING FORMAT: COLOUR PALETTE, not raw class IDs. 19-class
        Cityscapes palette (SEG_CITYSCAPES_PALETTE constant, SSOT in
        seg_encoder.py). Rationale: raw id/18 imposes false ordinal ordering
        on categorical labels; palette gives every class a distinct well-separated
        RGB identity — same approach as ControlNet-Seg.
      • LOCKED MODEL: nvidia/segformer-b5-finetuned-cityscapes-1024-1024
        (b5 chosen for best seg map accuracy; see §9 for rationale).
      • PARITY GUARANTEE: calculate_segmentation_map.py uses encoder.label_ids()
        (shares _predict_ids with forward()); dataset colourises with the SAME
        palette; inference calls forward() — all three produce identical maps.
      • HF NORMALIZATION: done manually (ImageNet mean/std as [1,3,1,1] buffers)
        rather than via SegformerImageProcessor to avoid offline config.json
        dependency. Matches SegformerImageProcessor to 2.4e-7 (bit-exact).
  Q-CKPT How do val_steps and ckpt_steps interact in train.py? How many
    checkpoints + images result from a given config? VERIFIED ANSWER: ________

================================================================
## 5. IMAGE SIZE DECISION — LOCKED (user images 1280x800; repo trains 512x512)
Paper §5.1 (verified): COYO subset, short side >= 512, center-cropped to 512.
This is the PAPER'S data-pipeline choice, not an architecture requirement — the
model needs a 512x512 TENSOR; how you get there is free preprocessing choice.
The user's images are wide driver's-eye-view (1280x800, ratio 1.6:1) where
crop would cut off adjacent-lane/pedestrian content at the frame edges, so
CROP IS EXCLUDED for this project (decided twice, same reason: no content loss
is acceptable for driving scenes).

TERMINOLOGY FIX (resolved confusion): `transforms.Resize((size, size))` with a
TUPLE forces both dimensions independently = THIS IS STRETCH. There is no
separate third "resize" option — "resize" and "stretch" were the same
operation in the user's existing code. The real choice has only ever been TWO:
  - stretch: transforms.Resize((512,512)) directly — what the user's code
    already does. Keeps all content, distorts aspect ratio (shapes/angles).
  - letterbox: pad shorter dimension to square (no distortion), THEN resize to
    512x512 (e.g. transforms.Resize(512) on the padded square, or direct
    since it's already square). Keeps all content AND shape, costs some
    resolution to padding bars.

CRITICAL FINDING from Q-MIDAS verification (changes the stakes of this
decision): the repo's live `midas` encoder applies `better_resize()`
INTERNALLY and UNCONDITIONALLY — it does `side = min(H,W); center_crop to
[side,side]` on WHATEVER is fed to it, every single call, with no way to turn
it off from outside. This means:
  - If you feed the RAW 1280x800 image straight to the live encoder at
    inference, it will SELF-crop to 800x800 internally — silently cutting the
    left/right edges, the exact thing the user ruled out twice.
  - If you feed it an ALREADY-SQUARE image instead (via letterbox OR stretch),
    `side = min(H,W)` equals the full size, so `center_crop` is a geometric
    no-op — nothing is cut, because there's nothing non-square left to crop.
  -> CONCLUSION: "always feed the encoder a pre-made square" is now a HARD
     CORRECTNESS RULE, not a style preference — otherwise the encoder's
     internal crop will silently discard frame edges regardless of any
     upstream preprocessing decision. This applies EQUALLY at training-data
     -generation time (so saved depth maps match what live inference would
     see) and at live-inference time (square the image immediately before it
     reaches the encoder, every time).
  - HONEST NOTE: stretch ALSO produces a square, so it ALSO satisfies this
    hard rule, just via distortion instead of padding. Both remaining options
    (stretch, letterbox) independently avoid the encoder's forced crop; this
    finding does not by itself re-decide stretch vs letterbox — that remains
    the ORIGINAL tradeoff (silent permanent distortion vs visible fixable
    seam artifacts). Letterbox stays the locked default for that reason.
  - Note on avg_pool: only triggers when the square side is a multiple of 384
    with factor>1 (e.g. raw 800x800 -> factor=2). If you square to exactly
    512x512 before the encoder, factor = 512//384 = 1, so NO avg_pool runs —
    straight bilinear 512->384. Both offline and live paths should square to
    the SAME size (512x512) before hitting the encoder, so factor=1 in both,
    for the simplest possible parity.

DECISION (locked): resize_mode = "letterbox" (DEFAULT), with "stretch"
available as a config-selectable alternative. NOT hardcoded — implemented as
a YAML-controlled toggle so depth and segmentation pipelines can be switched
together and compared if needed. Example config key:
    resize_mode: letterbox   # letterbox (default) | stretch   [crop excluded]
IMPORTANT: squaring (letterbox or stretch) must happen BEFORE the image
reaches the midas encoder, both in the offline calculate_depth_map.py AND in
any live-inference sampling code, to exactly 512x512 (so factor=1, no
avg_pool — matching behaviour in both paths). This is now a correctness
requirement, not just a preference.

RATIONALE for letterbox as default (engineering-risk based, not aesthetic):
- Stretch's failure mode (geometric distortion of every shape/angle) is
  SILENT and PERMANENT — it looks like normal output and gets baked into
  every training example without being visually obvious.
- Letterbox's failure mode (model confusion at the padding seam) is VISIBLE
  and LOCALIZED — checkable by eye in the first depth/seg map produced, and
  fixable (mask pad region from loss, use edge-padding instead of solid
  black, etc.).
- Since depth is the FOUNDATION of training (per user), prefer the failure
  mode that can be caught early over the one that silently corrupts results.
- Both letterbox and stretch must be applied IDENTICALLY to: raw image, depth
  map, and (later) segmentation map, so all three stay pixel-aligned. This is
  non-negotiable regardless of which mode is active.

ACTION ITEMS for implementation (apply to BOTH depth and seg pipelines):
- `size` (currently 512, read from YAML) is the FINAL square output for BOTH
  modes — this does not change. What differs is the step BEFORE the final
  resize:
    stretch:   transforms.Resize((size, size)) directly on the raw image
               (current code — one step).
    letterbox: (1) pad the shorter dimension of the raw image to make it
               square (for 1280x800, pad height since width is longer), using
               edge/reflect padding not solid black/zero (see below), THEN
               (2) transforms.Resize((size, size)) on the now-square padded
               image (same Resize call, just fed a pre-padded square instead
               of the raw rectangle).
  `size` is read from YAML either way; letterbox just adds a pre-step.
- RECORD PADDING AS A FRACTION, not raw pixels: compute pad amount relative to
  the ORIGINAL image dimensions (e.g. "12.5% of height was padding") BEFORE
  the final resize to `size`. Recording raw pixel counts is a bug risk — those
  numbers become wrong once the image has been shrunk to 512x512. Storing a
  fraction lets you correctly reconstruct/crop the padding off a generated
  output of any later resolution.
- Add `resize_mode` to the experiment YAML (configs/experiment/*.yaml), not
  hardcoded in the script.
- Letterbox implementation: use edge-padding or reflect-padding rather than
  solid black/zero, to reduce the chance a model reads the pad boundary as a
  real depth/seg edge — VERIFY this choice once the first real depth map is
  inspected.
- Stretch implementation: direct Resize((size,size)) as already coded — keep
  as the alternate path, not deleted.
- After implementing, the user should VISUALLY INSPECT the first few depth
  maps produced under letterbox mode specifically for seam artifacts at the
  padding boundary. This is the real verification step — not theoretical.
- This same resize_mode toggle must be threaded through calculate_seg_map.py
  (Stage C) identically, once that stage is built.
- LIVE-INFERENCE REMINDER: whatever squaring is chosen must ALSO be applied to
  the raw image immediately before it is passed to the live midas encoder at
  sampling time — sample.py/sample_two.py currently feed images straight to
  encoder() with no squaring step. This needs to be added there too, or
  inference will silently hit the encoder's internal center-crop.

================================================================
## 6. Canonical project structure (VERIFIED ON DISK 2026-06-29 — final locations)
```
# REPO ROOT (all entrypoints are at the repo root, not in a preprocessing/ subfolder)
pre_depth_calculations.py      # STAGE A: image -> MiDaS depth -> data/raw_depth/ + data/depth_training/{train,val,test}.json
calculate_segmentation_map.py  # STAGE C: image -> SegFormer   -> data/raw_seg/   + data/seg_training/{train,val,test}.json
train_depth.py                 # STAGE B: train on saved depth maps
train_seg.py                   # STAGE D: train on saved seg maps
inference_depth.py             # STAGE B live inference (live MiDaS at sample time)
inference_seg.py               # STAGE D live inference (live SegFormer at sample time)

src/data/
  local.py        # stock dataset (ImageFolderDataset — used by train.py and style training)
  local_depth.py  # STAGE B dataset: loads (image, depth_map, prompt) from depth_training/*.json
  local_seg.py    # STAGE D dataset: loads (image, seg_map,   prompt) from seg_training/*.json
  transforms.py   # shared transforms: SquarePad, TopCrop, build_seg_square_preprocess()

src/encoders/
  seg_encoder.py  # STAGE C/D: SegmentationEncoder (live SegFormer at inference, label_ids at calc)

src/annotators/
  midas.py        # STAGE A/B: DepthEstimator (live MiDaS at inference; used by train.py stock path)

configs/
  train_depth.yaml       # STAGE B base config (size, lr defaults)
  train_seg.yaml         # STAGE D base config (mirrors train_depth.yaml + seg grid knobs)
  inference_depth.yaml   # STAGE B inference config
  inference_seg.yaml     # STAGE D inference config
  data/
    local_depth.yaml     # depth data module config
    local_seg.yaml       # seg data module config (val_batch_size:4)
  lora/encoder/
    midas.yaml           # encoder config: DepthEstimator, dpt-hybrid-midas
    segformer.yaml       # encoder config: SegmentationEncoder, segformer-b5-cityscapes
  experiment/
    train_depth.yaml     # STAGE B unified experiment config (12GB GPU, cluster options commented)
    train_seg.yaml       # STAGE D unified experiment config (supersedes 12gb+cluster variants)

data/
  train.json  val.json  test.json     # ORIGINAL source manifests (prompt + source image)
  raw/                                 # source RGB images
  raw_depth/                           # STAGE A output: MiDaS depth PNGs (letterboxed, per-image min-max)
  raw_seg/    (created by STAGE C)     # STAGE C output: SegFormer class-ID PNGs (8-bit L, 0..18)
  depth_training/{train,val,test}.json # STAGE B manifests
  seg_training/{train,val,test}.json   # STAGE D manifests (created by STAGE C)

checkpoints/local_models/
  stable-diffusion-v1-5/   dpt-hybrid-midas/   segformer-b5-cityscapes/   taesd/
```
Rules: one concern per file; calculation never holds training logic; depth and
seg never share code paths; training data (manifests, saved maps) under data/;
model checkpoints under checkpoints/; all generated outputs under outputs/.

================================================================
## 7. Pinned external models
### Depth — MiDaS (Stage A) [VERIFIED — see Q-MIDAS above]
- Intel/dpt-hybrid-midas, internal model_size=384, per-image min-max [0,1]
  output, 3-channel replicated, final size from config (512).
- Encoder unconditionally center-crops non-square input — see section 5.

### Segmentation — SegFormer Cityscapes (Stage C/D)
LOCKED MODEL (2026-06-29): nvidia/segformer-b5-finetuned-cityscapes-1024-1024
  b5 (MiT-B5 backbone, 82M params) chosen over b0 (3.7M) for significantly
  better boundary precision on driving-scene classes — pedestrians, vehicles,
  traffic lights — where seg quality is the point of the depth-vs-seg comparison.
  Local path: checkpoints/local_models/segformer-b5-cityscapes
Inference via SegmentationEncoder (src/encoders/seg_encoder.py):
  encoder = SegmentationEncoder(size=512, model=MODEL_OR_PATH)
  colour_map = encoder(imgs_minus1_to_1)   # [B,3,512,512] in [0,1]
  ids        = encoder.label_ids(imgs_...)  # [B,512,512] long (offline calc only)
- Output = COLOUR MAP (SEG_CITYSCAPES_PALETTE, 19 Cityscapes classes).
  DECIDED: colour palette, NOT raw id/18 ramp (decided 2026-06-29, same session
  as encoder design). Rationale in Q-SEGENC above.
- "1024x1024" in the model name = image size the model was trained ON for
  segmentation (input resolution), NOT output generation size. SD 1.5 always
  generates 512x512. The b5 benefit is better quality seg MAPS as conditioning.

================================================================
## 8. Feature requirements (build modular + heavily commented)
- CHECKPOINT MONITORING IMAGES (UPDATED 2026-06-29 per user): when a checkpoint
  is SAVED, that same checkpoint generates N validation images so the user can
  judge it by eye. DECISIONS (locked):
    • N SEPARATE files, NOT one stacked grid (a 10-row grid is too small to read).
      Each file = ONE labeled "explained" image: ORIGINAL | DEPTH MAP | PREDICTED.
      Filenames sample_<NN>_<fixed|new>.jpg.
    • Saved INSIDE the checkpoint's OWN folder (next to its weights), e.g.
      checkpoint-1000/sample_00_fixed.jpg — NOT a separate image_grid/ folder.
      So each model and its images live together.
    • ONE prompts.txt per checkpoint (all N prompts, tagged [fixed]/[new]) — NOT
      one txt per image.
    • DEPTH TRAINING GENERATES WITH THE PROMPT ONLY BY DEFAULT. The
      grid_include_empty_prompt flag is KEPT (user: keep it) but DEFAULTS FALSE,
      so no empty-prompt image unless explicitly turned on. When true, each scene
      image gets a 4th "RAW DEPTH GEN" panel (extra generation with empty prompt).
- YAML-controlled values:
    n_grid_images: int             # validation images per checkpoint (default 10)
    grid_include_empty_prompt: bool # default FALSE; true adds the 4th empty-prompt panel
- N SPLIT 50/50 (user decision): half FIXED scenes (seeded once, reused every
  checkpoint → watch the same scenes improve) + half RE-RANDOMIZED each checkpoint
  (global RNG → fresh generalization peek). prompts.txt tags which is which.
- SOURCE OF THE N IMAGES = VALIDATION set only. NOT train (already seen), NOT
  test (keep as honest final benchmark; don't contaminate). Fixed half via
  random.Random(cfg.seed).sample(...); fresh half via random.sample(pool) where
  pool excludes the fixed half (no duplicates).
- COUPLING: a single helper save_ckpt_and_grid(stem,is_best,info_lines) saves the
  checkpoint AND its images together; called for step / epoch / best_model / final
  checkpoint (the final one previously had NO images). Grid gen is best-effort
  (logged, never blocks the checkpoint save).
- CLI: train_depth.py / inference_depth.py use Hydra-native local_files_only=true
  (NO argv shim — user rejected it). Only the calc script uses --local_files_only.
- Report expected #checkpoints and #images for the chosen config (disk usage).
- Structured, timestamped training logs + general logs (file-based, not just
  stdout). Extend the existing tensorboard logging (loss, lr, val images).
- Tensorboard must show training curves; build on log_with="tensorboard".

================================================================
## 9. Decisions log (append-only — keeps every session consistent)
- [x] Q-MIDAS variant + preprocessing verified: Intel/dpt-hybrid-midas, model_size=384,
       center-crop+bilinear to 384, bicubic upsample to 512, per-image min-max [0,1], 3ch
- [x] Image-size strategy: letterbox (default) / stretch (alt), YAML-toggled via
       resize_mode; crop excluded. Squaring now confirmed REQUIRED before the
       encoder (training AND live inference) to prevent its internal forced crop.
- [x] Q-SKIPENC wiring verified: plain bool param on forward()/forward_easy()/sample_custom();
       sample_easy()/model.sample() has NO skip_encode — always live encoder (correct for inference)
- [x] resize_mode=letterbox IMPLEMENTED (2026-06-26). Files changed:
       src/data/transforms.py — SquarePad rewritten as plain callable, edge-padding,
         stores padding fracs as attribute; dropped v2 Transform inheritance + fill=255.
       configs/data/local_depth.yaml — SquarePad prepended to transform list.
       pre_depth_calculations.py — SquarePad + Resize replacing bare Resize in preprocess.
       inference_depth.py — same preprocess change.
       configs/experiment/train_depth_12gb.yaml — informational comment added.
       configs/experiment/train_depth_cluster.yaml — informational comment added.
       Parity verified 2026-06-26: 4 image sizes (1280x800, 800x600, 512x512, 600x800)
       all PASS — offline and live preprocessing tensors are pixel-identical (max diff=0.0);
       uint8 round-trip error = 0.000000 (well within 1/255 tolerance).
- [x] END-TO-END DEPTH PIPELINE VERIFIED (2026-06-28). calc -> train -> infer
       re-read from source (not memory). Confirmed: letterbox live in all 3
       preprocess sites; MiDaS path unchanged vs Q-MIDAS; skip_encode=True in
       train_depth.py (saved maps), live encoder() at inference (no skip).
       Cross-stage parity EXECUTED on data/raw/000000/raw_image.jpg (1280x800):
       calc-float vs infer-float max abs diff = 0.00000000 (bit-identical);
       reloaded training PNG vs live infer float = 0.00392154 (<= 1/255). PASS.
       Code-quality: added PARITY-CRITICAL drift warnings to the 3 duplicated
       preprocess sites; documented why DepthJsonDataset.depth_transform has no
       SquarePad (PNG already a letterboxed square); added loud KeyError for a
       missing depth_path in __getitem__.
       OPEN (not bugs): (1) preprocess is TRIPLICATED not shared -> drift risk;
       (2) train_depth_12gb.yaml epochs:1 + val_steps/ckpt_steps:100 contradict
       its own doc block (looks like leftover quick-test values); (3) ckpt_steps
       is DEAD everywhere (train_depth.py never reads it; checkpointing is tied
       to val_steps + per-epoch); (4) feature knobs n_grid_images /
       grid_include_empty_prompt from §8 NOT wired (grid is hardcoded 1 image,
       empty-prompt always on); (5) BUG-A1 stem check in _verify is weak when
       all stems are "raw_image" (folder structure + zip lockstep are the real
       guard, and those are correct).
- [x] CHECKPOINT MONITORING GRID implemented (2026-06-29). train_depth.py:
       helpers _build_grid (N-row), _generate_checkpoint_grid, _save_checkpoint_grid;
       nested save_ckpt_and_grid() COUPLES checkpoint + grid in one place, called for
       step / epoch / best_model / final checkpoint (final previously had NO grid).
       Grid = N rows x [ORIGINAL | DEPTH MAP | PREDICTED | RAW DEPTH GEN]; 4th column
       toggled by grid_include_empty_prompt. ROW SELECTION (user decision): N=10 split
       50/50 — half FIXED (seeded once, reused every ckpt -> track improvement), half
       RE-RANDOMIZED each ckpt (global RNG -> generalization peek); .txt tags rows
       [fixed]/[new]. Source = validation set only. YAML knobs added to
       configs/experiment/train_depth.yaml: n_grid_images (10), grid_include_empty_prompt
       (true). Saved to <run>/image_grid/<stem>.jpg + .txt (+ best_model/preview_grid.jpg).
       Verified offline (no model): build+save produce correct 10-row 4-col/3-col grids
       and tagged txt; fixed half constant, new half varies per ckpt, no dup rows.
       ALSO (2026-06-28..29) config refactor: deleted train_depth_12gb.yaml +
       train_depth_cluster.yaml -> single configs/experiment/train_depth.yaml (cluster
       options commented; 4x12GB via `accelerate launch --num_processes=4`). Local-vs-
       internet model loading: YAML lists base_model_name/path + depth_model_name/path;
       local_files_only flag picks (true=local folders under checkpoints/local_models,
       false=HF hub). Pick done in train_depth.py / inference_depth.py main() (no custom
       resolvers, no hf_setup.py). Uniform CLI: all 3 entrypoints accept
       --local_files_only True/False (Hydra ones via an argv shim). Models live in
       checkpoints/local_models/{stable-diffusion-v1-5,dpt-hybrid-midas}. Training reads
       data/depth_training/{train,val}.json.
- [x] TENSORBOARD: tfevents filename no longer leaks hostname/username — gethostname
       overridden with cfg.tag before init_trackers (was ...tfevents.<time>.aditya.<pid>).
       Tags namespaced: train/loss, train/lr, val/sample_grid, val/prompts (dropped the
       duplicate loss/lr that the old val block re-logged).
- [x] SUPERSEDES the two entries above (2026-06-29, later same day, per user):
       MONITORING REDESIGN: N SEPARATE labeled images, NOT one stacked grid (10 rows
       too small to read). Each scene -> its own sample_<NN>_<fixed|new>.jpg (labeled
       ORIGINAL | DEPTH MAP | PREDICTED), saved INSIDE that checkpoint's OWN folder
       (next to weights), NOT image_grid/. ONE prompts.txt per checkpoint (tagged
       [fixed]/[new]). grid_include_empty_prompt KEPT but DEFAULT FALSE (depth
       training generates WITH prompt only; true adds a 4th RAW DEPTH GEN panel).
       Helpers now: _scene_image, _save_checkpoint_images (replaced _build_grid/
       _generate_checkpoint_grid/_save_checkpoint_grid).
       __main__ SIMPLIFIED: argv shim REMOVED from train_depth.py + inference_depth.py
       (user disliked it) -> just main(). Hydra apps use local_files_only=true (native);
       only the calc script uses --local_files_only True. (Models confirmed downloaded to
       checkpoints/local_models/{stable-diffusion-v1-5,dpt-hybrid-midas,segformer-b0-cityscapes,taesd}.)
       NORMAL VALIDATION ADDED (user: "i cannot validate model otherwise"):
       _validation_loss() runs the same denoising loss on held-out val_dataloader
       (no grad, eval, RNG seeded+restored for a comparable curve, reduced across
       GPUs) over cfg.val_batches batches -> logged as val/loss. Flow per checkpoint
       (do_validation): VALIDATE (val/loss) -> SAVE checkpoint + N images -> track
       best_model by LOWEST val/loss (was: training-loss average). val_batches is now
       LIVE. Final checkpoint = snapshot via save_ckpt_and_grid (no extra val).
       Still DEAD in train_depth.py: use_empty_prompt_eval, n_samples,
       save_grid, log_cond, ignore_check.
     • CHECKPOINT FOLDER LAYOUT (user request): grouped per epoch ->
       checkpoint-epoch{N}/step{global_step}/ for step saves, and
       checkpoint-epoch{N}/checkpoint-epoch{N}/ for the epoch-end save (no
       "-end"/"-final" suffixes). Final snapshot reuses the epoch-end folder name.
       best_model/ stays top-level. Each folder = weights + N images + prompts.txt.
     • FIXED-SCENE SELECTION now random PER RUN: _fixed_val_idxs uses
       random.Random() (OS entropy), not random.Random(cfg.seed) — so each run
       picks a different fixed half (logged for reproducibility); still constant
       across that run's checkpoints. The other half re-randomizes each ckpt.
       Generation seed stays cfg.seed (fixed scenes only change as weights change).
     • val_steps / ckpt_steps now DECOUPLED (user chose this): do_validation()
       = validation-only (compute+log val/loss, update best_model) every val_steps;
       save_ckpt_and_grid() (weights+images) fires separately every ckpt_steps.
       End-of-epoch always does BOTH. ckpt_steps is now LIVE (was dead).
- [x] Q-SEGENC encoder interface verified: SegmentationEncoder (see Q-SEGENC answer
      above). Colour palette chosen. LOCKED model b5. Two entry points sharing
      _predict_ids() for train/inference parity. (2026-06-29)
- [x] Q-CKPT val_steps vs ckpt_steps: DECOUPLED — val_steps=val/loss cadence,
      ckpt_steps=disk-save cadence. ~40 steps/epoch (639 train / bs4 / accum4).
- [x] Depth normalization scheme: confirmed = per-image min-max [0,1] (see Q-MIDAS)
- [x] SegFormer model ID: nvidia/segformer-b5-finetuned-cityscapes-1024-1024
      (locked 2026-06-29; was b0 in old backup — updated everywhere)
- [x] Seg conditioning format: COLOUR PALETTE (SEG_CITYSCAPES_PALETTE, 19 classes)
      NOT raw id/18. See Q-SEGENC for rationale. Decided 2026-06-29.
- [ ] Prompt generation method (which captioner): __________
- [x] n_grid_images value: 10 (5 fixed + 5 re-randomized each checkpoint)
- [x] grid_include_empty_prompt: DEFAULT FALSE for both depth and seg training
      (generates WITH prompt only; true adds 4th empty-prompt panel)
- [x] grid image source confirmed = validation set (not train/test): yes
- [x] Base diffusion model: SD1.5 (confirmed from train_struct_sd15.yaml)
- [ ] Live-inference squaring step added to sample.py/sample_two.py before
       encoder() call: __________
- [x] SEG PIPELINE COMPLETE + MOVED (2026-06-29). All files written, verified,
       moved to final project locations, segmentation_backup/ deleted. Final locations
       listed in §6. Verified in-place 2026-06-29 (8-step report). Key facts:
         calculate_segmentation_map.py — local_files_only, b5, out_dir=data/seg_training/,
           parse_bool, run_seg_directory_mode, precompute_segmentation_maps,
           build_segmentation_training_jsons, _verify_segmentation_training_json
         train_seg.py — complete rewrite mirroring train_depth.py: _seg_label_bar,
           _seg_scene_image, _save_checkpoint_segmentation_images,
           _segmentation_validation_loss, do_segmentation_validation,
           save_seg_ckpt_and_grid; decoupled val_steps/ckpt_steps; best_model by
           val/loss; OS-entropy 50/50 fixed/fresh split; TensorBoard hostname override
         inference_seg.py — rewrite mirroring inference_depth.py: _seg_label_bar,
           make_seg_inference_grid; local_files_only branch; build_seg_square_preprocess
         src/encoders/seg_encoder.py — no jaxtyping, SEG_CITYSCAPES_PALETTE,
           seg_palette_tensor, seg_colorize_ids, b5, _seg_mean/_seg_std, local_files_only=True
         src/data/local_seg.py — _seg_resolve, seg_palette, NEAREST resize, val_batch_size=4
         src/data/transforms.py (REAL) — build_seg_square_preprocess() appended
         configs/train_seg.yaml — n_grid_images:10, grid_include_empty_prompt:false, local_files_only:true
         configs/experiment/train_seg.yaml — NEW unified config (supersedes 12gb+cluster)
         configs/lora/encoder/segformer.yaml — b0 -> b5
         configs/data/local_seg.yaml — val_batch_size: 1 -> 4
         configs/inference_seg.yaml — local_files_only, base/seg model name+path pairs
"""
seg_training.py
------------
STAGE D — train the segmentation-conditioned LoRAdapter on PRE-SAVED Cityscapes
colour maps. This is the segmentation twin of depth_training.py.

STRUCTURE: this file is a deliberate line-for-line mirror of depth_training.py, with
exactly three categories of change:
  1. "depth" -> "seg" in names and keys (batch["seg"] not batch["depth"],
     cfg.seg_model_name/path not cfg.depth_model_name/path, panel labels).
  2. Segmentation-specific naming (every function/class has "seg" or "segmentation"
     so the two pipelines are visually distinct when files sit side by side).
  3. Nothing else differs — epochs, val_steps, ckpt_steps, monitoring design,
     val/loss computation, checkpoint layout, TensorBoard tags, model loading logic,
     fixed/fresh split, coupling helper — ALL identical to depth. This is required
     for the depth-vs-seg comparison to be valid (references.md §4).

KEY SEG-SPECIFIC POINTS:
  • batch["seg"] is a [B,3,H,W] colour map in [0,1] produced by SegJsonDataset
    (src/data/local_seg.py) from the saved class-ID PNGs.
  • skip_encode=True bypasses the live SegmentationEncoder during training, exactly
    as depth bypasses MiDaS. The pre-saved map goes straight to the mapper.
  • Monitoring panel labels: "SEG MAP" (was "DEPTH MAP"), "RAW SEG GEN" (was "RAW DEPTH GEN").
  • Model loading: cfg.seg_model_name/seg_model_path (was depth_model_name/path).
"""

import hydra
import math
import os
import random
import signal
import traceback
from functools import reduce
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
from hydra.utils import get_original_cwd
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from src.model import ModelBase
from src.utils import add_lora_from_config, save_checkpoint


torch.set_float32_matmul_precision("high")

stop_training = False


def signal_handler(sig, frame):
    global stop_training
    stop_training = True
    print("got stop signal")


# ── Checkpoint-monitoring images (references.md §8) ───────────────────────────
# Every time a checkpoint is saved, that SAME checkpoint generates N validation
# images so you can judge it by eye. Each scene is saved as its OWN labeled file
# (a 3-panel "explained" image: ORIGINAL | SEG MAP | PREDICTED) INSIDE the
# checkpoint's own folder, next to its weights. One prompts.txt lists all N
# prompts. The same structure as depth — only the panel label "DEPTH MAP" ->
# "SEG MAP" changes.

def _seg_label_bar(width: int, text: str, bar_h: int = 24) -> np.ndarray:
    """
    Dark bar with centred yellow text label. Returns [bar_h, width, 3] uint8.
    The 'seg_' prefix marks this as segmentation-pipeline code.
    Mirrors depth's _label_bar with identical implementation.
    """
    bar  = Image.new("RGB", (width, bar_h), color=(25, 25, 25))
    draw = ImageDraw.Draw(bar)
    bbox = draw.textbbox((0, 0), text)
    draw.text(((width - bbox[2]) // 2, 4), text, fill=(255, 220, 60))
    return np.asarray(bar)


def _seg_scene_image(
    orig_11:  torch.Tensor,    # [3,H,W] in [-1,1]  — raw validation image
    seg_01:   torch.Tensor,    # [3,H,W] in [0,1]   — pre-computed seg colour map
    pred_pil: Image.Image,     # PIL — generation WITH the text prompt
    size:     int,
    raw_pil:  Image.Image | None = None,  # PIL or None — generation WITHOUT prompt
) -> Image.Image:
    """
    One EXPLAINED image for a single validation scene — labeled panels side by side:
        ORIGINAL | SEG MAP | PREDICTED   (and | RAW SEG GEN if raw_pil given)

    Saved as its own file so each image is large enough to judge by eye. Returns
    a PIL.Image. Mirrors depth's _scene_image with "SEG MAP" / "RAW SEG GEN" labels.

    Args:
        orig_11 : [3,H,W] tensor in [-1,1]   (raw validation image from dataset)
        seg_01  : [3,H,W] tensor in [0,1]    (pre-computed seg conditioning colour map)
        pred_pil: PIL                         (generation WITH the text prompt)
        size    : square side for each panel in pixels
        raw_pil : PIL or None                 (generation WITHOUT a prompt; only
                                               when grid_include_empty_prompt=true)
    """
    orig_np = np.asarray(
        TF.to_pil_image(((orig_11.float() + 1) / 2).clamp(0, 1).cpu())
        .resize((size, size)).convert("RGB")
    )
    seg_np  = np.asarray(
        TF.to_pil_image(seg_01.float().clamp(0, 1).cpu())
        .resize((size, size)).convert("RGB")
    )
    pred_np = np.asarray(pred_pil.resize((size, size)).convert("RGB"))

    texts   = ["ORIGINAL", "SEG MAP", "PREDICTED"]
    columns = [orig_np, seg_np, pred_np]

    if raw_pil is not None:
        texts.append("RAW SEG GEN")
        columns.append(np.asarray(raw_pil.resize((size, size)).convert("RGB")))

    labels = np.concatenate([_seg_label_bar(size, t) for t in texts], axis=1)
    panels = np.concatenate(columns, axis=1)
    return Image.fromarray(np.concatenate([labels, panels], axis=0))


def _save_checkpoint_segmentation_images(
    model, val_dataset, idxs, kinds, n_loras, cfg, cfg_mask, device, out_dir, include_empty
):
    """
    Generate + save the monitoring images for one checkpoint into out_dir (the
    SAME folder as that checkpoint's weights). One labeled file per scene, plus
    a single prompts.txt. Returns (prompts, [np_images]) for TensorBoard.

    The 'segmentation' in the name marks this as segmentation-pipeline code.
    Mirrors depth's _save_checkpoint_images exactly, with seg-specific field names:
      • item["seg"] instead of item["depth"]
      • skip_encode=True (pre-saved colour map -> mapper, same as training)
      • _seg_scene_image() for the labeled panel image

    Per scene: load (jpg, seg, caption) from the VALIDATION set only (never
    train/test — references.md §8), feed the pre-computed seg colour map to the
    mapper via skip_encode=True, generate WITH the prompt, save a labeled
    image sample_NN_fixed/new.jpg.

    include_empty: when True, a 4th "RAW SEG GEN" panel is added (generation with
    empty prompt, pure seg adherence). Default False — seg training generates WITH
    prompt only, matching depth's behaviour.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts, images = [], []

    for n, (idx, kind) in enumerate(zip(idxs, kinds)):
        item   = val_dataset[idx]
        seg    = item["seg"].unsqueeze(0).to(device)    # [1,3,H,W] in [0,1]
        cs     = [seg] * n_loras
        prompt = cfg.prompt if cfg.get("prompt") else item["caption"]

        # Generate WITH the prompt; fixed seed per scene so checkpoints are comparable.
        pred = model.sample_custom(
            prompt=[prompt], num_images_per_prompt=1, cs=cs,
            generator=torch.Generator(device=device).manual_seed(cfg.seed),
            cfg_mask=cfg_mask, skip_encode=True,
        )[0]

        # Generate WITHOUT the prompt (pure seg adherence) — only when requested.
        raw = None
        if include_empty:
            raw = model.sample_custom(
                prompt=[""], num_images_per_prompt=1, cs=cs,
                generator=torch.Generator(device=device).manual_seed(cfg.seed),
                cfg_mask=cfg_mask, skip_encode=True,
            )[0]

        img = _seg_scene_image(item["jpg"], seg[0], pred, cfg.size, raw_pil=raw)
        img.save(out_dir / f"sample_{n:02d}_{kind}.jpg", quality=95)
        prompts.append(prompt)
        images.append(np.asarray(img))

    # ONE prompts.txt for all scenes (tagged [fixed]/[new]), next to the images.
    (out_dir / "prompts.txt").write_text(
        "\n".join(f"[{n}] [{k}] {p}" for n, (k, p) in enumerate(zip(kinds, prompts))),
        encoding="utf-8",
    )
    return prompts, images


def _segmentation_validation_loss(
    model, val_dataloader, n_loras, cfg, cfg_mask, accelerator, max_batches
):
    """
    Standard validation: run the SAME denoising loss as training on HELD-OUT
    validation data WITHOUT backprop, averaged over up to max_batches batches.

    The 'segmentation' in the name marks this as segmentation-pipeline code.
    Mirrors depth's _validation_loss exactly, with batch["seg"] instead of
    batch["depth"]. Same RNG save/restore trick for a comparable val curve.

    This gives you:
      (a) A val/loss curve to watch: if it diverges from train/loss, you're
          overfitting. If it keeps falling, the model is still generalising.
      (b) An objective best_model criterion (lowest val/loss), not a biased
          training-loss average.

    Uses skip_encode=True (pre-computed seg colour map -> mapper, same as training).
    Returns the global mean loss (float), reduced across all processes.
    """
    device = accelerator.device

    # Snapshot training RNG state -> seed for reproducible val loss -> restore.
    # This makes the noise/timesteps drawn at each checkpoint identical, so val/loss
    # curves from different checkpoints are directly comparable.
    cpu_rng  = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(cfg.seed)

    model.unet.eval()
    for m in model.mappers:  m.eval()
    for e in model.encoders: e.eval()

    total = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= max_batches:
                break
            imgs  = batch["jpg"].to(device).clip(-1.0, 1.0)
            B     = imgs.shape[0]
            seg   = batch["seg"].to(device)            # [B,3,H,W] colour map in [0,1]
            cs    = [seg] * n_loras
            prompts = [cfg.prompt] * B if cfg.get("prompt") else batch["caption"]
            _, loss, _, _ = model.forward_easy(
                imgs, prompts, cs,
                cfg_mask=[True for _ in cfg_mask],
                skip_encode=True,   # pre-saved seg map -> mapper, no live SegFormer
                batch=batch,
            )
            total += loss.detach()
            count += 1

    model.unet.train()
    for m in model.mappers:  m.train()
    for e in model.encoders: e.train()

    torch.set_rng_state(cpu_rng)                        # restore training RNG
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)

    total = accelerator.reduce(total, reduction="sum")  # global sum across GPUs
    count = accelerator.reduce(count, reduction="sum")
    return (total / torch.clamp(count, min=1.0)).item()
# ─────────────────────────────────────────────────────────────────────────────


@hydra.main(config_path="configs", config_name="train_seg", version_base=None)
def main(cfg):
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, signal_handler)

    # ── Pick LOCAL model folders vs HUB ids from the local_files_only flag ──────
    # YAML lists both (base_model_name/path, seg_model_name/path); here we choose.
    # Local paths are made absolute from the repo root (get_original_cwd), because
    # Hydra has already chdir'd into the run directory by now.
    # When offline we also export HF_HUB_OFFLINE so NOTHING can touch the network.
    _root = get_original_cwd()
    if cfg.local_files_only:
        os.environ["HF_HUB_OFFLINE"]      = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        cfg.model.model_name              = os.path.join(_root, cfg.base_model_path)
        cfg.lora.struct.encoder.model     = os.path.join(_root, cfg.seg_model_path)
    else:
        cfg.model.model_name              = cfg.base_model_name
        cfg.lora.struct.encoder.model     = cfg.seg_model_name
    print(f"[model] base             = {cfg.model.model_name}")
    print(f"[model] seg encoder      = {cfg.lora.struct.encoder.model}")
    print(f"[model] local_files_only = {cfg.local_files_only}")

    # Suppress expected-but-noisy warnings.
    import logging
    logging.getLogger("transformers.utils.loading_report").setLevel(logging.ERROR)
    logging.getLogger("diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion").setLevel(logging.ERROR)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    accelerator = Accelerator(
        project_dir=output_path / "logs",
        log_with="tensorboard",
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.get("bf16", True) else "no",
    )

    logger = get_logger(__name__)
    logger.info("==================================")
    logger.info(cfg)
    logger.info(output_path)

    cfg = hydra.utils.instantiate(cfg)
    model: ModelBase = cfg.model
    model = model.to(accelerator.device)
    model.pipe.to(accelerator.device)
    n_loras = len(cfg.lora.keys())

    cfg_mask = add_lora_from_config(model, cfg, accelerator.device)

    if cfg.get("gradient_checkpointing", False):
        model.unet.enable_gradient_checkpointing()

    dm = cfg.data
    train_dataloader = dm.train_dataloader()
    val_dataloader   = dm.val_dataloader()

    mappers_params = list(
        filter(lambda p: p.requires_grad,
               reduce(lambda x, y: x + list(y.parameters()), model.mappers, []))
    )
    encoder_params = list(
        filter(lambda p: p.requires_grad,
               reduce(lambda x, y: x + list(y.parameters()), model.encoders, []))
    )
    optimizer = torch.optim.AdamW(
        model.params_to_optimize + mappers_params + encoder_params,
        lr=cfg.learning_rate,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.gradient_accumulation_steps
    )
    max_train_steps = cfg.epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=cfg.get("lr_warmup_steps", 0) * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    logger.info(f"Mapper params:  {sum(p.numel() for p in mappers_params):,}")
    logger.info(f"Encoder params: {sum(p.numel() for p in encoder_params):,}")
    logger.info(f"LoRA params:    {sum(p.numel() for p in model.params_to_optimize):,}")

    # ── SEGMENTATION TRAINING STARTUP BANNER ──────────────────────────────────
    # Printed once at launch so you can identify this run at a glance in the log
    # file, distinguish it from a depth run, and know exactly where to point TensorBoard.
    if accelerator.is_main_process:
        tb_dir = output_path / "logs" / "tensorboard"
        logger.info("")
        logger.info("=" * 64)
        logger.info("  PIPELINE   :  SEGMENTATION  (SegFormer-b5 Cityscapes conditioning)")
        logger.info(f"  Output     :  {output_path}")
        logger.info(f"  TensorBoard:  tensorboard --logdir \"{tb_dir}\"")
        logger.info(f"  Train      :  {len(dm.train_dataset):,} images  |  Val: {len(dm.val_dataset):,} images")
        logger.info(f"  Schedule   :  {cfg.epochs} epochs x {num_update_steps_per_epoch} steps = {max_train_steps} total optimizer steps")
        logger.info(f"  LR         :  {cfg.learning_rate}  scheduler={cfg.lr_scheduler}  warmup={cfg.get('lr_warmup_steps', 0)} steps")
        logger.info(f"  val_steps  :  {cfg.val_steps}  (val/loss + best_model update)")
        logger.info(f"  ckpt_steps :  {cfg.ckpt_steps}  (weights + {cfg.get('n_grid_images', 10)} grid images saved)")
        logger.info(f"  TensorBoard scalars: train/loss  train/lr  train/grad_norm  train/epoch  val/loss")
        logger.info(f"  TensorBoard images : val/sample_00 … val/sample_{cfg.get('n_grid_images', 10) - 1:02d}  (ORIGINAL | SEG MAP | PREDICTED)")
        logger.info("=" * 64)
        logger.info("")

    if accelerator.is_main_process:
        # Keep personal identifiers OUT of the tensorboard event filename.
        # Override gethostname with the experiment tag (matches depth's approach).
        import socket as _socket
        _socket.gethostname = lambda: str(cfg.get("tag", "loradapter"))
        accelerator.init_trackers("tensorboard")

    prepared = accelerator.prepare(
        *model.mappers, *model.encoders, model.unet,
        optimizer, train_dataloader, val_dataloader, lr_scheduler,
    )
    mappers  = prepared[: len(model.mappers)]
    encoders = prepared[len(model.mappers): len(model.mappers) + len(model.encoders)]
    (unet, optimizer, train_dataloader, val_dataloader, lr_scheduler) = prepared[
        len(model.mappers) + len(model.encoders):
    ]
    model.unet     = unet
    model.mappers  = mappers
    model.encoders = encoders

    try:
        if cfg.get("max_train_steps", None) is not None:
            max_train_steps = cfg.max_train_steps
    except Exception:
        pass

    global_step  = 0
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_main_process,
    )
    progress_bar.set_description("Steps")
    best_loss = float("inf")

    # ── Checkpoint-monitoring images setup (references.md §8) ─────────────────
    # n_grid_images segmentation scenes per checkpoint, split 50/50:
    #   • FIXED half: drawn ONCE with OS entropy at the start of this run, then
    #     reused every checkpoint -> watch the SAME scenes improve over training.
    #     (Different fixed scenes each RUN; logged for reproducibility.)
    #   • FRESH half: re-drawn at each checkpoint from non-fixed pool ->
    #     quick generalization peek on scenes the fixed set never shows.
    # SOURCE = validation set only (never train/test), per references.md §8.
    n_grid_images = max(2, min(int(cfg.get("n_grid_images", 10)), len(dm.val_dataset)))
    include_empty = bool(cfg.get("grid_include_empty_prompt", False))
    n_fixed  = n_grid_images // 2          # consistent half across checkpoints
    n_random = n_grid_images - n_fixed     # fresh half at each checkpoint

    # random.Random() with NO seed = OS entropy -> different fixed set each run.
    _fixed_val_idxs = random.Random().sample(range(len(dm.val_dataset)), n_fixed)
    logger.info(
        f"Seg grid: {n_grid_images} val scenes = {n_fixed} fixed (OS entropy) "
        f"{_fixed_val_idxs} + {n_random} re-randomized each checkpoint "
        f"(include_empty={include_empty}, val size={len(dm.val_dataset)})"
    )

    def save_seg_ckpt_and_grid(stem, is_best=False, info_lines=None):
        """
        Save the CURRENT model as a checkpoint AND its monitoring images together,
        so every checkpoint always has a matching set of validation images.

        The 'seg' prefix marks this as segmentation-pipeline code. Mirrors depth's
        save_ckpt_and_grid with "seg" substituted throughout.

        Grid generation is best-effort: errors are logged but never block the
        checkpoint save. Main process only.

          stem      : checkpoint subfolder path (e.g. "checkpoint-epoch2/step1000").
          is_best   : save into best_model/ and write info.txt.
          info_lines: extra lines for best_model/info.txt.
        """
        if not accelerator.is_main_process:
            return

        # Images saved INSIDE the checkpoint's own folder (next to weights), so
        # the model and its monitoring images always live together.
        ckpt_dir = (output_path / "best_model") if is_best else (output_path / stem)
        save_checkpoint(
            model.get_lora_state_dict(accelerator.unwrap_model(unet)),
            [accelerator.unwrap_model(m).state_dict() for m in mappers],
            None, ckpt_dir,
        )

        try:
            unet.eval()
            for m in mappers:  m.eval()
            for e in encoders: e.eval()

            # Fixed half reused; fresh half drawn from the non-fixed pool (no dups).
            pool     = [i for i in range(len(dm.val_dataset))
                        if i not in set(_fixed_val_idxs)]
            new_idxs = random.sample(pool, min(n_random, len(pool)))
            idxs     = list(_fixed_val_idxs) + new_idxs
            kinds    = ["fixed"] * len(_fixed_val_idxs) + ["new"] * len(new_idxs)

            with torch.no_grad():
                prompts, images = _save_checkpoint_segmentation_images(
                    model, dm.val_dataset, idxs, kinds, n_loras, cfg, cfg_mask,
                    accelerator.device, ckpt_dir, include_empty,
                )

            if is_best:
                (ckpt_dir / "info.txt").write_text(
                    "\n".join(info_lines or []), encoding="utf-8"
                )

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    for n, img in enumerate(images):
                        tracker.writer.add_image(
                            f"val/sample_{n:02d}", img, global_step, dataformats="HWC"
                        )
                    tracker.writer.add_text("val/prompts", " | ".join(prompts), global_step)

            logger.info(f"[seg grid] {stem}: {len(prompts)} scene images -> {ckpt_dir}")

        except Exception as e:
            print("!!! ERROR generating segmentation checkpoint images !!!")
            print(e)
            print(traceback.format_exc())
        finally:
            unet.train()
            for m in mappers:  m.train()
            for e in encoders: e.train()

    def do_segmentation_validation(label):
        """
        VALIDATION ONLY — the CHEAP half, run every val_steps:
          1. Compute val/loss on held-out seg data.
          2. Log to TensorBoard as val/loss.
          3. If it's the best so far, save best_model/ (weights + images + info.txt).

        The 'segmentation' in the name marks this as seg-pipeline code. Mirrors
        depth's do_validation exactly with _segmentation_validation_loss and seg keys.
        Does NOT save a regular checkpoint — that is decoupled and controlled
        separately by ckpt_steps (see save_seg_ckpt_and_grid calls in the loop).

          label : human tag for the log, e.g. "step150" or "epoch2".
        """
        nonlocal best_loss
        val_loss = _segmentation_validation_loss(
            model, val_dataloader, n_loras, cfg, cfg_mask,
            accelerator, int(cfg.get("val_batches", 8)),
        )
        accelerator.log({"val/loss": val_loss}, step=global_step)
        if accelerator.is_main_process:
            logger.info(f"[seg val] {label}: val/loss = {val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            save_seg_ckpt_and_grid(
                "best_model", is_best=True,
                info_lines=[f"from:     {label}", f"val_loss: {val_loss:.6f}"],
            )
            if accelerator.is_main_process:
                logger.info(
                    f"New best segmentation model — {label}, val/loss={val_loss:.6f}"
                )

    # ── Training loop ──────────────────────────────────────────────────────────
    logger.info("[SEG] start segmentation training")
    for epoch in range(cfg.epochs):
        logger.info(f"[SEG] Epoch {epoch + 1}/{cfg.epochs} started  (global_step={global_step})")
        unet.train()
        for m in mappers:  m.train()
        for e in encoders: e.train()

        for step, batch in enumerate(train_dataloader):
            _grad_norm = None   # only set on true optimizer steps (sync_gradients=True)

            with accelerator.accumulate(unet, *mappers, *encoders):
                imgs = batch["jpg"].to(accelerator.device).clip(-1.0, 1.0)
                B    = imgs.shape[0]

                # ── SEG CHANGE: pre-saved colour seg map as conditioning ──────
                # batch["seg"] is the colourised class-ID map ([B,3,H,W] in [0,1])
                # produced by SegJsonDataset._load_seg_colormap() at load time.
                # skip_encode=True -> SegmentationEncoder NOT called; the map goes
                # straight to the mapper, exactly mirroring depth's batch["depth"].
                seg_maps = batch["seg"].to(accelerator.device)
                cs = [seg_maps] * n_loras
                # ─────────────────────────────────────────────────────────────

                prompts = (
                    [cfg.prompt] * B
                    if cfg.get("prompt", None) is not None
                    else batch["caption"]
                )

                model_pred, loss, x0, _ = model.forward_easy(
                    imgs, prompts, cs,
                    cfg_mask=[True for _ in cfg_mask],
                    skip_encode=True,   # seg map already computed — bypass SegFormer
                    batch=batch,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    all_params = [p for g in optimizer.param_groups for p in g["params"]]
                    # clip_grad_norm_ returns the total norm BEFORE clipping — log it
                    # to detect instability: spiky/high values mean the model is
                    # struggling; a smoothly decreasing norm means healthy convergence.
                    _g = accelerator.clip_grad_norm_(all_params, max_norm=1.0)
                    _grad_norm = float(_g)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            loss_val = loss.detach().item()
            lr_val   = lr_scheduler.get_last_lr()[0]
            # epoch as a float (e.g. 2.5 = halfway through epoch 3) so TensorBoard
            # shows epoch boundaries without having to manually count steps.
            epoch_frac = epoch + step / max(len(train_dataloader), 1)
            log_dict = {
                "train/loss":  loss_val,
                "train/lr":    lr_val,
                "train/epoch": epoch_frac,
            }
            if _grad_norm is not None:
                # Only logged on true optimizer steps (not gradient-accumulation micro-steps).
                # What to watch: starts high (~1.0 at clip), drops over training.
                # Red flag: stays above 1.0 for many steps, or spikes repeatedly.
                log_dict["train/grad_norm"] = _grad_norm
            progress_bar.set_postfix(loss=loss_val, lr=f"{lr_val:.2e}", gnorm=f"{_grad_norm or 0:.3f}", refresh=False)
            accelerator.log(log_dict, step=global_step)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # ── DECOUPLED step-level triggers ────────────────────────────
                # val_steps  = how often to compute val/loss (cheap; also updates
                #              best_model when val/loss improves).
                # ckpt_steps = how often to write a checkpoint to disk (heavy:
                #              weights + N monitoring images), grouped per epoch as
                #              checkpoint-epoch{N}/step{global_step}/.
                # These are INDEPENDENT — e.g. validate every 500 but save every 1000.
                if global_step % cfg.val_steps == 0 or stop_training:
                    do_segmentation_validation(f"step{global_step}")
                if global_step % cfg.ckpt_steps == 0 or stop_training:
                    save_seg_ckpt_and_grid(
                        f"checkpoint-epoch{epoch + 1}/step{global_step}"
                    )

            if stop_training:
                break

        # ── END-OF-EPOCH: ALWAYS validate AND save this epoch's checkpoint ───
        do_segmentation_validation(f"epoch{epoch + 1}")
        save_seg_ckpt_and_grid(
            f"checkpoint-epoch{epoch + 1}/checkpoint-epoch{epoch + 1}"
        )

        if stop_training:
            break

    # ── Final snapshot of the last weights ────────────────────────────────────
    # Reuses the end-of-epoch folder (no extra folder), capturing early-stop too.
    accelerator.wait_for_everyone()
    save_seg_ckpt_and_grid(
        f"checkpoint-epoch{epoch + 1}/checkpoint-epoch{epoch + 1}"
    )


if __name__ == "__main__":
    main()

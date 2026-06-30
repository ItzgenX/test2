import hydra
import math
from hydra.utils import get_original_cwd
from src.model import ModelBase
from diffusers.optimization import get_scheduler
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import torchvision.transforms.functional as TF
from accelerate.logging import get_logger
import signal
import os
import traceback
import random
from functools import reduce
from PIL import Image, ImageDraw

from src.utils import add_lora_from_config, save_checkpoint


torch.set_float32_matmul_precision("high")


stop_training = False


def signal_handler(sig, frame):
    global stop_training
    stop_training = True
    print("got stop signal")


# ── Checkpoint-monitoring images (references.md §8) ────────────────────────────
# Every time a checkpoint is saved, that SAME checkpoint generates N validation
# images so you can judge it by eye. Each scene is saved as its OWN labeled file
# (a 3-panel "explained" image: ORIGINAL | DEPTH MAP | PREDICTED) — big enough to
# read — INSIDE the checkpoint's own folder, next to its weights. One prompts.txt
# lists all N prompts. Depth training generates WITH the prompt only (no
# empty-prompt / "raw depth" generation).

def _label_bar(width, text, bar_h=24):
    """Dark bar with centred text label. Returns [bar_h, width, 3] uint8 array."""
    bar  = Image.new("RGB", (width, bar_h), color=(25, 25, 25))
    draw = ImageDraw.Draw(bar)
    bbox = draw.textbbox((0, 0), text)
    draw.text(((width - bbox[2]) // 2, 4), text, fill=(255, 220, 60))
    return np.asarray(bar)


def _scene_image(orig_11, depth_01, pred_pil, size, raw_pil=None):
    """
    One EXPLAINED image for a single validation scene — labeled panels side by side:
        ORIGINAL | DEPTH MAP | PREDICTED   (and | RAW DEPTH GEN if raw_pil given)
    Saved as its own file so each image is large enough to judge. Returns a PIL.
        orig_11 : [3,H,W] tensor in [-1,1]   (raw validation image)
        depth_01: [3,H,W] tensor in [0,1]    (pre-computed depth conditioning)
        pred_pil: PIL                        (generation WITH the text prompt)
        raw_pil : PIL or None                (generation WITHOUT a prompt; only when
                                              grid_include_empty_prompt=true)
    """
    orig_np  = np.asarray(TF.to_pil_image(((orig_11.float()+1)/2).clamp(0,1).cpu()).resize((size,size)).convert("RGB"))
    depth_np = np.asarray(TF.to_pil_image(depth_01.float().clamp(0,1).cpu()).resize((size,size)).convert("RGB"))
    pred_np  = np.asarray(pred_pil.resize((size,size)).convert("RGB"))
    texts   = ["ORIGINAL", "DEPTH MAP", "PREDICTED"]
    columns = [orig_np, depth_np, pred_np]
    if raw_pil is not None:
        texts.append("RAW DEPTH GEN")
        columns.append(np.asarray(raw_pil.resize((size, size)).convert("RGB")))
    labels = np.concatenate([_label_bar(size, t) for t in texts], axis=1)
    panels = np.concatenate(columns, axis=1)
    return Image.fromarray(np.concatenate([labels, panels], axis=0))


def _save_checkpoint_images(model, val_dataset, idxs, kinds, n_loras, cfg, cfg_mask, device, out_dir, include_empty):
    """
    Generate + save the monitoring images for one checkpoint into out_dir (the SAME
    folder as that checkpoint's weights). One labeled file per scene, plus a single
    prompts.txt. Returns (prompts, [np_images]) so the caller can log to TensorBoard.

    Per scene: load (jpg, depth, caption) from the VALIDATION set only (never
    train/test — references.md §8), feed the PRE-COMPUTED depth to the mapper
    (skip_encode=True, mirroring training), generate WITH the prompt, save a labeled
    image sample_<n>_<kind>.jpg.
    include_empty: depth training defaults this OFF (generate WITH prompt only). When
    on (grid_include_empty_prompt=true) a 4th RAW DEPTH GEN panel is added (an extra
    generation with an empty prompt).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts, images = [], []
    for n, (idx, kind) in enumerate(zip(idxs, kinds)):
        item   = val_dataset[idx]
        depth  = item["depth"].unsqueeze(0).to(device)        # [1,3,H,W] in [0,1]
        cs     = [depth] * n_loras
        prompt = cfg.prompt if cfg.get("prompt") else item["caption"]
        pred = model.sample_custom(
            prompt=[prompt], num_images_per_prompt=1, cs=cs,
            generator=torch.Generator(device=device).manual_seed(cfg.seed),
            cfg_mask=cfg_mask, skip_encode=True,
        )[0]
        raw = None
        if include_empty:
            raw = model.sample_custom(
                prompt=[""], num_images_per_prompt=1, cs=cs,
                generator=torch.Generator(device=device).manual_seed(cfg.seed),
                cfg_mask=cfg_mask, skip_encode=True,
            )[0]
        img = _scene_image(item["jpg"], depth[0], pred, cfg.size, raw_pil=raw)
        img.save(out_dir / f"sample_{n:02d}_{kind}.jpg", quality=95)
        prompts.append(prompt)
        images.append(np.asarray(img))
    # ONE prompts file for all scenes (tagged fixed/new), next to the images.
    (out_dir / "prompts.txt").write_text(
        "\n".join(f"[{n}] [{k}] {p}" for n, (k, p) in enumerate(zip(kinds, prompts))),
        encoding="utf-8")
    return prompts, images


def _validation_loss(model, val_dataloader, n_loras, cfg, cfg_mask, accelerator, max_batches):
    """
    Standard validation: run the SAME denoising loss as training, but on HELD-OUT
    validation data and WITHOUT backprop, averaged over up to max_batches batches.
    This is the quantitative signal that lets you (a) watch train/loss vs val/loss
    diverge = overfitting, and (b) pick best_model by a real held-out metric.

    Uses skip_encode=True (pre-computed depth -> mapper, mirroring training).
    The torch RNG is SEEDED then RESTORED, so the noise/timesteps drawn for the loss
    are identical at every checkpoint (a comparable val curve) WITHOUT disturbing
    the training RNG stream. Returns the global mean loss (float), reduced across
    all processes so it is correct on multi-GPU too.
    """
    device = accelerator.device

    # Snapshot RNG -> seed for a reproducible val loss -> restore afterwards.
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
            depth = batch["depth"].to(device)
            cs    = [depth] * n_loras
            prompts = [cfg.prompt] * B if cfg.get("prompt") else batch["caption"]
            _, loss, _, _ = model.forward_easy(
                imgs, prompts, cs,
                cfg_mask=[True for _ in cfg_mask],
                skip_encode=True, batch=batch,
            )
            total += loss.detach()
            count += 1

    model.unet.train()
    for m in model.mappers:  m.train()
    for e in model.encoders: e.train()

    torch.set_rng_state(cpu_rng)                       # restore training RNG
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)

    total = accelerator.reduce(total, reduction="sum")  # global sum across GPUs
    count = accelerator.reduce(count, reduction="sum")
    return (total / torch.clamp(count, min=1.0)).item()
# ─────────────────────────────────────────────────────────────────────────────


@hydra.main(config_path="configs", config_name="train_depth", version_base=None)
def main(cfg):
    if hasattr(signal, "SIGUSR1"):   # Linux/Mac only
        signal.signal(signal.SIGUSR1, signal_handler)

    # ── Pick LOCAL model folders vs HUB ids from the local_files_only flag ──────
    # The YAML lists both (base_model_name/path, depth_model_name/path); here we
    # choose. Local paths are made absolute from the repo root (get_original_cwd),
    # because Hydra has already chdir'd into the run directory by now.
    # When offline we also export HF_HUB_OFFLINE so NOTHING can touch the network.
    _root = get_original_cwd()
    if cfg.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        cfg.model.model_name          = os.path.join(_root, cfg.base_model_path)
        cfg.lora.struct.encoder.model = os.path.join(_root, cfg.depth_model_path)
    else:
        cfg.model.model_name          = cfg.base_model_name
        cfg.lora.struct.encoder.model = cfg.depth_model_name
    print(f"[model] base  = {cfg.model.model_name}")
    print(f"[model] depth = {cfg.lora.struct.encoder.model}")
    print(f"[model] local_files_only = {cfg.local_files_only}")

    # Suppress expected-but-noisy warnings:
    # 1. transformers LOAD REPORT: "position_ids UNEXPECTED" — this key exists in the
    #    SD 1.5 CLIP checkpoint but was removed from newer transformers CLIP architecture.
    #    It is harmless (the model works correctly without it).
    # 2. diffusers safety checker: we intentionally disable it for training/research.
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
        filter(lambda p: p.requires_grad, reduce(lambda x, y: x + list(y.parameters()), model.mappers, []))
    )
    encoder_params = list(
        filter(lambda p: p.requires_grad, reduce(lambda x, y: x + list(y.parameters()), model.encoders, []))
    )

    optimizer = torch.optim.AdamW(
        model.params_to_optimize + mappers_params + encoder_params,
        lr=cfg.learning_rate,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    max_train_steps = cfg.epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.get("lr_warmup_steps", 0) * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    logger.info(f"Number params Mapper Network(s) {sum(p.numel() for p in mappers_params):,}")
    logger.info(f"Number params Encoder Network(s) {sum(p.numel() for p in encoder_params):,}")
    logger.info(f"Number params all LoRAs(s) {sum(p.numel() for p in model.params_to_optimize):,}")

    logger.info("init trackers")
    if accelerator.is_main_process:
        # Keep personal/machine identifiers OUT of the tensorboard event filename.
        # torch's SummaryWriter embeds socket.gethostname() in the tfevents name
        # (it was "events.out.tfevents.<time>.aditya.<pid>.0" — "aditya" = this
        # machine's hostname). Override it with the generic experiment tag so the
        # logs are shareable without leaking the username/hostname.
        import socket as _socket
        _socket.gethostname = lambda: str(cfg.get("tag", "loradapter"))
        accelerator.init_trackers("tensorboard")

    logger.info("prepare network")

    prepared = accelerator.prepare(
        *model.mappers,
        *model.encoders,
        model.unet,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )

    mappers  = prepared[: len(model.mappers)]
    encoders = prepared[len(model.mappers) : len(model.mappers) + len(model.encoders)]
    (unet, optimizer, train_dataloader, val_dataloader, lr_scheduler) = prepared[
        len(model.mappers) + len(model.encoders) :
    ]
    model.unet     = unet
    model.mappers  = mappers
    model.encoders = encoders

    try:
        if cfg.get("max_train_steps", None) is not None:
            max_train_steps = cfg.max_train_steps
    except:
        pass

    global_step = 0
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_main_process,
    )
    progress_bar.set_description("Steps")

    best_loss = float("inf")

    # ── Checkpoint-monitoring images setup (references.md §8) ──────────────────
    # n_grid_images validation scenes per checkpoint, split 50/50:
    #   • HALF are FIXED scenes: chosen RANDOMLY ONCE at the start of THIS run, then
    #     reused at every checkpoint of this run → watch the SAME scenes improve.
    #     (Different scenes each run; the exact indices are logged below so you can
    #     reproduce a run by re-using them if ever needed.)
    #   • HALF are NEW scenes (re-drawn randomly at each checkpoint) → see how the
    #     model does on fresh, never-pinned scenes (a quick generalization peek).
    # SOURCE = validation set only (never train/test), per references.md §8.
    n_grid_images = max(2, min(int(cfg.get("n_grid_images", 10)), len(dm.val_dataset)))
    # OFF by default for depth training (generate WITH prompt only); when true,
    # each scene image also gets a 4th RAW DEPTH GEN panel (empty-prompt generation).
    include_empty = bool(cfg.get("grid_include_empty_prompt", False))
    n_fixed  = n_grid_images // 2                 # consistent half
    n_random = n_grid_images - n_fixed            # fresh half
    # random.Random() with NO seed = OS entropy → a different fixed set each run.
    _fixed_val_idxs = random.Random().sample(range(len(dm.val_dataset)), n_fixed)
    logger.info(f"Grid: {n_grid_images} val scenes = {n_fixed} fixed (random per run) {_fixed_val_idxs} "
                f"+ {n_random} re-randomized each checkpoint "
                f"(include_empty={include_empty}, val size={len(dm.val_dataset)})")

    def save_ckpt_and_grid(stem, is_best=False, info_lines=None):
        """
        Save the CURRENT model as a checkpoint AND its monitoring images together, so
        every checkpoint always has a matching grid (the two are coupled here in
        ONE place — step, epoch, best, and final all call this).

        Grid generation is best-effort: if it errors it is logged but never blocks
        the checkpoint save. Main process only.
          stem      : checkpoint folder path, e.g. "checkpoint-epoch2/step1000" / "checkpoint-epoch2/checkpoint-epoch2".
          is_best   : save into best_model/ and also write info.txt (epoch, loss).
          info_lines: extra lines for best_model/info.txt.
        """
        if not accelerator.is_main_process:
            return
        # Images are saved INSIDE the checkpoint's own folder (next to its weights),
        # so each checkpoint's model and its monitoring images live together.
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

            # Scene indices = the FIXED half + a FRESH random half (re-drawn now, so
            # those scenes differ at every checkpoint). The fresh half is sampled
            # from val indices NOT in the fixed half (no duplicates).
            pool = [i for i in range(len(dm.val_dataset)) if i not in set(_fixed_val_idxs)]
            new_idxs = random.sample(pool, min(n_random, len(pool)))   # global RNG -> varies per checkpoint
            idxs  = list(_fixed_val_idxs) + new_idxs
            kinds = ["fixed"] * len(_fixed_val_idxs) + ["new"] * len(new_idxs)

            with torch.no_grad():
                prompts, images = _save_checkpoint_images(
                    model, dm.val_dataset, idxs, kinds, n_loras, cfg, cfg_mask,
                    accelerator.device, ckpt_dir, include_empty)
            if is_best:
                (ckpt_dir / "info.txt").write_text(
                    "\n".join(info_lines or []), encoding="utf-8")
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    for n, img in enumerate(images):
                        tracker.writer.add_image(f"val/sample_{n:02d}", img,
                                                 global_step, dataformats="HWC")
                    tracker.writer.add_text("val/prompts", " | ".join(prompts), global_step)
            logger.info(f"[grid] {stem}: {len(prompts)} scene images -> {ckpt_dir}")
        except Exception as e:
            print("!!! ERROR generating checkpoint images !!!")
            print(e)
            print(traceback.format_exc())
        finally:
            unet.train()
            for m in mappers:  m.train()
            for e in encoders: e.train()

    def do_validation(label):
        """
        VALIDATION ONLY — the CHEAP half, run every `val_steps`:
          1. Compute val/loss on held-out data (the standard diffusion metric).
          2. Log it to TensorBoard (the val/loss curve).
          3. If it's the best so far, save best_model/ (weights + images + info.txt).
        It does NOT save a regular checkpoint — that is decoupled and controlled
        separately by `ckpt_steps` (see save_ckpt_and_grid calls in the loop).
          label : a human tag for the log line / best_model's info.txt, e.g. "step150".
        """
        nonlocal best_loss
        val_loss = _validation_loss(model, val_dataloader, n_loras, cfg, cfg_mask,
                                    accelerator, int(cfg.get("val_batches", 8)))
        accelerator.log({"val/loss": val_loss}, step=global_step)
        if accelerator.is_main_process:
            logger.info(f"[val] {label}: val/loss = {val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            save_ckpt_and_grid("best_model", is_best=True,
                               info_lines=[f"from:     {label}",
                                           f"val_loss: {val_loss:.6f}"])
            if accelerator.is_main_process:
                logger.info(f"New best model — {label}, val/loss={val_loss:.6f}")

    logger.info("start training")
    for epoch in range(cfg.epochs):
        logger.info("new epoch")
        unet.train()
        for m in mappers:  m.train()
        for e in encoders: e.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, *mappers, *encoders):
                imgs = batch["jpg"].to(accelerator.device).clip(-1.0, 1.0)
                B = imgs.shape[0]

                # ── DEPTH CHANGE: use pre-computed depth maps instead of images ──
                # batch["depth"] is loaded from the cached PNG (precompute_depth.py).
                # skip_encode=True means the DepthEstimator is NOT called here —
                # the depth tensor goes directly to the mapper network.
                depth_maps = batch["depth"].to(accelerator.device)
                cs = [depth_maps] * n_loras
                # ─────────────────────────────────────────────────────────────────

                if cfg.get("prompt", None) is not None:
                    prompts = [cfg.prompt] * B
                else:
                    prompts = batch["caption"]

                model_pred, loss, x0, _ = model.forward_easy(
                    imgs,
                    prompts,
                    cs,
                    cfg_mask=[True for _ in cfg_mask],
                    skip_encode=True,   # depth already computed — bypass DPT
                    batch=batch,
                )

                accelerator.backward(loss)

                # Clip gradients to prevent exploding gradients during training.
                # max_norm=1.0 is the standard value for diffusion fine-tuning.
                if accelerator.sync_gradients:
                    all_params = [p for group in optimizer.param_groups for p in group["params"]]
                    accelerator.clip_grad_norm_(all_params, max_norm=1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            loss_val = loss.detach().item()
            lr_val   = lr_scheduler.get_last_lr()[0]
            progress_bar.set_postfix(loss=loss_val, lr=lr_val, refresh=False)
            # TensorBoard tags namespaced train/ vs val/ (val/* set in do_validation)
            accelerator.log({"train/loss": loss_val, "train/lr": lr_val}, step=global_step)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # ── DECOUPLED step-level triggers ──────────────────────────────
                # val_steps  = how often to compute val/loss (cheap; also updates
                #              best_model when val/loss improves).
                # ckpt_steps = how often to write a checkpoint to disk (heavy:
                #              weights + N monitoring images), grouped per epoch as
                #              checkpoint-epoch{N}/step{global_step}/.
                # They are independent — e.g. validate every 50 but save every 100.
                if global_step % cfg.val_steps == 0 or stop_training:
                    do_validation(f"step{global_step}")
                if global_step % cfg.ckpt_steps == 0 or stop_training:
                    save_ckpt_and_grid(f"checkpoint-epoch{epoch + 1}/step{global_step}")

            if stop_training:
                break

        # ── END-OF-EPOCH: ALWAYS validate AND save this epoch's checkpoint ──────
        # (independent of val_steps/ckpt_steps so every epoch gets both a fresh
        # val/loss and its checkpoint-epoch{N}/checkpoint-epoch{N}/ end folder).
        do_validation(f"epoch{epoch + 1}")
        save_ckpt_and_grid(f"checkpoint-epoch{epoch + 1}/checkpoint-epoch{epoch + 1}")

        if stop_training:
            break

    # ── Final snapshot of the last weights ─────────────────────────────────────
    # `epoch` keeps its last value after the loop. On normal completion this is the
    # same weights the end-of-epoch save just wrote, so we reuse that exact folder
    # name (harmless re-save, no extra folder); it still captures an early-stop.
    accelerator.wait_for_everyone()
    save_ckpt_and_grid(f"checkpoint-epoch{epoch + 1}/checkpoint-epoch{epoch + 1}")


if __name__ == "__main__":
    main()

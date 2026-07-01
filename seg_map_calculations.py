"""
seg_map_calculations.py
-----------------------------
STAGE C — pre-compute and CACHE semantic-segmentation maps for training images
using SegFormer-b5-Cityscapes, then build seg_training/{train,val,test}.json.

This is the segmentation twin of depth_map_calculations.py. It mirrors that
script's STRUCTURE (relative-path-preserving output tree, one-pass atomic JSON
building, loud self-verification) but is a SEPARATE file — depth and segmentation
never share a calculation file (references.md §6). The seg-specific differences
vs depth, each deliberate:

  • OUTPUT = raw class-ID 8-bit PNG (values 0..18), NOT a grayscale depth ramp.
    We store the raw IDs (canonical, hand-editable, re-palette-able) and let the
    dataset colourise them at LOAD time with SEG_CITYSCAPES_PALETTE (the SSOT in
    src/encoders/seg_encoder.py). This script never writes colour — only labels.

  • The SegmentationEncoder (in src/encoders/seg_encoder.py) is IMPORTED here and
    used via label_ids(), so the maps saved for training are produced by exactly the
    same _predict_ids() code the live inference encoder runs — the train/inference
    parity rule.

  • local_files_only is threaded from CLI to the encoder, matching depth's pattern.
    Default = True (offline; the b5 checkpoint must be in checkpoints/local_models/).

LOCKED MODEL: nvidia/segformer-b5-finetuned-cityscapes-1024-1024
  (references.md §9 — b5 chosen for best segmentation accuracy)

TYPICAL WORKFLOW (data_dir mode — recommended, mirrors depth):
  # builds data/seg_training/{train,val,test}.jsonl from data/{train,val,test}.jsonl
  python seg_map_calculations.py --data_dir data/

  # then train:
  python seg_training.py experiment=train_seg

QUICK COMMANDS (run from repo root with conda loradapter env active):
  # --- Dry run: 15 images, verify pipeline before committing to full dataset ---
  python seg_map_calculations.py --data_dir data/ --dry_run_n 15

  # --- Full run: all images (639 train + 137 val + 137 test) ---
  python seg_map_calculations.py --data_dir data/

  # --- Re-compute everything (force overwrite of existing PNGs) ---
  python seg_map_calculations.py --data_dir data/ --no_skip

JSON ENTRY FORMAT produced:
  {"raw_image_path": "data/raw/000417/raw_image.jpg",
   "seg_path":       "data/raw_seg/000417/raw_image.png",
   "prompt":         "..."}

    ###python seg_map_calculations.py   --dataset_dir D:/MyWorkplace/WorkStation/custome_dataset --data_dir data/ --image_path target
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.data.transforms import build_seg_square_preprocess
from src.encoders.seg_encoder import SEG_CITYSCAPES_PALETTE, SegmentationEncoder

# LOCKED model (references.md §9). Use --local_files_only False for first download.
DEFAULT_SEG_MODEL = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"


def parse_bool(value):
    """Parse True/False (any case, plus 1/0, yes/no) from the CLI into a bool."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(
        f"--local_files_only expects True or False, got: {value!r}"
    )


# ============================================================================ #
#  HELPERS (mirrors depth_map_calculations.py — adapted, not copy-pasted)      #
# ============================================================================ #

def _get_image_path(entry: dict, image_path: str = "source") -> str:
    """
    Return the source image path from a JSONL entry.

    Checks `image_path` first (default "source", override with --image_path).
    Falls back to "raw_image_path" so seg_training/ JSONLs also work.
    Change the key your manifests use without touching any other code:
        python seg_map_calculations.py --data_dir data/ --image_path target
    """
    if image_path in entry:
        return entry[image_path]
    if "raw_image_path" in entry:
        return entry["raw_image_path"]
    raise KeyError(
        f"Entry has neither '{image_path}' nor 'raw_image_path'. Keys: {list(entry.keys())}"
    )


def _seg_out_path(src_path: str, output_dir: Path, input_dir: Path | None) -> Path:
    """
    Compute where to save the seg-ID PNG for src_path, mirroring the input tree.

    When input_dir is given the relative folder structure is preserved:
        input_dir/000417/raw_image.jpg  ->  output_dir/000417/raw_image.png
    This avoids stem collisions when all images share the same filename
    (every scene here is named 'raw_image.jpg', so the sub-folder IS the identity).
    Same logic as depth's _depth_out_path — kept identical because the folder
    layout is identical; only the variable names differ.
    """
    src = Path(src_path)
    if input_dir is not None:
        try:
            rel = src.relative_to(input_dir)
            return output_dir / rel.parent / (rel.stem + ".png")
        except ValueError:
            pass
    return output_dir / (src.stem + ".png")


def precompute_segmentation_maps(
    image_paths: list,
    output_dir: Path,
    size: int = 512,
    batch_size: int = 4,
    model_name: str = DEFAULT_SEG_MODEL,
    device: str = "cuda",
    resize_mode: str = "letterbox",
    skip_existing: bool = True,
    input_dir: Path = None,
    local_files_only: bool = True,
    out_path_fn=None,
) -> dict:
    """
    Core routine: segment a list of images, save each as a raw class-ID PNG.

    The "segmentation" in the name satisfies the visual-identity rule: you can
    tell at a glance this function belongs to the segmentation pipeline.

    Returns: dict {image path (str) -> saved seg PNG path (str)} for every image
    that was processed or found already cached.

    Notes:
      • RGB preprocessing uses build_seg_square_preprocess() — the SHARED function
        from src/data/transforms.py — so it is byte-identical to what
        seg_inference.py will use. This is what guarantees train/inference parity.
      • encoder.label_ids() returns raw class IDs; we save them as PIL mode "L"
        (8-bit grayscale, values 0..num_classes-1). Colour palette is applied
        later, at load time, by SegJsonDataset._load_seg_colormap().
      • local_files_only: True = load strictly from local disk (offline); the b5
        checkpoint must be in checkpoints/local_models/segformer-b5-cityscapes.
        False = allow downloading from HF hub on first use.
      • out_path_fn: optional callable(image_path: Path) -> Path. When given, it
        FULLY decides where each seg PNG is saved (overrides output_dir/input_dir).
        Used by dataset-scan mode to save the map IN the image's own folder as
        <folder>_seg_map.png. When None, the default _seg_out_path placement is used.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # One place decides the output path for an image, honouring out_path_fn.
    def _resolve_out(p):
        return out_path_fn(Path(p)) if out_path_fn is not None else _seg_out_path(p, output_dir, input_dir)

    # ---- Load the SAME encoder used at live inference ----------------------- #
    # Using SegmentationEncoder (not SegformerForSemanticSegmentation directly)
    # guarantees the offline maps and the live encoder share _predict_ids() byte-
    # for-byte — the train/inference parity guarantee.
    print(f"\nLoading segmentation encoder: {model_name}")
    print(f"  (local_files_only={local_files_only})")
    encoder = SegmentationEncoder(
        size=size, model=model_name, local_files_only=local_files_only
    )
    encoder = encoder.to(device).eval()
    num_classes = encoder.num_classes
    print(f"Segmentation encoder ready ({num_classes} classes).\n")

    # ---- SHARED RGB preprocessing ----------------------------------------- #
    # build_seg_square_preprocess is the SINGLE SOURCE OF TRUTH for squaring.
    # Using it here (offline calc) AND in seg_inference.py (live inference) is
    # what guarantees the seg map the network sees at inference matches the
    # map saved for training. Do not inline this or duplicate it.
    preprocess = build_seg_square_preprocess(size=size, resize_mode=resize_mode)

    path_to_seg = {}
    processed = skipped = errors = 0

    pbar = tqdm(range(0, len(image_paths), batch_size), desc="Computing seg maps")
    for batch_start in pbar:
        batch_paths = image_paths[batch_start: batch_start + batch_size]

        # Cache-hit check: if the output PNG already exists, reuse it.
        to_process = []
        for p in batch_paths:
            out = _resolve_out(p)
            if skip_existing and out.exists():
                path_to_seg[str(p)] = str(out)
                skipped += 1
            else:
                to_process.append(p)
        if not to_process:
            continue

        tensors, valid_paths = [], []
        for p in to_process:
            # Split into two try/except blocks so a failure tells us WHICH step
            # broke: loading the file (Pillow) vs. preprocessing it
            # (build_seg_square_preprocess's SquarePad/Resize/ToTensor/Normalize
            # chain). The old single try block blamed every failure on "could
            # not load image" even when the real failure was in preprocess().
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                p_obj = Path(p)
                exists = p_obj.exists()
                print(f"\n[WARN] Image LOAD failed (Pillow could not open/decode the file) — tried:\n"
                      f"         {p}\n"
                      f"         reason: {e}\n"
                      f"         path.exists() = {exists}"
                      + ("" if exists else "  -> the path itself is wrong (case, mount point, or "
                         "missing --image_root — depending on which mode you're running)")
                      + (f"\n         file size = {p_obj.stat().st_size} bytes"
                         "  -> 0 bytes or a tiny file usually means a broken symlink or failed copy"
                         if exists else ""))
                errors += 1
                continue

            try:
                tensors.append(preprocess(img))      # [3, size, size] in [-1,1]
                valid_paths.append(p)
            except Exception as e:
                print(f"\n[WARN] Image PREPROCESS failed (SquarePad/Resize/ToTensor/Normalize) "
                      f"— the file loaded fine, the transform chain threw — path:\n"
                      f"         {p}\n"
                      f"         image size/mode: {img.size} {img.mode}\n"
                      f"         reason: {e}")
                errors += 1
        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)   # [B, 3, size, size]
        with torch.no_grad():
            # label_ids -> [B, size, size] long ids in [0, num_classes-1]
            # This calls _predict_ids(), the SAME path the live encoder uses.
            id_maps = encoder.label_ids(batch_tensor)

        for i, src_path in enumerate(valid_paths):
            ids_np = id_maps[i].to(torch.uint8).cpu().numpy()   # [size, size] uint8

            # Self-check per image: no id may exceed the class table. A bad id
            # here would silently corrupt colourisation at load — fail loudly.
            if ids_np.max() >= num_classes:
                raise ValueError(
                    f"{Path(src_path).name}: predicted id {ids_np.max()} >= "
                    f"num_classes {num_classes}. Encoder/model mismatch."
                )

            out = _resolve_out(src_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            # "L" mode = 8-bit single channel. Values ARE class ids, not brightness.
            Image.fromarray(ids_np, mode="L").save(out)

            path_to_seg[str(src_path)] = str(out)
            processed += 1

        pbar.set_postfix(processed=processed, skipped=skipped, errors=errors)

    print(f"\n{'='*52}")
    print(f"  Processed : {processed} images")
    print(f"  Skipped   : {skipped}   (already existed)")
    print(f"  Errors    : {errors}")
    print(f"  Output    : {output_dir}")
    print(f"{'='*52}\n")
    return path_to_seg


def _verify_segmentation_training_jsonl(
    jsonl_path: Path, num_classes: int
) -> tuple[int, int]:
    """
    Verify every entry in a seg-training output JSONL file.

    The "segmentation" in the name satisfies the visual-identity rule.

    For each entry checks:
      1. All three required keys present: raw_image_path, seg_path, prompt.
      2. seg_path exists on disk (the PNG was actually written).
      3. seg_path.stem == raw_image_path.stem — guards against any index
         mismatch that would silently pair the wrong seg map with an image.
      4. The PNG opens, is single-channel, and contains only valid class ids
         (< num_classes). This is seg-specific: continuous depth doesn't need
         this check; discrete class IDs can go out of range silently.

    Returns (n_passed, n_failed) and prints a one-line PASS/FAIL summary.
    """
    cwd = Path.cwd()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    passed = failed = 0
    for i, entry in enumerate(entries):
        missing = [k for k in ("raw_image_path", "seg_path", "prompt")
                   if k not in entry]
        if missing:
            print(f"  [FAIL] entry {i}: missing keys {missing}")
            failed += 1
            continue

        raw_p = Path(entry["raw_image_path"])
        seg_p = Path(entry["seg_path"])

        # seg PNG must exist on disk.
        abs_seg = seg_p if seg_p.is_absolute() else cwd / seg_p
        if not abs_seg.exists():
            print(f"  [FAIL] entry {i}: seg PNG not on disk: {seg_p}")
            failed += 1
            continue

        # Stem must match (raw_image.jpg -> raw_image.png).
        if raw_p.stem != seg_p.stem:
            print(f"  [FAIL] entry {i}: stem mismatch — "
                  f"raw={raw_p.stem!r}  seg={seg_p.stem!r}")
            failed += 1
            continue

        # Seg-specific content check: PNG must be valid labels in [0, num_classes).
        try:
            arr = np.asarray(Image.open(abs_seg).convert("L"))
        except Exception as e:
            print(f"  [FAIL] entry {i}: cannot open seg PNG {seg_p}: {e}")
            failed += 1
            continue
        if arr.max() >= num_classes:
            print(f"  [FAIL] entry {i}: seg id {arr.max()} >= num_classes {num_classes}")
            failed += 1
            continue

        passed += 1

    status = "PASS" if failed == 0 else "FAIL"
    print(f"  [{status}] {jsonl_path.name}: {passed}/{len(entries)} entries valid"
          + (f", {failed} FAILED" if failed else ""))
    return passed, failed


def _find_split_jsonl(data_dir: Path, split: str) -> "Path | None":
    """
    Locate the JSONL file for a given split in data_dir.
    Exact match first (data_dir/{split}.jsonl), then any *.jsonl whose stem
    contains the split name; among those pick the shortest stem.
    """
    exact = data_dir / f"{split}.jsonl"
    if exact.exists():
        return exact
    candidates = sorted(
        [p for p in data_dir.glob("*.jsonl") if split.lower() in p.stem.lower()],
        key=lambda p: len(p.stem),
    )
    return candidates[0] if candidates else None


def build_segmentation_training_jsons(
    data_dir: Path,
    raw_dir: Path,
    seg_dir: Path,
    output_dir: Path,
    size: int = 512,
    batch_size: int = 4,
    model_name: str = DEFAULT_SEG_MODEL,
    device: str = "cuda",
    resize_mode: str = "letterbox",
    skip_existing: bool = True,
    subset_n: int = None,
    local_files_only: bool = True,
    image_path: str = "source",
    image_root: Path = None,
) -> None:
    """
    Build data/seg_training/{train,val,test}.jsonl from data/{train,val,test}.jsonl.

    The "segmentation" in the name satisfies the visual-identity rule.

    Mirrors build_depth_training_jsons() exactly in CONTROL FLOW (so depth and
    seg manifests stay structurally identical and comparable), differing only in
    that it computes/saves segmentation IDs and writes a 'seg_path' field.

    The critical no-mismatch rule (same as depth): output entries are built in a
    SINGLE pass over zip(source_entries, abs_image_paths) — each output entry is
    assembled from its OWN source entry, never by independently iterating the
    path_to_seg dict (whose iteration order is not guaranteed to match entry order).

    Args:
      data_dir         : folder with train.jsonl / val.jsonl / test.jsonl (e.g. data/).
                         Uses _find_split_jsonl() so non-standard names are supported.
      raw_dir          : root of the raw image tree (data/raw/) — used to mirror the
                         sub-folder structure into seg_dir (parity with depth).
      seg_dir          : where seg-ID PNGs are saved (data/raw_seg/).
      output_dir       : where the three-field JSONLs are written (data/seg_training/).
      subset_n         : if set, only the first N entries per JSONL (dry run).
      local_files_only : passed through to precompute_segmentation_maps -> encoder.
    """
    cwd         = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir_abs = raw_dir if raw_dir.is_absolute() else cwd / raw_dir

    # num_classes for verification: derive from the palette constant (19 for Cityscapes).
    num_classes = len(SEG_CITYSCAPES_PALETTE)

    for split in ["train", "val", "test"]:
        src_path = _find_split_jsonl(data_dir, split)
        if src_path is None:
            print(f"\n[WARN] No JSONL for split '{split}' found in {data_dir} — skipping.")
            continue

        with open(src_path, "r", encoding="utf-8") as f:
            all_entries = [json.loads(line) for line in f if line.strip()]
        entries = all_entries[:subset_n] if subset_n else all_entries

        print(f"\n{'='*56}")
        print(f"  {src_path.name}: processing {len(entries)} entries"
              + (f" (dry-run subset of {len(all_entries)})" if subset_n else ""))
        print(f"{'='*56}")

        # Step 1: absolute image paths, built in LOCKSTEP with `entries`.
        # Priority: image_root / relative_path  >  cwd / relative_path
        # If --image_root /mnt/data is given and JSONL says "images/foo.jpg",
        # the actual path becomes /mnt/data/images/foo.jpg.
        _root = image_root if image_root is not None else cwd
        abs_image_paths = []
        for entry in entries:
            p     = Path(_get_image_path(entry, image_path))
            abs_p = p if p.is_absolute() else _root / p
            abs_image_paths.append(abs_p)
            if not abs_p.exists():
                print(f"  [WARN] Image not found on disk: {abs_p}")

        # Step 2: compute / look up seg PNGs (mirrors data/raw/ tree into seg_dir).
        path_to_seg = precompute_segmentation_maps(
            image_paths=[str(p) for p in abs_image_paths],
            output_dir=seg_dir,
            size=size,
            batch_size=batch_size,
            model_name=model_name,
            device=device,
            resize_mode=resize_mode,
            skip_existing=skip_existing,
            input_dir=raw_dir_abs,
            local_files_only=local_files_only,
        )

        # Step 3: build output entries — ONE PASS, each atomic from its source.
        # zip(entries, abs_image_paths) is the ONLY correct pattern: iterating
        # path_to_seg separately would risk a silent index mismatch.
        out_entries = []
        n_skipped   = 0
        for entry, abs_img_p in zip(entries, abs_image_paths):
            seg_abs_str = path_to_seg.get(str(abs_img_p))
            if seg_abs_str is None:
                print(f"  [WARN] No seg result for {abs_img_p} — entry skipped.")
                n_skipped += 1
                continue

            raw_rel = Path(_get_image_path(entry, image_path)).as_posix()
            try:
                seg_rel = Path(seg_abs_str).relative_to(cwd).as_posix()
            except ValueError:
                seg_rel = Path(seg_abs_str).as_posix()

            out_entries.append({
                "raw_image_path": raw_rel,
                "seg_path":       seg_rel,
                "prompt":         entry["prompt"],
            })

        # Step 4: write output JSONL (one object per line).
        out_path = output_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for entry in out_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        skip_note = f"  ({n_skipped} skipped due to seg errors)" if n_skipped else ""
        print(f"\n  Written {len(out_entries)} entries -> {out_path}{skip_note}")

        # Step 5: verify every entry (loud PASS/FAIL).
        _verify_segmentation_training_jsonl(out_path, num_classes=num_classes)


# ============================================================================ #
#  DATASET-SCAN MODE (segmentation)                                            #
#  Mirrors depth_map_calculations.py exactly: find raw_image.jpg by SCANNING,  #
#  save each seg map into a SIBLING folder (<dataset_dir>_seg_map/) that       #
#  MIRRORS the internal folder structure; then rebuild the split manifests    #
#  from the originals so every prompt + split assignment stays exactly mapped.#
# ============================================================================ #

def _sibling_map_path(
    image_path: Path, dataset_dir: Path, sibling_root: Path,
    suffix: str, ext: str = ".png",
) -> Path:
    """
    Given an image at    <dataset_dir>/000417/raw_image.jpg
    and sibling_root  =  <dataset_dir's parent>/<dataset_dir.name><suffix>
                         (e.g. .../custome_dataset_seg_map)
    return the map path  <sibling_root>/000417/000417<suffix><ext>

    The output tree MIRRORS dataset_dir's internal structure, living as a
    SIBLING of dataset_dir — the original dataset folder is never written into.
    """
    rel_dir     = image_path.parent.relative_to(dataset_dir)
    folder_name = image_path.parent.name
    return sibling_root / rel_dir / (folder_name + suffix + ext)


def _verify_scan_seg_training_jsonl(jsonl_path: Path, map_key: str, suffix: str) -> tuple[int, int]:
    """
    Verify a scan-mode seg output JSONL. For each entry checks:
      1. Required keys present: raw_image_path, <map_key>, prompt.
      2. The map file exists on disk.
      3. The map lives in a SIBLING tree, NOT inside the image's own folder.
      4. The map's leaf folder name matches the image's leaf folder name
         (mirrored structure) and the map is named <folder_name><suffix>.png.
    (Per-pixel class-ID validity is already enforced at write time inside
    precompute_segmentation_maps, so it is not re-checked here.)
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    passed = failed = 0
    for i, entry in enumerate(entries):
        missing = [k for k in ("raw_image_path", map_key, "prompt") if k not in entry]
        if missing:
            print(f"  [FAIL] entry {i}: missing keys {missing}")
            failed += 1
            continue

        raw_p = Path(entry["raw_image_path"])
        map_p = Path(entry[map_key])

        if not map_p.exists():
            print(f"  [FAIL] entry {i}: map file not on disk: {map_p}")
            failed += 1
            continue

        if map_p.parent == raw_p.parent:
            print(f"  [FAIL] entry {i}: map was written INTO the source dataset "
                  f"folder (expected a sibling tree): {map_p.parent}")
            failed += 1
            continue

        if map_p.parent.name != raw_p.parent.name:
            print(f"  [FAIL] entry {i}: mirrored leaf folder name mismatch — "
                  f"image={raw_p.parent.name!r}  map={map_p.parent.name!r}")
            failed += 1
            continue

        expected = raw_p.parent.name + suffix + ".png"
        if map_p.name != expected:
            print(f"  [FAIL] entry {i}: map name {map_p.name!r} != expected {expected!r}")
            failed += 1
            continue

        passed += 1

    status = "PASS" if failed == 0 else "FAIL"
    print(f"  [{status}] {jsonl_path.name}: {passed}/{len(entries)} entries valid"
          + (f", {failed} FAILED" if failed else ""))
    return passed, failed


def build_seg_training_from_scan(
    dataset_dir: Path,
    data_dir: Path,
    output_dir: Path,
    image_name: str = "raw_image.jpg",
    image_path: str = "target",
    size: int = 512,
    batch_size: int = 4,
    model_name: str = DEFAULT_SEG_MODEL,
    device: str = "cuda",
    resize_mode: str = "letterbox",
    skip_existing: bool = True,
    local_files_only: bool = True,
    subset_n: int = None,
) -> None:
    """
    DATASET-SCAN MODE for segmentation. Mirrors
    depth_map_calculations.build_depth_training_from_scan step for step:

    1. SCAN dataset_dir recursively for files named exactly `image_name`
       (default 'raw_image.jpg'). Every OTHER file is ignored, so the source
       dataset can freely contain other images per folder.
    2. Compute a seg map for each and save it into a SIBLING folder that
       MIRRORS dataset_dir's internal structure — the source dataset folder
       itself is NEVER written into:
           dataset_dir  = .../custome_dataset
           sibling_root = .../custome_dataset_seg_map
           .../custome_dataset/000417/raw_image.jpg
             -> .../custome_dataset_seg_map/000417/000417_seg_map.png
    3. Read the original split manifests (train/val/test .jsonl) in data_dir to
       recover each image's PROMPT and SPLIT, matching by ABSOLUTE image path.
    4. Write data/seg_training/{train,val,test}.jsonl with ABSOLUTE
       raw_image_path + seg_path + prompt. Verify each file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sibling output tree: a NEW folder next to dataset_dir, same name + suffix.
    sibling_root = dataset_dir.parent / (dataset_dir.name + "_seg_map")
    sibling_root.mkdir(parents=True, exist_ok=True)
    print(f"Seg maps will be saved to (sibling, mirrored structure): {sibling_root}")

    # ---- Step 1: SCAN for the target filename only ------------------------- #
    print(f"\nScanning for '{image_name}' under: {dataset_dir}")
    found = sorted(dataset_dir.rglob(image_name))
    if subset_n:
        found = found[:subset_n]
    if not found:
        print(f"[ERROR] No '{image_name}' found under {dataset_dir}. Nothing to do.")
        return
    print(f"Found {len(found)} '{image_name}' file(s)"
          + (f"  (dry-run: scan capped at first {subset_n})" if subset_n else "") + ".")

    # ---- Step 2: compute seg, saving to the mirrored sibling tree ---------- #
    out_fn = lambda p: _sibling_map_path(Path(p), dataset_dir, sibling_root, "_seg_map")
    path_to_seg = precompute_segmentation_maps(
        image_paths=[str(p) for p in found],
        output_dir=output_dir,            # not used for saving (out_fn overrides)
        size=size,
        batch_size=batch_size,
        model_name=model_name,
        device=device,
        resize_mode=resize_mode,
        skip_existing=skip_existing,
        local_files_only=local_files_only,
        out_path_fn=out_fn,
    )

    # ---- Step 3: index computed seg maps by NORMALISED absolute image path - #
    seg_by_img = {
        os.path.normcase(str(Path(img).resolve())): seg
        for img, seg in path_to_seg.items()
    }

    # ---- Step 4: rebuild each split manifest, matching by absolute path ---- #
    for split in ("train", "val", "test"):
        src = _find_split_jsonl(data_dir, split)
        if src is None:
            print(f"\n[WARN] No JSONL for split '{split}' in {data_dir} — skipping.")
            continue

        with open(src, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]

        out_entries = []
        n_skipped = 0
        for entry in entries:
            raw_str = _get_image_path(entry, image_path)
            key = os.path.normcase(str(Path(raw_str).resolve()))
            seg = seg_by_img.get(key)
            if seg is None:
                n_skipped += 1
                continue
            out_entries.append({
                "raw_image_path": Path(raw_str).resolve().as_posix(),
                "seg_path":       Path(seg).resolve().as_posix(),
                "prompt":         entry["prompt"],
            })

        out_path = output_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for e in out_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        if subset_n:
            note = f"  (dry-run: {n_skipped} entries not in the scanned subset — expected)"
        else:
            note = f"  ({n_skipped} entries had no seg on disk — skipped)" if n_skipped else ""
        print(f"\n  Written {len(out_entries)} entries -> {out_path}{note}")
        _verify_scan_seg_training_jsonl(out_path, "seg_path", "_seg_map")


def run_seg_directory_mode(args):
    """
    DIRECTORY MODE: scan --input_dir recursively for images, segment them, save
    class-ID PNGs preserving the relative folder structure. Use when you just want
    the label maps without (re)building the training JSONs.

    The "seg" prefix marks this as segmentation-pipeline code.
    """
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir
        else input_dir.parent / "raw_seg"
    )

    valid_exts  = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    )
    if not image_paths:
        print(f"[ERROR] No images found under: {input_dir}")
        return

    print(f"Found {len(image_paths)} images under: {input_dir}")
    precompute_segmentation_maps(
        image_paths=[str(p) for p in image_paths],
        output_dir=output_dir,
        size=args.size,
        batch_size=args.batch_size,
        model_name=args.model,
        device=args.device,
        resize_mode=args.resize_mode,
        skip_existing=not args.no_skip,
        input_dir=input_dir,
        local_files_only=args.local_files_only,
    )
    print("\nNext step — build training JSONLs with --data_dir, then seg_training.py")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute SegFormer-b5-Cityscapes segmentation label maps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Folder with train.jsonl/val.jsonl/test.jsonl (e.g. data/). "
             "Any *.jsonl whose stem contains 'train'/'val'/'test' is also accepted. "
             "Activates seg-training JSONL mode -> data/seg_training/*.jsonl.",
    )
    parser.add_argument(
        "--raw_dir", type=str, default=None,
        help="Root of raw image tree (default: <data_dir>/raw).",
    )
    parser.add_argument(
        "--seg_dir", type=str, default=None,
        help="Where to save seg-ID PNGs (default: <data_dir>/raw_seg).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output dir for the JSONLs (default: <data_dir>/seg_training) "
             "or for PNGs in --input_dir mode.",
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory-mode: scan this folder for images instead of using JSONs.",
    )
    parser.add_argument(
        "--dry_run_n", type=int, default=None,
        help="Process only the first N entries per JSONL (sanity run before full dataset).",
    )
    parser.add_argument(
        "--size", type=int, default=512,
        help="Square size for seg maps. Default 512 (matches cfg.size).",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--model", type=str,
        default="checkpoints/local_models/segformer-b5-cityscapes",
        help=(
            "Seg model: local folder path (default, offline) OR a HF id like "
            f"{DEFAULT_SEG_MODEL} (use with --local_files_only False). "
            "Default: checkpoints/local_models/segformer-b5-cityscapes"
        ),
    )
    parser.add_argument(
        "--local_files_only", type=parse_bool, default=True,
        metavar="True|False",
        help=(
            "True (default) = load --model strictly from local disk (offline). "
            "False = allow download into checkpoints/local_models/. "
            "On first use, run with --local_files_only False to download the b5 model."
        ),
    )
    parser.add_argument(
        "--resize_mode", type=str, default="letterbox",
        choices=["letterbox", "stretch"],
        help="Squaring before the encoder (references.md §5). Default: letterbox.",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no_skip", action="store_true",
        help="Re-compute even if a seg PNG already exists.",
    )
    parser.add_argument(
        "--image_path", type=str, default="source",
        help="Key in your input JSONLs that holds the image path. "
             "Default: 'source'. Change to 'target' (or any other key) "
             "if your manifests use a different name. "
             "Example: --image_path target",
    )
    parser.add_argument(
        "--image_root", type=str, default=None,
        help="Root folder prepended to RELATIVE image paths from the JSONL. "
             "Use this when images are NOT under the repo root. "
             "Example: --image_root /mnt/dataset  (then JSONL path "
             "'images/foo.jpg' resolves to /mnt/dataset/images/foo.jpg). "
             "Absolute paths in the JSONL are always used as-is.",
    )

    # ---- Dataset-SCAN mode (find raw_image.jpg by scanning a folder) ------- #
    parser.add_argument(
        "--dataset_dir", type=str, default=None,
        help="Activates DATASET-SCAN mode. Recursively scans this folder for files "
             "named --image_name (default raw_image.jpg), computes a seg map for each, "
             "and saves it IN the image's own folder as <folder>_seg_map.png. Requires "
             "--data_dir (the folder holding train/val/test.jsonl) to recover each "
             "image's prompt + split. "
             "Example: --dataset_dir /data/custome_dataset --data_dir data/ --image_path target",
    )
    parser.add_argument(
        "--image_name", type=str, default="raw_image.jpg",
        help="(scan mode) Exact filename to process in each folder. Every other file "
             "is ignored — including *_depth_map.png / *_seg_map.png this pipeline writes. "
             "Default: raw_image.jpg",
    )
    args = parser.parse_args()

    if not args.dataset_dir and not args.data_dir and not args.input_dir:
        parser.error(
            "Provide one of:\n"
            "  --dataset_dir /data/custome_dataset --data_dir data/  (scan mode — saves maps in-folder)\n"
            "  --data_dir data/   (builds data/seg_training/*.json from JSONL paths)\n"
            "  --input_dir data/raw   (directory mode — PNGs only, no JSON)"
        )
    if args.data_dir and args.input_dir:
        parser.error("--data_dir and --input_dir are mutually exclusive.")
    if args.dataset_dir and not args.data_dir:
        parser.error(
            "--dataset_dir (scan mode) also needs --data_dir pointing at the folder that "
            "holds train/val/test.jsonl — that is where each image's prompt + split come from.\n"
            "  Example: --dataset_dir /data/custome_dataset --data_dir data/ --image_path target"
        )

    # Belt-and-suspenders offline lock: set env vars so NOTHING touches the network,
    # on top of local_files_only=True being passed to from_pretrained.
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"]      = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print(f"Device           : {args.device}")
    print(f"Size             : {args.size}x{args.size}   resize_mode: {args.resize_mode}")
    print(f"Model            : {args.model}")
    print(f"local_files_only : {args.local_files_only}")

    if args.dataset_dir:
        # DATASET-SCAN mode: find raw_image.jpg by scanning, save maps in-folder,
        # rebuild seg_training/{train,val,test}.jsonl from the original splits.
        dataset_dir = Path(args.dataset_dir).resolve()
        data_dir    = Path(args.data_dir).resolve()
        out_dir     = Path(args.output_dir).resolve() if args.output_dir else data_dir / "seg_training"

        if not dataset_dir.exists():
            parser.error(f"--dataset_dir not found: {dataset_dir}")
        if args.dry_run_n:
            print(f"\n[DRY RUN] Scan capped at first {args.dry_run_n} images.")

        build_seg_training_from_scan(
            dataset_dir=dataset_dir,
            data_dir=data_dir,
            output_dir=out_dir,
            image_name=args.image_name,
            image_path=args.image_path,
            size=args.size,
            batch_size=args.batch_size,
            model_name=args.model,
            device=args.device,
            resize_mode=args.resize_mode,
            skip_existing=not args.no_skip,
            local_files_only=args.local_files_only,
            subset_n=args.dry_run_n,
        )

    elif args.data_dir:
        data_dir  = Path(args.data_dir).resolve()
        raw_dir   = Path(args.raw_dir).resolve()   if args.raw_dir  else data_dir / "raw"
        seg_dir   = Path(args.seg_dir).resolve()   if args.seg_dir  else data_dir / "raw_seg"
        # output_dir: data/seg_training/ — mirrors depth's data/depth_training/ layout.
        out_dir   = Path(args.output_dir).resolve() if args.output_dir else data_dir / "seg_training"
        if args.dry_run_n:
            print(f"\n[DRY RUN] First {args.dry_run_n} entries per JSON.")
        image_root = Path(args.image_root).resolve() if args.image_root else None
        if image_root:
            print(f"Image root : {image_root}  (prepended to relative image paths)")
        build_segmentation_training_jsons(
            data_dir=data_dir, raw_dir=raw_dir if args.raw_dir else (image_root or data_dir / "raw"),
            seg_dir=seg_dir, output_dir=out_dir,
            size=args.size, batch_size=args.batch_size, model_name=args.model,
            device=args.device, resize_mode=args.resize_mode,
            skip_existing=not args.no_skip, subset_n=args.dry_run_n,
            local_files_only=args.local_files_only,
            image_path=args.image_path,
            image_root=image_root,
        )
    else:
        run_seg_directory_mode(args)


if __name__ == "__main__":
    main()

"""
depth_map_calculations.py
-------------------
Pre-compute and CACHE depth maps for training images using Intel DPT-Hybrid MiDaS.

WHY WE NEED THIS:
  The depth estimator (DPT transformer) is slow to run at every training step.
  Since training images don't change, we compute depths ONCE and save them as PNGs.
  Training then loads PNGs directly — no DPT overhead per step.

TWO INPUT MODES:
  1. DIRECTORY MODE: scan --input_dir, save depths to --output_dir
     Then auto-updates any *.json files found beside the image folder.

  2. JSON MODE: read --json_file (each entry has "raw_image_path" + "prompt"),
     compute depths, ADD "depth_path" field to each entry, save updated JSON.

JSON MANIFEST FORMAT:
  [
    {"raw_image_path": "data/images/cat.jpg",  "prompt": "a cute cat on a chair"},
    {"raw_image_path": "data/images/dog.jpg",  "prompt": "a brown dog on grass"}
  ]
  After running: "depth_path" key is added to each entry automatically.

TYPICAL WORKFLOW (--data_dir mode — recommended):
  # Step 1: compute depths for all images, builds depth_training/train+val+test.jsonl
  python depth_map_calculations.py --data_dir data/
  # Step 2: train
  python depth_training.py experiment=train_depth

ALTERNATIVE (JSON mode — when you already have a single JSON file):
  python depth_map_calculations.py --json_file data/train.jsonl --output_dir data/depths

QUICK COMMANDS (run from repo root with conda loradapter env active):
  # --- Dry run: 15 images, verify pipeline before committing to full dataset ---
  python depth_map_calculations.py --data_dir data/ --dry_run_n 15

  # --- Full run: all images (639 train + 137 val + 137 test) ---
  python depth_map_calculations.py --data_dir data/

  # --- Re-compute everything (force overwrite of existing PNGs) ---
  python depth_map_calculations.py --data_dir data/ --no_skip

  ###python depth_map_calculations.py --dataset_dir D:/MyWorkplace/WorkStation/custome_dataset --data_dir data/ --image_path target
"""

import argparse
import json
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.annotators.midas import DepthEstimator
from src.data.transforms import SquarePad


def parse_bool(value):
    """Parse True/False (any case, plus 1/0, yes/no) from the CLI into a bool."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"--local_files_only expects True or False, got: {value!r}")


# ============================================================================ #
#  HELPER: read image path from a JSON entry regardless of which key it uses   #
# ============================================================================ #

def _get_image_path(entry: dict, image_path: str = "source") -> str:
    """
    Return the image path string from a JSONL entry.

    Checks `image_path` first (default "source", override with --image_path).
    Falls back to "raw_image_path" so depth_training/ JSONLs also work.
    Change the key your manifests use without touching any other code:
        python depth_map_calculations.py --data_dir data/ --image_path target
    """
    if image_path in entry:
        return entry[image_path]
    if "raw_image_path" in entry:
        return entry["raw_image_path"]
    raise KeyError(
        f"Entry has neither '{image_path}' nor 'raw_image_path' key. "
        f"Keys present: {list(entry.keys())}"
    )


def _depth_out_path(src_path: str, output_dir: Path, input_dir: Path | None) -> Path:
    """
    Compute where to save the depth PNG for src_path.

    When input_dir is given the relative folder structure is preserved:
        input_dir/A/scene_001/img.jpg  →  output_dir/A/scene_001/img.png
    This avoids stem collisions when many images share the same filename
    (e.g. all named 'original_sample_img.jpg' in different sub-folders).
    """
    src = Path(src_path)
    if input_dir is not None:
        try:
            rel = src.relative_to(input_dir)
            return output_dir / rel.parent / (rel.stem + ".png")
        except ValueError:
            pass
    return output_dir / (src.stem + ".png")


def precompute_depths_from_paths(
    image_paths: list,
    output_dir: Path,
    size: int = 512,
    batch_size: int = 4,
    model_name: str = "Intel/dpt-hybrid-midas",
    device: str = "cuda",
    skip_existing: bool = True,
    input_dir: Path = None,
    local_files_only: bool = False,
    out_path_fn=None,
) -> dict:
    """
    Core function: compute depth maps for a list of image paths, save as grayscale PNGs.

    input_dir: when provided, depth PNGs are saved preserving the relative folder
               structure under output_dir (avoids stem collisions with nested datasets).
    local_files_only: True = load the depth model strictly from local disk (offline,
               no network); pass a local folder path as `model_name` for this.
               False = allow downloading the model (into checkpoints/local_models).
    out_path_fn: optional callable(image_path: Path) -> Path. When given, it FULLY
               decides where each depth PNG is saved (overrides output_dir/input_dir
               logic). Used by dataset-scan mode to save the map IN the image's own
               folder as <folder>_depth_map.png. When None, the default
               _depth_out_path(output_dir, input_dir) placement is used.

    Returns:
        dict mapping original image path → saved depth PNG path
        (only includes entries that were actually processed)
    """
    # One place decides the output path for an image, honouring out_path_fn.
    def _resolve_out(p):
        return out_path_fn(Path(p)) if out_path_fn is not None else _depth_out_path(p, output_dir, input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load DPT model -------------------------------------------------- #
    print(f"\nLoading depth estimator: {model_name}  (local_files_only={local_files_only})")
    depth_estimator = DepthEstimator(size=size, model=model_name, local_files_only=local_files_only)
    depth_estimator = depth_estimator.to(device).eval()
    print("Depth estimator ready.\n")

    # ---- Image preprocessing --------------------------------------------- #
    # DepthEstimator expects [B, 3, H, W] in [-1, 1].
    # resize_mode: letterbox — pad the shorter side to square with edge-
    # replication, THEN resize to (size × size).  This keeps the full image
    # frame intact (no content cropped) while satisfying the encoder's
    # requirement for a square input (references.md §5).
    #
    # PARITY-CRITICAL DUPLICATION WARNING: this exact chain is repeated in
    # THREE places and they MUST stay byte-for-byte identical, or training
    # data and live inference silently diverge:
    #   1. here (offline depth precompute, this file)
    #   2. depth_inference.py        (live inference preprocess)
    #   3. configs/data/local_depth.yaml  (training-time RGB transform)
    # If you change one, change all three. Parity is verified by the Step-4
    # cross-stage check (float-vs-float diff must be 0.0).
    preprocess = transforms.Compose([
        SquarePad(),                                       # shorter side → square (edge-replicated)
        transforms.Resize((size, size)),                   # square → size × size
        transforms.ToTensor(),                             # [0,255] → [0,1]
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # [0,1] → [-1,1]
    ])

    path_to_depth = {}   # result mapping: image path (str) → depth path (str)
    processed = skipped = errors = 0

    pbar = tqdm(range(0, len(image_paths), batch_size), desc="Computing depth maps")

    for batch_start in pbar:
        batch_paths = image_paths[batch_start : batch_start + batch_size]

        to_process = []
        for p in batch_paths:
            out = _resolve_out(p)
            if skip_existing and out.exists():
                path_to_depth[str(p)] = str(out)
                skipped += 1
            else:
                to_process.append(p)

        if not to_process:
            continue

        tensors, valid_paths = [], []
        for p in to_process:
            # Split into two try/except blocks so a failure tells us WHICH step
            # broke: loading the file (Pillow) vs. preprocessing it (SquarePad/
            # Resize/ToTensor/Normalize). The old single try block blamed every
            # failure on "could not load image" even when the real failure was
            # in preprocess() — misleading when the file is perfectly valid.
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
                tensors.append(preprocess(img))     # [3, H, W] in [-1, 1]
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

        batch_tensor = torch.stack(tensors).to(device)   # [B, 3, H, W]
        with torch.no_grad():
            # depth_maps: [B, 3, H, W] in [0, 1]  (all 3 channels identical)
            depth_maps = depth_estimator(batch_tensor)

        for i, src_path in enumerate(valid_paths):
            depth_single = depth_maps[i, 0]    # [H, W] in [0, 1]
            depth_np = (depth_single.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            out = _resolve_out(src_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(depth_np, mode="L").save(out)   # "L" = 8-bit grayscale

            path_to_depth[str(src_path)] = str(out)
            processed += 1

        pbar.set_postfix(processed=processed, skipped=skipped, errors=errors)

    print(f"\n{'='*52}")
    print(f"  Processed : {processed} images")
    print(f"  Skipped   : {skipped}   (already existed)")
    print(f"  Errors    : {errors}")
    print(f"  Output    : {output_dir}")
    print(f"{'='*52}\n")

    return path_to_depth


def _verify_depth_training_jsonl(jsonl_path: Path) -> tuple[int, int]:
    """
    Verify every entry in a depth-training output JSONL file.

    For each entry checks:
      1. All three required keys present: raw_image_path, depth_path, prompt.
      2. depth_path exists on disk (the PNG was actually written).
      3. depth_path.stem == raw_image_path.stem — guards against any index
         mismatch that would silently pair the wrong depth with an image.

    Returns (n_passed, n_failed) and prints a one-line PASS/FAIL summary
    plus details for any failing entries.
    """
    cwd = Path.cwd()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    passed = failed = 0
    for i, entry in enumerate(entries):

        # ---- check required keys ----------------------------------------- #
        missing = [k for k in ("raw_image_path", "depth_path", "prompt")
                   if k not in entry]
        if missing:
            print(f"  [FAIL] entry {i}: missing keys {missing}")
            failed += 1
            continue

        raw_p   = Path(entry["raw_image_path"])
        depth_p = Path(entry["depth_path"])

        # ---- depth PNG must exist on disk --------------------------------- #
        abs_depth = depth_p if depth_p.is_absolute() else cwd / depth_p
        if not abs_depth.exists():
            print(f"  [FAIL] entry {i}: depth PNG not on disk: {depth_p}")
            failed += 1
            continue

        # ---- stem must match (raw_image.jpg -> raw_image.png) ------------ #
        if raw_p.stem != depth_p.stem:
            print(f"  [FAIL] entry {i}: stem mismatch — "
                  f"raw={raw_p.stem!r}  depth={depth_p.stem!r}")
            failed += 1
            continue

        passed += 1

    status = "PASS" if failed == 0 else "FAIL"
    print(f"  [{status}] {jsonl_path.name}: {passed}/{len(entries)} entries valid"
          + (f", {failed} FAILED" if failed else ""))
    return passed, failed


def _find_split_jsonl(data_dir: Path, split: str) -> Path | None:
    """
    Locate the JSONL file for a given split ("train", "val", or "test") in data_dir.

    Search order:
      1. Exact name: data_dir/{split}.jsonl
      2. Any *.jsonl file whose stem contains the split name (case-insensitive).
         If multiple match, pick the shortest name (most specific).
    Returns the Path if found, else None.
    """
    exact = data_dir / f"{split}.jsonl"
    if exact.exists():
        return exact
    candidates = sorted(
        [p for p in data_dir.glob("*.jsonl") if split.lower() in p.stem.lower()],
        key=lambda p: len(p.stem),
    )
    return candidates[0] if candidates else None


def build_depth_training_jsons(
    data_dir: Path,
    raw_dir: Path,
    depth_dir: Path,
    output_dir: Path,
    size: int = 512,
    batch_size: int = 4,
    model_name: str = "Intel/dpt-hybrid-midas",
    device: str = "cuda",
    skip_existing: bool = True,
    subset_n: int = None,
    local_files_only: bool = False,
    image_path: str = "source",
    image_root: Path = None,
) -> None:
    """
    Create depth_training/{train,val,test}.jsonl from data/{train,val,test}.jsonl.

    For each source JSONL this function:
      1. Reads every entry's image path + prompt via _get_image_path() —
         handles both the 'source' key (existing data/ JSONLs) and the
         'raw_image_path' key (DepthJsonDataset format).
      2. Runs precompute_depths_from_paths() to compute (or look up cached)
         depth PNGs for all images in that JSONL.
      3. In a SINGLE PASS over zip(source_entries, abs_image_paths), builds
         each output entry from its own source entry — never from a separately
         sorted list that could silently drift in index.  Each output entry
         carries exactly three fields:
           raw_image_path  — forward-slash relative path to the source image
           depth_path      — forward-slash relative path to the depth PNG
           prompt          — text caption, copied verbatim
      4. Writes depth_training/<name>.jsonl (one JSON object per line).
      5. Calls _verify_depth_training_jsonl() and prints PASS/FAIL.

    Args:
        data_dir   : folder containing train.jsonl / val.jsonl / test.jsonl.
                     Uses _find_split_jsonl() to discover files whose names may
                     differ — any *.jsonl whose stem contains "train"/"val"/"test"
                     is accepted if the exact default name is absent.
        raw_dir    : root of the raw image tree (data/raw/).
                     Used by _depth_out_path() to mirror the folder structure
                     into depth_dir (data/raw/000417/img.jpg ->
                     depth_dir/000417/img.png).
        depth_dir  : where depth PNGs are saved (data/raw_depth/).
        output_dir : where the new three-field JSONLs are written
                     (default data/depth_training/).
        subset_n   : if set, only process the first N entries per JSONL —
                     use this for a dry run before committing to the full dataset.
    """
    cwd = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir_abs = raw_dir if raw_dir.is_absolute() else cwd / raw_dir

    for split in ["train", "val", "test"]:
        src_path = _find_split_jsonl(data_dir, split)
        if src_path is None:
            print(f"\n[WARN] No JSONL for split '{split}' found in {data_dir} — skipping.")
            continue

        with open(src_path, "r", encoding="utf-8") as f:
            all_entries = [json.loads(line) for line in f if line.strip()]

        # Optionally slice to the first N for a dry run
        entries = all_entries[:subset_n] if subset_n else all_entries

        print(f"\n{'='*56}")
        print(f"  {src_path.name}: processing {len(entries)} entries"
              + (f" (dry-run subset of {len(all_entries)} total)" if subset_n else ""))
        print(f"{'='*56}")

        # ---- Step 1: resolve absolute image paths in entry order ----------- #
        # Priority: image_root / relative_path  >  cwd / relative_path
        # If --image_root /mnt/data is given and JSONL says "images/foo.jpg",
        # the actual path becomes /mnt/data/images/foo.jpg.
        # Without --image_root, falls back to resolving relative to CWD as before.
        _root = image_root if image_root is not None else cwd
        abs_image_paths = []
        for entry in entries:
            raw_str = _get_image_path(entry, image_path)
            p = Path(raw_str)
            abs_p = p if p.is_absolute() else _root / p
            abs_image_paths.append(abs_p)
            if not abs_p.exists():
                print(f"  [WARN] Image not found on disk: {abs_p}")

        # ---- Step 2: compute / look up depth PNGs -------------------------- #
        # _depth_out_path() inside here mirrors data/raw/<id>/img.jpg ->
        # depth_dir/<id>/img.png, preserving the sub-folder structure.
        path_to_depth = precompute_depths_from_paths(
            image_paths=[str(p) for p in abs_image_paths],
            output_dir=depth_dir,
            size=size,
            batch_size=batch_size,
            model_name=model_name,
            device=device,
            skip_existing=skip_existing,
            input_dir=raw_dir_abs,
            local_files_only=local_files_only,
        )

        # ---- Step 3: build output entries — ONE PASS, atomic per source ---- #
        # zip(entries, abs_image_paths) is the only correct pattern here.
        # DO NOT iterate path_to_depth independently — its iteration order is
        # not guaranteed to match entry order, and any separate zip/merge would
        # be the exact silent-mismatch bug this design rules out.
        out_entries = []
        n_skipped = 0
        for entry, abs_img_p in zip(entries, abs_image_paths):
            depth_abs_str = path_to_depth.get(str(abs_img_p))

            if depth_abs_str is None:
                print(f"  [WARN] No depth result for {abs_img_p} — entry skipped.")
                n_skipped += 1
                continue

            # Normalize to forward-slash, relative to cwd
            raw_rel = Path(_get_image_path(entry, image_path)).as_posix()
            try:
                depth_rel = Path(depth_abs_str).relative_to(cwd).as_posix()
            except ValueError:
                # depth_abs_str is already relative or on a different drive
                depth_rel = Path(depth_abs_str).as_posix()

            # One entry, built atomically from its own source entry
            out_entries.append({
                "raw_image_path": raw_rel,
                "depth_path":     depth_rel,
                "prompt":         entry["prompt"],
            })

        # ---- Step 4: write output JSONL (one object per line) ------------- #
        out_path = output_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for entry in out_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        skip_note = f"  ({n_skipped} skipped due to depth errors)" if n_skipped else ""
        print(f"\n  Written {len(out_entries)} entries -> {out_path}{skip_note}")

        # ---- Step 5: verify every entry ------------------------------------ #
        _verify_depth_training_jsonl(out_path)


# ============================================================================ #
#  DATASET-SCAN MODE                                                           #
#  Find raw_image.jpg by SCANNING a dataset folder (not by trusting JSONL      #
#  paths); save each depth map into a SIBLING folder                          #
#  (<dataset_dir>_depth_map/) that MIRRORS the internal folder structure,      #
#  named after the leaf folder; then rebuild the split manifests from the     #
#  originals so every prompt + split assignment stays exactly mapped.         #
# ============================================================================ #

def _sibling_map_path(
    image_path: Path, dataset_dir: Path, sibling_root: Path,
    suffix: str, ext: str = ".png",
) -> Path:
    """
    Given an image at    <dataset_dir>/000417/raw_image.jpg
    and sibling_root  =  <dataset_dir's parent>/<dataset_dir.name><suffix>
                         (e.g. .../custome_dataset_depth_map)
    return the map path  <sibling_root>/000417/000417<suffix><ext>

    The output tree MIRRORS dataset_dir's internal structure (any nesting
    depth), living as a SIBLING of dataset_dir — the original dataset folder
    is never written into. The map filename is still based on the leaf folder
    name (not the fixed "raw_image" filename) so it stays unique and readable.
    """
    rel_dir     = image_path.parent.relative_to(dataset_dir)  # e.g. "000417" or "A/000417"
    folder_name = image_path.parent.name                      # e.g. "000417"
    return sibling_root / rel_dir / (folder_name + suffix + ext)


def _verify_scan_training_jsonl(jsonl_path: Path, map_key: str, suffix: str) -> tuple[int, int]:
    """
    Verify a scan-mode output JSONL. For each entry checks:
      1. Required keys present: raw_image_path, <map_key>, prompt.
      2. The map file exists on disk.
      3. The map lives in a SIBLING tree, NOT inside the image's own folder
         (the source dataset folder must never be written into).
      4. The map's leaf folder name matches the image's leaf folder name
         (mirrored structure) and the map is named <folder_name><suffix>.png.
    Returns (n_passed, n_failed) and prints a one-line PASS/FAIL summary.
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


def build_depth_training_from_scan(
    dataset_dir: Path,
    data_dir: Path,
    output_dir: Path,
    image_name: str = "raw_image.jpg",
    image_path: str = "target",
    size: int = 512,
    batch_size: int = 4,
    model_name: str = "Intel/dpt-hybrid-midas",
    device: str = "cuda",
    skip_existing: bool = True,
    local_files_only: bool = False,
    subset_n: int = None,
) -> None:
    """
    DATASET-SCAN MODE (Stage A, robust variant).

    1. SCAN dataset_dir recursively for files named exactly `image_name`
       (default 'raw_image.jpg'). Every OTHER file is ignored, so the source
       dataset can freely contain other images per folder.
    2. Compute a depth map for each found image and save it into a SIBLING
       folder that MIRRORS dataset_dir's internal structure — the source
       dataset folder itself is NEVER written into:
           dataset_dir      = .../custome_dataset
           sibling_root      = .../custome_dataset_depth_map
           .../custome_dataset/000417/raw_image.jpg
             -> .../custome_dataset_depth_map/000417/000417_depth_map.png
    3. Read the original split manifests (train/val/test .jsonl) in data_dir to
       recover each image's PROMPT and which SPLIT it belongs to, matching by
       ABSOLUTE image path (case-normalised) — so the prompt<->image<->split
       mapping is preserved exactly.
    4. Write data/depth_training/{train,val,test}.jsonl with ABSOLUTE
       raw_image_path + depth_path + prompt. Verify each file.

    Why scan instead of trusting JSONL paths to FIND images: if raw_image.jpg
    exists on disk it WILL be found, regardless of how the JSONL path was
    written. The JSONL is used only to attach prompt + split, matched by path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sibling output tree: a NEW folder next to dataset_dir, same name + suffix.
    # e.g. .../custome_dataset  ->  .../custome_dataset_depth_map
    sibling_root = dataset_dir.parent / (dataset_dir.name + "_depth_map")
    sibling_root.mkdir(parents=True, exist_ok=True)
    print(f"Depth maps will be saved to (sibling, mirrored structure): {sibling_root}")

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

    # ---- Step 2: compute depth, saving to the mirrored sibling tree --------- #
    out_fn = lambda p: _sibling_map_path(Path(p), dataset_dir, sibling_root, "_depth_map")
    path_to_depth = precompute_depths_from_paths(
        image_paths=[str(p) for p in found],
        output_dir=output_dir,            # not used for saving (out_fn overrides)
        size=size,
        batch_size=batch_size,
        model_name=model_name,
        device=device,
        skip_existing=skip_existing,
        local_files_only=local_files_only,
        out_path_fn=out_fn,
    )

    # ---- Step 3: index computed depths by NORMALISED absolute image path --- #
    depth_by_img = {
        os.path.normcase(str(Path(img).resolve())): dep
        for img, dep in path_to_depth.items()
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
            dep = depth_by_img.get(key)
            if dep is None:
                # Image not among the scanned/computed set (in a full run this
                # means the file was missing on disk; in a dry run it just wasn't
                # in the capped subset). Skip to keep the mapping honest.
                n_skipped += 1
                continue
            out_entries.append({
                "raw_image_path": Path(raw_str).resolve().as_posix(),
                "depth_path":     Path(dep).resolve().as_posix(),
                "prompt":         entry["prompt"],
            })

        out_path = output_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for e in out_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        if subset_n:
            note = f"  (dry-run: {n_skipped} entries not in the scanned subset — expected)"
        else:
            note = f"  ({n_skipped} entries had no depth on disk — skipped)" if n_skipped else ""
        print(f"\n  Written {len(out_entries)} entries -> {out_path}{note}")
        _verify_scan_training_jsonl(out_path, "depth_path", "_depth_map")


def run_json_mode(args):
    """
    JSON MODE:
      Read --json_file (list of {raw_image_path, prompt} entries).
      Compute depth for each image.
      Add / update "depth_path" field in each entry.
      Save updated JSON back to the same file.
    """
    json_path = Path(args.json_file).resolve()
    if not json_path.exists():
        print(f"[ERROR] JSON file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(entries)} entries from: {json_path}")

    # Resolve each image path relative to cwd or json file's directory
    image_paths = []
    for entry in entries:
        p = Path(entry["raw_image_path"])
        if not p.is_absolute():
            # Try relative to cwd first (project root), then relative to JSON location
            abs_p = Path.cwd() / p
            if abs_p.exists():
                p = abs_p
            else:
                p = json_path.parent / p
        image_paths.append(p)
        if not p.exists():
            print(f"[WARN] Image not found: {p}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else json_path.parent / "depth"

    path_to_depth = precompute_depths_from_paths(
        image_paths=[str(p) for p in image_paths],
        output_dir=output_dir,
        size=args.size,
        batch_size=args.batch_size,
        model_name=args.model,
        device=args.device,
        skip_existing=not args.no_skip,
        local_files_only=args.local_files_only,
    )

    # Update each entry with its depth path
    updated = 0
    for entry, img_path in zip(entries, image_paths):
        depth_p = path_to_depth.get(str(img_path))
        if depth_p is not None:
            try:
                rel = str(Path(depth_p).relative_to(Path.cwd()))
            except ValueError:
                rel = str(depth_p)
            entry["depth_path"] = rel.replace("\\", "/")
            updated += 1
        elif "depth_path" not in entry:
            print(f"[WARN] No depth for: {img_path}")

    # Write updated JSONL (one object per line)
    with open(json_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Updated {updated}/{len(entries)} entries with depth paths.")
    print(f"Saved updated JSONL -> {json_path}")
    print("\nNext step:")
    print(f"  python depth_training.py experiment=train_depth data.json_file={json_path}")


def update_json_files(json_dir: Path, path_to_depth_rel: dict, input_dir: Path = None) -> None:
    """
    Scan json_dir for *.json files (train.json, val.json, test.json, ...).
    For each entry, fill the "depth_path" field by matching the source image path.

    Args:
        json_dir          : directory to scan for JSON files (e.g. data/)
        path_to_depth_rel : {relative_src_path: relative_depth_path}
                            Keys are relative to input_dir
                            (e.g. "A/scene_001/original_sample_img.jpg")
        input_dir         : absolute path to the image root used during precompute.
                            Required for correct matching with nested folder layouts.
    """
    json_files = sorted(json_dir.glob("*.jsonl"))
    if not json_files:
        print(f"  No JSONL files found in: {json_dir}")
        return

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                entries = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            print(f"  [WARN] Could not read {json_path.name}: {e}")
            continue

        updated = 0
        for entry in entries:
            if "raw_image_path" not in entry:
                continue

            raw_path = Path(entry["raw_image_path"])

            # Derive the lookup key: path relative to input_dir.
            # JSON entries store paths relative to the project root
            # (e.g. "data/images/A/scene_001/original_sample_img.jpg").
            # input_dir is e.g. /abs/path/to/data/images, so the key is
            # "A/scene_001/original_sample_img.jpg".
            if input_dir is not None:
                abs_raw = (Path.cwd() / raw_path) if not raw_path.is_absolute() else raw_path
                try:
                    rel_key = str(abs_raw.relative_to(input_dir)).replace("\\", "/")
                except ValueError:
                    rel_key = raw_path.name
            else:
                rel_key = raw_path.stem   # legacy flat-layout fallback

            depth_rel = path_to_depth_rel.get(rel_key)
            if depth_rel:
                entry["depth_path"] = depth_rel
                updated += 1
            elif "depth_path" not in entry:
                print(f"  [WARN] No depth for: {raw_path}")

        if updated > 0:
            with open(json_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"  Updated {updated}/{len(entries)} entries in {json_path.name}")
        else:
            print(f"  No matching images found in {json_path.name} (skipped)")


def run_directory_mode(args):
    """
    DIRECTORY MODE:
      Scan --input_dir for images.
      Compute depths, save to --output_dir.
      Auto-update any *.json files in the data directory with depth paths.
    """
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_dir.parent / "depths"   # default: data/depths/
    )

    # Recursive scan — supports nested folder layouts like
    # data/images/A/scene_001/original_sample_img.jpg
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    )

    if not image_paths:
        print(f"[ERROR] No images found under: {input_dir}")
        return

    print(f"Found {len(image_paths)} images under: {input_dir}")

    path_to_depth = precompute_depths_from_paths(
        image_paths=[str(p) for p in image_paths],
        output_dir=output_dir,
        size=args.size,
        batch_size=args.batch_size,
        model_name=args.model,
        device=args.device,
        skip_existing=not args.no_skip,
        input_dir=input_dir,  # preserve relative path structure → no stem collisions
        local_files_only=args.local_files_only,
    )

    # Build relative-source-path → relative-depth-path dict for JSON update.
    # Keys are relative to input_dir (e.g. "A/scene_001/original_sample_img.jpg")
    # so multiple images with the same filename but different parent folders are
    # disambiguated correctly.
    path_to_depth_rel = {}
    for img_p, depth_p in path_to_depth.items():
        try:
            rel_src = str(Path(img_p).relative_to(input_dir)).replace("\\", "/")
        except ValueError:
            rel_src = Path(img_p).name
        try:
            rel_depth = str(Path(depth_p).relative_to(Path.cwd())).replace("\\", "/")
        except ValueError:
            rel_depth = str(depth_p).replace("\\", "/")
        path_to_depth_rel[rel_src] = rel_depth

    # Auto-update JSON files in the data directory with depth paths
    json_dir = Path(args.json_dir).resolve() if args.json_dir else input_dir.parent
    print(f"\nUpdating JSON files in: {json_dir}")
    update_json_files(json_dir, path_to_depth_rel, input_dir)

    print("\nNext step — train:")
    print("  python depth_training.py experiment=train_depth_12gb")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute depth maps from a directory of images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Primary arguments ----------------------------------------------- #
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory containing images (.jpg/.jpeg/.png). Required unless --json_file is used.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to save depth PNGs. Default: data/depths/ beside --input_dir.",
    )

    # ---- Depth-training JSON mode (new — creates depth_training/*.json) --- #
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Folder containing train.jsonl / val.jsonl / test.jsonl (e.g. data/). "
             "Any *.jsonl whose stem contains 'train'/'val'/'test' is accepted if the "
             "exact default name is absent. Activates depth-training JSONL mode: "
             "computes depth maps and writes depth_training/{train,val,test}.jsonl. "
             "Cannot be combined with --input_dir.",
    )
    parser.add_argument(
        "--raw_dir", type=str, default=None,
        help="Root of the raw image tree (e.g. data/raw/). Used to mirror the "
             "sub-folder structure into --depth_dir. Default: <data_dir>/raw.",
    )
    parser.add_argument(
        "--depth_dir", type=str, default=None,
        help="Where to save depth PNGs in depth-training JSON mode "
             "(e.g. data/raw_depth/). Default: <data_dir>/raw_depth.",
    )
    parser.add_argument(
        "--dry_run_n", type=int, default=None,
        help="(depth-training JSON mode) Process only the first N entries per JSON "
             "file, then verify. Use to sanity-check before running the full dataset.",
    )

    # ---- Optional: JSON mode (only needed if you already have a JSON file) #
    parser.add_argument(
        "--json_file", type=str, default=None,
        help="(Optional) Path to an existing JSON manifest. If given, reads image "
             "paths from the JSON and updates depth_path fields in it.",
    )
    parser.add_argument(
        "--json_dir", type=str, default=None,
        help="(Optional) Folder to scan for *.jsonl files to update after computing depths. "
             "Defaults to the parent of --input_dir (e.g. data/).",
    )

    # ---- Processing options ---------------------------------------------- #
    parser.add_argument("--size",       type=int, default=512,
                        help="Square spatial size for depth maps. Default: 512")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Images per GPU batch. Default: 4")
    # --model accepts EITHER a local folder path (offline, the default) OR a
    # HuggingFace id (when --local_files_only False, downloads into checkpoints/local_models).
    parser.add_argument("--model",      type=str,
                        default="checkpoints/local_models/dpt-hybrid-midas",
                        help="Depth model: local folder path (default, offline) OR a HF id "
                             "like Intel/dpt-hybrid-midas (use with --local_files_only False). "
                             "Default: checkpoints/local_models/dpt-hybrid-midas")
    parser.add_argument("--local_files_only", type=parse_bool, default=True,
                        metavar="True|False",
                        help="True (default) = load --model strictly from local disk (offline), "
                             "never download; pass a local folder path as --model. "
                             "False = allow download into checkpoints/local_models.")
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: 'cuda' or 'cpu'. Default: cuda if available")
    parser.add_argument("--no_skip", action="store_true",
                        help="Re-compute even if depth PNG already exists.")
    parser.add_argument("--image_path", type=str, default="source",
                        help="Key in your input JSONLs that holds the image path. "
                             "Default: 'source'. Change to 'target' (or any other key) "
                             "if your manifests use a different name. "
                             "Example: --image_path target")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Root folder prepended to RELATIVE image paths from the JSONL. "
                             "Use this when images are NOT under the repo root. "
                             "Example: --image_root /mnt/dataset  (then JSONL path "
                             "'images/foo.jpg' resolves to /mnt/dataset/images/foo.jpg). "
                             "Absolute paths in the JSONL are always used as-is.")

    # ---- Dataset-SCAN mode (find raw_image.jpg by scanning a folder) ------- #
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Activates DATASET-SCAN mode. Recursively scans this folder "
                             "for files named --image_name (default raw_image.jpg), computes "
                             "a depth map for each, and saves it IN the image's own folder as "
                             "<folder>_depth_map.png. Requires --data_dir (the folder holding "
                             "train/val/test.jsonl) to recover each image's prompt + split. "
                             "Example: --dataset_dir /data/custome_dataset --data_dir data/ --image_path target")
    parser.add_argument("--image_name", type=str, default="raw_image.jpg",
                        help="(scan mode) Exact filename to process in each folder. "
                             "Every other file is ignored — including *_depth_map.png / "
                             "*_seg_map.png this pipeline writes. Default: raw_image.jpg")

    args = parser.parse_args()

    # When offline, export the HF offline env vars so NOTHING touches the network
    # (belt-and-suspenders on top of local_files_only passed to from_pretrained).
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Validate: need at least one input mode
    if not args.dataset_dir and not args.data_dir and not args.input_dir and not args.json_file:
        parser.error(
            "Provide one of:\n"
            "  --dataset_dir /data/custome_dataset --data_dir data/  (scan mode — saves maps in-folder)\n"
            "  --data_dir data/              (builds depth_training/*.jsonl from JSONL paths)\n"
            "  --input_dir data/raw          (directory mode)\n"
            "  --json_file data/train.jsonl  (single-JSONL mode)"
        )

    if args.data_dir and args.input_dir:
        parser.error("--data_dir and --input_dir are mutually exclusive.")

    if args.dataset_dir and not args.data_dir:
        parser.error(
            "--dataset_dir (scan mode) also needs --data_dir pointing at the folder that "
            "holds train/val/test.jsonl — that is where each image's prompt + split come from.\n"
            "  Example: --dataset_dir /data/custome_dataset --data_dir data/ --image_path target"
        )

    print(f"Device : {args.device}")
    print(f"Size   : {args.size}x{args.size}")
    print(f"Batch  : {args.batch_size}")

    if args.dataset_dir:
        # DATASET-SCAN mode: find raw_image.jpg by scanning, save maps in-folder,
        # rebuild depth_training/{train,val,test}.jsonl from the original splits.
        dataset_dir = Path(args.dataset_dir).resolve()
        data_dir    = Path(args.data_dir).resolve()
        out_dir     = Path(args.output_dir).resolve() if args.output_dir else data_dir / "depth_training"

        if not dataset_dir.exists():
            parser.error(f"--dataset_dir not found: {dataset_dir}")
        if args.dry_run_n:
            print(f"\n[DRY RUN] Scan capped at first {args.dry_run_n} images.")

        build_depth_training_from_scan(
            dataset_dir=dataset_dir,
            data_dir=data_dir,
            output_dir=out_dir,
            image_name=args.image_name,
            image_path=args.image_path,
            size=args.size,
            batch_size=args.batch_size,
            model_name=args.model,
            device=args.device,
            skip_existing=not args.no_skip,
            local_files_only=args.local_files_only,
            subset_n=args.dry_run_n,
        )

    elif args.data_dir:
        # Depth-training JSON mode: read data/{train,val,test}.json,
        # compute depths, write depth_training/{train,val,test}.json
        data_dir  = Path(args.data_dir).resolve()
        raw_dir   = Path(args.raw_dir).resolve()  if args.raw_dir   else data_dir / "raw"
        depth_dir = Path(args.depth_dir).resolve() if args.depth_dir else data_dir / "raw_depth"
        out_dir   = Path(args.output_dir).resolve() if args.output_dir else data_dir / "depth_training"

        if args.dry_run_n:
            print(f"\n[DRY RUN] Processing first {args.dry_run_n} entries per JSON.")

        image_root = Path(args.image_root).resolve() if args.image_root else None
        if image_root:
            print(f"Image root : {image_root}  (prepended to relative image paths)")
        build_depth_training_jsons(
            data_dir=data_dir,
            raw_dir=raw_dir if args.raw_dir else (image_root or data_dir / "raw"),
            depth_dir=depth_dir,
            output_dir=out_dir,
            size=args.size,
            batch_size=args.batch_size,
            model_name=args.model,
            device=args.device,
            skip_existing=not args.no_skip,
            subset_n=args.dry_run_n,
            local_files_only=args.local_files_only,
            image_path=args.image_path,
            image_root=image_root,
        )

    elif args.json_file and not args.input_dir:
        # JSON-only mode: image paths come from a single JSON file
        run_json_mode(args)
    else:
        # Directory mode: scan input_dir for images (primary usage)
        run_directory_mode(args)


if __name__ == "__main__":
    main()

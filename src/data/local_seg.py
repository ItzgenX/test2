"""
src/data/local_seg.py
---------------------
STAGE D dataset: load (RGB image, pre-computed segmentation map, prompt) triplets
for training the segmentation-conditioned LoRAdapter.

This is the segmentation twin of the Depth*JsonDataset classes in src/data/local.py.
It mirrors their STRUCTURE (JSON manifest, _seg_resolve, early missing-file check,
{"jpg","seg","caption"} return dict) but differs where the segmentation SIGNAL
genuinely demands it — two seg-specific points, both critical:

  1. The saved map is a RAW CLASS-ID PNG (8-bit, values 0..18), NOT a continuous
     depth map. We COLOURISE it at load time with the pinned Cityscapes palette
     (SEG_CITYSCAPES_PALETTE from src/encoders/seg_encoder.py, the SSOT), producing
     a 3-channel RGB map in [0,1]. This matches the live SegmentationEncoder.forward()
     output exactly (both call the same seg_colorize_ids + palette), so training and
     inference conditioning are pixel-identical.

  2. Resizing a class-ID map MUST use NEAREST interpolation. Depth uses bilinear
     (correct — averaging continuous depth values is fine), but averaging categorical
     class ids is meaningless: the mean of "road"=0 and "car"=13 is 6.5, which is
     "traffic light" — a fabricated class that doesn't exist in the image. NEAREST
     preserves exact labels. In practice the PNG is already at the right size (saved
     by seg_map_calculations.py at cfg.size), so this resize is usually a
     no-op alignment step — but we still force NEAREST so it is correct for any size
     and never silently corrupts labels.

Returned by __getitem__:
  {
    "jpg"    : RGB tensor   [3, H, W] in [-1, 1]   (standard training format)
    "seg"    : colour map   [3, H, W] in [0, 1]    (Cityscapes-palette RGB)
    "caption": prompt string
  }
The "seg" key parallels depth's "depth" key so seg_training.py reads batch["seg"].
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.encoders.seg_encoder import SEG_CITYSCAPES_PALETTE, seg_palette_tensor, seg_colorize_ids


class SegJsonDataset(Dataset):
    """
    Load image + pre-computed segmentation-ID map pairs from a JSON manifest.

    JSONL format — one JSON object per line (written by seg_map_calculations.py):
        {"raw_image_path": "...jpg", "seg_path": "...png", "prompt": "..."}
        {"raw_image_path": "...jpg", "seg_path": "...png", "prompt": "..."}

    Paths resolve: absolute as-is, else relative to project_root, else to json_dir.
    This mirrors the depth dataset's path resolution exactly.
    """

    def __init__(
        self,
        json_file: Path,
        image_transform,           # torchvision Compose for the RGB image (-> [-1,1])
        size: int = 512,           # square side for the conditioning colour map
        project_root: Path = None,
        palette: list = None,      # class-id -> RGB; defaults to Cityscapes SSOT
        image_root: Path = None,
    ):
        self.json_file    = Path(json_file)
        self.json_dir     = self.json_file.parent
        self.project_root = Path(project_root) if project_root else self.json_dir
        self.image_root   = Path(image_root) if image_root else None
        self.size         = size
        self.image_transform = image_transform

        # Build the palette tensor ONCE here from the shared constant so every
        # sample colourises identically, and identically to the live encoder.
        # Using seg_palette_tensor keeps the [0,1] conversion in one function.
        self.seg_palette  = seg_palette_tensor(
            palette if palette is not None else SEG_CITYSCAPES_PALETTE
        )
        self.num_classes  = self.seg_palette.shape[0]   # 19 for Cityscapes

        with open(self.json_file, "r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]

        # Early, loud missing-file check — catch a bad manifest before the
        # dataloader workers surface a confusing deep stack trace mid-training.
        missing_img = missing_seg = 0
        for item in self.items:
            if not self._seg_resolve(item["raw_image_path"]).exists():
                print(f"[SegJsonDataset] WARN image not found: {item['raw_image_path']}")
                missing_img += 1
            if not item.get("seg_path", ""):
                missing_seg += 1
            elif not self._seg_resolve(item["seg_path"]).exists():
                print(f"[SegJsonDataset] WARN seg not found: {item['seg_path']}")
                missing_seg += 1
        if missing_seg:
            print(
                f"[SegJsonDataset] {missing_seg}/{len(self.items)} entries missing "
                f"seg_path. Run: python seg_map_calculations.py --data_dir data/"
            )

    def _seg_resolve(self, p: str) -> Path:
        """
        Resolve a path: absolute as-is; else project_root relative; else json_dir.

        The 'seg_' prefix marks this as segmentation-pipeline code (naming rule).
        Same resolution logic as depth's dataset _resolve.
        """
        p = Path(p)
        if p.is_absolute():
            return p
        if self.image_root:
            abs_p = self.image_root / p
            if abs_p.exists():
                return abs_p
        abs_p = self.project_root / p
        return abs_p if abs_p.exists() else self.json_dir / p

    def __len__(self):
        return len(self.items)

    def _load_seg_colormap(self, seg_path: Path) -> torch.Tensor:
        """
        Load a raw class-ID PNG and return a colourised map [3, size, size] in [0,1].

        Steps (the two seg-specific points live here):
          1. Open as "L" (8-bit single channel) — pixel values ARE class ids.
          2. NEAREST resize to (size, size) — label-preserving (NEVER bilinear).
          3. seg_colorize_ids() with the shared palette -> [1,3,size,size] in [0,1].

        Returns [3, size, size] float tensor in [0, 1].
        """
        ids_pil = Image.open(seg_path).convert("L")

        # NEAREST is REQUIRED for a class-ID map: bilinear would average class
        # ids and produce fabricated class values. PIL.Image.NEAREST is the flag.
        if ids_pil.size != (self.size, self.size):
            ids_pil = ids_pil.resize((self.size, self.size), Image.NEAREST)

        ids = torch.from_numpy(np.asarray(ids_pil, dtype=np.int64))   # [size, size]

        # seg_colorize_ids expects [B, H, W]; add/remove the batch dim around it.
        colour = seg_colorize_ids(ids.unsqueeze(0), self.seg_palette)   # [1,3,size,size]
        return colour[0]                                                  # [3,size,size]

    def __getitem__(self, idx: int):
        item    = self.items[idx]
        caption = item.get("prompt", "")

        # ---- RGB image -> [-1, 1] -------------------------------------------
        image = Image.open(self._seg_resolve(item["raw_image_path"])).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)

        # ---- Segmentation colour map -> [0,1], 3-channel -------------------
        # Fail loudly rather than propagating a KeyError deep in a worker.
        if "seg_path" not in item:
            raise KeyError(
                f"Entry {idx} in {self.json_file.name} has no 'seg_path'. "
                f"Run seg_map_calculations.py --data_dir data/ to build JSONs."
            )
        seg = self._load_seg_colormap(self._seg_resolve(item["seg_path"]))

        return {"jpg": image, "seg": seg, "caption": caption}


class SegJsonDataModule:
    """
    Data module reading a JSON manifest for train + optional validation.

    Twin of DepthJsonDataModule. Exposes train_dataloader()/val_dataloader() and
    .train_dataset / .val_dataset (seg_training.py indexes val_dataset directly for
    the fixed-scene monitoring images). val_json_file MUST point at the real
    validation set — NEVER test.json (references.md §8).
    """

    def __init__(
        self,
        json_file: str,
        transform: list,               # Hydra-instantiated image transforms (-> [-1,1])
        size: int = 512,
        val_json_file: str = None,
        batch_size: int = 8,
        val_batch_size: int = 4,       # 4 = safe under no_grad; matches depth default
        workers: int = 4,
        val_workers: int = 2,
        palette: list = None,
        image_root: str = None,
    ):
        # project_root: three levels up from this file (src/data/ -> src/ -> root).
        project_root = Path(os.path.abspath(__file__)).parent.parent.parent
        image_tfm    = transforms.Compose(transform)
        _img_root    = Path(project_root, image_root) if image_root else project_root

        self.batch_size     = batch_size
        self.val_batch_size = val_batch_size
        self.workers        = workers
        self.val_workers    = val_workers

        self.train_dataset = SegJsonDataset(
            json_file=Path(project_root, json_file),
            image_transform=image_tfm, size=size,
            project_root=_img_root, palette=palette,
        )

        if val_json_file:
            self.val_dataset = SegJsonDataset(
                json_file=Path(project_root, val_json_file),
                image_transform=image_tfm, size=size,
                project_root=_img_root, palette=palette,
            )
        else:
            # No val set provided -> fall back to train set.
            # Prefer always providing val_json_file; this fallback is only for quick
            # debugging runs where no val split is available.
            print("[SegJsonDataModule] WARNING: no val_json_file — using train set as val.")
            self.val_dataset = self.train_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.val_batch_size,
            shuffle=False, num_workers=self.val_workers,
        )

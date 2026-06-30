"""
check_resize_quality.py
-----------------------
Visual sanity check: does 1280x800 -> 512x512 (SquarePad + Resize) preserve features?

Picks N random images from data/raw, generates ONE quality grid per image, saves to
outputs/viz/resize_check/<scene_id>_quality.png

Each grid:
  TOP ROW   : Original full-res  |  512x512 output
              (coloured boxes mark the 4 zoom regions)
  BOTTOM ROW: 4 side-by-side zoom crops — LEFT = from original, RIGHT = from 512x512
              (same region, same display size — shows exactly what detail survives)

Usage:
  python check_resize_quality.py              # 5 random images
  python check_resize_quality.py --n 10       # more images
  python check_resize_quality.py --seed 42    # fixed random seed
  python check_resize_quality.py --ids 000331 000417  # specific scenes
"""
import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms

from src.data.transforms import SquarePad

OUT_DIR = Path("outputs/viz/resize_check")

TO_512 = transforms.Compose([
    SquarePad(),
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),
])

# 4 zoom regions: (cx, cy, frac, label, box_colour)
ZOOM_REGIONS = [
    (0.18, 0.35, 0.28, "Top-left",     (220,  60,  60)),
    (0.52, 0.42, 0.28, "Centre",       ( 60, 200,  60)),
    (0.78, 0.50, 0.28, "Right side",   ( 60, 130, 255)),
    (0.35, 0.78, 0.28, "Bottom/ground",(230, 180,   0)),
]


def _crop_pair(orig, out512, cx, cy, frac, render=460):
    """
    Cut the same relative region from orig and out512.
    Returns (orig_patch, out_patch) both resized to render x render PIL images.
    """
    ow, oh = orig.size

    # --- from original ---
    side_o = int(min(ow, oh) * frac)
    l = max(0, min(int(cx * ow - side_o // 2), ow - side_o))
    t = max(0, min(int(cy * oh - side_o // 2), oh - side_o))
    patch_o = orig.crop((l, t, l + side_o, t + side_o)).resize(
        (render, render), Image.LANCZOS
    )

    # --- from 512 output ---
    # Content in 512 sits inside SquarePad bands.
    if ow > oh:   # landscape: top + bottom bands
        pad_top = round((ow - oh) / 2 / ow * 512)
        pad_bot = 512 - round(oh / ow * 512) - pad_top
        cx0, cy0 = 0, pad_top
        cw,  ch  = 512, 512 - pad_top - pad_bot
    else:         # portrait: left + right bands
        pad_lft = round((oh - ow) / 2 / oh * 512)
        pad_rgt = 512 - round(ow / oh * 512) - pad_lft
        cx0, cy0 = pad_lft, 0
        cw,  ch  = 512 - pad_lft - pad_rgt, 512

    side_s = int(frac * min(cw, ch))
    sx = cx0 + max(0, min(int(cx * cw - side_s // 2), cw - side_s))
    sy = cy0 + max(0, min(int(cy * ch - side_s // 2), ch - side_s))
    patch_s = out512.crop((sx, sy, sx + side_s, sy + side_s)).resize(
        (render, render), Image.LANCZOS
    )
    return patch_o, patch_s


def _annotate(img, regions, line_w=5):
    """Draw coloured boxes on a PIL copy."""
    out = img.copy()
    d   = ImageDraw.Draw(out)
    ow, oh = img.size
    for cx, cy, frac, _, col in regions:
        side = int(min(ow, oh) * frac)
        l = max(0, min(int(cx * ow - side // 2), ow - side))
        t = max(0, min(int(cy * oh - side // 2), oh - side))
        d.rectangle((l, t, l + side, t + side), outline=col, width=line_w)
    return out


def _annotate_512(out512, orig_size, regions, line_w=3):
    """Draw the equivalent boxes on the 512 output."""
    ow, oh = orig_size
    img = out512.copy()
    d   = ImageDraw.Draw(img)
    if ow > oh:
        pad_top = round((ow - oh) / 2 / ow * 512)
        cx0, cy0 = 0, pad_top
        cw, ch   = 512, 512 - 2 * pad_top
    else:
        pad_lft = round((oh - ow) / 2 / oh * 512)
        cx0, cy0 = pad_lft, 0
        cw, ch   = 512 - 2 * pad_lft, 512

    for cx, cy, frac, _, col in regions:
        side = int(frac * min(cw, ch))
        sx = cx0 + max(0, min(int(cx * cw - side // 2), cw - side))
        sy = cy0 + max(0, min(int(cy * ch - side // 2), ch - side))
        d.rectangle((sx, sy, sx + side, sy + side), outline=col, width=line_w)
    return img


def make_grid(img_path: Path, out_dir: Path) -> Path:
    orig   = Image.open(img_path).convert("RGB")
    out512 = TO_512(orig)
    ow, oh = orig.size

    # Display both at the same pixel height (800px) so you can visually compare.
    # Original keeps its natural width; 512x512 is square.
    DISPLAY_H = 800
    orig_disp = orig.resize((int(ow / oh * DISPLAY_H), DISPLAY_H), Image.LANCZOS)
    out_disp  = out512.resize((DISPLAY_H, DISPLAY_H), Image.LANCZOS)

    # Width ratio matches actual image widths so neither gets distorted.
    orig_w = orig_disp.size[0]
    out_w  = out_disp.size[0]

    fig, (ax0, ax1) = plt.subplots(
        1, 2,
        figsize=((orig_w + out_w) / 100, DISPLAY_H / 100 + 0.8),
        gridspec_kw={"width_ratios": [orig_w, out_w]},
        facecolor="#111118",
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.01, wspace=0.02)
    fig.suptitle(img_path.parent.name, color="white", fontsize=13, fontweight="bold")

    ax0.imshow(np.asarray(orig_disp))
    ax0.set_title(f"Original  {ow} x {oh}", color="#aaaacc", fontsize=11, pad=5)
    ax0.axis("off")

    ax1.imshow(np.asarray(out_disp))
    ax1.set_title("512 x 512", color="#aaaacc", fontsize=11, pad=5)
    ax1.axis("off")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{img_path.parent.name}_quality.png"
    fig.savefig(save_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n",    type=int, default=5,
                        help="Number of random images (default 5)")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed (default 7)")
    parser.add_argument("--ids",  nargs="*", default=None,
                        help="Specific scene IDs e.g. 000331 000417")
    parser.add_argument("--data_dir", default="data/raw",
                        help="Root of raw image tree (default data/raw)")
    args = parser.parse_args()

    all_imgs = sorted(Path(args.data_dir).glob("*/raw_image.jpg"))
    if not all_imgs:
        print(f"[ERROR] No images found under {args.data_dir}")
        return

    if args.ids:
        chosen = [Path(args.data_dir) / sid / "raw_image.jpg" for sid in args.ids]
        missing = [p for p in chosen if not p.exists()]
        if missing:
            print(f"[ERROR] Not found: {missing}")
            return
    else:
        random.seed(args.seed)
        chosen = random.sample(all_imgs, min(args.n, len(all_imgs)))

    print(f"Generating quality grids for {len(chosen)} image(s)...")
    for img_path in chosen:
        save_path = make_grid(img_path, OUT_DIR)
        print(f"  {img_path.parent.name}  ->  {save_path}")

    print(f"\nDone. All grids saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

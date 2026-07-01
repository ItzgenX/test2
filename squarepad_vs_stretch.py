"""
SquarePad vs direct stretch for one image.
3 columns: Original | With SquarePad (current) | Without SquarePad (stretch)

Usage:
  python squarepad_vs_stretch.py
  python squarepad_vs_stretch.py --image data/raw/000417/raw_image.jpg
  python squarepad_vs_stretch.py --image data/raw/000703/raw_image.jpg --out outputs/viz/squarepad_compare.png
  python squarepad_vs_stretch.py --image data\wp3419589.jpg 
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.data.transforms import SquarePad


def make_comparison(img_path: Path, out_path: Path):
    orig = Image.open(img_path).convert("RGB")

    with_squarepad = transforms.Compose([
        SquarePad(),
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),
    ])(orig)

    without_squarepad = transforms.Resize(
        (512, 512), interpolation=transforms.InterpolationMode.LANCZOS
    )(orig)

    H = 800
    orig_d = orig.resize((int(orig.size[0] / orig.size[1] * H), H), Image.LANCZOS)
    sq_d   = with_squarepad.resize((H, H), Image.LANCZOS)
    st_d   = without_squarepad.resize((H, H), Image.LANCZOS)

    ow = orig_d.size[0]

    fig, axes = plt.subplots(1, 3, figsize=(28, 7),
                              gridspec_kw={"width_ratios": [ow, H, H]},
                              facecolor="#111118")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.02)
    fig.suptitle("SquarePad vs direct stretch — what is the trade-off?",
                 color="white", fontsize=14, fontweight="bold")

    axes[0].imshow(np.asarray(orig_d))
    axes[0].set_title(f"Original  {orig.size[0]} x {orig.size[1]}",
                      color="#aaaacc", fontsize=12, pad=6)
    axes[0].axis("off")

    axes[1].imshow(np.asarray(sq_d))
    axes[1].set_title("WITH SquarePad  (current)\nPads shorter side with edge pixels → no distortion\nbut you see the padding bands",
                      color="#4ec9b0", fontsize=11, pad=6)
    axes[1].axis("off")

    axes[2].imshow(np.asarray(st_d))
    axes[2].set_title("WITHOUT SquarePad  (stretch)\nNo padding bands\nbut image is SQUISHED to fit square",
                      color="#ff9f43", fontsize=11, pad=6)
    axes[2].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default="data/raw/000331/raw_image.jpg",
                        help="Input image path (default: data/raw/000331/raw_image.jpg)")
    parser.add_argument("--out", default="outputs/viz/squarepad_vs_stretch.png",
                        help="Output PNG path")
    args = parser.parse_args()

    make_comparison(Path(args.image), Path(args.out))


if __name__ == "__main__":
    main()

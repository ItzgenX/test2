import torchvision.transforms.functional as TF
from torchvision.transforms.v2 import Transform
import torchvision.transforms.v2.functional as Fv2


class SquarePad:
    """
    Pad the shorter dimension of a PIL image to produce a square using
    edge-replication (the border pixel is repeated, not solid black).

    Works as a plain callable so it is compatible with both v1 and v2
    torchvision.transforms.Compose chains and with Hydra instantiation.

    Input:  PIL.Image of any size (H × W)
    Output: PIL.Image of size (max(H,W) × max(H,W))

    After each call, `last_padding_fracs` holds the padding amounts as
    fractions of the ORIGINAL image dimensions in (left, top, right, bottom)
    order.  Storing fractions rather than pixel counts means the values
    remain correct even after the image is later resized to 512 × 512.
    Use them to identify or mask the padded region in generated outputs.

    Edge-replication rationale (vs solid black / zero):
      A solid-colour boundary looks like a real depth discontinuity to the
      DPT model and produces a sharp artefact ring in the depth map at the
      pad boundary.  Repeating the edge pixel makes the transition smooth so
      depth estimation near the border is not disturbed.
    """

    def __init__(self):
        self.last_padding_fracs = (0.0, 0.0, 0.0, 0.0)

    def __call__(self, img):
        w, h = img.size          # PIL uses (width, height)

        if h == w:
            self.last_padding_fracs = (0.0, 0.0, 0.0, 0.0)
            return img

        if w > h:                # landscape (e.g. 1280 × 800): pad top + bottom
            pad_total  = w - h
            pad_top    = pad_total // 2
            pad_bottom = pad_total - pad_top
            pad_left = pad_right = 0
        else:                    # portrait: pad left + right
            pad_total  = h - w
            pad_left   = pad_total // 2
            pad_right  = pad_total - pad_left
            pad_top = pad_bottom = 0

        # Record as fractions so the values stay valid after the downstream Resize.
        self.last_padding_fracs = (
            pad_left   / w,
            pad_top    / h,
            pad_right  / w,
            pad_bottom / h,
        )

        # TF.pad: padding = [left, top, right, bottom], padding_mode='edge'
        return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom],
                      padding_mode='edge')


class TopCrop(Transform):
    # use standard crop transform of v2
    # but always crops from the top

    def __init__(self, size):
        super().__init__()
        self.size = size

    def _transform(self, inpt, params):
        return Fv2.crop(inpt, 0, 0, self.size, self.size)


# ============================================================================ #
#  SEGMENTATION PIPELINE — shared preprocessing builder                        #
# ============================================================================ #

from torchvision import transforms as _tv   # local alias: avoids shadowing outer scope


def build_seg_square_preprocess(size: int, resize_mode: str = "letterbox"):
    """
    Build the ONE canonical RGB preprocessing pipeline for the SEGMENTATION pipeline.

    WHY THIS EXISTS (and why it belongs here, not in a seg-specific file):
      Both calculate_segmentation_map.py (offline calc) and inference_seg.py (live
      inference) must apply byte-for-byte identical preprocessing so the seg map the
      network sees at inference exactly matches what was saved for training. The only
      way to guarantee that is to import the SAME function in both places — this is it.
      (Depth triplicated this preprocessing and references.md flags that as a drift
      risk; segmentation fixes it with this single source of truth.)

    The "seg" prefix distinguishes it from any hypothetical depth equivalent and
    satisfies the user's visual-identity rule: segmentation code has "seg" in name.

    INPUT  (of the returned callable): PIL.Image of any size, any aspect ratio.
    OUTPUT (of the returned callable): float tensor [3, size, size] in [-1, 1].
      This is the range every encoder slot in src/model.py expects
      (asserted by SegmentationEncoder._predict_ids).

    Args:
        size:        final square side in pixels (e.g. 512). Must match cfg.size
                     and the size used when the offline seg PNGs were computed.
        resize_mode: "letterbox" (default) — pads the shorter side to a square
                       with edge-replication BEFORE resizing to (size, size).
                       Keeps the full driving-scene frame, no content cropped,
                       no aspect distortion (references.md §5).
                     "stretch" — resizes straight to (size, size), distorting
                       aspect ratio but also producing a correct square.
                     "crop" is EXCLUDED for this project (would cut driving-scene
                       frame edges — ruled out in references.md §5).

    Correctness note: both modes produce a (size, size) square PIL image before
    the ToTensor step, so SegmentationEncoder always receives exactly (size, size)
    input and never triggers the kind of internal forced-crop that MiDaS does.
    """
    # Tail shared by both modes: PIL [0,255] -> tensor [0,1] -> [-1,1].
    # mean=std=0.5 maps [0,1] linearly to [-1,1] (the SD1.5 VAE / encoder range).
    seg_to_tensor_tail = [
        _tv.ToTensor(),                                    # [0,255] -> [0,1]
        _tv.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [0,1] -> [-1,1]
    ]

    if resize_mode == "letterbox":
        # SquarePad pads the shorter side (edge-replicated) to make the image
        # square; THEN Resize is a uniform scale (no distortion) because the
        # input is already square after padding.
        seg_head = [SquarePad(), _tv.Resize((size, size))]
    elif resize_mode == "stretch":
        # Direct Resize to a square — keeps all content but distorts aspect ratio.
        seg_head = [_tv.Resize((size, size))]
    else:
        # Fail loudly: a silent fallback here would corrupt train/inference parity.
        raise ValueError(
            f"resize_mode must be 'letterbox' or 'stretch' (got {resize_mode!r}). "
            f"'crop' is excluded — it cuts driving-scene frame edges (references.md §5)."
        )

    return _tv.Compose(seg_head + seg_to_tensor_tail)

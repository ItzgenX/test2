"""
src/encoders/seg_encoder.py
---------------------------
LIVE semantic-segmentation encoder for the conditional-LoRA pipeline.

WHY THIS FILE EXISTS:
  The depth pipeline reused the repo's existing `midas` encoder slot — depth
  conditioning was already a first-class citizen of LoRAdapter. Segmentation is
  NOT in the paper and there is NO stock seg encoder, so we build one here. Its
  job is to satisfy the EXACT same contract the midas encoder satisfies, so it
  drops into the same `lora.struct.encoder` slot and the same
  encoder -> mapper -> DataProvider chain in src/model.py with zero changes to
  model.py or the mapper (see Q-SEGENC in references.md):

      ENCODER-SLOT CONTRACT
        INPUT : [B, 3, H, W] float in [-1, 1]
                (model.forward passes the raw image tensor; in sample_easy() the
                 batch is cat([zeros, img]) for CFG, so the encoder must also
                 tolerate an all-zeros half — it does, zeros are in range).
        OUTPUT: [B, 3, size, size] float in [0, 1].
                midas returns a 3x-replicated grayscale depth map in [0,1];
                we return a 3-channel RGB colour segmentation map in [0,1].
                The mapper (FixedStructureMapper15) is Conv2d(3, ...), so any
                [B,3,size,size] tensor in [0,1] plugs in unchanged.

KEY DESIGN DECISION — colour palette vs raw class IDs (references.md §9):
  SegFormer predicts a DISCRETE class-ID per pixel (19 Cityscapes classes).
  Feeding the raw id/18 grayscale ramp into a conv mapper built for depth's
  CONTINUOUS gradient would impose a false ordinal ordering on categorical
  labels: class 5 "pole" is not semantically between classes 4 "fence" and
  6 "traffic light", yet id/18 makes them equidistant. A fixed colour palette
  gives every class a DISTINCT, well-separated RGB identity — the seg analogue
  of depth's smooth signal, and exactly what ControlNet-Seg does.

LOCKED MODEL: nvidia/segformer-b5-finetuned-cityscapes-1024-1024
  b5's MiT-B5 backbone (82M params) gives significantly better boundary
  precision than b0 (3.7M) on driving-scene classes (pedestrians, vehicles,
  traffic lights). Since segmentation QUALITY is the point of the
  depth-vs-seg comparison, b5 is the correct choice. Cost is offline calc
  time only — training uses pre-saved maps (skip_encode=True).

NORMALIZATION: done manually here (÷255 + ImageNet mean/std) rather than via
  SegformerImageProcessor because (a) the cached checkpoint may have no
  preprocessor_config.json so `from_pretrained` can fail offline, and (b)
  midas likewise ignores DPTImageProcessor and normalizes by hand. The manual
  path matches SegformerImageProcessor to 2.4e-7 (bit-exact).
"""

import torch
from torch import nn
import torch.nn.functional as F


# ============================================================================ #
#  SEG_CITYSCAPES_PALETTE — single source of truth for class-ID -> RGB colour  #
# ============================================================================ #
# Canonical Cityscapes 19-class colours, indexed by trainId 0..18 in the exact
# order of the checkpoint's id2label (road..bicycle). These are the standard,
# semantically-conventional Cityscapes colours (road=purple, vegetation=green,
# sky=steel-blue, car=deep-blue, person=red, ...). They are deliberately spread
# across colour space so the mapper can distinguish classes.
#
# "SEG_" prefix: satisfies the visual-identity rule — you can tell at a glance
# this constant belongs to the segmentation pipeline, not depth.
#
# DO NOT reorder: index == class id. If you ever switch to a model with a
# different class set, replace this whole table; num_classes follows it.
SEG_CITYSCAPES_PALETTE: list[tuple[int, int, int]] = [
    (128,  64, 128),   # 0  road
    (244,  35, 232),   # 1  sidewalk
    ( 70,  70,  70),   # 2  building
    (102, 102, 156),   # 3  wall
    (190, 153, 153),   # 4  fence
    (153, 153, 153),   # 5  pole
    (250, 170,  30),   # 6  traffic light
    (220, 220,   0),   # 7  traffic sign
    (107, 142,  35),   # 8  vegetation
    (152, 251, 152),   # 9  terrain
    ( 70, 130, 180),   # 10 sky
    (220,  20,  60),   # 11 person
    (255,   0,   0),   # 12 rider
    (  0,   0, 142),   # 13 car
    (  0,   0,  70),   # 14 truck
    (  0,  60, 100),   # 15 bus
    (  0,  80, 100),   # 16 train
    (  0,   0, 230),   # 17 motorcycle
    (119,  11,  32),   # 18 bicycle
]


def seg_palette_tensor(
    palette: list[tuple[int, int, int]] = SEG_CITYSCAPES_PALETTE,
) -> torch.Tensor:
    """
    Convert the integer RGB palette into a lookup tensor in [0, 1].

    WHY: colourising a class-ID map is a gather/index operation; having the
    palette as a [num_classes, 3] float tensor in [0,1] lets us do
    `palette[ids]` to produce a [..., 3] colour image directly in the output
    range the mapper expects.

    The "seg_" prefix marks this as part of the segmentation pipeline.

    Returns: FloatTensor [num_classes, 3] in [0, 1].
    """
    return torch.tensor(palette, dtype=torch.float32) / 255.0


def seg_colorize_ids(
    ids: torch.Tensor,          # [B, H, W] long — class IDs
    palette: torch.Tensor,      # [K, 3] float in [0,1] — from seg_palette_tensor()
) -> torch.Tensor:              # [B, 3, H, W] float in [0,1]
    """
    Map a class-ID map to a 3-channel RGB colour image in [0, 1].

    The "seg_" prefix marks this as segmentation-pipeline code.

    WHY IT EXISTS IN THE PIPELINE:
      This is the SINGLE colourisation step shared by:
        (a) the live encoder (SegmentationEncoder.forward, for live inference),
        (b) the training dataset (src/data/local_seg.py colourises the saved
            raw-ID PNG at load time with this same function),
        (c) inference visualisation.
      Sharing it guarantees a class always gets the identical colour everywhere
      — the parity guarantee that training and inference conditioning match.

    Inputs:
      ids     : Long tensor [B, H, W] of class ids in [0, num_classes-1].
      palette : Float tensor [num_classes, 3] in [0,1] (from seg_palette_tensor()).
    Output:
      Float tensor [B, 3, H, W] in [0, 1].

    Fails loudly if any id would index outside the palette — a silent
    out-of-range gather is exactly the kind of quiet corruption to avoid.
    """
    max_id = int(ids.max()) if ids.numel() else 0
    if max_id >= palette.shape[0]:
        raise ValueError(
            f"class id {max_id} >= palette size {palette.shape[0]}. "
            f"The ID map and SEG_CITYSCAPES_PALETTE disagree on class count."
        )
    palette = palette.to(ids.device)
    # palette[ids] -> [B, H, W, 3]; permute to channels-first [B, 3, H, W].
    colour = palette[ids.long()]          # [B, H, W, 3] in [0,1]
    return colour.permute(0, 3, 1, 2).contiguous()


# ============================================================================ #
#  LIVE ENCODER                                                                 #
# ============================================================================ #

class SegmentationEncoder(nn.Module):
    """
    Live SegFormer-Cityscapes encoder that plugs into the LoRAdapter encoder slot.

    Mirrors src/annotators/midas.DepthEstimator in role (live conditioning at
    inference) and interface, but produces a COLOUR segmentation map instead of
    a depth map. It is an nn.Module so accelerate's .prepare()/.to()/.eval()
    treat it identically to the midas encoder.

    During TRAINING: this encoder is NOT called at all — train_seg.py feeds
    pre-saved colour maps via skip_encode=True, bypassing this entirely. It
    runs ONLY at live inference inside model.sample() -> sample_easy() ->
    encoder(c). (Same pattern as depth: DepthEstimator runs live at inference,
    not during training.)

    Two public entry points sharing ONE internal predictor (_predict_ids):
      • forward(imgs)    -> colour map [B,3,size,size] in [0,1]
                            (what src/model.py calls as `encoder(c)` at inference)
      • label_ids(imgs)  -> raw class-IDs [B,size,size] long
                            (used ONLY by calculate_segmentation_map.py, which
                             saves raw IDs to disk; colourisation happens at
                             dataset load time in local_seg.py)

    Sharing _predict_ids guarantees the saved IDs and live inference IDs are
    byte-for-byte identical — the train/inference parity rule.

    Args:
      size             : final square side of the conditioning map (matches
                         cfg.size, e.g. 512). Output spatial size.
      model            : HuggingFace checkpoint id or local folder path.
                         LOCKED = nvidia/segformer-b5-finetuned-cityscapes-1024-1024.
      seg_input_size   : square size the RGB is resized to before SegFormer.
                         512 = a multiple of SegFormer's /4 patch stride and
                         matches cfg.size, so no extra hop; keeps calc and
                         inference on identical code paths.
      local_files_only : offline mode (no network). Mirrors midas.
      palette          : the class-ID -> RGB table. Defaults to the pinned
                         SEG_CITYSCAPES_PALETTE (the SSOT for this project).
    """

    # ImageNet normalization constants SegFormer was trained with. Registered as
    # buffers (below) so they move with .to(device) and match autocast dtype.
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        size: int,
        model: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        seg_input_size: int = 512,
        local_files_only: bool = True,
        palette: list[tuple[int, int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.model_name   = model
        self.size         = size
        self.seg_input_size = seg_input_size

        # Import here (not at module level) so the rest of the project can import
        # seg_encoder.py without requiring transformers to be installed — only the
        # actual encoder construction path needs it.
        from transformers import SegformerForSemanticSegmentation

        # The frozen SegFormer. requires_grad_(False) + eval() because this is a
        # FIXED annotator, never trained — exactly like DepthEstimator.
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained(
            model, local_files_only=local_files_only
        )
        self.seg_model.requires_grad_(False)
        self.seg_model.eval()

        self.num_classes = self.seg_model.config.num_labels   # 19 for Cityscapes

        # Palette as a [num_classes, 3] buffer in [0,1]. Buffer (not parameter)
        # so it is NOT optimised but DOES follow .to(device)/dtype automatically.
        pal = seg_palette_tensor(palette if palette is not None else SEG_CITYSCAPES_PALETTE)
        if pal.shape[0] != self.num_classes:
            raise ValueError(
                f"SEG_CITYSCAPES_PALETTE has {pal.shape[0]} colours but the "
                f"checkpoint has {self.num_classes} classes — they must match exactly."
            )
        self.register_buffer("palette", pal, persistent=False)

        # ImageNet mean/std as [1,3,1,1] buffers for broadcast normalization.
        self.register_buffer(
            "_seg_mean",
            torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_seg_std",
            torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def _predict_ids(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Core predictor: [-1,1] RGB batch -> class-ID map [B, size, size] (long).

        This is the ONE place SegFormer runs. Both forward() (colour map) and
        label_ids() (raw IDs for offline save) call it, so the offline maps
        saved for training and the live maps at inference are produced by
        byte-for-byte the same computation — the train/inference parity guarantee.

        Steps:
          1. Assert input contract: 4-D, 3-channel, in [-1, 1] (same asserts as midas).
          2. (x + 1.0) / 2.0  ->  [0, 1].
          3. Resize to seg_input_size (bilinear) — SegFormer's expected input scale.
             A 512x512 input is already square (letterboxed upstream), so this is
             a uniform rescale, never a crop. (Unlike MiDaS, SegFormer has no
             forced internal center-crop — squaring upstream is still required to
             prevent the LoRAdapter encoder call in sample_easy from receiving a
             non-square tensor, but SegFormer itself handles any H×W cleanly.)
          4. ImageNet normalize: (x - mean) / std. Buffers broadcast and match
             the device/dtype of x automatically.
          5. SegFormer forward: logits [B, 19, H/4, W/4].
          6. Bilinear-upsample logits to (size, size), THEN argmax(dim=1).
             (Upsample logits then argmax produces smoother class boundaries than
             argmax-then-nearest-upsample — the standard HF recipe.)
          7. Return [B, size, size] long ids in [0, num_classes-1].
        """
        assert imgs.dim() == 4,          f"expected [B,3,H,W], got {tuple(imgs.shape)}"
        assert imgs.shape[1] == 3,       "segmentation encoder input must have 3 channels"
        assert imgs.min() >= -1.0 - 1e-4, "segmentation encoder input must be >= -1"
        assert imgs.max() <=  1.0 + 1e-4, "segmentation encoder input must be <= 1"

        x = (imgs + 1.0) / 2.0                       # [-1,1] -> [0,1]

        # Resize to SegFormer's input scale. Already-square input (letterboxed
        # upstream) makes this a uniform rescale, no distortion.
        x = F.interpolate(
            x, size=(self.seg_input_size, self.seg_input_size),
            mode="bilinear", align_corners=False,
        )

        # ImageNet normalize: buffers broadcast and match device/dtype of x.
        x = (x - self._seg_mean.to(x.dtype)) / self._seg_std.to(x.dtype)

        logits = self.seg_model(pixel_values=x).logits   # [B, 19, H/4, W/4]

        # Upsample class logits to the conditioning canvas, then pick the winner.
        logits = F.interpolate(
            logits.float(), size=(self.size, self.size),
            mode="bilinear", align_corners=False,
        )
        ids = logits.argmax(dim=1)   # [B, size, size] long
        return ids

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Encoder-slot entry point. [-1,1] RGB -> [0,1] colour seg map [B,3,size,size].

        This is what src/model.py calls as `cond = encoder(c)` during LIVE
        inference (sample_easy). The returned colour map goes to the mapper,
        exactly where the depth map would go. Output range/shape match midas
        so no downstream change is needed.

        Input:  [B, 3, H, W] in [-1, 1]
        Output: [B, 3, size, size] in [0, 1]
        """
        ids = self._predict_ids(imgs)             # [B, size, size]
        return seg_colorize_ids(ids, self.palette)  # [B, 3, size, size] in [0,1]

    @torch.no_grad()
    def label_ids(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Offline-only entry point: return the raw class-ID map (NOT colourised).

        Used ONLY by calculate_segmentation_map.py, which saves the raw IDs as
        an 8-bit PNG (canonical, hand-editable, re-palette-able). The dataset
        (local_seg.py) then colourises at load time using seg_colorize_ids.

        WHY SEPARATE FROM forward(): saving raw IDs (not colour) means a palette
        change never requires re-running SegFormer, and resizing the saved map can
        use NEAREST (label-preserving). forward() and label_ids() share
        _predict_ids(), so the saved IDs ARE identical to what the live encoder
        would predict — the parity guarantee.

        Input:  [B, 3, H, W] in [-1, 1]
        Output: [B, size, size] long, ids in [0, num_classes-1]
        """
        return self._predict_ids(imgs)

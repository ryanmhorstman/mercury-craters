"""Trained detector: a CenterNet-style network that predicts crater centres and radii.

The network looks at a greyscale tile and outputs, at 1/4 resolution:
  heat    probability that a crater centre is at each output pixel
  offset  sub-pixel correction of that centre (x, y)
  logr    log of the crater radius, in input pixels
A crater is read off as a local maximum of `heat`. Because the output is a
centre and a radius, the detector produces circles directly.

One crater can light up two local maxima: the trained models put a weaker
second peak 8-9 output cells (~1.5 radii) west of the real one, too far apart
for the 3x3 peak window. `suppress_twins` keeps only the stronger of two
similar-sized circles whose centres are that close, and `overlap_rule` applies
the same test when merging tiles (docs/phase3-notes.md).

Backbone: torchvision ResNet-34 (ImageNet weights, BSD licence), with its
first layer collapsed to one input channel, plus a small feature-pyramid
decoder back to stride 4.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34

STRIDE = 4
MAX_OBJECTS = 256


# ---------------------------------------------------------------- model

def conv_bn_relu(c_in, c_out, k=3):
    return nn.Sequential(nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


def head(c_in, c_out, bias=0.0):
    layer = nn.Sequential(nn.Conv2d(c_in, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, c_out, 1))
    nn.init.constant_(layer[-1].bias, bias)
    return layer


class CraterNet(nn.Module):
    def __init__(self, pretrained=True, width=128):
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        grey = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        grey.weight = nn.Parameter(backbone.conv1.weight.sum(dim=1, keepdim=True))
        self.stem = nn.Sequential(grey, backbone.bn1, backbone.relu, backbone.maxpool)   # stride 4
        self.layers = nn.ModuleList([backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4])
        self.lateral = nn.ModuleList([nn.Conv2d(c, width, 1) for c in (64, 128, 256, 512)])
        self.smooth = conv_bn_relu(width, width)
        self.heat = head(width, 1, bias=-2.19)     # starts at ~10% probability everywhere (CenterNet init)
        self.offset = head(width, 2)
        self.logr = head(width, 1, bias=3.0)       # starts at radius ~20 px

    def forward(self, x):
        features = []
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        p = self.lateral[3](features[3])
        for i in (2, 1, 0):
            p = F.interpolate(p, size=features[i].shape[-2:], mode="nearest") + self.lateral[i](features[i])
        p = self.smooth(p)
        return {"heat": self.heat(p), "offset": self.offset(p), "logr": self.logr(p)}


# ---------------------------------------------------------------- targets and loss

def standardize(image):
    """Per-image zero mean, unit spread; makes the model indifferent to mosaic brightness."""
    image = image.astype(np.float32)
    return (image - image.mean()) / (image.std() + 1e-6)


def make_targets(craters, ignore, size, ignore_background=False):
    """Training targets for one crop.

    craters: (n, 3) array of col, row, radius in crop pixels, to be detected.
    ignore:  (m, 3) array of circles the model shouldn't be punished for finding
             (catalog craters outside the trained size range).
    ignore_background: no "no crater here" loss anywhere in the crop, only the
             labelled craters count (crops where the catalog is known to be incomplete).
    """
    out = size // STRIDE
    heat = np.zeros((out, out), np.float32)
    ignore_mask = np.full((out, out), 1.0 if ignore_background else 0.0, np.float32)
    index = np.zeros(MAX_OBJECTS, np.int64)
    offset = np.zeros((MAX_OBJECTS, 2), np.float32)
    logr = np.zeros(MAX_OBJECTS, np.float32)
    present = np.zeros(MAX_OBJECTS, np.float32)
    ys, xs = np.mgrid[0:out, 0:out]

    for col, row, r in ignore:
        cx, cy = col / STRIDE - 0.5, row / STRIDE - 0.5
        reach = max(2.0, 0.5 * r / STRIDE)
        ignore_mask[(xs - cx) ** 2 + (ys - cy) ** 2 <= reach ** 2] = 1.0

    k = 0
    for col, row, r in craters:
        cx, cy = col / STRIDE - 0.5, row / STRIDE - 0.5
        ix, iy = int(round(cx)), int(round(cy))
        if not (0 <= ix < out and 0 <= iy < out) or k >= MAX_OBJECTS:
            continue
        sigma = max(1.0, r / STRIDE / 4.0)
        heat = np.maximum(heat, np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma ** 2)))
        index[k] = iy * out + ix
        offset[k] = (cx - ix, cy - iy)
        logr[k] = np.log(r)
        present[k] = 1.0
        k += 1

    ignore_mask[heat > 0.01] = 0.0   # a true crater's neighbourhood is never ignored
    return {"heat": heat[None], "ignore": ignore_mask[None], "index": index,
            "offset": offset, "logr": logr, "present": present}


def focal_loss(logits, target, ignore):
    """CenterNet's penalty-reduced focal loss; ignored pixels contribute nothing."""
    pred = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    positive = target.eq(1).float()
    negative = (1 - positive) * (1 - ignore)
    pos_loss = torch.log(pred) * (1 - pred) ** 2 * positive
    neg_loss = torch.log(1 - pred) * pred ** 2 * (1 - target) ** 4 * negative
    n_pos = positive.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / n_pos


def gather(feature_map, index):
    """(B, C, H, W) values at flat indices (B, K) -> (B, K, C)."""
    b, c = feature_map.shape[:2]
    flat = feature_map.float().view(b, c, -1)
    return flat.gather(2, index.unsqueeze(1).expand(-1, c, -1)).permute(0, 2, 1)


def loss_fn(output, batch):
    present = batch["present"].unsqueeze(-1)
    n = present.sum().clamp(min=1.0)
    heat = focal_loss(output["heat"], batch["heat"], batch["ignore"])
    offset = (F.l1_loss(gather(output["offset"], batch["index"]), batch["offset"], reduction="none") * present).sum() / n
    logr = (F.l1_loss(gather(output["logr"], batch["index"]), batch["logr"].unsqueeze(-1), reduction="none") * present).sum() / n
    return heat + offset + logr, {"heat": heat.item(), "offset": offset.item(), "logr": logr.item()}


# ---------------------------------------------------------------- decoding and inference

def suppress_twins(circles, radius_frac, ratio_min, ratio_max):
    """Drop each circle whose centre lies within radius_frac x the radius of a
    higher-scoring circle of similar size (radius ratio within [ratio_min, ratio_max]).

    circles: DataFrame of col, row, radius_px, score. Greedy, best score first.
    """
    circles = circles.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    xy = circles[["col", "row"]].to_numpy()
    r = circles["radius_px"].to_numpy()
    distance = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    ratio = r[None, :] / r[:, None]                      # [kept, other]: other's radius over kept's
    covers = (distance <= radius_frac * r[:, None]) & (ratio >= ratio_min) & (ratio <= ratio_max)
    keep = np.ones(len(circles), dtype=bool)
    for i in range(len(circles)):
        if keep[i]:
            keep[i + 1:] &= ~covers[i, i + 1:]
    return circles[keep].reset_index(drop=True)


@torch.no_grad()
def decode(output, min_score=0.05, top_k=600, twins=None):
    """Network output for one image -> DataFrame of col, row, radius_px, score (input pixels).

    twins: optional dict radius_frac, ratio_min, ratio_max for `suppress_twins`.
    """
    heat = torch.sigmoid(output["heat"][0, 0].float())
    peaks = heat == F.max_pool2d(heat[None, None], 3, stride=1, padding=1)[0, 0]
    scores = (heat * peaks).flatten()
    k = min(top_k, scores.numel())
    values, flat_index = scores.topk(k)
    keep = values >= min_score
    values, flat_index = values[keep], flat_index[keep]
    width = heat.shape[1]
    iy, ix = flat_index // width, flat_index % width
    off = output["offset"][0].float()
    col = (ix.float() + off[0, iy, ix] + 0.5) * STRIDE
    row = (iy.float() + off[1, iy, ix] + 0.5) * STRIDE
    radius = torch.exp(output["logr"][0, 0].float()[iy, ix])
    circles = pd.DataFrame({"col": col.cpu().numpy(), "row": row.cpu().numpy(),
                            "radius_px": radius.cpu().numpy(), "score": values.cpu().numpy()})
    return suppress_twins(circles, **twins) if twins else circles


def overlap_rule(match, twins):
    """Rule for merging detections across overlapping tiles: the usual same-crater
    rule, widened to the twin rule so a twin split across two tiles is merged too."""
    if not twins:
        return match
    return {"center_frac": max(match["center_frac"], twins["radius_frac"] / 2.0),   # radii -> diameters
            "ratio_min": twins["ratio_min"], "ratio_max": twins["ratio_max"]}


class Detector:
    """Callable for detect.run_detector: (image, tile, level_cfg) -> circles in tile pixels."""

    def __init__(self, checkpoint, downsample, device=None, min_score=0.05, twins=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CraterNet(pretrained=False).to(self.device).eval()
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model"] if "model" in state else state)
        self.downsample = downsample          # {level name: factor}
        self.min_score = min_score
        self.twins = twins

    @classmethod
    def from_model(cls, model, downsample, min_score=0.05, twins=None):
        detector = cls.__new__(cls)
        detector.device = str(next(model.parameters()).device)
        detector.model, detector.downsample, detector.min_score, detector.twins = model, downsample, min_score, twins
        return detector

    @torch.no_grad()
    def __call__(self, image, tile, level_cfg):
        import cv2

        f = self.downsample.get(tile.level, 1)
        if f > 1:
            image = cv2.resize(image, (image.shape[1] // f, image.shape[0] // f), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(standardize(image))[None, None].to(self.device)
        with torch.autocast(device_type=self.device.split(":")[0], enabled=self.device.startswith("cuda")):
            output = self.model(x)
        circles = decode(output, self.min_score, twins=self.twins)
        circles[["col", "row", "radius_px"]] *= f
        return circles


class LevelDetector:
    """Dispatches each tile to the detector for its level, falling back to a default."""

    def __init__(self, default, by_level):
        self.default, self.by_level = default, by_level

    def __call__(self, image, tile, level_cfg):
        return self.by_level.get(tile.level, self.default)(image, tile, level_cfg)


def build_detector(params, models_dir, min_score):
    """The configured model, with per-level replacements from detectors.cnn.level_models."""
    def one(name):
        return Detector(models_dir / f"{name}_best.pt", params["level_downsample"], min_score=min_score,
                        twins=params["twin_suppression"])

    by_level = {level: one(name) for level, name in (params.get("level_models") or {}).items()}
    return LevelDetector(one(params["model_name"]), by_level) if by_level else one(params["model_name"])

"""Train the CNN detector on the training split; choose the checkpoint on V1 F-beta.

Each training sample is a "view" of a cached tile:
  fine tiles   random 512 px crop at 166 m/px, labelled for 5-40 km craters
  coarse tiles whole tile shrunk 2x to 512 px (1,330 m/px), labelled for 30-400 km
Craters outside a view's size range are marked "ignore", not "no crater".
Augmentation: random scale, the 8 flips/rotations, brightness/contrast/gamma.

Labels are the original catalog, or (detectors.cnn.use_registered_labels) the
north-south registered copy built by `python -m craters.phase4 register-labels`.

Test regions T1-T3 are never read here.
"""
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .detect import load_tile_image, run_detector
from .detectors.cnn import CraterNet, Detector, loss_fn, make_targets, overlap_rule, standardize
from .metrics import evaluate
from .tiling import catalog_in_tile, tile_from_row


class EpochSampler(Sampler):
    """Yields new sample numbers every epoch: epoch e gets e*n .. e*n + n - 1.

    Each sample number seeds that sample's random choices, so runs are
    reproducible and no two epochs repeat, even with persistent workers.
    """

    def __init__(self, n):
        self.n, self.epoch = n, 0

    def __iter__(self):
        return iter(range(self.epoch * self.n, (self.epoch + 1) * self.n))

    def __len__(self):
        return self.n


class TileViews(Dataset):
    def __init__(self, tiles, catalog, cfg, params, seed):
        self.cfg, self.p, self.seed = cfg, params, seed
        self.tiles = {"fine": [], "coarse": []}
        for _, row in tiles.iterrows():
            tile = tile_from_row(row)
            craters = catalog_in_tile(tile, catalog, margin_px=64)
            circles = craters[["col", "row", "radius_px"]].to_numpy(np.float32)
            self.tiles[tile.level].append((tile, circles, craters["diameter_km"].to_numpy()))

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i))
        p = self.p
        level = "coarse" if rng.random() < p["coarse_fraction"] else "fine"
        tile, circles, diameter_km = self.tiles[level][rng.integers(len(self.tiles[level]))]
        image = load_tile_image(tile, self.cfg)
        lo, hi = p["label_km"][level]
        crop = p["crop_px"]

        if level == "coarse":
            f = image.shape[0] / crop
            view = cv2.resize(image, (crop, crop), interpolation=cv2.INTER_AREA)
            c = circles / f
        else:
            scale = float(np.exp(rng.uniform(*np.log(p["scale_jitter"]))))
            src = min(int(round(crop * scale)), image.shape[0])
            x0, y0 = rng.integers(0, image.shape[1] - src + 1), rng.integers(0, image.shape[0] - src + 1)
            view = image[y0:y0 + src, x0:x0 + src]
            interp = cv2.INTER_AREA if src > crop else cv2.INTER_LINEAR
            view = cv2.resize(view, (crop, crop), interpolation=interp)
            c = (circles - [x0, y0, 0]) * (crop / src)

        label = (diameter_km >= lo) & (diameter_km <= hi)
        view, c = dihedral(view, c, rng.integers(8), crop)
        view = photometric(view, rng)
        targets = make_targets(c[label], c[~label], crop)
        targets["image"] = standardize(view)[None]
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in targets.items()}


def dihedral(image, circles, k, size):
    """One of the 8 flips/rotations of a square image, applied to circle centres too."""
    col, row = circles[:, 0].copy(), circles[:, 1].copy()
    if k & 1:
        image, col = image[:, ::-1], size - col
    if k & 2:
        image, row = image[::-1, :], size - row
    if k & 4:
        image, col, row = image.T, row, col
    return image, np.column_stack([col, row, circles[:, 2]]) if len(circles) else circles


def photometric(image, rng):
    x = image.astype(np.float32) / 255.0
    x = np.clip(x * rng.uniform(0.75, 1.3) + rng.uniform(-0.1, 0.1), 0, 1) ** rng.uniform(0.7, 1.4)
    return x * 255.0


def f_beta(precision, recall, beta):
    """F-beta score; beta > 1 weights recall above precision, beta = 1 is F1."""
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall) if precision + recall > 0 else 0.0


def validate(model, cfg, params, tiles, catalog, images):
    """End-to-end V1 F1 and F-beta at each score cutoff, through the same path as the test run."""
    model.eval()
    twins = params["twin_suppression"]
    detector = Detector.from_model(model, params["level_downsample"], min_score=min(params["score_cutoffs"]), twins=twins)
    boxes = list(cfg["regions"]["validation"].values())
    detections = run_detector(detector, tiles, boxes, cfg, images=images, progress=False,
                              merge=overlap_rule(cfg["match"], twins))
    rows = []
    for cutoff in params["score_cutoffs"]:
        kept = detections[detections["score"] >= cutoff]
        overall = evaluate(catalog, kept, boxes, cfg["match"], cfg["evaluation"]["diameter_bins_km"])[0].iloc[0]
        rows.append({"score_cutoff": cutoff, **overall[["TP", "FP", "FN", "precision", "recall", "f1"]].to_dict(),
                     "fbeta": f_beta(overall["precision"], overall["recall"], params["selection_beta"])})
    model.train()
    return pd.DataFrame(rows)


def train(cfg):
    params = cfg["detectors"]["cnn"]
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tiles = pd.read_parquet(cfg["paths"]["tiles"])
    labels = cfg["paths"]["catalog_registered_labels"] if params["use_registered_labels"] else cfg["paths"]["catalog"]
    catalog = pd.read_parquet(labels)
    print(f"labels: {labels}", flush=True)
    train_tiles = tiles[tiles["split"] == "train"]
    dataset = TileViews(train_tiles, catalog, cfg, params, cfg["seed"])
    sampler = EpochSampler(params["samples_per_epoch"])
    loader = DataLoader(dataset, batch_size=params["batch_size"], sampler=sampler, num_workers=params["workers"],
                        persistent_workers=params["workers"] > 0, drop_last=True)

    from .phase3 import preload
    val_images = preload(tiles, list(cfg["regions"]["validation"].values()), cfg)

    model = CraterNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    steps = params["epochs"] * len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=params["lr"], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    out_dir = cfg["paths"]["models"]
    out_dir.mkdir(parents=True, exist_ok=True)
    name = params["model_name"]
    history, best = [], {"fbeta": -1.0}
    for epoch in range(1, params["epochs"] + 1):
        sampler.epoch = epoch
        start, totals = time.time(), {"loss": 0.0, "heat": 0.0, "offset": 0.0, "logr": 0.0}
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                output = model(batch["image"])
            loss, parts = loss_fn(output, batch)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            totals["loss"] += loss.item()
            for k, v in parts.items():
                totals[k] += v
        record = {"epoch": epoch, **{k: v / len(loader) for k, v in totals.items()}, "seconds": time.time() - start}

        if epoch % params["eval_every"] == 0 or epoch == params["epochs"]:
            table = validate(model, cfg, params, tiles, catalog, val_images)
            top = table.loc[table["fbeta"].idxmax()]
            record.update({"v1_fbeta": top["fbeta"], "v1_f1": top["f1"], "v1_precision": top["precision"], "v1_recall": top["recall"],
                           "v1_cutoff": top["score_cutoff"]})
            if top["fbeta"] > best["fbeta"]:
                best = {"fbeta": float(top["fbeta"]), "beta": params["selection_beta"], "f1": float(top["f1"]),
                        "epoch": epoch, "score_cutoff": float(top["score_cutoff"])}
                torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / f"{name}_best.pt")
        history.append(record)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)

    pd.DataFrame(history).to_csv(out_dir / f"{name}_history.csv", index=False)
    (out_dir / f"{name}_best.json").write_text(json.dumps(best, indent=2))
    print(f"best V1 F{params['selection_beta']:g} {best['fbeta']:.3f} (F1 {best['f1']:.3f}) at epoch {best['epoch']} (score >= {best['score_cutoff']})")
    return best

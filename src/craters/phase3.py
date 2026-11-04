"""Phase 3: run detectors. Each is tuned on V1 only, then run once on T1-T3.

    python -m craters.phase3 cache-tiles     # render train + V1 tiles once
    python -m craters.phase3 hough
    python -m craters.phase3 cnn             # train (~1 h on an RTX 3060 Ti), then run
    python -m craters.phase3 global          # run the chosen CNN over the whole planet

Writes data/detections/<detector>.parquet for Phase 2 to score, and the
tuning record to outputs/phase3/. The global run writes
data/detections/global_<model_name>.parquet; see docs/global-run.md.
"""
import argparse
import json
import time

import pandas as pd

from .config import load_config
from .detect import detect_on_tiles, load_tile_image, run_detector, tiles_covering
from .detectors import hough
from .metrics import evaluate
from .tiling import make_cap_tiles, tile_from_row, tiles_frame


def preload(tiles, boxes, cfg):
    """Render (or load cached) every tile touching the boxes, once."""
    images = {}
    for level in cfg["tiling"]["levels"]:
        for _, row in tiles_covering(tiles, boxes, level).iterrows():
            images[row["tile_id"]] = load_tile_image(tile_from_row(row), cfg)
    return images


def tune_hough(cfg, tiles, catalog):
    boxes = list(cfg["regions"]["validation"].values())
    bins = cfg["evaluation"]["diameter_bins_km"]
    images = preload(tiles, boxes, cfg)
    records = []
    for params in cfg["detectors"]["hough"]["grid"]:
        start = time.time()
        detections = run_detector(lambda im, t, lvl: hough.detect(im, t, lvl, params), tiles, boxes, cfg,
                                  images=images, progress=False)
        for cutoff in cfg["detectors"]["hough"]["score_cutoffs"]:
            kept = detections[detections["score"] >= cutoff]
            overall = evaluate(catalog, kept, boxes, cfg["match"], bins)[0].iloc[0]
            records.append({**params, "score_cutoff": cutoff, "n_detections": len(kept),
                            **overall[["TP", "FP", "FN", "precision", "recall", "f1"]].to_dict()})
        best_here = max(r["f1"] for r in records[-len(cfg["detectors"]["hough"]["score_cutoffs"]):])
        print(f"  {params}: {len(detections)} raw detections, best V1 F1 {best_here:.3f} ({time.time() - start:.0f}s)", flush=True)
    table = pd.DataFrame(records).sort_values("f1", ascending=False)
    best = table.iloc[0]
    params = {k: best[k] for k in hough.DEFAULTS if k in best and pd.notna(best[k])}
    return params, float(best["score_cutoff"]), table


def run_hough(cfg):
    tiles = pd.read_parquet(cfg["paths"]["tiles"])
    catalog = pd.read_parquet(cfg["paths"]["catalog"])
    out_dir = cfg["paths"]["phase2"].parent / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("tuning on V1")
    params, cutoff, table = tune_hough(cfg, tiles, catalog)
    params = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
    table.to_csv(out_dir / "hough_tuning_V1.csv", index=False)
    (out_dir / "hough_params.json").write_text(json.dumps({"params": params, "score_cutoff": cutoff}, indent=2))
    print(f"chosen: {params}, score >= {cutoff}")
    print(table.head(5).round(3).to_string(index=False))

    print("running on T1-T3")
    test_boxes = list(cfg["regions"]["test"].values())
    detections = run_detector(lambda im, t, lvl: hough.detect(im, t, lvl, params), tiles, test_boxes, cfg)
    detections = detections[detections["score"] >= cutoff].reset_index(drop=True)
    cfg["paths"]["detections"].mkdir(parents=True, exist_ok=True)
    detections.to_parquet(cfg["paths"]["detections"] / "hough.parquet", index=False)
    print(f"wrote {len(detections)} detections")


def run_cnn(cfg, skip_training=False):
    from .detectors.cnn import Detector
    from .train_cnn import train

    params = cfg["detectors"]["cnn"]
    models, name = cfg["paths"]["models"], params["model_name"]
    best = json.loads((models / f"{name}_best.json").read_text()) if skip_training else train(cfg)

    print(f"running on T1-T3 with epoch {best['epoch']} checkpoint, score >= {best['score_cutoff']}")
    tiles = pd.read_parquet(cfg["paths"]["tiles"])
    detector = Detector(models / f"{name}_best.pt", params["level_downsample"], min_score=best["score_cutoff"])
    test_boxes = list(cfg["regions"]["test"].values())
    detections = run_detector(detector, tiles, test_boxes, cfg)
    cfg["paths"]["detections"].mkdir(parents=True, exist_ok=True)
    detections.to_parquet(cfg["paths"]["detections"] / f"{name}.parquet", index=False)
    print(f"wrote {len(detections)} detections")


def _cache_one(row):
    cfg = load_config()
    load_tile_image(tile_from_row(row), cfg)


def cache_tiles(cfg, splits, workers=6):
    """Render every tile of the given splits into the tile cache, in parallel."""
    tiles = pd.read_parquet(cfg["paths"]["tiles"])
    print(f"caching tiles of {', '.join(splits)}", flush=True)
    cache_tile_rows(tiles[tiles["split"].isin(splits)], workers)


def cache_tile_rows(tiles, workers=6):
    from multiprocessing import Pool

    rows = [row for _, row in tiles.iterrows()]
    print(f"caching {len(rows)} tiles", flush=True)
    with Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_cache_one, rows, chunksize=4), start=1):
            if i % 200 == 0:
                print(f"  {i}/{len(rows)}", flush=True)


def global_tiles(cfg):
    """The Phase 1 tiles (|lat| <= lat_limit, ids unchanged) plus polar-cap rows."""
    tiles = pd.read_parquet(cfg["paths"]["tiles"])
    caps = [tiles_frame(make_cap_tiles(name, level, cfg["tiling"]["lat_limit"], cfg["regions"]))
            for name, level in cfg["tiling"]["levels"].items()]
    tiles = pd.concat([tiles, *caps], ignore_index=True)
    tiles.to_parquet(cfg["paths"]["tiles_global"], index=False)
    print(pd.crosstab(tiles["level"], tiles["split"], margins=True).to_string(), flush=True)
    return tiles


def run_global(cfg):
    """Run the chosen CNN over every tile on the planet; same levels, merging and cutoff as T1-T3."""
    from .detectors.cnn import Detector

    params = cfg["detectors"]["cnn"]
    models, name = cfg["paths"]["models"], params["model_name"]
    best = json.loads((models / f"{name}_best.json").read_text())
    tiles = global_tiles(cfg)

    start = time.time()
    cache_tile_rows(tiles)          # ~5.6 GB in total; tiles already cached are just checked
    render_s = time.time() - start

    print(f"detecting with {name} (epoch {best['epoch']}, score >= {best['score_cutoff']})", flush=True)
    start = time.time()
    detector = Detector(models / f"{name}_best.pt", params["level_downsample"], min_score=best["score_cutoff"])
    detections = detect_on_tiles(detector, tiles, cfg, progress=True)
    detect_s = time.time() - start

    columns = ["lon", "lat", "diameter_km", "score", "level", "n_tiles", "tile_id"]
    out = cfg["paths"]["detections"] / f"global_{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    detections[columns].to_parquet(out, index=False)
    timing = {"model": name, "tiles": len(tiles), "detections": len(detections),
              "render_s": round(render_s), "detect_and_merge_s": round(detect_s)}
    cfg["paths"]["global"].mkdir(parents=True, exist_ok=True)
    (cfg["paths"]["global"] / "run_timing.json").write_text(json.dumps(timing, indent=2))
    print(f"wrote {len(detections)} detections to {out}; {timing}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("detector", choices=["hough", "cnn", "cache-tiles", "global"])
    parser.add_argument("--skip-training", action="store_true", help="cnn: reuse the saved best checkpoint")
    args = parser.parse_args()
    cfg = load_config()
    if args.detector == "hough":
        run_hough(cfg)
    elif args.detector == "cnn":
        run_cnn(cfg, args.skip_training)
    elif args.detector == "cache-tiles":
        cache_tiles(cfg, ["train", "V1"])
    elif args.detector == "global":
        run_global(cfg)


if __name__ == "__main__":
    main()

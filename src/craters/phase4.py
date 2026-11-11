"""Phase 4: adjudicate disagreements between the catalog and the detectors.

    python -m craters.phase4 register  # correct the catalog's north-south registration (writes a working copy)
    python -m craters.phase4 register-labels  # same, extended over train + V1, for retraining the CNN
    python -m craters.phase4 build     # make the review queue from data/detections/*.parquet
    python -m craters.phase4 review    # open the review app at http://localhost:8765
    python -m craters.phase4 report    # original vs adjusted metrics, from the verdicts so far

    python -m craters.phase4 build --calibrated    # queue of the published calibrated run's unmatched detections
    python -m craters.phase4 review --calibrated   # review them (any first N of the queue is a random sample)
    python -m craters.phase4 report --calibrated   # how many of its detections are false

Verdicts live in adjudication/verdicts.csv, which is tracked in git. They are
hand-made data, unlike everything under data/ and outputs/.
"""
import argparse
import json

import pandas as pd

from .adjudicate import adjusted_metrics, build_items, catalog_summary, false_positive_summary, verdicts_for
from .catalog import load_eval_catalog
from .config import load_config, run_name
from .metrics import bin_labels
from .registration import matched_offsets, register


def detectors(cfg):
    paths = sorted(cfg["paths"]["detections"].glob("*.parquet"))
    return {path.stem: pd.read_parquet(path) for path in paths
            if not path.stem.startswith("global_")}     # the whole-planet run is compared in compare_global


def register_catalog(cfg):
    """Correct the catalog's north-south registration and map the shifts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = cfg["registration"]
    catalog = pd.read_parquet(cfg["paths"]["catalog"])
    found = {k: v for k, v in detectors(cfg).items() if k in params["detectors"]}
    corrected, history = register(catalog, found, cfg["match"], params)
    corrected.to_parquet(cfg["paths"]["catalog_registered"], index=False)

    out = cfg["paths"]["phase4"]
    out.mkdir(parents=True, exist_ok=True)
    history.to_csv(out / "registration_history.csv", index=False)
    print(history.to_string(index=False))
    moved = corrected[corrected["registered"]]
    print(f"\n{len(moved)} catalog craters corrected; north shift (km) by region:")
    print(moved.groupby("split")["north_shift_km"].describe()[["count", "min", "50%", "max"]].round(2).to_string())

    fig, axes = plt.subplots(1, len(cfg["regions"]["test"]), figsize=(15, 4.6))
    for ax, (name, box) in zip(axes, cfg["regions"]["test"].items()):
        rows = moved[moved["split"] == name]
        points = ax.scatter(rows["lon"], rows["lat_catalog"], c=rows["north_shift_km"], s=6, cmap="viridis", vmin=-7, vmax=0)
        ax.set_xlim(box[0], box[1])
        ax.set_ylim(box[2], box[3])
        ax.set_title(f"{name}: catalog north shift applied", fontsize=10)
        ax.set_xlabel("longitude E")
        ax.set_ylabel("latitude N")
    fig.colorbar(points, ax=axes, label="shift applied (km, negative = moved south)")
    fig.savefig(out / "registration_shift_map.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {cfg['paths']['catalog_registered']} and {out / 'registration_shift_map.png'}")


def label_detector(cfg, name):
    """The already-tuned detector behind data/detections/<name>.parquet, as (detect_fn, score cutoff)."""
    if name == "hough":
        from .detectors import hough

        chosen = json.loads((cfg["paths"]["phase2"].parent / "phase3" / "hough_params.json").read_text())
        return (lambda im, t, lvl: hough.detect(im, t, lvl, chosen["params"])), chosen["score_cutoff"]
    from .detectors.cnn import Detector

    models = cfg["paths"]["models"]
    best = json.loads((models / f"{name}_best.json").read_text())
    params = cfg["detectors"]["cnn"]
    detector = Detector(models / f"{name}_best.pt", params["level_downsample"], min_score=best["score_cutoff"],
                        twins=params["twin_suppression"])
    return detector, best["score_cutoff"]


def register_training_labels(cfg):
    """Registration as in `register`, with the shift field also covering train + V1.

    The test-region shift field comes from test-region detections only, so train
    and V1 craters were never moved. Here the same detectors run over every train
    tile and every tile touching V1, and the registration is rebuilt from all of
    them, for CNN labels only. Scoring keeps using catalog_registered.
    """
    from .detect import detect_on_tiles, tiles_covering
    from .detectors.cnn import overlap_rule
    from .geo import in_box

    params = cfg["registration"]
    tiles = pd.read_parquet(cfg["paths"]["tiles"])
    v1 = list(cfg["regions"]["validation"].values())
    chosen = pd.concat([tiles[tiles["split"] == "train"],
                        *(tiles_covering(tiles, v1, level) for level in cfg["tiling"]["levels"])])
    chosen = chosen.drop_duplicates("tile_id")
    out_dir = cfg["paths"]["detections_labels"]
    out_dir.mkdir(parents=True, exist_ok=True)

    found = {}
    for name in params["detectors"]:
        path = out_dir / f"{name}.parquet"
        if not path.exists():
            print(f"running {name} over {len(chosen)} train + V1 tiles", flush=True)
            detect_fn, cutoff = label_detector(cfg, name)
            merge = None if name == "hough" else overlap_rule(cfg["match"], cfg["detectors"]["cnn"]["twin_suppression"])
            detections = detect_on_tiles(detect_fn, chosen, cfg, merge=merge)
            detections[detections["score"] >= cutoff].reset_index(drop=True).to_parquet(path, index=False)
        found[name] = pd.concat([pd.read_parquet(cfg["paths"]["detections"] / f"{name}.parquet"),
                                 pd.read_parquet(path)], ignore_index=True)

    catalog = pd.read_parquet(cfg["paths"]["catalog"])
    # The old CNN trained on unshifted train labels; if it had learned the shift, its
    # train-region offsets would sit near 0 while Hough's stay negative.
    offsets = matched_offsets(catalog, found, cfg["match"], params["min_diameter_km"])
    offsets["area"] = "train / other"
    for name, box in {**cfg["regions"]["test"], **cfg["regions"]["validation"]}.items():
        offsets.loc[in_box(offsets["lat"].to_numpy(), offsets["lon"].to_numpy(), box), "area"] = name
    print("median north offset before correction (km), area x detector:")
    print(offsets.pivot_table(index="area", columns="detector", values="north_km", aggfunc="median").round(2).to_string())

    corrected, history = register(catalog, found, cfg["match"], params)
    corrected.to_parquet(cfg["paths"]["catalog_registered_labels"], index=False)
    print(history.to_string(index=False))
    summary = corrected.groupby("split").agg(craters=("registered", "size"), moved=("registered", "sum"),
                                             median_shift_km=("north_shift_km", "median"),
                                             largest_shift_km=("north_shift_km", "min"))
    print(summary.round(2).to_string())
    print(f"wrote {cfg['paths']['catalog_registered_labels']}")


def build(cfg):
    catalog = load_eval_catalog(cfg)
    boxes = list(cfg["regions"]["test"].values())
    found = detectors(cfg)
    items, _ = build_items(catalog, found, boxes, cfg["match"], cfg["evaluation"]["diameter_bins_km"], cfg["seed"])
    items.to_csv(cfg["paths"]["review_items"], index=False)
    to_review = items[items["auto_verdict"] == ""]
    print(f"detectors: {', '.join(found)}")
    print(f"{len(items)} items, {len(to_review)} to review "
          f"({(items['auto_verdict'] == 'duplicate').sum()} auto-marked duplicates)")
    print(pd.crosstab(to_review["kind"], to_review["bin"]).to_string())


def calibrated_items(cfg):
    """The published calibrated global run's unmatched detections in T1-T3, in random order, and its metrics table.

    Only detections are queued, so the review answers one question: how many of
    the run's detections the thesis doesn't list are real craters.
    """
    name = f"global_{run_name(cfg['detectors']['cnn'])}_calibrated"
    detections = pd.read_parquet(cfg["paths"]["detections"] / f"{name}.parquet")
    items, tables = build_items(load_eval_catalog(cfg), {name: detections}, list(cfg["regions"]["test"].values()),
                                cfg["match"], cfg["evaluation"]["diameter_bins_km"], cfg["seed"])
    items = items[items["kind"] == "detection"].reset_index(drop=True)
    items["position"] = range(1, len(items) + 1)
    return name, items, tables[name]


def build_calibrated(cfg):
    name, items, _ = calibrated_items(cfg)
    items.to_csv(cfg["paths"]["review_items_calibrated"], index=False)
    to_review = items[items["auto_verdict"] == ""]
    print(f"{name}: {len(items)} unmatched detections in T1-T3, {len(to_review)} to review "
          f"({len(items) - len(to_review)} auto-marked duplicates)")
    print(to_review["bin"].value_counts().reindex(bin_labels(cfg["evaluation"]["diameter_bins_km"])).to_string())


def report_calibrated(cfg):
    name, items, table = calibrated_items(cfg)
    verdicts, _ = verdicts_for(items, cfg["paths"]["verdicts"])
    summary = false_positive_summary(items, verdicts, table, cfg["evaluation"]["diameter_bins_km"])
    out = cfg["paths"]["phase4"]
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "calibrated_false_positives.csv", index=False)
    print(f"{name}, T1-T3: matched detections count as real, reviewed ones as judged, "
          f"the rest at their bin's reviewed rate (95% interval)")
    print(summary.round(3).to_string(index=False))


def report(cfg):
    catalog = load_eval_catalog(cfg)
    bins = cfg["evaluation"]["diameter_bins_km"]
    boxes = list(cfg["regions"]["test"].values())
    found = detectors(cfg)
    items, tables = build_items(catalog, found, boxes, cfg["match"], bins, cfg["seed"])
    verdicts, stale = verdicts_for(items, cfg["paths"]["verdicts"])
    print(f"catalog: {'registered' if 'north_shift_km' in catalog else 'original'} positions; "
          f"{len(verdicts)} verdicts apply, {stale} stale (their circle has moved)")

    first = next(iter(tables.values())).set_index("bin")["n_catalog"]
    summary = catalog_summary(items, verdicts, first.to_dict(), bins)
    adjusted = adjusted_metrics(items, verdicts, tables, bins)

    out = cfg["paths"]["phase4"]
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "catalog_summary.csv", index=False)
    adjusted.to_csv(out / "adjusted_metrics.csv", index=False)

    reviewed = summary.set_index("bin").loc["all"]
    print(f"reviewed {int(reviewed['reviewed'])} of {int(reviewed['to_review'])} items "
          f"({'exact' if reviewed['reviewed'] == reviewed['to_review'] else 'estimates extrapolate within kind x size bin'})\n")
    print("How complete and clean is the catalog (T1-T3)?")
    print(summary.round(3).to_string(index=False))
    print("\nOriginal vs adjusted metrics:")
    cols = ["detector", "bin", "precision", "precision_adj", "recall", "recall_adj", "f1", "f1_adj"]
    order = {b: i for i, b in enumerate(["all", *bin_labels(bins)])}
    print(adjusted.sort_values(["detector", "bin"], key=lambda s: s.map(order) if s.name == "bin" else s)[cols]
          .round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["register", "register-labels", "build", "review", "report"])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--calibrated", action="store_true",
                        help="build / review / report: the published calibrated run's unmatched detections only")
    args = parser.parse_args()
    cfg = load_config()
    if args.calibrated and args.step in ("build", "report"):
        return build_calibrated(cfg) if args.step == "build" else report_calibrated(cfg)
    if args.step == "register":
        register_catalog(cfg)
    elif args.step == "register-labels":
        register_training_labels(cfg)
    elif args.step == "build":
        build(cfg)
    elif args.step == "report":
        report(cfg)
    else:
        from .review_app import ReviewState, serve

        items_path = cfg["paths"]["review_items_calibrated" if args.calibrated else "review_items"]
        if not items_path.exists():
            build_calibrated(cfg) if args.calibrated else build(cfg)
        state = ReviewState(items_path, cfg["paths"]["verdicts"], cfg["paths"]["basemap"],
                            cfg["paths"]["basemap"].parent / "chips")
        serve(state, args.port)


if __name__ == "__main__":
    main()

"""Phase 4: adjudicate disagreements between the catalog and the detectors.

    python -m craters.phase4 register  # correct the catalog's north-south registration (writes a working copy)
    python -m craters.phase4 build     # make the review queue from data/detections/*.parquet
    python -m craters.phase4 review    # open the review app at http://localhost:8765
    python -m craters.phase4 report    # original vs adjusted metrics, from the verdicts so far

Verdicts live in adjudication/verdicts.csv, which is tracked in git. They are
hand-made data, unlike everything under data/ and outputs/.
"""
import argparse

import pandas as pd

from .adjudicate import adjusted_metrics, build_items, catalog_summary, verdicts_for
from .catalog import load_eval_catalog
from .config import load_config
from .metrics import bin_labels
from .registration import register


def detectors(cfg):
    return {path.stem: pd.read_parquet(path) for path in sorted(cfg["paths"]["detections"].glob("*.parquet"))}


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
    parser.add_argument("step", choices=["register", "build", "review", "report"])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    cfg = load_config()
    if args.step == "register":
        register_catalog(cfg)
    elif args.step == "build":
        build(cfg)
    elif args.step == "report":
        report(cfg)
    else:
        from .review_app import ReviewState, serve

        if not cfg["paths"]["review_items"].exists():
            build(cfg)
        state = ReviewState(cfg["paths"]["review_items"], cfg["paths"]["verdicts"], cfg["paths"]["basemap"],
                            cfg["paths"]["basemap"].parent / "chips")
        serve(state, args.port)


if __name__ == "__main__":
    main()

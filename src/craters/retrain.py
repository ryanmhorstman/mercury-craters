"""Does training on the supplemented catalog help? Full retrains, several seeds, EMA on.

    python -m craters.retrain             # train every run not finished yet (GPU, ~1 h each), then report
    python -m craters.retrain --report    # only the comparison of the finished runs

Fine-tuning cnn_reg_f2 couldn't answer this, because any further training of
that checkpoint loses accuracy (docs/self-training.md). So here every model is
trained from ImageNet weights with the recipe of cnn_reg_f2, plus weight
averaging (detectors.cnn.ema), in two arms:

  catalog     labels: the registered thesis catalog
  supplement  the same, plus the supplemented catalog's new craters in the train area

Each arm is trained once per seed in selftrain.retrain.seeds; a seed sets the
weight initialisation of the heads and the order and augmentation of the
samples. Each model then goes through the global run and the calibration, and
is scored in V1 and T1-T3 against the thesis and the supplemented catalog
(craters.supplement). Checkpoints and calibration are chosen on V1 against the
thesis catalog in both arms, and new craters outside the train area are never
trained on.

Models are named retrain_<arm>_s<seed>. Writes, in outputs/selftrain/retrain/:
scores_<model>.csv, runs.csv (one row per model) and summary.csv (mean and
spread over seeds). A finished model is skipped, so an interrupted batch can be
started again.
"""
import argparse
import copy

import pandas as pd

from .calibrate import calibrate
from .config import load_config, run_name
from .supplement import THESIS, score_run, scoring_catalogs

ARMS = ("catalog", "supplement")
SCORED = {"v1_thesis": (THESIS, "V1"), "test_thesis": (THESIS, "T1-T3"),
          "v1_supplemented": ("supplemented", "V1"), "test_supplemented": ("supplemented", "T1-T3")}


def out_dir(cfg):
    return cfg["paths"]["selftrain"] / "retrain"


def model_name(arm, seed):
    return f"retrain_{arm}_s{seed}"


def supplement_labels(cfg):
    """The supplemented catalog's new craters in the train area, as extra training labels."""
    supplemented = pd.read_parquet(cfg["paths"]["catalog_supplemented"])
    new = supplemented[(supplemented["source"] != THESIS) & (supplemented["split"] == "train")]
    path = cfg["paths"]["catalog"].parent / "selftrain" / "supplement_train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    new[["lat", "lon", "diameter_km", "source"]].assign(ignore=False).to_parquet(path, index=False)
    return path


def run_config(cfg, arm, seed, labels_path):
    """The config of one retrain: cnn_reg_f2's recipe from ImageNet weights, with EMA and its own seed."""
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed
    cnn = cfg["detectors"]["cnn"]
    cnn.update(model_name=model_name(arm, seed), level_models={}, init_from=None,
               extra_labels=str(labels_path) if arm == "supplement" else None)
    cnn["ema"]["enabled"] = True
    return cfg


def train_one(cfg):
    """Train, run over the planet, calibrate and score one model."""
    from .phase3 import run_global
    from .train_cnn import train

    name = cfg["detectors"]["cnn"]["model_name"]
    out = out_dir(cfg)
    train(cfg)
    run_global(cfg, timing_path=out / f"run_timing_{name}.json")
    calibrate(cfg, record_path=out / f"calibration_{name}.json", quiet=True)
    detections = pd.read_parquet(cfg["paths"]["detections"] / f"global_{name}_calibrated.parquet")
    score_run(cfg, detections, scoring_catalogs(cfg)).round(4).to_csv(out / f"scores_{name}.csv", index=False)


def f1_row(scores):
    """All-sizes F1 of one model's score table, one value per entry of SCORED."""
    f1 = scores[scores["bin"] == "all"].set_index(["catalog", "region"])["f1"]
    return {column: float(f1[key]) for column, key in SCORED.items()}


def summarize(runs):
    """Per arm and score: mean, standard deviation, lowest and highest F1 over seeds."""
    long = runs.melt(id_vars=["arm", "seed"], value_vars=list(SCORED), var_name="score", value_name="f1")
    stats = long.groupby(["arm", "score"], sort=False)["f1"].agg(["mean", "std", "min", "max", "count"])
    return stats.rename(columns={"count": "seeds"}).reset_index()


def beats(summary, reference_v1_f1):
    """True when the supplement arm's mean V1 F1 against the thesis catalog is above the
    reference by more than its own spread over seeds (one standard deviation)."""
    row = summary[(summary["arm"] == "supplement") & (summary["score"] == "v1_thesis")]
    return bool(len(row)) and float(row["mean"].iloc[0] - row["std"].iloc[0]) > reference_v1_f1


def report(cfg):
    out = out_dir(cfg)
    rows = []
    for seed in cfg["selftrain"]["retrain"]["seeds"]:
        for arm in ARMS:
            path = out / f"scores_{model_name(arm, seed)}.csv"
            if path.exists():
                rows.append({"model": model_name(arm, seed), "arm": arm, "seed": seed, **f1_row(pd.read_csv(path))})
    if not rows:
        print("no finished retrains yet")
        return
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv", index=False)
    summary = summarize(runs)
    summary.round(4).to_csv(out / "summary.csv", index=False)

    published = run_name(cfg["detectors"]["cnn"])
    detections = pd.read_parquet(cfg["paths"]["detections"] / f"global_{published}_calibrated.parquet")
    reference = f1_row(score_run(cfg, detections, scoring_catalogs(cfg)))
    print("F1, all sizes (V1 and T1-T3, against the thesis and the supplemented catalog):")
    print(runs.round(4).to_string(index=False))
    print("\nover seeds:\n" + summary.round(4).to_string(index=False))
    print(f"\n{published}: " + ", ".join(f"{k} {v:.4f}" for k, v in reference.items()))
    verdict = "beats" if beats(summary, reference["v1_thesis"]) else "does not beat"
    print(f"the supplement arm {verdict} {published} on V1 beyond its seed spread")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="store_true", help="only compare the finished runs")
    args = parser.parse_args()
    cfg = load_config()
    out_dir(cfg).mkdir(parents=True, exist_ok=True)
    if not args.report:
        labels_path = supplement_labels(cfg)
        for seed in cfg["selftrain"]["retrain"]["seeds"]:      # both arms of a seed before the next seed
            for arm in ARMS:
                name = model_name(arm, seed)
                if (out_dir(cfg) / f"scores_{name}.csv").exists():
                    print(f"{name}: already done", flush=True)
                    continue
                print(f"\n=== {name} ===", flush=True)
                train_one(run_config(cfg, arm, seed, labels_path))
    report(cfg)


if __name__ == "__main__":
    main()

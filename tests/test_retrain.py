import pandas as pd
import pytest
import torch

from craters.retrain import beats, f1_row, model_name, run_config, summarize
from craters.train_cnn import WeightAverage


def test_weight_average_follows_the_model_slowly_and_copies_integer_buffers():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    average = WeightAverage(model, decay=0.5)
    start = model[0].weight.detach().clone()
    for step in range(1, 31):            # past the warm-up, so the decay is 0.5
        with torch.no_grad():
            model[0].weight.fill_(1.0)
            model[1].num_batches_tracked.fill_(step)
        average.update(model)
    assert torch.allclose(average.model[0].weight, torch.ones(2, 2), atol=1e-4)
    assert average.model[1].num_batches_tracked.item() == 30
    assert not average.model.training and not average.model[0].weight.requires_grad

    with torch.no_grad():
        model[0].weight.fill_(3.0)
    average.update(model)                # one step moves the average half-way
    assert torch.allclose(average.model[0].weight, torch.full((2, 2), 2.0), atol=1e-4)
    assert not torch.equal(start, average.model[0].weight)


def test_run_config_sets_the_arm_without_touching_the_base_config():
    base = {"seed": 42, "detectors": {"cnn": {"model_name": "cnn_reg_f2", "ema": {"enabled": False, "decay": 0.999},
                                              "level_models": {"coarse": "x"}, "init_from": "old.pt"}}}
    cfg = run_config(base, "supplement", 2, "labels.parquet")
    cnn = cfg["detectors"]["cnn"]
    assert cfg["seed"] == 2 and cnn["model_name"] == model_name("supplement", 2) == "retrain_supplement_s2"
    assert cnn["ema"]["enabled"] and cnn["extra_labels"] == "labels.parquet"
    assert cnn["init_from"] is None and cnn["level_models"] == {}          # from ImageNet weights, one model for both levels
    assert run_config(base, "catalog", 2, "labels.parquet")["detectors"]["cnn"]["extra_labels"] is None
    assert base["seed"] == 42 and not base["detectors"]["cnn"]["ema"]["enabled"]


def test_f1_row_reads_the_all_sizes_rows():
    scores = pd.DataFrame({"catalog": ["thesis", "thesis", "supplemented", "supplemented", "thesis"],
                           "region": ["V1", "T1-T3", "V1", "T1-T3", "V1"],
                           "bin": ["all", "all", "all", "all", "5-10"],
                           "f1": [0.6, 0.7, 0.62, 0.72, 0.1]})
    assert f1_row(scores) == {"v1_thesis": 0.6, "test_thesis": 0.7, "v1_supplemented": 0.62, "test_supplemented": 0.72}


def test_summary_and_the_replacement_rule():
    runs = pd.DataFrame({"arm": ["catalog", "supplement"] * 3, "seed": [1, 1, 2, 2, 3, 3],
                         "v1_thesis": [0.60, 0.68, 0.62, 0.70, 0.64, 0.72],
                         "test_thesis": 0.6, "v1_supplemented": 0.6, "test_supplemented": 0.6})
    summary = summarize(runs)
    row = summary[(summary["arm"] == "supplement") & (summary["score"] == "v1_thesis")].iloc[0]
    assert row["mean"] == pytest.approx(0.70) and row["std"] == pytest.approx(0.02) and row["seeds"] == 3
    assert beats(summary, 0.67)          # 0.70 - 0.02 > 0.67
    assert not beats(summary, 0.69)      # better on average, but inside the seed spread

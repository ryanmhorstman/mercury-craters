import numpy as np
import pandas as pd

from craters.calibrate import apply_diameter_correction, apply_size_cutoffs
from craters.config import run_name
from craters.detectors.cnn import LevelDetector


def test_diameter_correction_interpolates_in_log_diameter_per_level():
    detections = pd.DataFrame({"level": ["fine", "fine", "fine", "coarse"],
                               "diameter_km": [5.0, np.sqrt(50.0), 20.0, 100.0]})
    correction = {"fine": {"centre_km": [5.0, 10.0], "log_ratio": [0.0, np.log(1.2)], "pairs": [20, 20]},
                  "coarse": {"centre_km": [], "log_ratio": [], "pairs": []}}
    out = apply_diameter_correction(detections, correction)
    expected = [5.0, np.sqrt(50.0) * np.sqrt(1.2), 20.0 * 1.2, 100.0]   # midpoint in log D; held flat past the ends
    assert np.allclose(out["diameter_km"], expected)


def test_size_cutoffs_apply_per_band():
    detections = pd.DataFrame({"diameter_km": [6.0, 6.0, 12.0, 12.0, 500.0], "score": [0.25, 0.35, 0.16, 0.14, 0.2]})
    cutoffs = [{"min_km": 5, "max_km": 10, "score_cutoff": 0.3},
               {"min_km": 10, "max_km": 2000, "score_cutoff": 0.15}]
    kept = apply_size_cutoffs(detections, cutoffs)
    assert kept["score"].tolist() == [0.35, 0.16, 0.2]


def test_level_detector_dispatches_by_level():
    class Tile:
        def __init__(self, level):
            self.level = level

    detector = LevelDetector(lambda *a: "base", {"coarse": lambda *a: "tuned"})
    assert detector(None, Tile("fine"), None) == "base"
    assert detector(None, Tile("coarse"), None) == "tuned"
    assert run_name({"model_name": "m", "level_models": {"coarse": "m_ft"}}) == "m+coarse-m_ft"
    assert run_name({"model_name": "m", "level_models": {}}) == "m"

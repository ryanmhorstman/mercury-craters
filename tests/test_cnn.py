import numpy as np
import pandas as pd
import pytest
import torch

from craters.detectors.cnn import STRIDE, CraterNet, decode, loss_fn, make_targets, suppress_twins
from craters.train_cnn import dihedral


def fake_output(targets):
    """Network output that reproduces the targets exactly (heat as near-certain logits)."""
    out = targets["heat"].shape[-1]
    heat = torch.full((1, 1, out, out), -20.0)
    offset = torch.zeros((1, 2, out, out))
    logr = torch.zeros((1, 1, out, out))
    for k in np.flatnonzero(targets["present"]):
        iy, ix = divmod(int(targets["index"][k]), out)
        heat[0, 0, iy, ix] = 20.0
        offset[0, :, iy, ix] = torch.from_numpy(targets["offset"][k])
        logr[0, 0, iy, ix] = float(targets["logr"][k])
    return {"heat": heat, "offset": offset, "logr": logr}


def test_targets_decode_back_to_the_same_circles():
    craters = np.array([[100.3, 210.9, 20.0], [400.0, 77.5, 55.0], [256.0, 256.0, 15.0]], np.float32)
    targets = make_targets(craters, np.zeros((0, 3), np.float32), 512)
    found = decode(fake_output(targets), min_score=0.5).sort_values("radius_px").to_numpy()
    expected = craters[np.argsort(craters[:, 2])]
    assert np.allclose(found[:, :3], expected, atol=1e-3)


def test_ignored_craters_do_not_add_loss_but_real_ones_do():
    real = np.array([[128.0, 128.0, 20.0]], np.float32)
    ignored = np.array([[384.0, 384.0, 8.0]], np.float32)
    targets = make_targets(real, ignored, 512)
    batch = {k: torch.from_numpy(v)[None] for k, v in targets.items()}
    perfect = fake_output(targets)
    loss_perfect = loss_fn(perfect, batch)[0].item()

    with_extra = fake_output(targets)
    iy, ix = int(384 / STRIDE - 0.5), int(384 / STRIDE - 0.5)
    with_extra["heat"][0, 0, iy, ix] = 20.0           # a confident peak on the ignored crater
    assert loss_fn(with_extra, batch)[0].item() == pytest.approx(loss_perfect, abs=1e-4)

    with_false = fake_output(targets)
    with_false["heat"][0, 0, 20, 100] = 20.0           # a confident peak on empty ground
    assert loss_fn(with_false, batch)[0].item() > loss_perfect + 1.0


def test_dihedral_moves_circles_with_the_image():
    image = np.zeros((64, 64), np.float32)
    image[10, 40] = 1.0                                # pixel centre at col 40.5, row 10.5
    circles = np.array([[40.5, 10.5, 3.0]], np.float32)
    for k in range(8):
        out_image, out_circles = dihedral(image, circles, k, 64)
        row, col = np.argwhere(out_image == 1.0)[0]
        assert (col + 0.5, row + 0.5) == pytest.approx(tuple(out_circles[0, :2]))


def test_model_output_shapes():
    model = CraterNet(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 256, 256))
    assert out["heat"].shape == (1, 1, 64, 64)
    assert out["offset"].shape == (1, 2, 64, 64)
    assert out["logr"].shape == (1, 1, 64, 64)


def test_f_beta_is_f1_at_beta_1_and_favours_recall_above_it():
    from craters.train_cnn import f_beta

    assert f_beta(0.6, 0.4, 1) == pytest.approx(2 * 0.6 * 0.4 / 1.0)
    assert f_beta(0.4, 0.6, 2) > f_beta(0.6, 0.4, 2)
    assert f_beta(0.0, 0.0, 2) == 0.0


TWINS = {"radius_frac": 2.0, "ratio_min": 0.75, "ratio_max": 1.33}


def two_peak_output(peaks, size=128):
    """Heatmap with a peak per (ix, iy, logit) and radius 20 px everywhere."""
    out = size // STRIDE
    heat = torch.full((1, 1, out, out), -20.0)
    for ix, iy, logit in peaks:
        heat[0, 0, iy, ix] = logit
    return {"heat": heat, "offset": torch.zeros((1, 2, out, out)), "logr": torch.full((1, 1, out, out), np.log(20.0))}


def test_a_second_weaker_peak_on_the_same_crater_is_suppressed():
    # The strong peak and a weaker one 8 cells (32 px, 1.6 radii) west, as the trained models produce.
    output = two_peak_output([(20, 16, 3.0), (12, 16, 1.0)])
    assert len(decode(output, min_score=0.5)) == 2               # the 3x3 peak window keeps both
    found = decode(output, min_score=0.5, twins=TWINS)
    assert len(found) == 1
    assert found["col"].iloc[0] == pytest.approx((20 + 0.5) * STRIDE)


def test_separate_craters_of_the_same_size_stay_separate():
    # Neighbours 11 cells (44 px, 2.2 radii) away, west and north: rims nearly touching.
    output = two_peak_output([(20, 16, 3.0), (9, 16, 2.0), (20, 5, 2.0)])
    assert len(decode(output, min_score=0.5, twins=TWINS)) == 3


def test_twin_suppression_ignores_circles_of_different_size():
    circles = pd.DataFrame({"col": [100.0, 70.0], "row": [100.0, 100.0], "radius_px": [20.0, 8.0], "score": [0.9, 0.5]})
    assert len(suppress_twins(circles, **TWINS)) == 2


def test_overlap_rule_merges_twins_split_across_tiles():
    from craters.dedupe import merge_overlap_detections
    from craters.detectors.cnn import overlap_rule

    match = {"center_frac": 0.5, "ratio_min": 0.75, "ratio_max": 1.33}
    assert overlap_rule(match, None) == match
    # Same 6 km crater from two tiles, the second circle 0.8 D west (a twin): 0.5 D merging keeps both.
    km_per_deg = np.radians(1.0) * 2439.4
    twins = pd.DataFrame({"lat": [0.0, 0.0], "lon": [10.0, 10.0 - 0.8 * 6.0 / km_per_deg],
                          "diameter_km": [6.0, 6.0], "score": [0.3, 0.2]})
    assert len(merge_overlap_detections(twins, match)) == 2
    assert len(merge_overlap_detections(twins, overlap_rule(match, TWINS))) == 1

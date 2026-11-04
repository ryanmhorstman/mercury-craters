import numpy as np
import pytest
import torch

from craters.detectors.cnn import STRIDE, CraterNet, decode, loss_fn, make_targets
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

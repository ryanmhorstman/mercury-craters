import pytest

from craters.mapview import tile_window

W, H = 92160, 46080     # the 166 m basemap: 180 W - 180 E, 90 N - 90 S


def test_zoom_0_tiles_swap_the_raster_halves():
    """Map x runs 0-360 E but the raster starts at 180 W: tile (0, 0, 0) is the raster's right half."""
    assert tile_window(0, 0, 0, W, H) == (W / 2, 0.0, W / 2, H)
    assert tile_window(0, 1, 0, W, H) == (0.0, 0.0, W / 2, H)


def test_tiles_meeting_at_0_degrees_are_contiguous_in_the_raster():
    z = 5
    last = tile_window(z, 2 ** (z + 1) - 1, 3, W, H)     # ends at 360 E
    first = tile_window(z, 0, 3, W, H)                   # starts at 0 E
    assert last[0] + last[2] == pytest.approx(first[0])  # no gap or overlap: no seam at 0/360
    assert last[1] == first[1]


def test_tile_position_at_depth():
    z, span = 7, 180 / 2**7
    col, row, cols, rows = tile_window(z, 100, 40, W, H)
    assert col == pytest.approx(((100 * span + 180) % 360) / 360 * W)
    assert row == pytest.approx(40 * span / 180 * H)
    assert cols == pytest.approx(rows) == pytest.approx(span / 180 * H)


def test_tiles_repeat_east_west():
    assert tile_window(3, -1, 2, W, H) == tile_window(3, 15, 2, W, H)
    assert tile_window(0, 2, 0, W, H) == tile_window(0, 0, 0, W, H)


@pytest.mark.parametrize("z, x, y", [(0, 0, 1), (3, 0, 8), (3, 0, -1), (-1, 0, 0)])
def test_tiles_above_or_below_the_map(z, x, y):
    assert tile_window(z, x, y, W, H) is None

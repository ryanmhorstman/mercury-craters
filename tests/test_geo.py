import numpy as np

from craters.geo import great_circle_km, in_box, local_transformers, lon_180, lon_360


def test_longitude_conventions():
    assert np.allclose(lon_360([-90, 0, 359.5, 360, 540]), [270, 0, 359.5, 0, 180])
    assert np.allclose(lon_180([270, 0, 180, 90]), [-90, 0, -180, 90])


def test_stereographic_round_trip_at_high_latitude():
    forward, inverse = local_transformers(55.0, 315.0)
    lon = np.array([310.0, 315.0, 320.0])
    lat = np.array([52.0, 55.0, 58.0])
    x, y = forward.transform(lon, lat)
    lon2, lat2 = inverse.transform(x, y)
    assert np.allclose(lon_360(lon2), lon, atol=1e-9)
    assert np.allclose(lat2, lat, atol=1e-9)


def test_great_circle_quarter_circumference():
    quarter = great_circle_km([0], [0], [0], [90])[0]
    assert np.isclose(quarter, np.pi / 2 * 2439.4)


def test_box_wraps_through_zero():
    box = [350, 10, -5, 5]
    assert in_box(np.array([0, 0, 0]), np.array([355, 5, 20]), box).tolist() == [True, True, False]

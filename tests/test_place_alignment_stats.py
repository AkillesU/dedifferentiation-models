"""Tests for place_alignment_stats.py: trend window, weighted slopes, place index, and progress slopes."""

import numpy as np
import pandas as pd
import pytest

import place_alignment_stats as place


def test_trend_window_stops_at_first_level_reaching_human_drop():
    assert place.trend_window_mask([20, 40, 60, 80], 47.0, "matched").tolist() == [True, True, True, False]
    assert place.trend_window_mask([20, 40], 47.0, "matched").all()
    with pytest.raises(ValueError, match="trend_window"):
        place.trend_window_mask([20], 47.0, "some")


def test_weighted_slopes_recovers_linear_trend():
    x = np.array([0.0, 10.0, 20.0, 30.0])
    assert place.weighted_slopes(x, 2.0 * x + 1.0, np.array([200, 200, 100, 50])) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="vary"):
        place.weighted_slopes(np.ones(3), np.arange(3.0), np.ones(3))


def test_place_index_contrasts_example():
    contrasts = place.place_index_contrasts(np.array([4.0, 2.0, -6.0]))  # face, object, place
    assert contrasts["place_specificity_index"] == pytest.approx(-9.0)
    assert contrasts["place_minus_face"] == pytest.approx(-10.0)
    assert contrasts["place_minus_object"] == pytest.approx(-8.0)


def test_progress_slopes_divide_by_human_drop():
    overall = np.array([0.0, 10.0, 20.0])
    category_means = np.stack([overall, overall, 0.5 * overall], axis=-1)  # face, object, place drops
    slopes = place.progress_slopes(overall, category_means, np.ones(3), np.array([50.0, 50.0, 50.0]))
    assert np.allclose(slopes, [0.2, 0.2, 0.1])


def test_flag_closest_level_and_holm():
    level_index = pd.DataFrame({
        "damage_layer": ["V1"] * 2 + ["V2"] * 2, "damage_level": [0.1, 0.2, 0.1, 0.2],
        "overall_drop_percent": [30.0, 50.0, 46.0, 60.0], "quantity": "place_specificity_index",
        "bootstrap_p_two_sided": [0.5, 0.04, 0.02, 0.9],
    })
    flagged = place.flag_closest_level(level_index, human_overall=47.0)
    closest = flagged[flagged["is_closest_numerically"]]
    assert closest["damage_level"].tolist() == [0.2, 0.1]
    assert np.allclose(closest["bootstrap_p_holm_layers"], [0.04, 0.04])

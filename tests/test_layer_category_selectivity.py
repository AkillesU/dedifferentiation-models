"""Tests for layer_category_selectivity.py: the top-5% U* score, layer choice, and the configs (no model runs)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import layer_category_selectivity as layer_choice

CONFIGS = sorted(Path("configs/selectivity").glob("*.yaml"))
LABELS = ["animals"] * 20 + ["faces"] * 20 + ["objects"] * 20 + ["places"] * 20


def test_top_fraction_u_star_finds_a_perfect_unit():
    rng = np.random.default_rng(0)
    activations = rng.random((80, 40))
    activations[:, 0] = np.repeat([0.0, 2.0, 0.0, 0.0], 20)  # unit 0 responds only to faces
    # 5% of 40 units = 2 units: the perfect unit (U* = 1) and the best random one
    u_star = layer_choice.top_fraction_u_star(activations, LABELS)
    assert set(u_star) == {"animals", "faces", "objects", "places"}
    assert 0.75 < u_star["faces"] < 1.0 and u_star["faces"] > u_star["animals"]


def test_top_fraction_u_star_does_not_depend_on_chunking():
    activations = np.random.default_rng(1).random((80, 57))
    whole = layer_choice.top_fraction_u_star(activations, LABELS, chunk_size=1000)
    chunked = layer_choice.top_fraction_u_star(activations, LABELS, chunk_size=10)
    assert whole == pytest.approx(chunked)


def test_top_fraction_u_star_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\[N,U\]"):
        layer_choice.top_fraction_u_star(np.ones((79, 5)), LABELS)


def test_score_layers_averages_categories_and_marks_the_best():
    scores = pd.DataFrame({"layer": ["a", "b"], "n_units": [10, 10],
                           "u_star_faces": [0.6, 0.9], "u_star_places": [0.8, 0.7]})
    scored = layer_choice.score_layers(scores)
    assert scored["score"].tolist() == pytest.approx([0.7, 0.8])
    assert scored["chosen"].tolist() == [False, True]


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.stem)
def test_readout_layer_is_a_candidate(config_path):
    with open(config_path, encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    assert set(config["layer_paths"]) <= set(config["candidate_layers"])

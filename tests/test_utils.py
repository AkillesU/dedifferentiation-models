"""Tests for the pure functions in utils.py: differentiation, RDM storage, unit selection, damage, and curves."""

import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

import utils


def toy_similarity(n_per_category: int = 4) -> tuple[np.ndarray, list[str]]:
    """Return a similarity matrix with within-category r = 0.8 and between-category r = 0.2."""
    names = [f"{c}{i + 1}.jpg" for c in utils.CATEGORIES for i in range(n_per_category)]
    labels = utils.assign_categories(names)
    similarity = np.where(labels[:, None] == labels[None, :], 0.8, 0.2)
    np.fill_diagonal(similarity, 1.0)
    return similarity, names


def test_assign_categories_strips_numbers_and_folders():
    assert utils.assign_categories(["face1.jpg", "objects/object12.png", "place_3.jpg"]).tolist() == ["face", "object", "place"]


def test_calc_within_between_excludes_diagonal():
    similarity, names = toy_similarity()
    result = utils.calc_within_between(similarity, utils.assign_categories(names), "face")
    assert result["within"] == pytest.approx(0.8)
    assert result["between"] == pytest.approx(0.2)
    assert result["differentiation"] == pytest.approx(0.6)


def test_calc_within_between_rejects_single_image_category():
    similarity, names = toy_similarity(n_per_category=1)
    with pytest.raises(ValueError, match="at least two images"):
        utils.calc_within_between(similarity, utils.assign_categories(names), "face")


def test_selective_rdm_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    activations = rng.normal(size=(3, 8, 20))
    rdms = np.stack([np.corrcoef(a) for a in activations])
    names = [f"face{i}.jpg" for i in range(8)]
    utils.save_selective_rdms(tmp_path / "damaged_0.1.npz", rdms, names, ["0", "1", "2"])
    loaded, loaded_names, replicates = utils.load_selective_rdms(tmp_path / "damaged_0.1.npz")
    assert loaded_names == names and replicates == ["0", "1", "2"]
    assert np.allclose(loaded, rdms, atol=1e-15)


def test_upper_to_rdm_rejects_wrong_size():
    with pytest.raises(ValueError, match="upper triangles"):
        utils.upper_to_rdm(np.zeros((1, 5)), n_images=4)


def test_canonical_layer_name_maps_equivalent_spellings():
    assert utils.canonical_layer_name("module._modules.IT._modules.output") == "IT"
    assert utils.canonical_layer_name("classifier_2") == "classifier.2"
    assert utils.canonical_layer_name("module.IT") == "IT"


def test_select_top_units_keeps_rounded_fraction():
    table = pd.DataFrame({"layer": ["module.IT"] * 40 + ["module.V4"] * 10, "unit": list(range(40)) + list(range(10)),
                          "mw_animals": 0.5, "mw_faces": np.r_[np.arange(40) / 40, np.zeros(10)],
                          "mw_objects": 0.5, "mw_places": 0.5})
    assert utils.select_top_units(table, "IT", "face", 0.05).tolist() == [39, 38]
    with pytest.raises(ValueError, match="No selectivity rows"):
        utils.select_top_units(table, "V1", "face", 0.05)


def test_finite_correlation_matrix_zeroes_undefined_pairs():
    activations = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [3.0, 2.0, 1.0]])
    matrix = utils.finite_correlation_matrix(activations)
    assert np.all(np.isfinite(matrix)) and matrix[0, 1] == 0.0 and matrix[0, 2] == pytest.approx(-1.0)
    assert np.allclose(np.diag(matrix), 1.0)


def test_build_selective_rdms_from_activations(tmp_path):
    rng = np.random.default_rng(1)
    names = [f"{c}{i + 1}.jpg" for c in utils.CATEGORIES for i in range(3)]
    for level in ("0.0", "0.5"):
        for simulation in ("0", "1"):
            utils.save_activations(tmp_path / "act" / f"damaged_{level}" / f"{simulation}.npz", rng.normal(size=(12, 30)), names)
    table = pd.DataFrame({"layer": "IT", "unit": range(30), **{c: rng.random(30) for c in utils.SELECTIVITY_COLUMNS.values()}})
    utils.build_selective_rdms(tmp_path / "act", tmp_path / "rdm", table, "IT", 0.1)
    rdms, loaded_names, replicates = utils.load_selective_rdms(tmp_path / "rdm" / "place_selective" / "damaged_0.5.npz")
    assert rdms.shape == (2, 12, 12) and loaded_names == names and replicates == ["0", "1"]


def test_sever_connections_and_delete_units_counts():
    generator = torch.Generator().manual_seed(0)
    conv = nn.Conv2d(4, 10, 3)
    utils.sever_connections([("conv", conv)], 0.25, generator)
    assert int((conv.weight == 0).sum()) == int(0.25 * conv.weight.numel())
    conv = nn.Conv2d(4, 10, 3)
    utils.delete_units([("conv", conv)], 0.3, generator, include_bias=True)
    deleted = [i for i in range(10) if torch.all(conv.weight[i] == 0)]
    assert len(deleted) == 3 and torch.all(conv.bias[deleted] == 0)


def test_add_weight_noise_scales_with_layer_sd():
    torch.manual_seed(0)
    conv = nn.Conv2d(64, 64, 3)
    original = conv.weight.detach().clone()
    sds = utils.weight_standard_deviations([("conv", conv)])
    utils.add_weight_noise([("conv", conv)], 2.0, sds, torch.Generator().manual_seed(1))
    assert float((conv.weight - original).std()) == pytest.approx(2.0 * sds["conv"], rel=0.02)


def test_eccentricity_probability_peripheral_and_foveal():
    peripheral = utils.eccentricity_deletion_probability(9, 9, 0.5, 1.0, 0.8)
    foveal = utils.eccentricity_deletion_probability(9, 9, 0.0, 0.5, 0.8, reverse=True)
    assert peripheral[4, 4] == 0.0 and peripheral[0, 0] == pytest.approx(0.8)
    assert foveal[4, 4] == pytest.approx(0.8) and foveal[0, 0] == 0.0
    with pytest.raises(ValueError, match="r_start"):
        utils.eccentricity_deletion_probability(9, 9, 0.6, 0.5, 0.8)


def test_sever_connections_by_eccentricity_matches_conv_when_intact():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 6, 3, padding=1)
    x = torch.randn(2, 3, 8, 8)
    expected = conv(x)
    model = nn.Sequential(copy.deepcopy(conv))
    utils.sever_connections_by_eccentricity(model, "0", 0.5, 1.0, 0.0, torch.Generator().manual_seed(0))
    assert torch.allclose(model(x), expected)
    damaged = nn.Sequential(copy.deepcopy(conv))
    utils.sever_connections_by_eccentricity(damaged, "0", 0.0, 1.0, 1.0, torch.Generator().manual_seed(0))
    # With p_max = 1 every connection at the corners is severed, leaving only the bias
    corner = damaged(x)[:, :, 0, 0]
    assert torch.allclose(corner, conv.bias.expand_as(corner))


def make_condition(levels=(0.0, 0.5, 1.0), n_simulations=3) -> pd.DataFrame:
    """Return differentiation rows with value = (1 - level) * (category index + 1) for every simulation."""
    rows = [{"damage_type": "connections", "damage_area": "V1", "readout_layer": "IT", "top_fraction": 0.05,
             "selection_mode": "percentage", "category": c, "damage_level": level, "replicate": str(r),
             "replicate_order": r, "within": 0.0, "between": 0.0, "differentiation": (1 - level) * (i + 1)}
            for level in levels for i, c in enumerate(utils.CATEGORIES) for r in range(n_simulations)]
    return pd.DataFrame(rows)


def test_level_values_total_and_scaled():
    condition = make_condition()
    total = utils.level_values(condition, "total")
    assert total[0.0].tolist() == [2.5, 2.5, 2.5] and total[0.5].tolist() == [1.25] * 3
    scaled = utils.level_values(condition, "face", scaled_percent=True)
    assert np.allclose(scaled[0.5], 50.0)


def test_matched_category_values_scales_each_simulation():
    matched = utils.matched_category_values(make_condition(), 0.5, ["face", "place"])
    assert list(matched.columns) == ["replicate", "face", "place"]
    assert np.allclose(matched[["face", "place"]], 50.0)


def test_auc_loss_and_curve_summary():
    assert utils.auc_loss([0, 0.5, 1], [0.2, 0.1, 0.0]) == pytest.approx(0.5)
    auc, sd = utils.curve_auc_summary(utils.level_values(make_condition(), "total"))
    assert auc == pytest.approx(0.5) and sd == pytest.approx(0.0)
    with pytest.raises(ValueError, match="starting at 0"):
        utils.auc_loss([0.1, 0.5], [1.0, 0.5])


def test_holm_adjust():
    assert np.allclose(utils.holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_load_config_jobs_merges_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("fit:\n  defaults: {seed: 1, model: m}\n  jobs:\n    - {seed: 2}\n    - {damage_type: noise}\n", encoding="utf-8")
    assert utils.load_config_jobs(path, "fit") == [{"seed": 2, "model": "m"}, {"seed": 1, "model": "m", "damage_type": "noise"}]
    with pytest.raises(ValueError, match="no mapping section"):
        utils.load_config_jobs(path, "missing")


def test_load_best_fit_levels(tmp_path):
    path = tmp_path / "descriptives.csv"
    pd.DataFrame({"source": ["Human", "Model", "Model"], "damage_type": ["", "connections", "noise"],
                  "damaged_area": ["", "Whole model", "V1"], "damage_level": [np.nan, 0.15, 0.9]}).to_csv(path, index=False)
    assert utils.load_best_fit_levels(path) == {("connections", "All"): 0.15, ("noise", "V1"): 0.9}
    assert utils.resolve_damage_level("best_fit", "noise", "V1", path) == 0.9


def test_save_figure_keeps_dots_in_file_name(tmp_path):
    paths = utils.save_figure(plt.figure(), tmp_path / "figure_S8_retained_fovea_0.1", dpi=10)
    assert [p.name for p in paths] == ["figure_S8_retained_fovea_0.1.png", "figure_S8_retained_fovea_0.1.svg"]
    assert all(p.exists() for p in paths)

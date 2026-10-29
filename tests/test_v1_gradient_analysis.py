"""Tests for v1_gradient_analysis.py: normalised maps, eccentricity bands, within-image Cliff's delta, and the config (no model runs)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

import utils
import v1_gradient_analysis as gradients

CONFIG = "configs/gradients/cornet_rt5_top5.yaml"


def test_normalised_maps_sum_to_one_and_ignore_overall_scale():
    maps = np.random.default_rng(0).random((2, 3, 5, 6))
    normalised = gradients.normalised_maps(maps)
    assert normalised.shape == maps.shape
    np.testing.assert_allclose(normalised.sum(axis=(-2, -1)), 1.0)
    np.testing.assert_allclose(gradients.normalised_maps(7.5 * maps), normalised)


def test_normalised_maps_reject_empty_maps():
    maps = np.ones((3, 4, 4))
    maps[1] = 0.0
    with pytest.raises(ValueError, match=r"\[\[1\]\] are all zero"):
        gradients.normalised_maps(maps)


def test_eccentricity_bands_span_centre_to_corner():
    eccentricity, edges = gradients.eccentricity_bands(5, 5, 4)
    assert eccentricity[2, 2] == 0 and eccentricity[0, 0] == pytest.approx(np.sqrt(8))
    assert edges[0] == 0 and edges[-1] == pytest.approx(np.sqrt(8)) and len(edges) == 5


def test_band_densities_average_each_band_and_drop_corners():
    eccentricity, edges = gradients.eccentricity_bands(5, 5, 2)
    uniform = gradients.normalised_maps(np.ones((3, 2, 5, 5)))
    densities = gradients.band_densities(uniform, eccentricity, edges)
    # A uniform map has the same density (1/25) in every band; the 4 corners fall outside the last band
    assert densities.shape == (3, 2, 2) and np.allclose(densities, 1 / 25)
    band = np.digitize(eccentricity.ravel(), edges) - 1
    assert (band == len(edges) - 1).sum() == 4


def test_band_densities_reject_wrong_shape_and_empty_bands():
    eccentricity, edges = gradients.eccentricity_bands(5, 5, 2)
    with pytest.raises(ValueError, match=r"\[\.\.\.,H,W\]"):
        gradients.band_densities(np.ones((2, 4, 4)), eccentricity, edges)
    eccentricity, edges = gradients.eccentricity_bands(4, 4, 12)
    with pytest.raises(ValueError, match="no positions"):
        gradients.band_densities(np.ones((2, 4, 4)), eccentricity, edges)


def test_delta_matches_hand_computed_example():
    # Image 0: signs [1, 1, -1, 0] -> 0.25.  Image 1: signs [1, 1, 1, -1] -> 0.5.
    selective = np.array([[0.5], [0.3]])
    random_sets = np.array([[0.1, 0.2, 0.6, 0.5], [0.1, 0.1, 0.1, 0.9]]).T[:, :, None]
    result = gradients.within_image_delta(selective, random_sets, 100, "0|test")
    np.testing.assert_allclose(result["image_delta"][:, 0], [0.25, 0.5])
    np.testing.assert_allclose(result["cliffs_delta"], [0.375])
    np.testing.assert_allclose(result["mean_diff"], [(0.5 - 0.35 + 0.3 - 0.3) / 2])


def test_image_delta_equals_cliffs_delta_from_mann_whitney_u():
    rng = np.random.default_rng(2)
    selective, random_sets = rng.random((5, 3)), rng.random((40, 5, 3))
    image_delta = gradients.within_image_signs(selective, random_sets).mean(axis=1)
    for image in range(5):
        for position in range(3):
            u = mannwhitneyu([selective[image, position]], random_sets[:, image, position]).statistic
            assert image_delta[image, position] == pytest.approx(2 * u / 40 - 1)


def test_uniformly_more_sensitive_selective_units_give_zero_delta():
    # Scaling by a power of two is exact, so the normalised maps tie exactly
    random_maps = np.random.default_rng(1).random((1, 3, 5, 5))
    selective = gradients.normalised_maps(4.0 * random_maps[0])
    result = gradients.within_image_delta(selective, gradients.normalised_maps(random_maps), 20, "0|test")
    np.testing.assert_array_equal(result["cliffs_delta"], 0.0)


def test_delta_is_one_when_selective_exceeds_every_random_set():
    selective = np.full((3, 2, 2), 0.9)
    random_sets = np.random.default_rng(3).random((10, 3, 2, 2)) * 0.5
    result = gradients.within_image_delta(selective, random_sets, 50, "0|test")
    np.testing.assert_array_equal(result["cliffs_delta"], 1.0)
    np.testing.assert_array_equal(result["ci_low"], 1.0)
    assert result["image_delta"].shape == (3, 2, 2)


def test_signs_reject_mismatched_shapes():
    with pytest.raises(ValueError, match=r"random_sets must be \[N_sets, \*selective.shape\]"):
        gradients.within_image_signs(np.ones((3, 4)), np.ones((10, 2, 4)))


def test_crossed_bootstrap_equals_resampling_images_and_sets():
    signs = np.random.default_rng(4).integers(-1, 2, size=(6, 8, 3)).astype(np.int8)
    n_bootstrap = 25
    draws = gradients.crossed_bootstrap_delta(signs, n_bootstrap, np.random.default_rng(5), chunk_size=n_bootstrap)
    reference_rng = np.random.default_rng(5)
    image_counts = reference_rng.multinomial(6, np.full(6, 1 / 6), size=n_bootstrap)
    set_counts = reference_rng.multinomial(8, np.full(8, 1 / 8), size=n_bootstrap)
    for b in range(n_bootstrap):
        images, sets = np.repeat(np.arange(6), image_counts[b]), np.repeat(np.arange(8), set_counts[b])
        np.testing.assert_allclose(draws[b], signs[np.ix_(images, sets)].mean(axis=(0, 1)), atol=1e-6)


def test_crossed_bootstrap_is_reproducible_and_bounded():
    signs = np.random.default_rng(6).integers(-1, 2, size=(5, 7, 4)).astype(np.int8)
    first = gradients.crossed_bootstrap_delta(signs, 30, np.random.default_rng(9), chunk_size=7)
    second = gradients.crossed_bootstrap_delta(signs, 30, np.random.default_rng(9), chunk_size=7)
    np.testing.assert_array_equal(first, second)
    assert np.all((first >= -1) & (first <= 1))


def test_crossed_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError, match=r"\[N_img,N_sets,X\]"):
        gradients.crossed_bootstrap_delta(np.ones((3, 4), dtype=np.int8), 10, np.random.default_rng(0))
    with pytest.raises(ValueError, match="n_bootstrap must be at least 1"):
        gradients.crossed_bootstrap_delta(np.ones((3, 4, 2), dtype=np.int8), 0, np.random.default_rng(0))


def test_within_image_delta_is_seeded_and_brackets_the_estimate():
    rng = np.random.default_rng(7)
    selective, random_sets = rng.normal(0.5, 1.0, (16, 3)), rng.normal(0.0, 1.0, (20, 16, 3))
    first = gradients.within_image_delta(selective, random_sets, 200, "0|place")
    second = gradients.within_image_delta(selective, random_sets, 200, "0|place")
    np.testing.assert_array_equal(first["ci_low"], second["ci_low"])
    assert np.all(first["ci_low"] <= first["cliffs_delta"]) and np.all(first["cliffs_delta"] <= first["ci_high"])


def test_position_delta_table_has_one_row_per_position():
    rng = np.random.default_rng(8)
    result = gradients.within_image_delta(rng.random((4, 2, 3)), rng.random((5, 4, 2, 3)), 20, "0|test")
    table = gradients.position_delta_table(result)
    assert list(table.columns) == ["fy", "fx", "mean_diff", "cliffs_delta", "ci_low", "ci_high"]
    assert len(table) == 6 and table.loc[5, "cliffs_delta"] == result["cliffs_delta"][1, 2]


def test_run_stats_writes_maps_and_eccentricity_bias(tmp_path):
    rng = np.random.default_rng(10)
    job = utils.load_config_jobs(CONFIG, "v1_gradients")[0] | {
        "gradient_dir": str(tmp_path / "gradients"), "stats_dir": str(tmp_path / "stats"), "n_bootstrap": 20}
    gradient_dir = tmp_path / "gradients" / gradients.analysis_name(job)
    gradient_dir.mkdir(parents=True)
    np.save(gradient_dir / "maps_selective.npy", rng.random((3, 24, 24)).astype(np.float32))
    np.save(gradient_dir / "maps_random.npy", rng.random((4, 3, 24, 24)).astype(np.float32))
    gradients.run_stats([job])
    name = gradients.analysis_name(job)
    assert len(pd.read_csv(tmp_path / "stats" / f"{name}_position_delta.csv")) == 24 * 24
    assert np.load(tmp_path / "stats" / f"{name}_image_delta_maps.npy").shape == (3, 24, 24)
    bias = pd.read_csv(tmp_path / "stats" / "eccentricity_bias.csv")
    assert len(bias) == job["n_bands"] and set(bias["category"]) == {job["units"]}


def test_load_maps_rejects_mismatched_shapes(tmp_path):
    np.save(tmp_path / "maps_selective.npy", np.ones((3, 6, 6)))
    np.save(tmp_path / "maps_random.npy", np.ones((4, 2, 6, 6)))
    with pytest.raises(ValueError, match=r"\[N_sets,N_img,H,W\]"):
        gradients.load_maps(tmp_path)


def test_category_images_use_file_name_order(tmp_path):
    for name in ["place1.jpg", "place2.jpg", "place10.jpg", "face1.jpg"]:
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in utils.category_images(tmp_path, "place")] == ["place1.jpg", "place10.jpg", "place2.jpg"]


def test_config_lists_all_sixteen_analyses():
    jobs = utils.load_config_jobs(CONFIG, "v1_gradients")
    for job in jobs:
        gradients.validate_job(job)
    pairs = {(job["units"], job["images"]) for job in jobs}
    assert len(jobs) == 16 and pairs == {(u, i) for u in utils.CATEGORIES for i in utils.CATEGORIES}


def test_validate_job_rejects_unknown_category():
    job = utils.load_config_jobs(CONFIG, "v1_gradients")[0] | {"units": "tool"}
    with pytest.raises(ValueError, match="units must be one of"):
        gradients.validate_job(job)


def test_plot_gradient_delta_maps_saves_values(tmp_path):
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    table = pd.DataFrame({"fy": np.repeat(np.arange(3), 3), "fx": np.tile(np.arange(3), 3), "mean_diff": 0.0,
                          "cliffs_delta": np.linspace(-1, 1, 9), "ci_low": -1.0, "ci_high": 1.0})
    table.to_csv(stats_dir / "place_units_place_images_position_delta.csv", index=False)
    values = utils.plot_gradient_delta_maps([{"units": "place", "images": "place", "title": "Places"}], "maps",
                                            plot_dir=tmp_path, stats_dir=stats_dir)
    assert len(values) == 9 and (tmp_path / "maps.png").exists() and (tmp_path / "maps_values.csv").exists()


def test_plot_eccentricity_bias_saves_values(tmp_path):
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    rows = [{"category": c, "band": b, "band_center": b + 0.5, "cliffs_delta": 0.1, "ci_low": 0.0, "ci_high": 0.2}
            for c in ("animal", "place") for b in range(3)]
    pd.DataFrame(rows).to_csv(stats_dir / "eccentricity_bias.csv", index=False)
    values = utils.plot_eccentricity_bias(["place"], {"place": "#0072B2"}, {"place": "^"}, "bias",
                                          plot_dir=tmp_path, stats_dir=stats_dir)
    assert len(values) == 3 and (tmp_path / "bias.png").exists() and (tmp_path / "bias_values.csv").exists()

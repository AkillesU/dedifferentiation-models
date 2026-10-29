"""Tests for fit_haupt_damage_levels.py: human reference values, bootstrap helpers, and level selection."""

import argparse

import numpy as np
import pandas as pd
import pytest

import fit_haupt_damage_levels as fit
from tests.test_utils import make_condition

HUMAN_CSV = "data/reference/haupt_2025/haupt2025_fig6B_derived.csv"
TABLE_S5_CSV = "data/reference/haupt_2025/table_s5_raw.csv"


def test_relative_drop_percent_is_ratio_of_summed_means():
    assert fit.relative_drop_percent([0.4, 0.6], [0.2, 0.3]) == pytest.approx(50.0)
    with pytest.raises(ValueError, match="positive"):
        fit.relative_drop_percent([0.0, 0.0], [0.1, 0.1])


def test_human_drops_equal_table_s5_after_centring():
    centre = fit.load_table_s5_cell_means(TABLE_S5_CSV)
    drops = fit.human_drop_estimates(fit.load_human_subject_values(HUMAN_CSV, center_means=centre))
    # Table S1 human row of the manuscript (FFA, LOC, PPA, and their equal mean)
    assert drops["face"] == pytest.approx(27.49, abs=0.01)
    assert drops["object"] == pytest.approx(59.21, abs=0.01)
    assert drops["place"] == pytest.approx(54.70, abs=0.01)
    assert drops["overall"] == pytest.approx(47.13, abs=0.01)


def test_bootstrap_p_two_sided_uses_plus_one_form():
    assert fit.bootstrap_p_two_sided(np.ones(99)) == pytest.approx(2 / 100)
    assert fit.bootstrap_p_two_sided(np.r_[np.ones(50), -np.ones(50)]) == 1.0


def test_select_levels_brackets_the_human_drop():
    rows = [{"model": "m", "damage_type": "noise", "damage_layer": "V1", "activation_layer": "IT", "damage_level": level,
             "target": "overall", "difference_model_minus_human": diff, "absolute_difference": abs(diff)}
            for level, diff in [(0.1, -20.0), (0.2, -3.0), (0.3, 5.0), (0.4, 20.0)]]
    roles = fit.select_levels(pd.DataFrame(rows), tolerance=1e-8).set_index("damage_level")["selection_roles"]
    assert roles[0.2] == "closest_numerically;closest_below_human" and roles[0.3] == "closest_above_human"


def test_load_layer_candidates_reads_the_table(tmp_path):
    table = make_condition(levels=(0.0, 0.25, 0.5))
    (tmp_path / "differentiation").mkdir()
    table.to_csv(tmp_path / "differentiation" / "toy.csv.gz", index=False)
    args = argparse.Namespace(model_dir="toy", differentiation_dir=tmp_path / "differentiation", damage_type="connections",
                              activation_layer="IT", selectivity_fraction=0.05, baseline_level=0.0, tolerance=1e-8)
    candidates = fit.load_layer_candidates(args, "V1")
    assert [c.damage_level for c in candidates] == [0.25, 0.5]
    drops = fit.to_relative_drop_distributions(candidates[1].matched)
    assert np.allclose(drops["overall"], 50.0)

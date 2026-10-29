"""Tests for compute_differentiation.py: parsing selective-RDM paths and building the differentiation table."""

import numpy as np
import pytest

import compute_differentiation
import utils


def write_condition(model_dir, damage_type="connections", area="V1", levels=("0.0", "0.5"), n_simulations=3):
    """Write toy selective RDMs (within r = 0.8 - level, between r = 0.2) for every category and level."""
    names = [f"{c}{i + 1}.jpg" for c in utils.CATEGORIES for i in range(4)]
    labels = utils.assign_categories(names)
    for level in levels:
        similarity = np.where(labels[:, None] == labels[None, :], 0.8 - float(level), 0.2)
        np.fill_diagonal(similarity, 1.0)
        for category in utils.CATEGORIES:
            utils.save_selective_rdms(
                model_dir / damage_type / area / "RDM_0.05_percentage" / "IT" / f"{category}_selective" / f"damaged_{level}.npz",
                np.repeat(similarity[None], n_simulations, axis=0), names, [str(i) for i in range(n_simulations)])


def test_build_differentiation_table(tmp_path):
    model_dir = tmp_path / "cornet_rt5_c"
    write_condition(model_dir)
    table = compute_differentiation.build_differentiation_table(model_dir)
    assert list(table.columns) == compute_differentiation.TABLE_COLUMNS
    assert len(table) == 2 * 4 * 3
    face = table[(table.category == "face") & (table.damage_level == 0.5)]
    assert np.allclose(face.differentiation, 0.1) and face.replicate_order.tolist() == [0, 1, 2]
    assert set(table.top_fraction) == {0.05} and set(table.readout_layer) == {"IT"}


def test_build_differentiation_table_requires_rdms(tmp_path):
    with pytest.raises(FileNotFoundError, match="No selective-RDM files"):
        compute_differentiation.build_differentiation_table(tmp_path)

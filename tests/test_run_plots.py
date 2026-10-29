"""Tests for run_plots.py: figure configs name valid functions and parameters, and a curve plot draws from a table."""

import inspect
from pathlib import Path

import matplotlib
import pytest
import yaml

matplotlib.use("Agg")

import run_plots
import utils
from tests.test_utils import make_condition


@pytest.mark.parametrize("config_path", sorted(Path("configs/figures").glob("*.yaml")))
def test_figure_configs_match_function_signatures(config_path):
    with open(config_path, encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    names = []
    for figure in config["figures"]:
        function = run_plots.PLOT_FUNCTIONS[figure["function"]]
        signature = inspect.signature(function)
        signature.bind(plot_dir="plots", **figure["parameters"])
        names.append(figure["parameters"]["file_name"])
    assert len(names) == len(set(names)), f"Duplicate file_name in {config_path}"


def test_run_figures_rejects_unknown_figure(tmp_path):
    path = tmp_path / "figures.yaml"
    path.write_text("plot_dir: plots\nfigures: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not in"):
        run_plots.run_figures(str(path), only=["missing"])


def test_plot_differentiation_curves_draws_and_saves_values(tmp_path):
    (tmp_path / "differentiation").mkdir()
    make_condition().to_csv(tmp_path / "differentiation" / "toy.csv.gz", index=False)
    values = utils.plot_differentiation_curves(
        model_dir="toy", damage_type="connections", damage_areas=["V1"], readout_layer="IT", categories=["total"],
        colours={"V1": "C0"}, markers={"V1": "o"}, file_name="curves", plot_dir=tmp_path / "plots",
        differentiation_dir=tmp_path / "differentiation")
    assert (tmp_path / "plots" / "curves.png").exists() and (tmp_path / "plots" / "curves.svg").exists()
    assert values["auc_loss"].iloc[0] == pytest.approx(0.5)

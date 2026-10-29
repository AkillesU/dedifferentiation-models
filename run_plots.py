"""
Draw the manuscript figures listed in a YAML figure config.

Each config lists figures as a plotting function from utils.py and its parameters. The
figures read the differentiation tables (data/differentiation/) and the statistics
outputs (stats/), so run compute_differentiation.py (or download the data release),
fit_haupt_damage_levels.py, place_alignment_stats.py, and v1_gradient_analysis.py stats
first. Every figure is saved as PNG and SVG with a CSV of the plotted values.

Usage:
    python run_plots.py configs/figures/figure_1.yaml
    python run_plots.py configs/figures/supplementary.yaml --only figure_S7_top10_connections
"""

import argparse

import yaml

from utils import (
    plot_category_panels,
    plot_damage_group_auc,
    plot_differentiation_curves,
    plot_eccentricity_bias,
    plot_gradient_delta_maps,
    plot_gradient_overlays,
    plot_place_progress_panels,
    plot_total_differentiation,
)

PLOT_FUNCTIONS = {
    "plot_differentiation_curves": plot_differentiation_curves,
    "plot_total_differentiation": plot_total_differentiation,
    "plot_category_panels": plot_category_panels,
    "plot_place_progress_panels": plot_place_progress_panels,
    "plot_damage_group_auc": plot_damage_group_auc,
    "plot_gradient_delta_maps": plot_gradient_delta_maps,
    "plot_gradient_overlays": plot_gradient_overlays,
    "plot_eccentricity_bias": plot_eccentricity_bias,
}


def run_figures(config_path: str, only: list[str] | None = None) -> None:
    """
    Draw every figure (or the selected ones) of a figure config.

    Parameters:
        config_path (str): YAML with plot_dir and a figures list of {function, parameters}.
        only (list[str] | None): file_name values of the figures to draw; None draws all.

    Returns:
        None
    """
    with open(config_path, encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    figures = config.get("figures", [])
    names = [figure["parameters"]["file_name"] for figure in figures]
    unknown = sorted(set(only or []) - set(names))
    if unknown:
        raise ValueError(f"--only {unknown} not in {config_path}; figures are {names}.")
    for figure, name in zip(figures, names):
        if only and name not in only:
            continue
        if figure["function"] not in PLOT_FUNCTIONS:
            raise ValueError(f"function must be one of {sorted(PLOT_FUNCTIONS)}; got {figure['function']!r}.")
        print(f"Drawing {name} with {figure['function']}")
        PLOT_FUNCTIONS[figure["function"]](plot_dir=config.get("plot_dir", "plots"), **figure["parameters"])


def main() -> None:
    """Draw the figures of a config from the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Figure config YAML, e.g. configs/figures/figure_1.yaml.")
    parser.add_argument("--only", nargs="+", help="Draw only the figures with these file_name values.")
    args = parser.parse_args()
    run_figures(args.config, args.only)


if __name__ == "__main__":
    main()

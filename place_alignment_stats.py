"""
Test whether uniform damage leaves place dedifferentiation short of its human level (Table S2).

Places are often the most dedifferentiated category in older adults. The question here
is whether a uniformly damaged model, at the damage level whose overall drop matches
older adults, leaves places further below their human drop than faces and objects are
below theirs. Relative drops, the human data, and the bootstrap are defined and loaded as
in fit_haupt_damage_levels.py (FFA -> face, LOC -> object, PPA -> place).

Place specificity index (PSI, Methods 2.5.2)
    At every tested level, each category's deviation from its human drop is the model's
    mean drop minus the human drop (percentage points), and
    PSI = place deviation - (face deviation + object deviation) / 2.
    Negative values mean places are less dedifferentiated than in older adults relative
    to the other categories. Table S2 and the Fig 1d side panels report the PSI at the
    tested level whose overall drop is closest to the human overall drop (the
    human-matched level of fit_haupt_damage_levels.py). Its two-sided p-value is
    Holm-adjusted across the damaged areas of a damage type. place_minus_face and
    place_minus_object are given as a breakdown.

Place slope difference (progress regression, Methods 2.5.3, Fig S3)
    Each category's progress towards its own human level (mean model drop / human drop,
    1 = human level) is regressed on the mean overall drop by weighted least squares
    (weights = simulations per level), over the levels up to and including the first
    that reaches the human overall drop (--trend-window matched; "all" uses every level).
    Slopes are per 10 percentage points of overall drop. The place slope difference is
    the place slope minus the mean face and object slope; negative means places approach
    their human level more slowly. Its two-sided p-value is Holm-adjusted across damaged
    areas.

Statistics
    Percentile bootstraps (100,000 resamples). Model simulations are resampled within
    each damage level, with the same per-level streams as fit_haupt_damage_levels.py, and
    human participants within each comparison-by-group cell, with the same human draws
    for every area. CIs therefore include human variability. p-values use the
    (extreme + 1) / (n + 1) form. p >= alpha does not establish equivalence.

Outputs in --output-dir (default stats/place_alignment), named like the Haupt fit:
    *_level-curves.csv     per area and level: mean and SD of the overall, category, and
                           place-excess drops, and whether the level is in the trend window
    *_level-index.csv      per area and level: PSI and breakdown with CIs and p-values;
                           is_closest_numerically marks the human-matched level
    *_progress-trend.csv   per area: category progress slopes (with the human drop) and
                           the place slope difference with its breakdown
    <model>_table-S2.csv   with a YAML config whose jobs share one model and output
                           directory: PSI and place slope difference per damage type and area

Usage:
    python place_alignment_stats.py --config configs/stats/cornet_rt5_haupt.yaml
    python place_alignment_stats.py --model cornet-rt --model-dir cornet_rt5_c \
        --damage-type connections --damage-layer All V1 V2 V4 IT
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from fit_haupt_damage_levels import (
    FIT_CATEGORIES,
    CandidateData,
    add_haupt_comparison_arguments,
    bootstrap_mean_distribution,
    bootstrap_p_two_sided,
    check_simulation_count,
    load_human_reference,
    load_layer_candidates,
    percentile_interval,
    safe_tag,
    sort_by_damage_area,
    stable_seed,
    to_relative_drop_distributions,
    validate_fit_args,
)
from utils import DAMAGE_AREA_LABELS, holm_adjust, load_config_jobs, namespace_from_mapping

TREND_WINDOWS = ("matched", "all")
# Report trend slopes per 10 percentage points of overall drop
SLOPE_SCALE = 10.0
CURVE_TARGETS = ("overall",) + FIT_CATEGORIES + ("place_excess",)
SETUP_COLUMNS = ["model", "damage_type", "damage_layer", "activation_layer"]
# Primary contrast first; the pairwise contrasts break it down
PLACE_INDEX_CONTRASTS = ("place_specificity_index", "place_minus_face", "place_minus_object")
# The same contrasts applied to progress slopes; the first is the place slope difference
PROGRESS_CONTRASTS = ("progress_place_minus_others", "progress_place_minus_face", "progress_place_minus_object")


@dataclass
class LayerAlignment:
    """Level curves, place index at every level, and progress slopes with bootstrap draws for one damaged area."""

    damage_layer: str
    curves: pd.DataFrame
    level_index: pd.DataFrame
    progress_slope: np.ndarray
    progress_slope_draws: np.ndarray


# ------------------------- Analysis -------------------------

def place_excess(values: Mapping) -> object:
    """
    Return the place drop minus the mean face and object drop.

    Example:
        face 27.5, object 59.2, place 54.7 -> 54.7 - (27.5 + 59.2) / 2 = 11.35
    """
    return values["place"] - 0.5 * (values["face"] + values["object"])


def curve_targets(category_drops: np.ndarray) -> dict[str, np.ndarray]:
    """
    Return overall, category, and place-excess drops from category drops.

    Parameters:
        category_drops (np.ndarray): Drops [..., 3] in FIT_CATEGORIES order.

    Returns:
        dict[str, np.ndarray]: One array [...] per CURVE_TARGETS entry; overall is the
        equal mean of the three categories, as in the Haupt fit.
    """
    category_drops = np.asarray(category_drops, dtype=float)
    if category_drops.shape[-1] != len(FIT_CATEGORIES):
        raise ValueError(f"Category drops need a last axis of {len(FIT_CATEGORIES)}; got shape {category_drops.shape}.")
    values = {category: category_drops[..., index] for index, category in enumerate(FIT_CATEGORIES)}
    values["overall"] = category_drops.mean(axis=-1)
    values["place_excess"] = place_excess(values)
    return {target: values[target] for target in CURVE_TARGETS}


def trend_window_mask(overall_means: np.ndarray, human_overall: float, window: str) -> np.ndarray:
    """
    Return which damage levels enter the progress regression.

    Example:
        overall means [20, 40, 60, 80] with human 47 -> "matched" keeps the first three levels.

    Parameters:
        overall_means (np.ndarray): Mean overall drop per level [L], in increasing level order.
        human_overall (float): Human overall relative drop (%).
        window (str): "matched" keeps levels up to and including the first that reaches the
            human drop (every level if none does); "all" keeps every level.

    Returns:
        np.ndarray: Boolean mask [L].
    """
    if window not in TREND_WINDOWS:
        raise ValueError(f"trend_window must be one of {list(TREND_WINDOWS)}; got {window!r}.")
    mask = np.ones(np.shape(overall_means), dtype=bool)
    if window == "matched":
        reached = np.flatnonzero(np.asarray(overall_means, dtype=float) >= human_overall)
        if reached.size:
            mask[reached[0] + 1:] = False
    return mask


def weighted_slopes(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Return weighted least-squares slopes of y on x along the last axis.

    With x constant within a level and y the level mean, weighting by the simulations per
    level gives the ordinary least-squares slope of the per-simulation values.

    Parameters:
        x (np.ndarray): Predictor per level [..., L].
        y (np.ndarray): Mean outcome per level [..., L].
        weights (np.ndarray): Simulations per level [L].

    Returns:
        np.ndarray: Slopes [...].
    """
    x, y, weights = (np.asarray(a, dtype=float) for a in (x, y, weights))
    if x.shape != y.shape or x.shape[-1] != len(weights):
        raise ValueError(f"x and y need the same shape [..., {len(weights)}]; got {x.shape} and {y.shape}.")
    weights = weights / weights.sum()
    x_centred = x - (x * weights).sum(axis=-1, keepdims=True)
    y_centred = y - (y * weights).sum(axis=-1, keepdims=True)
    variance = (weights * x_centred**2).sum(axis=-1)
    if np.any(variance <= 0):
        raise ValueError("The trend predictor must vary across levels to fit a slope.")
    return (weights * x_centred * y_centred).sum(axis=-1) / variance


def place_index_contrasts(deviation: np.ndarray) -> dict[str, np.ndarray]:
    """
    Return the place specificity index and its pairwise breakdown.

    Example:
        deviations face +4, object +2, place -6 -> index -9, place minus face -10,
        place minus object -8.

    Parameters:
        deviation (np.ndarray): Model minus human values [..., 3] in FIT_CATEGORIES order.

    Returns:
        dict[str, np.ndarray]: One array [...] per PLACE_INDEX_CONTRASTS entry.
    """
    deviation = np.asarray(deviation, dtype=float)
    face, object_, place = (deviation[..., FIT_CATEGORIES.index(c)] for c in ("face", "object", "place"))
    return {"place_specificity_index": place - 0.5 * (face + object_),
            "place_minus_face": place - face,
            "place_minus_object": place - object_}


def progress_slopes(overall: np.ndarray, category_means: np.ndarray, weights: np.ndarray,
                    human_categories: np.ndarray) -> np.ndarray:
    """
    Return how fast each category approaches its human drop as damage grows.

    Progress is the model drop divided by the human drop of the same category, so its
    slope on the overall drop is the drop slope divided by the human drop.

    Example:
        drop slope 1.0 and human drop 50 -> 10 * 1.0 / 50 = 0.2: ten points of overall
        drop cover a fifth of the way to the human level.

    Parameters:
        overall (np.ndarray): Mean overall drop per level [..., L].
        category_means (np.ndarray): Mean category drops [..., L, 3].
        weights (np.ndarray): Simulations per level [L].
        human_categories (np.ndarray): Human category drops [..., 3].

    Returns:
        np.ndarray: Progress per 10 points of overall drop [..., 3]; NaN where a human drop is not positive.
    """
    drop_slopes = np.stack([weighted_slopes(overall, category_means[..., i], weights)
                            for i in range(len(FIT_CATEGORIES))], axis=-1)
    human_categories = np.asarray(human_categories, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = SLOPE_SCALE * drop_slopes / human_categories
    return np.where(human_categories > 0, slopes, np.nan)


def level_curve_table(levels: np.ndarray, level_drops: list[np.ndarray], in_window: np.ndarray) -> pd.DataFrame:
    """
    Return the mean and SD of every CURVE_TARGETS drop per damage level.

    Parameters:
        levels (np.ndarray): Damage levels [L].
        level_drops (list[np.ndarray]): Simulation category drops [N,3] per level.
        in_window (np.ndarray): Trend-window mask [L].

    Returns:
        pd.DataFrame: One row per level with n_simulations, in_trend_window, and
        <target>_mean and <target>_sd columns.
    """
    rows = []
    for level, drops, used in zip(levels, level_drops, in_window):
        row = {"damage_level": float(level), "n_simulations": len(drops), "in_trend_window": bool(used)}
        for target, values in curve_targets(drops).items():
            row[f"{target}_mean"] = float(values.mean())
            row[f"{target}_sd"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def contrast_row(estimate: float, draws: np.ndarray) -> dict[str, float]:
    """
    Return the estimate, 95% CI, bootstrap SE and z, and two-sided p-value of one contrast.

    The bootstrap z (estimate / SE) is descriptive; the CI and p-value are the test.
    Non-finite draws (a human drop of zero) are left out.
    """
    finite = np.asarray(draws, dtype=float)
    finite = finite[np.isfinite(finite)]
    lower, upper = percentile_interval(finite) if finite.size else (np.nan, np.nan)
    standard_error = float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan
    return {"estimate": float(estimate), "ci95_lower": lower, "ci95_upper": upper,
            "bootstrap_se": standard_error,
            "bootstrap_z": float(estimate) / standard_error if standard_error > 0 else np.nan,
            "bootstrap_p_two_sided": bootstrap_p_two_sided(finite) if finite.size else np.nan}


def level_index_table(levels: np.ndarray, point_means: np.ndarray, draw_means: np.ndarray,
                      human_point: Mapping[str, float], human_bootstrap: pd.DataFrame) -> pd.DataFrame:
    """
    Return the place index contrasts at every tested damage level, without interpolation.

    Parameters:
        levels (np.ndarray): Damage levels [L].
        point_means (np.ndarray): Mean category drops [L,3].
        draw_means (np.ndarray): Bootstrap mean category drops [D,L,3].
        human_point (Mapping[str, float]): Human relative drop per target.
        human_bootstrap (pd.DataFrame): Human bootstrap drops [D, targets].

    Returns:
        pd.DataFrame: One row per level and PLACE_INDEX_CONTRASTS entry with damage_level,
        overall_drop_percent, quantity, and the contrast_row() columns.
    """
    human_categories = np.array([human_point[c] for c in FIT_CATEGORIES])
    human_draws = np.column_stack([human_bootstrap[c].to_numpy() for c in FIT_CATEGORIES])
    rows = []
    for index, level in enumerate(levels):
        point = place_index_contrasts(point_means[index] - human_categories)
        draws = place_index_contrasts(draw_means[:, index] - human_draws)
        for contrast in PLACE_INDEX_CONTRASTS:
            rows.append({"damage_level": float(level), "overall_drop_percent": float(point_means[index].mean()),
                         "quantity": contrast, **contrast_row(point[contrast], draws[contrast])})
    return pd.DataFrame(rows)


def analyze_layer(candidates: list[CandidateData], human_point: Mapping[str, float], human_bootstrap: pd.DataFrame,
                  damage_type: str, trend_window: str, baseline_level: float, seed: int,
                  expected_simulations: int) -> LayerAlignment:
    """
    Compute level curves, the place index at every level, and progress slopes for one damaged area.

    Parameters:
        candidates (list[CandidateData]): Every damage level of one area, increasing, without the intact level.
        human_point (Mapping[str, float]): Human relative drop per target.
        human_bootstrap (pd.DataFrame): Human bootstrap drops; its length sets the model resamples.
        damage_type (str): Damage label, e.g. "connections".
        trend_window (str): One of TREND_WINDOWS.
        baseline_level (float): Intact level.
        seed (int): Base seed; each level uses the Haupt fit's stream.
        expected_simulations (int): Required simulations per level; 0 skips the check.

    Returns:
        LayerAlignment: Curves, level index, and progress slopes with bootstrap draws.
    """
    damage_layer = candidates[0].damage_layer
    levels = np.array([candidate.damage_level for candidate in candidates])
    if np.any(np.diff(levels) <= 0) or levels[0] <= baseline_level:
        raise ValueError(f"{damage_type}/{damage_layer} levels must increase and exceed {baseline_level:g}; got {levels.tolist()}.")

    level_drops, level_draws = [], []
    for candidate in candidates:
        drops = to_relative_drop_distributions(candidate.matched, FIT_CATEGORIES)
        check_simulation_count(len(drops), f"{candidate.source} level {candidate.damage_level:g}", expected_simulations)
        values = drops.loc[:, list(FIT_CATEGORIES)].to_numpy(dtype=float)
        # Same resampling stream as fit_haupt_damage_levels.py at this level
        rng = np.random.default_rng(stable_seed(seed, damage_type, damage_layer, f"{candidate.damage_level:.10g}"))
        level_drops.append(values)
        level_draws.append(bootstrap_mean_distribution(values, len(human_bootstrap), rng))

    n_simulations = np.array([len(values) for values in level_drops])
    point_means = np.stack([values.mean(axis=0) for values in level_drops])
    draw_means = np.stack(level_draws, axis=1)
    point_overall = curve_targets(point_means)["overall"]
    in_window = trend_window_mask(point_overall, human_point["overall"], trend_window)
    if in_window.sum() < 2:
        raise ValueError(f"{damage_type}/{damage_layer}: the {trend_window!r} trend window holds "
                         f"{int(in_window.sum())} level(s); at least two are needed.")
    weights = n_simulations[in_window]
    human_categories = np.array([human_point[c] for c in FIT_CATEGORIES])
    human_draws = np.column_stack([human_bootstrap[c].to_numpy() for c in FIT_CATEGORIES])

    return LayerAlignment(
        damage_layer=damage_layer,
        curves=level_curve_table(levels, level_drops, in_window),
        level_index=level_index_table(levels, point_means, draw_means, human_point, human_bootstrap),
        progress_slope=progress_slopes(point_overall[in_window], point_means[in_window], weights, human_categories),
        progress_slope_draws=progress_slopes(curve_targets(draw_means)["overall"][:, in_window],
                                             draw_means[:, in_window], weights, human_draws),
    )


# ------------------------- Tables -------------------------

def progress_table(results: list[LayerAlignment], setup: Mapping[str, str], human_point: Mapping[str, float]) -> pd.DataFrame:
    """
    Return progress slopes and the place slope difference per damaged area.

    Rows per area, by role: category_slope (face, object, and place progress per 10
    points of overall drop, with the category's human drop), interaction (place minus
    the mean face and object slope; two-sided p Holm-adjusted across areas), and
    breakdown (place minus face, place minus object).
    """
    rows = []
    for result in results:
        base = {**setup, "damage_layer": result.damage_layer}
        for index, category in enumerate(FIT_CATEGORIES):
            rows.append({**base, "quantity": f"{category}_progress_slope", "role": "category_slope",
                         "human_drop_percent": human_point[category],
                         **contrast_row(result.progress_slope[index], result.progress_slope_draws[:, index])})
        points, draws = place_index_contrasts(result.progress_slope), place_index_contrasts(result.progress_slope_draws)
        for name, quantity in zip(PLACE_INDEX_CONTRASTS, PROGRESS_CONTRASTS):
            rows.append({**base, "quantity": quantity,
                         "role": "interaction" if quantity == PROGRESS_CONTRASTS[0] else "breakdown",
                         **contrast_row(points[name], draws[name])})
    table = pd.DataFrame(rows)
    interaction = table["role"] == "interaction"
    table.loc[interaction, "bootstrap_p_holm_layers"] = holm_adjust(table.loc[interaction, "bootstrap_p_two_sided"])
    return table


def flag_closest_level(level_index: pd.DataFrame, human_overall: float) -> pd.DataFrame:
    """
    Flag each area's human-matched level and Holm-adjust its place index across areas.

    The human-matched level has the mean overall drop nearest the human drop, first on
    ties, as closest_numerically in fit_haupt_damage_levels.py.

    Parameters:
        level_index (pd.DataFrame): level_index_table() rows of one job, with damage_layer.
        human_overall (float): Human overall relative drop (%).

    Returns:
        pd.DataFrame: Copy with is_closest_numerically and, on the closest
        place_specificity_index rows, bootstrap_p_holm_layers.
    """
    table = level_index.copy()
    distance = (table["overall_drop_percent"] - human_overall).abs()
    closest = {layer: table.loc[group.idxmin(), "damage_level"]
               for layer, group in distance.groupby(table["damage_layer"], sort=False)}
    table["is_closest_numerically"] = table["damage_level"] == table["damage_layer"].map(closest)
    primary = table["is_closest_numerically"] & (table["quantity"] == PLACE_INDEX_CONTRASTS[0])
    table.loc[primary, "bootstrap_p_holm_layers"] = holm_adjust(table.loc[primary, "bootstrap_p_two_sided"])
    return table


def build_table_s2(level_index: pd.DataFrame, progress: pd.DataFrame) -> pd.DataFrame:
    """
    Return Table S2: the place specificity index and place slope difference per area.

    Parameters:
        level_index (pd.DataFrame): flag_closest_level() rows of one or more jobs.
        progress (pd.DataFrame): progress_table() rows of the same jobs.

    Returns:
        pd.DataFrame: One row per damage type and area with psi_damage_level and psi_* /
        interaction_* estimate, ci95_lower, ci95_upper, p_two_sided, and p_holm.
    """
    keys = ["damage_type", "damage_layer"]
    renames = {"estimate": "estimate", "ci95_lower": "ci95_lower", "ci95_upper": "ci95_upper",
               "bootstrap_p_two_sided": "p_two_sided", "bootstrap_p_holm_layers": "p_holm"}
    sources = {
        "psi": (level_index[level_index["is_closest_numerically"]], "place_specificity_index", {"damage_level": "damage_level"}),
        "interaction": (progress, "progress_place_minus_others", {}),
    }
    parts = []
    for prefix, (table, quantity, extra) in sources.items():
        columns = {**extra, **renames}
        rows = table.loc[table["quantity"] == quantity, keys + list(columns)]
        if rows.duplicated(keys).any():
            raise ValueError(f"{quantity} has several rows for one damage type and area.")
        parts.append(rows.rename(columns={c: f"{prefix}_{name}" for c, name in columns.items()}).set_index(keys))
    table = sort_by_damage_area(pd.concat(parts, axis=1).reset_index())
    table.insert(1, "damage_area", table.pop("damage_layer").map(lambda l: DAMAGE_AREA_LABELS.get(l, l)))
    return table


# ------------------------- Running -------------------------

def setup_first(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the frame with SETUP_COLUMNS first."""
    return frame.loc[:, SETUP_COLUMNS + [c for c in frame.columns if c not in SETUP_COLUMNS]]


def run_alignment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run one damage type and write the level-curve, level-index, and progress tables.

    Parameters:
        args (argparse.Namespace): Parsed command-line or YAML job options.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Level-index rows (flagged) and progress rows,
        stacked across jobs by run_config() for Table S2.
    """
    _, human_point, human_bootstrap = load_human_reference(args)
    setup = {"model": args.model, "damage_type": args.damage_type, "activation_layer": args.activation_layer}
    results = [analyze_layer(load_layer_candidates(args, layer), human_point, human_bootstrap, args.damage_type,
                             args.trend_window, args.baseline_level, args.seed, args.expected_simulations)
               for layer in args.damage_layer]

    curves = setup_first(pd.concat([r.curves.assign(**setup, damage_layer=r.damage_layer) for r in results],
                                   ignore_index=True))
    level_index = setup_first(flag_closest_level(
        pd.concat([r.level_index.assign(**setup, damage_layer=r.damage_layer) for r in results], ignore_index=True),
        human_point["overall"]))
    progress = setup_first(progress_table(results, setup, human_point))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join([safe_tag(args.model), safe_tag(args.damage_type), safe_tag("-".join(args.damage_layer)),
                    safe_tag(args.activation_layer), f"top{args.selectivity_fraction:.2f}", "place-alignment"])
    curves.to_csv(output_dir / f"{tag}_level-curves.csv", index=False)
    level_index.to_csv(output_dir / f"{tag}_level-index.csv", index=False)
    progress.to_csv(output_dir / f"{tag}_progress-trend.csv", index=False)

    closest = level_index[level_index["is_closest_numerically"] & (level_index["quantity"] == PLACE_INDEX_CONTRASTS[0])]
    print("Place specificity index at the human-matched level (negative = places short of humans)")
    print(closest[["damage_layer", "damage_level", "estimate", "ci95_lower", "ci95_upper",
                   "bootstrap_p_two_sided", "bootstrap_p_holm_layers"]].to_string(index=False))
    print("\nPlace slope difference per 10 points of overall drop")
    print(progress.loc[progress["role"] == "interaction", ["damage_layer", "estimate", "ci95_lower", "ci95_upper",
                                                         "bootstrap_p_two_sided", "bootstrap_p_holm_layers"]].to_string(index=False))
    return level_index, progress


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML file with a place_alignment defaults/jobs section; replaces the single-run options.")
    add_haupt_comparison_arguments(parser)
    parser.set_defaults(output_dir="stats/place_alignment")
    parser.add_argument("--trend-window", choices=TREND_WINDOWS, default="matched",
                        help="Levels in the progress regression: 'matched' stops at the first level reaching the "
                             "human overall drop; 'all' uses every level.")
    return parser


def run_config(config_path: str | Path, parser: argparse.ArgumentParser) -> None:
    """Run all jobs of a YAML place_alignment section and write Table S2 when they share one model and output directory."""
    outputs, job_targets = [], set()
    for job_index, job in enumerate(load_config_jobs(config_path, "place_alignment"), start=1):
        damage_layers = job.pop("damage_layers", None)
        if not damage_layers:
            raise ValueError(f"place_alignment job {job_index} in {config_path} needs damage_layers.")
        args = namespace_from_mapping(parser, job)
        args.damage_layer = [str(layer) for layer in damage_layers]
        validate_fit_args(args)
        print(f"\nRunning place alignment | model={args.model} | type={args.damage_type} | layers={','.join(args.damage_layer)}")
        outputs.append(run_alignment(args))
        job_targets.add((args.model, args.output_dir))

    if len(job_targets) != 1:
        print(f"\nNo Table S2 written: jobs use different models or output directories {sorted(job_targets)}.")
        return
    model, output_dir = job_targets.pop()
    path = Path(output_dir) / f"{safe_tag(model)}_table-S2.csv"
    build_table_s2(pd.concat([o[0] for o in outputs], ignore_index=True),
                   pd.concat([o[1] for o in outputs], ignore_index=True)).to_csv(path, index=False)
    print(f"\nSaved Table S2 to {path.resolve()}")


def main() -> None:
    """Run the analysis from the command line or a YAML job matrix."""
    parser = build_parser()
    args = parser.parse_args()
    if args.config:
        run_config(args.config, parser)
        return
    validate_fit_args(args)
    run_alignment(args)


if __name__ == "__main__":
    main()

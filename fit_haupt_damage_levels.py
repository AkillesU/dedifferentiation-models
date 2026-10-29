"""
Compare model dedifferentiation with older adults (Haupt et al., 2025) and write Table S1.

Model input: the per-simulation differentiation table of one model
(data/differentiation/<model_dir>.csv.gz, from compute_differentiation.py). Human input:
the participant-level differentiation digitised from Haupt et al. Figure 6B (normalised
space, all runs; data/reference/haupt_2025/haupt2025_fig6B_derived.csv). ROIs map to model
categories as FFA -> face, LOC -> object, PPA -> place. Animals have no human ROI and are
left out (Methods 2.1.3).

Both sides are expressed as a relative drop in differentiation (Methods 2.5.1):
- Model: 100 - scaled differentiation per simulation, where each simulation is scaled by
  its own intact value. The overall drop is the equal mean of face, object, and place
  within each simulation.
- Human: 100 * (1 - sum(old means) / sum(young means)) over an ROI's three comparisons
  (ratio of group means, as in their Table S5). The overall drop is the equal mean of the
  three category drops. Younger and older adults are different people, so no
  per-participant drop exists.

By default each digitised comparison-by-group cell is shifted so its mean equals the
published Table S5 mean (young PPA P vs F has 20 of 21 dots), keeping the digitised spread;
--no-center-on-table-s5 keeps the digitised means.

Statistics are non-parametric bootstraps (100,000 resamples). Units of independence are
model simulations (resampled as whole rows, so categories stay paired) and human
participants within each comparison-by-group cell; cells are resampled independently
because participants cannot be followed across the rank-sorted dots. The human bootstrap
stream depends only on --seed and --n-bootstrap, which are shared with
place_alignment_stats.py, so the human CIs agree across Tables S1 and S2. Each damage level
has its own reproducible model stream. Two-sided p-values use the
(extreme + 1) / (n_bootstrap + 1) form; category p-values are Holm-adjusted across face,
object, and place at each level.

Outputs in --output-dir (default stats/haupt_fits):
- <tag>_descriptives.csv: one human row (mean, 95% CI) and, for each damaged area, the
  level whose overall drop is numerically closest to the human drop, with the model mean,
  SD, and 95% CI of the overall and category drops.
A YAML run whose jobs share one model and output directory also writes:
- <model>_haupt-fit_descriptives_all-jobs.csv: the descriptive tables of all jobs; the
  figure configs read the human-matched levels from it (damage_level: best_fit).
- <model>_table-S1_levels.csv: Table S1, every tested level with bootstrap 95% CIs and the
  flags is_bracket_below / is_bracket_above / is_closest_numerically.
- <model>_table-S1_human.csv: the human row of Table S1.

Usage:
    python fit_haupt_damage_levels.py --config configs/stats/cornet_rt5_haupt.yaml
    python fit_haupt_damage_levels.py --model cornet-rt --model-dir cornet_rt5_c \
        --damage-type noise --damage-layer All V1 V2 V4 IT
"""

from __future__ import annotations

import argparse
import math
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils import (
    DAMAGE_AREA_LABELS,
    holm_adjust,
    load_config_jobs,
    load_differentiation_table,
    matched_category_values,
    namespace_from_mapping,
    select_condition,
)

FIT_CATEGORIES = ("face", "object", "place")
FIT_TARGETS = ("overall",) + FIT_CATEGORIES
HUMAN_SUMMARY_TARGETS = FIT_TARGETS + ("pooled",)
ROI_CATEGORIES = {"FFA": "face", "LOC": "object", "PPA": "place"}
HUMAN_GROUPS = ("young", "old")
# Table S5 column holding each age group's mean differentiation
TABLE_S5_GROUP_COLUMNS = {"young": "mean_younger", "old": "mean_older"}
# Figure 6B comparison codes, e.g. "F_vs_P" -> face vs place.
COMPARISON_CODES = {"F": "face", "O": "object", "P": "place", "A": "animal"}
DESCRIPTIVE_LAYER_ORDER = ("V1", "V2", "V4", "IT", "All")
# Supplementary tables list the whole model first.
TABLE_LAYER_ORDER = ("All", "V1", "V2", "V4", "IT")
# select_levels role -> Table S1 flag column.
TABLE_S1_FLAGS = {
    "closest_below_human": "is_bracket_below",
    "closest_above_human": "is_bracket_above",
    "closest_numerically": "is_closest_numerically",
}
HUMAN_SOURCE_LABEL = "Human (Haupt et al., 2025)"
CI_LEVEL = 0.95
BOOTSTRAP_BLOCK_SIZE = 10000
SETUP_COLUMNS = ["model", "damage_type", "damage_layer", "activation_layer"]


@dataclass(frozen=True)
class CandidateData:
    """Matched model values (scaled %) for one damaged area and level."""

    damage_layer: str
    damage_level: float
    selectivity_fraction: float
    source: str
    matched: pd.DataFrame


def safe_tag(value: object) -> str:
    """Return a file-system-safe result tag."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def stable_seed(seed: int, *labels: object) -> list[int]:
    """
    Return seed entropy that is repeatable for the same seed and labels.

    Example:
        stable_seed(12345, "noise", "IT", 0.6) always gives the same list, so each
        damage level has its own reproducible bootstrap stream.
    """
    label = "|".join(str(item) for item in labels)
    return [int(seed), zlib.crc32(label.encode("utf-8"))]


# ------------------------- Human data -------------------------

def load_table_s5_cell_means(csv_path: str | Path,
                             analysis_variant: str = "normalized_all_runs") -> dict[tuple[str, str, str], float]:
    """
    Read Table S5 group means keyed like the Figure 6B cells.

    Example:
        PPA, place vs face, younger -> ("PPA", "P_vs_F", "young"): 0.4552

    Parameters:
        csv_path (str | Path): Path to table_s5_raw.csv.
        analysis_variant (str): Table S5 analysis, e.g. "normalized_all_runs".

    Returns:
        dict[tuple[str, str, str], float]: Mean differentiation keyed by (roi, comparison, group).
    """
    table = pd.read_csv(csv_path, encoding="utf-8-sig")
    variants = sorted(table["analysis_variant"].unique())
    if analysis_variant not in variants:
        raise ValueError(f"analysis_variant must be one of {variants}; got {analysis_variant!r}.")
    codes = {category: code for code, category in COMPARISON_CODES.items()}
    means = {}
    for row in table[table["analysis_variant"] == analysis_variant].itertuples():
        comparison = f"{codes[row.preferred_category]}_vs_{codes[row.comparison_category]}"
        for group, column in TABLE_S5_GROUP_COLUMNS.items():
            means[(row.roi, comparison, group)] = float(getattr(row, column))
    return means


def load_human_subject_values(csv_path: str | Path, value_column: str = "value_moment_matched",
                              center_means: dict[tuple[str, str, str], float] | None = None
                              ) -> dict[str, dict[str, list[np.ndarray]]]:
    """
    Read participant-level Haupt Figure 6B differentiation values.

    Parameters:
        csv_path (str | Path): Path to haupt2025_fig6B_derived.csv.
        value_column (str): Column holding the values, e.g. "value_moment_matched".
        center_means (dict | None): Target means keyed by (roi, comparison, group), e.g.
            from load_table_s5_cell_means(); each cell is shifted to that mean. None keeps
            the digitised values.

    Returns:
        dict[str, dict[str, list[np.ndarray]]]: values[category][group] is a list with one
        participant array per comparison, ordered by comparison name so young and old align.
    """
    table = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = {"roi", "comparison", "group", value_column} - set(table.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns {sorted(missing)}; has {list(table.columns)}.")
    unknown_rois = set(table["roi"]) - set(ROI_CATEGORIES)
    if unknown_rois:
        raise ValueError(f"roi must be one of {sorted(ROI_CATEGORIES)}; got {sorted(unknown_rois)}.")
    unknown_groups = set(table["group"]) - set(HUMAN_GROUPS)
    if unknown_groups:
        raise ValueError(f"group must be one of {list(HUMAN_GROUPS)}; got {sorted(unknown_groups)}.")
    values = pd.to_numeric(table[value_column], errors="coerce")
    if not np.isfinite(values).all():
        raise ValueError(f"{value_column!r} has non-finite values in rows {table.index[~np.isfinite(values)].tolist()[:10]}.")
    table = table.assign(**{value_column: values})

    subject_values: dict[str, dict[str, list[np.ndarray]]] = {}
    for roi, category in ROI_CATEGORIES.items():
        roi_rows = table[table["roi"] == roi]
        comparisons = {group: sorted(roi_rows.loc[roi_rows["group"] == group, "comparison"].unique())
                       for group in HUMAN_GROUPS}
        if not comparisons["young"] or comparisons["young"] != comparisons["old"]:
            raise ValueError(f"{roi} needs the same comparisons for young and old; got {comparisons}.")
        subject_values[category] = {}
        for group in HUMAN_GROUPS:
            cells = []
            for comparison in comparisons[group]:
                cell = roi_rows.loc[(roi_rows["group"] == group) & (roi_rows["comparison"] == comparison),
                                    value_column].to_numpy(dtype=float)
                if len(cell) < 2:
                    raise ValueError(f"{roi} {comparison} {group} needs at least two participants; got {len(cell)}.")
                if center_means is not None:
                    key = (roi, comparison, group)
                    if key not in center_means:
                        raise ValueError(f"center_means has no mean for {key}.")
                    # Keep the digitised spread but match the published mean
                    cell = cell - cell.mean() + center_means[key]
                cells.append(cell)
            subject_values[category][group] = cells
    return subject_values


def relative_drop_percent(young_means: np.ndarray, old_means: np.ndarray) -> np.ndarray:
    """
    Return the ratio-of-means relative drop over the last (comparison) axis.

    Parameters:
        young_means (np.ndarray): Young group means [..., n_comparisons].
        old_means (np.ndarray): Old group means, same shape.

    Returns:
        np.ndarray: 100 * (1 - sum(old) / sum(young)) with shape [...].
    """
    young_means, old_means = np.asarray(young_means, dtype=float), np.asarray(old_means, dtype=float)
    if young_means.shape != old_means.shape:
        raise ValueError(f"Young and old means need the same shape; got {young_means.shape} and {old_means.shape}.")
    young_total = young_means.sum(axis=-1)
    if np.any(young_total <= 0):
        raise ValueError(f"Young differentiation must be positive; minimum summed mean is {float(np.min(young_total)):.4g}.")
    return 100.0 * (1.0 - old_means.sum(axis=-1) / young_total)


def human_drop_estimates(subject_values: dict[str, dict[str, list[np.ndarray]]]) -> dict[str, float]:
    """Return the human relative drop for every HUMAN_SUMMARY_TARGETS entry."""
    drops = {}
    pooled = {group: [] for group in HUMAN_GROUPS}
    for category in FIT_CATEGORIES:
        means = {group: np.array([cell.mean() for cell in subject_values[category][group]]) for group in HUMAN_GROUPS}
        drops[category] = float(relative_drop_percent(means["young"], means["old"]))
        for group in HUMAN_GROUPS:
            pooled[group].extend(means[group])
    drops["overall"] = float(np.mean([drops[c] for c in FIT_CATEGORIES]))
    # Ratio of summed means over all nine comparisons
    drops["pooled"] = float(relative_drop_percent(pooled["young"], pooled["old"]))
    return {target: drops[target] for target in HUMAN_SUMMARY_TARGETS}


def bootstrap_mean_distribution(values: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    """
    Return bootstrap means of the rows of a sample.

    Parameters:
        values (np.ndarray): Sample [N] or [N,K]; rows are independent units.
        n_bootstrap (int): Number of resamples.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: Bootstrap means [n_bootstrap,K].
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"Bootstrap sample needs shape [N,K] with N >= 2; got {values.shape}.")
    n = values.shape[0]
    means = np.empty((n_bootstrap, values.shape[1]))
    # Draw in blocks so 100,000 resamples of 200 simulations stay small in memory
    for start in range(0, n_bootstrap, BOOTSTRAP_BLOCK_SIZE):
        stop = min(start + BOOTSTRAP_BLOCK_SIZE, n_bootstrap)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[indices].mean(axis=1)
    return means


def bootstrap_human_drops(subject_values: dict[str, dict[str, list[np.ndarray]]], n_bootstrap: int,
                          rng: np.random.Generator) -> pd.DataFrame:
    """
    Return the bootstrap distribution of human relative drops.

    Parameters:
        subject_values (dict): Output of load_human_subject_values().
        n_bootstrap (int): Number of resamples.
        rng (np.random.Generator): Random number generator.

    Returns:
        pd.DataFrame: Relative drops [n_bootstrap, HUMAN_SUMMARY_TARGETS].
    """
    drops = {}
    pooled = {group: [] for group in HUMAN_GROUPS}
    for category in FIT_CATEGORIES:
        means = {
            group: np.column_stack([bootstrap_mean_distribution(cell, n_bootstrap, rng)[:, 0]
                                    for cell in subject_values[category][group]])
            for group in HUMAN_GROUPS
        }
        drops[category] = relative_drop_percent(means["young"], means["old"])
        for group in HUMAN_GROUPS:
            pooled[group].append(means[group])
    frame = pd.DataFrame(drops)
    frame["overall"] = frame.loc[:, list(FIT_CATEGORIES)].mean(axis=1)
    frame["pooled"] = relative_drop_percent(np.hstack(pooled["young"]), np.hstack(pooled["old"]))
    return frame.loc[:, list(HUMAN_SUMMARY_TARGETS)]


def load_human_reference(args: argparse.Namespace
                         ) -> tuple[dict[str, dict[str, list[np.ndarray]]], dict[str, float], pd.DataFrame]:
    """
    Return the human participant values, point drops, and bootstrap drops.

    Parameters:
        args (argparse.Namespace): Options with human_csv, human_value_column,
            center_on_table_s5, table_s5_csv, n_bootstrap, and seed.

    Returns:
        tuple: load_human_subject_values() output, the point drop per HUMAN_SUMMARY_TARGETS
        entry, and the bootstrap drops [n_bootstrap, HUMAN_SUMMARY_TARGETS].
    """
    center_means = load_table_s5_cell_means(args.table_s5_csv) if args.center_on_table_s5 else None
    subject_values = load_human_subject_values(args.human_csv, args.human_value_column, center_means)
    human_bootstrap = bootstrap_human_drops(subject_values, args.n_bootstrap,
                                            np.random.default_rng(stable_seed(args.seed, "human")))
    return subject_values, human_drop_estimates(subject_values), human_bootstrap


# ------------------------- Model data -------------------------

def load_layer_candidates(args: argparse.Namespace, damage_layer: str) -> list[CandidateData]:
    """
    Return the matched model values at every damage level of one damaged area.

    Parameters:
        args (argparse.Namespace): Options with model_dir, differentiation_dir, damage_type,
            activation_layer, selectivity_fraction, baseline_level, and tolerance.
        damage_layer (str): Damaged area, e.g. "V1" or "All".

    Returns:
        list[CandidateData]: One entry per non-baseline level, in increasing order.
    """
    table = load_differentiation_table(args.model_dir, args.differentiation_dir)
    condition = select_condition(table, args.damage_type, damage_layer, args.activation_layer,
                                 args.selectivity_fraction)
    levels = sorted(level for level in condition["damage_level"].unique()
                    if not math.isclose(level, args.baseline_level, rel_tol=0.0, abs_tol=args.tolerance))
    source = f"{args.model_dir}/{args.damage_type}/{damage_layer}/{args.activation_layer}"
    return [
        CandidateData(
            damage_layer=damage_layer,
            damage_level=float(level),
            selectivity_fraction=args.selectivity_fraction,
            source=source,
            matched=matched_category_values(condition, level, FIT_CATEGORIES, args.baseline_level),
        )
        for level in levels
    ]


def check_simulation_count(n_simulations: int, source: str, expected_simulations: int) -> None:
    """Fail when a damage level does not have the expected number of complete simulations (0 skips)."""
    if expected_simulations > 0 and n_simulations != expected_simulations:
        raise ValueError(f"{source} has {n_simulations} complete simulations; expected {expected_simulations}.")


def to_relative_drop_distributions(matched: pd.DataFrame, categories: Iterable[str] = FIT_CATEGORIES) -> pd.DataFrame:
    """
    Convert matched scaled values to relative drops and add the equal-category overall drop.

    Parameters:
        matched (pd.DataFrame): Scaled differentiation (%) per simulation and category.
        categories (Iterable[str]): Categories to convert.

    Returns:
        pd.DataFrame: 100 - scaled value per category, plus "overall" [n_simulations, K+1].
    """
    categories = tuple(categories)
    missing = set(categories) - set(matched.columns)
    if missing:
        raise ValueError(f"Matched values are missing categories: {sorted(missing)}")
    if matched["replicate"].duplicated().any():
        raise ValueError("Matched values contain duplicate simulation identifiers.")
    drops = 100.0 - matched.loc[:, categories].astype(float)
    drops["overall"] = drops.loc[:, categories].mean(axis=1)
    return drops.reset_index(drop=True)


# ------------------------- Statistics -------------------------

def bootstrap_p_two_sided(bootstrap_values: np.ndarray) -> float:
    """
    Return the two-sided bootstrap p-value for a difference of zero.

    Each tail uses the (extreme + 1) / (n_bootstrap + 1) form, and the smaller tail is doubled.
    """
    bootstrap_values = np.asarray(bootstrap_values, dtype=float)
    n = len(bootstrap_values)
    lower = (np.sum(bootstrap_values <= 0.0) + 1) / (n + 1)
    upper = (np.sum(bootstrap_values >= 0.0) + 1) / (n + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    """Return the central CI_LEVEL percentile interval of bootstrap values."""
    tail = (1.0 - CI_LEVEL) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return float(lower), float(upper)


def compare_level_to_human(drops: pd.DataFrame, human_point: dict[str, float], human_bootstrap: pd.DataFrame,
                           rng: np.random.Generator, alpha: float) -> pd.DataFrame:
    """
    Compare one damage level's drops with the human drops.

    Parameters:
        drops (pd.DataFrame): Model relative drops [n_simulations, FIT_TARGETS].
        human_point (dict[str, float]): Human relative drop per fit target.
        human_bootstrap (pd.DataFrame): Human bootstrap drops [n_bootstrap, FIT_TARGETS];
            its length sets the number of model resamples.
        rng (np.random.Generator): Generator for resampling model simulations.
        alpha (float): Significance threshold for the labelled columns.

    Returns:
        pd.DataFrame: One row per fit target with model and human drops, the model minus
        human difference with its 95% CI, and two-sided bootstrap p-values.
    """
    targets = list(FIT_TARGETS)
    model_values = drops.loc[:, targets].to_numpy(dtype=float)
    human_draws = human_bootstrap.loc[:, targets].to_numpy(dtype=float)
    model_draws = bootstrap_mean_distribution(model_values, human_draws.shape[0], rng)

    model_mean = model_values.mean(axis=0)
    human = np.array([human_point[target] for target in targets])
    difference = model_mean - human
    difference_draws = model_draws - human_draws

    rows = []
    for j, target in enumerate(targets):
        difference_ci = percentile_interval(difference_draws[:, j])
        model_ci = percentile_interval(model_draws[:, j])
        human_ci = percentile_interval(human_draws[:, j])
        p_value = bootstrap_p_two_sided(difference_draws[:, j])
        rows.append({
            "target": target,
            "n_simulations": int(model_values.shape[0]),
            "human_relative_drop_percent": human[j],
            "human_ci95_lower": human_ci[0],
            "human_ci95_upper": human_ci[1],
            "model_mean_relative_drop_percent": model_mean[j],
            "model_sd": float(np.std(model_values[:, j], ddof=1)),
            "model_ci95_lower": model_ci[0],
            "model_ci95_upper": model_ci[1],
            "difference_model_minus_human": difference[j],
            "difference_ci95_lower": difference_ci[0],
            "difference_ci95_upper": difference_ci[1],
            "absolute_difference": abs(difference[j]),
            "bootstrap_p_two_sided": p_value,
            "significant_difference_at_alpha": p_value < alpha,
        })
    frame = pd.DataFrame(rows)
    is_category = frame["target"].isin(FIT_CATEGORIES)
    frame.loc[is_category, "bootstrap_p_holm_three_categories"] = holm_adjust(frame.loc[is_category, "bootstrap_p_two_sided"])
    return frame


def analyze_candidates(candidates: Iterable[CandidateData], model: str, damage_type: str, activation_layer: str,
                       human_point: dict[str, float], human_bootstrap: pd.DataFrame, expected_simulations: int,
                       alpha: float, seed: int) -> pd.DataFrame:
    """
    Compare every candidate damage level with the human drops.

    Parameters:
        candidates (Iterable[CandidateData]): Model levels to compare.
        model (str): Model label, e.g. "cornet-rt".
        damage_type (str): Damage label, e.g. "noise".
        activation_layer (str): Readout layer, e.g. "IT".
        human_point (dict[str, float]): Human relative drop per fit target.
        human_bootstrap (pd.DataFrame): Human bootstrap drops, shared by all levels.
        expected_simulations (int): Required simulations per level; 0 skips the check.
        alpha (float): Significance threshold.
        seed (int): Base seed; each level derives its own stream from it.

    Returns:
        pd.DataFrame: Per-target rows for every level.
    """
    frames = []
    for candidate in candidates:
        drops = to_relative_drop_distributions(candidate.matched, FIT_CATEGORIES)
        check_simulation_count(len(drops), f"{candidate.source} level {candidate.damage_level:g}", expected_simulations)
        rng = np.random.default_rng(stable_seed(seed, damage_type, candidate.damage_layer, f"{candidate.damage_level:.10g}"))
        frame = compare_level_to_human(drops, human_point, human_bootstrap, rng, alpha)
        setup = {"model": model, "damage_type": damage_type, "damage_layer": candidate.damage_layer,
                 "activation_layer": activation_layer, "damage_level": candidate.damage_level}
        for column, value in reversed(setup.items()):
            frame.insert(0, column, value)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def select_levels(all_levels: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """
    Label the levels whose overall drop is closest to the human drop, overall and from each side.

    Parameters:
        all_levels (pd.DataFrame): Output of analyze_candidates().
        tolerance (float): Differences within this distance of zero count as neither above nor below.

    Returns:
        pd.DataFrame: One row per selected setup and level with a semicolon-separated
        selection_roles column, e.g. "closest_numerically;closest_above_human".
    """
    overall = all_levels[all_levels["target"] == "overall"]
    selected = []
    for _, group in overall.groupby(SETUP_COLUMNS, sort=False):
        group = group.reset_index(drop=True)
        difference, absolute = group["difference_model_minus_human"], group["absolute_difference"]
        roles: dict[int, list[str]] = {}
        above, below = absolute[difference > tolerance], absolute[difference < -tolerance]
        for index, label in [(absolute.idxmin(), "closest_numerically"),
                             (above.idxmin() if not above.empty else None, "closest_above_human"),
                             (below.idxmin() if not below.empty else None, "closest_below_human")]:
            if index is not None:
                roles.setdefault(int(index), []).append(label)
        for index, labels in sorted(roles.items()):
            row = {column: group.loc[index, column] for column in SETUP_COLUMNS}
            row["damage_level"] = group.loc[index, "damage_level"]
            row["selection_roles"] = ";".join(labels)
            selected.append(row)
    return pd.DataFrame(selected)


# ------------------------- Tables -------------------------

def human_summary_table(human_point: dict[str, float], human_bootstrap: pd.DataFrame
                        ) -> dict[str, tuple[float, float, float]]:
    """Return {target: (mean, ci95_lower, ci95_upper)} for the human drops."""
    return {target: (human_point[target], *percentile_interval(human_bootstrap[target].to_numpy()))
            for target in HUMAN_SUMMARY_TARGETS}


def build_descriptive_table(model_rows: pd.DataFrame, human_summary: dict[str, tuple[float, float, float]],
                            human_n: str, decimals: int = 2, human_ci_decimals: int = 1) -> pd.DataFrame:
    """
    Return a table of the human drops and the model drops at the human-matched level.

    Parameters:
        model_rows (pd.DataFrame): One row per damage type, damaged area, and fit target at
            the chosen level (columns damage_type, damage_layer, damage_level, target,
            n_simulations, model_mean_relative_drop_percent, model_sd, model_ci95_lower,
            model_ci95_upper).
        human_summary (dict): Output of human_summary_table().
        human_n (str): Human sample description, e.g. "21 young, 21 old".
        decimals (int): Decimal places for the percentages.
        human_ci_decimals (int): Decimal places for the human CI bounds, which carry
            bootstrap Monte Carlo error of about 0.1 percentage points.

    Returns:
        pd.DataFrame: A human row, then one row per damage type and area, with
        <target>_mean, <target>_sd, <target>_ci95_lower, and <target>_ci95_upper.
    """
    required = {"damage_type", "damage_layer", "damage_level", "target", "n_simulations",
                "model_mean_relative_drop_percent", "model_sd", "model_ci95_lower", "model_ci95_upper"}
    missing = required - set(model_rows.columns)
    if missing:
        raise ValueError(f"model_rows is missing columns {sorted(missing)}.")

    human_row = {"source": HUMAN_SOURCE_LABEL, "damage_type": "", "damaged_area": "", "damage_level": np.nan, "n": human_n}
    for target in FIT_TARGETS:
        mean, lower, upper = human_summary[target]
        human_row.update({f"{target}_mean": mean, f"{target}_sd": np.nan,
                          f"{target}_ci95_lower": lower, f"{target}_ci95_upper": upper})

    layer_rank = {layer: i for i, layer in enumerate(DESCRIPTIVE_LAYER_ORDER)}
    rows = [human_row]
    setups = model_rows.groupby(["damage_type", "damage_layer"], sort=False)
    for damage_type, damage_layer in sorted(setups.groups, key=lambda key: (key[0], layer_rank.get(key[1], len(layer_rank)))):
        setup = setups.get_group((damage_type, damage_layer)).set_index("target")
        if set(FIT_TARGETS) - set(setup.index) or setup.index.duplicated().any() or setup["damage_level"].nunique() != 1:
            raise ValueError(f"{damage_type}/{damage_layer} needs one row per target {list(FIT_TARGETS)} at one level.")
        row = {"source": "Model", "damage_type": damage_type,
               "damaged_area": DAMAGE_AREA_LABELS.get(damage_layer, damage_layer),
               "damage_level": float(setup["damage_level"].iloc[0]), "n": int(setup["n_simulations"].iloc[0])}
        for target in FIT_TARGETS:
            row.update({f"{target}_mean": setup.loc[target, "model_mean_relative_drop_percent"],
                        f"{target}_sd": setup.loc[target, "model_sd"],
                        f"{target}_ci95_lower": setup.loc[target, "model_ci95_lower"],
                        f"{target}_ci95_upper": setup.loc[target, "model_ci95_upper"]})
        rows.append(row)

    table = pd.DataFrame(rows)
    value_columns = [c for c in table.columns if c.startswith(FIT_TARGETS)]
    table[value_columns] = table[value_columns].astype(float).round(decimals)
    human_ci_columns = [c for c in value_columns if "_ci95_" in c]
    table.loc[0, human_ci_columns] = table.loc[0, human_ci_columns].astype(float).round(human_ci_decimals)
    return table


def require_same_human_rows(human_rows: list[pd.DataFrame], label: str) -> None:
    """Fail unless every job produced the same human row (same human data, centring, bootstrap count, and seed)."""
    rows = [row.reset_index(drop=True) for row in human_rows]
    for index, row in enumerate(rows[1:], start=2):
        if not row.equals(rows[0]):
            raise ValueError(f"{label} {index} has a different human row from table 1; jobs must share the "
                             "human data, centring, bootstrap count, and seed.")


def combine_descriptive_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack per-job descriptive tables under their shared human row."""
    require_same_human_rows([table.iloc[[0]] for table in tables], "Descriptive table")
    return pd.concat([tables[0].iloc[[0]]] + [table.iloc[1:] for table in tables], ignore_index=True)


def sort_by_damage_area(frame: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    """
    Order rows by damage type (first appearance), TABLE_LAYER_ORDER, then extra columns.

    Parameters:
        frame (pd.DataFrame): Table with damage_type and damage_layer.
        extra (list[str] | None): Further sort columns, e.g. ["damage_level"].

    Returns:
        pd.DataFrame: Sorted copy with a fresh index.
    """
    type_rank = {name: i for i, name in enumerate(pd.unique(frame["damage_type"]))}
    layer_rank = {name: i for i, name in enumerate(TABLE_LAYER_ORDER)}
    keys = frame.assign(_type_rank=frame["damage_type"].map(type_rank),
                        _layer_rank=frame["damage_layer"].map(layer_rank).fillna(len(layer_rank)))
    ordered = keys.sort_values(["_type_rank", "_layer_rank", "damage_layer"] + (extra or []))
    return ordered.drop(columns=["_type_rank", "_layer_rank"]).reset_index(drop=True)


def build_table_s1_levels(all_levels: pd.DataFrame, roles: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """
    Return Table S1: model drops at every tested level with bracket flags.

    Parameters:
        all_levels (pd.DataFrame): Per-target rows from analyze_candidates().
        roles (pd.DataFrame): Output of select_levels().
        decimals (int): Decimal places for the percentages.

    Returns:
        pd.DataFrame: One row per damage type, damaged area, and level with n_simulations,
        <target>_mean, <target>_ci95_lower, <target>_ci95_upper, and the TABLE_S1_FLAGS columns.
    """
    level_columns = ["damage_type", "damage_layer", "damage_level"]
    role_lookup = {tuple(row[c] for c in level_columns): row["selection_roles"].split(";") for _, row in roles.iterrows()}
    rows = []
    for key, level in all_levels.groupby(level_columns, sort=False):
        level = level.set_index("target")
        if set(FIT_TARGETS) - set(level.index) or level.index.duplicated().any():
            raise ValueError(f"{key} needs one row per target {list(FIT_TARGETS)}; got {list(level.index)}.")
        row = dict(zip(level_columns, key))
        row["n_simulations"] = int(level["n_simulations"].iloc[0])
        for target in FIT_TARGETS:
            row[f"{target}_mean"] = level.loc[target, "model_mean_relative_drop_percent"]
            row[f"{target}_ci95_lower"] = level.loc[target, "model_ci95_lower"]
            row[f"{target}_ci95_upper"] = level.loc[target, "model_ci95_upper"]
        for role, column in TABLE_S1_FLAGS.items():
            row[column] = role in role_lookup.get(key, [])
        rows.append(row)

    table = sort_by_damage_area(pd.DataFrame(rows), extra=["damage_level"])
    value_columns = [c for c in table.columns if c.startswith(FIT_TARGETS)]
    table[value_columns] = table[value_columns].astype(float).round(decimals)
    table.insert(1, "damage_area", table.pop("damage_layer").map(lambda l: DAMAGE_AREA_LABELS.get(l, l)))
    return table


def build_table_s1_human(human_summary: dict[str, tuple[float, float, float]], human_n: str,
                         decimals: int = 2, ci_decimals: int = 1) -> pd.DataFrame:
    """
    Return the one-row Table S1 human reference with mean and 95% CI per target.

    Parameters:
        human_summary (dict): Output of human_summary_table().
        human_n (str): Human sample description, e.g. "21 young, 21 old".
        decimals (int): Decimal places for the means.
        ci_decimals (int): Decimal places for the CI bounds.

    Returns:
        pd.DataFrame: source, n, then <target>_mean, <target>_ci95_lower, <target>_ci95_upper.
    """
    row = {"source": HUMAN_SOURCE_LABEL, "n": human_n}
    for target in HUMAN_SUMMARY_TARGETS:
        mean, lower, upper = human_summary[target]
        row[f"{target}_mean"] = round(float(mean), decimals)
        row[f"{target}_ci95_lower"] = round(float(lower), ci_decimals)
        row[f"{target}_ci95_upper"] = round(float(upper), ci_decimals)
    return pd.DataFrame([row])


# ------------------------- Running -------------------------

def run_fit(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run one fit and write its descriptive table.

    Parameters:
        args (argparse.Namespace): Parsed command-line or YAML job options.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The descriptive table, the Table S1
        level rows, and the Table S1 human row (run_config() stacks them across jobs).
    """
    subject_values, human_point, human_bootstrap = load_human_reference(args)
    candidates = [c for layer in args.damage_layer for c in load_layer_candidates(args, layer)]
    all_levels = analyze_candidates(candidates, args.model, args.damage_type, args.activation_layer, human_point,
                                    human_bootstrap, args.expected_simulations, args.alpha, args.seed)
    roles = select_levels(all_levels, tolerance=args.tolerance)

    human_intervals = human_summary_table(human_point, human_bootstrap)
    # Largest cell per group, since one young PPA cell has a missing dot
    human_n = ", ".join(f"{max(len(cell) for c in FIT_CATEGORIES for cell in subject_values[c][group])} {group}"
                        for group in HUMAN_GROUPS)
    closest_levels = roles[roles["selection_roles"].str.split(";").apply(lambda r: "closest_numerically" in r)]
    closest = all_levels.merge(closest_levels[SETUP_COLUMNS + ["damage_level"]], on=SETUP_COLUMNS + ["damage_level"])
    descriptives = build_descriptive_table(closest, human_intervals, human_n)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join([safe_tag(args.model), safe_tag(args.damage_type), safe_tag("-".join(args.damage_layer)),
                    safe_tag(args.activation_layer), f"top{args.selectivity_fraction:.2f}", "haupt-fit"])
    descriptives.to_csv(output_dir / f"{tag}_descriptives.csv", index=False)

    print("Haupt relative drop (%) with bootstrap 95% CI")
    for target, (mean, lower, upper) in human_intervals.items():
        print(f"  {target:7s} {mean:6.2f} [{lower:6.2f}, {upper:6.2f}]")
    print(descriptives.to_string(index=False))
    return descriptives, build_table_s1_levels(all_levels, roles), build_table_s1_human(human_intervals, human_n)


def add_haupt_comparison_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the model, human-data, and bootstrap options shared with place_alignment_stats.py."""
    parser.add_argument("--model", help="Model label for result file names, e.g. cornet-rt.")
    parser.add_argument("--model-dir", help="Model folder of the differentiation table, e.g. cornet_rt5_c.")
    parser.add_argument("--differentiation-dir", default="data/differentiation",
                        help="Folder holding <model_dir>.csv.gz (default: data/differentiation).")
    parser.add_argument("--damage-type", help="Damage type, e.g. noise or connections.")
    parser.add_argument("--damage-layer", nargs="+", help="Damaged areas, e.g. All V1 V2 V4 IT.")
    parser.add_argument("--activation-layer", default="IT", help="Readout layer whose differentiation is compared.")
    parser.add_argument("--human-csv", default="data/reference/haupt_2025/haupt2025_fig6B_derived.csv",
                        help="Participant-level Haupt Figure 6B differentiation values.")
    parser.add_argument("--human-value-column", default="value_moment_matched",
                        help="Column of --human-csv holding the differentiation values.")
    parser.add_argument("--center-on-table-s5", action=argparse.BooleanOptionalAction, default=True,
                        help="Shift each human comparison-by-group cell to its Table S5 mean (default).")
    parser.add_argument("--table-s5-csv", default="data/reference/haupt_2025/table_s5_raw.csv",
                        help="Table S5 means used by --center-on-table-s5.")
    parser.add_argument("--selectivity-fraction", type=float, default=0.05,
                        help="Fraction of most selective units used per category (default: 0.05).")
    parser.add_argument("--baseline-level", type=float, default=0.0,
                        help="Intact level used as the 100%% reference and excluded from the fit.")
    parser.add_argument("--expected-simulations", type=int, default=200,
                        help="Required complete simulations per level; 0 disables the check.")
    parser.add_argument("--n-bootstrap", type=int, default=100000,
                        help="Bootstrap resamples; keep equal across scripts so the human draws are identical.")
    parser.add_argument("--seed", type=int, default=12345, help="Base random seed.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold.")
    parser.add_argument("--tolerance", type=float, default=1e-8,
                        help="Numerical tolerance for matching levels and bracketing.")
    parser.add_argument("--output-dir", default="stats/haupt_fits", help="Directory for result files.")


def validate_fit_args(args: argparse.Namespace) -> None:
    """Fail early on missing or out-of-range options."""
    missing = [name for name in ("model", "model_dir", "damage_type", "damage_layer") if not getattr(args, name, None)]
    if missing:
        raise ValueError(f"Missing required options: {', '.join('--' + m.replace('_', '-') for m in missing)}")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError(f"--alpha must be between 0 and 1; got {args.alpha!r}.")
    if args.expected_simulations < 0:
        raise ValueError(f"--expected-simulations must be nonnegative; got {args.expected_simulations!r}.")
    # The smallest attainable p-value is 2 / (n_bootstrap + 1)
    if args.n_bootstrap < 2.0 / args.alpha:
        raise ValueError(f"--n-bootstrap must be at least {math.ceil(2.0 / args.alpha)} for alpha={args.alpha}.")


def run_config(config_path: str | Path, parser: argparse.ArgumentParser) -> None:
    """
    Run all fitting jobs of a YAML haupt_fit section and write the combined tables.

    When every job shares one model and output directory, Table S1 and the combined
    descriptive table (read by the figure configs) are written.
    """
    outputs, job_targets = [], set()
    for job_index, job in enumerate(load_config_jobs(config_path, "haupt_fit"), start=1):
        damage_layers = job.pop("damage_layers", None)
        if not damage_layers:
            raise ValueError(f"haupt_fit job {job_index} in {config_path} needs damage_layers.")
        args = namespace_from_mapping(parser, job)
        args.damage_layer = [str(layer) for layer in damage_layers]
        validate_fit_args(args)
        print(f"\nRunning Haupt fit | model={args.model} | type={args.damage_type} | layers={','.join(args.damage_layer)}")
        outputs.append(run_fit(args))
        job_targets.add((args.model, args.output_dir))

    if len(job_targets) != 1:
        print(f"\nNo combined tables written: jobs use different models or output directories {sorted(job_targets)}.")
        return
    model, output_dir = job_targets.pop()
    prefix = Path(output_dir) / safe_tag(model)
    require_same_human_rows([output[2] for output in outputs], "Table S1 human row")
    outputs[0][2].to_csv(f"{prefix}_table-S1_human.csv", index=False)
    pd.concat([output[1] for output in outputs], ignore_index=True).to_csv(f"{prefix}_table-S1_levels.csv", index=False)
    combine_descriptive_tables([output[0] for output in outputs]).to_csv(f"{prefix}_haupt-fit_descriptives_all-jobs.csv", index=False)
    print(f"\nSaved Table S1 and the combined descriptive table under {Path(output_dir).resolve()}")


def main() -> None:
    """Run the fit from the command line or a YAML job matrix."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML file with a haupt_fit defaults/jobs section; replaces the single-run options.")
    add_haupt_comparison_arguments(parser)
    args = parser.parse_args()
    if args.config:
        run_config(args.config, parser)
        return
    validate_fit_args(args)
    run_fit(args)


if __name__ == "__main__":
    main()

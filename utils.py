"""
Shared functions for the dedifferentiation simulations, analyses, and figures.

Sections:
    - Selective RDMs and differentiation: storage of category-selective RDMs and the
      within-between differentiation metric (Methods 2.3).
    - Differentiation tables: loading the per-simulation tables written by
      compute_differentiation.py and deriving scaled and matched values from them.
    - Analysis configs and statistics: YAML defaults/jobs configs, Holm adjustment, and
      the human-matched damage levels shared by the statistics scripts and figures.
    - Category-selective units: top-unit selection from the localizer selectivity
      tables and selective RDMs built from saved activations (Methods 2.2).
    - Models and stimuli: model loading, module lookup, and image preprocessing
      (Methods 2.1).
    - Damage: connection severing, neuron deletion, weight noise, and eccentricity-graded
      V1 damage (Methods 2.4, 2.6.3).
    - Curves and summaries: per-level differentiation curves, scaled differentiation, AUC
      loss (Methods 2.3.1), and the human reference values used in the figures.
    - Plotting: the manuscript figure functions called by run_plots.py.
"""

import argparse
import hashlib
import math
import re
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from torchvision import transforms

CATEGORIES = ("animal", "face", "object", "place")
# Every RDM is a 64 x 64 image-by-image Pearson correlation matrix of the experimental stimuli.
N_STIMULI = 64
# Selectivity-table column holding each category's normalised Mann-Whitney U (Methods Eq. 1).
SELECTIVITY_COLUMNS = {"animal": "mw_animals", "face": "mw_faces", "object": "mw_objects", "place": "mw_places"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
# ImageNet normalisation used when the models were trained
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# Pretrained torchvision weights (the DEFAULT weights of torchvision 0.22)
TORCHVISION_WEIGHTS = {"alexnet": "IMAGENET1K_V1", "vgg16": "IMAGENET1K_V1"}
# Figure and table configs use this in place of a damage level to read the human-matched level
BEST_FIT_LEVEL = "best_fit"
# Tables label the whole-model damage scope "Whole model" instead of "All"
DAMAGE_AREA_LABELS = {"All": "Whole model"}


# ------------------------- Selective RDMs and differentiation -------------------------

def assign_categories(image_names) -> np.ndarray:
    """
    Assign a category label to each image from the non-numeric part of its file name.

    Example:
        "face1.jpg"    -> category "face"
        "object12.png" -> category "object"

    Parameters:
        image_names (list[str]): Image file names (or relative paths) in RDM order.

    Returns:
        np.ndarray: Category label for each image.
    """
    return np.array([re.sub(r"\d+", "", Path(name).stem).rstrip("_-. ") for name in image_names])


def calc_within_between(similarity_matrix: np.ndarray, categories_array: np.ndarray, category: str) -> dict:
    """
    Compute within-category and between-category similarity for one category.

    Within-category similarity is the mean correlation between different images of the
    category (diagonal excluded). Between-category similarity is the mean correlation
    between the category's images and all other images. Differentiation is their
    difference (Methods Eq. 2).

    Parameters:
        similarity_matrix (np.ndarray): Image-by-image Pearson correlation matrix [N,N].
        categories_array (np.ndarray): Category label of each image [N].
        category (str): Target category, e.g. "face".

    Returns:
        dict: {"within": float, "between": float, "differentiation": float}.
    """
    in_category = np.where(categories_array == category)[0]
    out_category = np.where(categories_array != category)[0]
    if len(in_category) < 2:
        raise ValueError(f"Category {category!r} needs at least two images; found {len(in_category)}.")

    within_block = similarity_matrix[np.ix_(in_category, in_category)]
    off_diagonal = ~np.eye(len(in_category), dtype=bool)
    within = float(np.mean(within_block[off_diagonal]))
    between = float(np.mean(similarity_matrix[np.ix_(in_category, out_category)].flatten()))
    return {"within": within, "between": between, "differentiation": within - between}


def rdm_to_upper(rdms: np.ndarray) -> np.ndarray:
    """
    Return the strict upper triangle of one or more symmetric RDMs.

    Parameters:
        rdms (np.ndarray): RDMs [R,N,N].

    Returns:
        np.ndarray: Upper-triangle values [R, N*(N-1)/2], row-major (np.triu_indices, k=1).
    """
    if rdms.ndim != 3 or rdms.shape[1] != rdms.shape[2]:
        raise ValueError(f"Expected RDMs of shape [R,N,N]; got {rdms.shape}.")
    rows, cols = np.triu_indices(rdms.shape[1], k=1)
    return rdms[:, rows, cols]


def upper_to_rdm(rdm_upper: np.ndarray, n_images: int = N_STIMULI) -> np.ndarray:
    """
    Rebuild full symmetric RDMs (diagonal = 1) from their upper triangles.

    The lower triangle mirrors the upper one. Correlation matrices from np.corrcoef are
    symmetric only to within floating-point rounding (~1e-16), so a rebuilt matrix can
    differ from the original at that level.

    Parameters:
        rdm_upper (np.ndarray): Upper-triangle values [R, N*(N-1)/2].
        n_images (int): Number of images N.

    Returns:
        np.ndarray: Full RDMs [R,N,N].
    """
    n_pairs = n_images * (n_images - 1) // 2
    if rdm_upper.ndim != 2 or rdm_upper.shape[1] != n_pairs:
        raise ValueError(f"Expected upper triangles of shape [R,{n_pairs}]; got {rdm_upper.shape}.")
    rows, cols = np.triu_indices(n_images, k=1)
    rdms = np.ones((rdm_upper.shape[0], n_images, n_images), dtype=rdm_upper.dtype)
    rdms[:, rows, cols] = rdm_upper
    rdms[:, cols, rows] = rdm_upper
    return rdms


def save_selective_rdms(path: str | Path, rdms: np.ndarray, image_names, replicates) -> None:
    """
    Save the selective RDMs of one damage level as a compressed .npz file.

    Parameters:
        path (str | Path): Output file, e.g. ".../face_selective/damaged_0.15.npz".
        rdms (np.ndarray): RDMs of every simulation at this level [R,N,N].
        image_names (list[str]): Image names in RDM order [N].
        replicates (list[str]): Simulation identifiers in replicate order [R].

    Returns:
        None
    """
    if len(replicates) != rdms.shape[0] or len(image_names) != rdms.shape[1]:
        raise ValueError(
            f"RDMs {rdms.shape} do not match {len(replicates)} replicates and {len(image_names)} images."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rdm_upper=rdm_to_upper(np.asarray(rdms, dtype=np.float64)),
        image_names=np.array(image_names, dtype=str),
        replicates=np.array(replicates, dtype=str),
    )


def load_selective_rdms(path: str | Path) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Load the selective RDMs of one damage level written by save_selective_rdms().

    Parameters:
        path (str | Path): .npz file.

    Returns:
        tuple: (rdms [R,N,N], image_names [N], replicates [R]).
    """
    with np.load(path, allow_pickle=False) as stored:
        image_names = stored["image_names"].tolist()
        rdms = upper_to_rdm(stored["rdm_upper"], n_images=len(image_names))
        replicates = stored["replicates"].tolist()
    return rdms, image_names, replicates


# ------------------------- Differentiation tables -------------------------

def load_differentiation_table(model_dir: str, differentiation_dir: str | Path = "data/differentiation") -> pd.DataFrame:
    """
    Load the per-simulation differentiation table of one model.

    Parameters:
        model_dir (str): Model folder name, e.g. "cornet_rt5_c".
        differentiation_dir (str | Path): Folder holding <model_dir>.csv.gz.

    Returns:
        pd.DataFrame: Columns damage_type, damage_area, readout_layer, top_fraction,
            selection_mode, category, damage_level, replicate, replicate_order, within,
            between, differentiation.
    """
    path = Path(differentiation_dir) / f"{model_dir}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run compute_differentiation.py table or download the data release.")
    return pd.read_csv(path, dtype={"replicate": str})


def select_condition(table: pd.DataFrame, damage_type: str, damage_area: str, readout_layer: str,
                     top_fraction: float = 0.05) -> pd.DataFrame:
    """
    Return the rows of one simulated condition.

    Parameters:
        table (pd.DataFrame): Table from load_differentiation_table().
        damage_type (str): e.g. "connections" or "eccentricity_weights_gradual_band0.50-1.00".
        damage_area (str): e.g. "V1", "All", or "block3".
        readout_layer (str): e.g. "IT" or "classifier_2".
        top_fraction (float): Fraction of selective units, e.g. 0.05.

    Returns:
        pd.DataFrame: Matching rows (a copy).
    """
    rows = table[
        (table["damage_type"] == damage_type) & (table["damage_area"] == damage_area)
        & (table["readout_layer"] == readout_layer) & np.isclose(table["top_fraction"], top_fraction)
    ]
    if rows.empty:
        raise ValueError(
            f"No rows for {damage_type}/{damage_area}/{readout_layer}/top {top_fraction}; available conditions: "
            f"{sorted(set(zip(table['damage_type'], table['damage_area'], table['readout_layer'], table['top_fraction'])))}"
        )
    return rows.copy()


def matched_category_values(condition: pd.DataFrame, damage_level: float, categories, baseline_level: float = 0.0,
                            value_mode: str = "scaled_percent") -> pd.DataFrame:
    """
    Return each simulation's category values at one level, matched to its intact baseline.

    A simulation's damaged differentiation is divided by the same simulation's value at
    the baseline level and only simulations complete in every category are kept, sorted
    by simulation identifier. Simulations whose baseline is zero are dropped.

    Parameters:
        condition (pd.DataFrame): Rows of one condition from select_condition().
        damage_level (float): Damage level to read.
        categories (list[str]): e.g. ["face", "object", "place"].
        baseline_level (float): Level used as the 100% reference, normally 0.
        value_mode (str): "scaled_percent" (100 * damaged / baseline) or "differentiation".

    Returns:
        pd.DataFrame: Column "replicate" plus one column per category.
    """
    allowed = ("scaled_percent", "differentiation")
    if value_mode not in allowed:
        raise ValueError(f"value_mode must be one of {allowed}; got {value_mode!r}.")

    def values_at(level: float) -> pd.DataFrame:
        rows = condition[np.isclose(condition["damage_level"], level, rtol=0, atol=1e-9)]
        if rows.empty:
            raise ValueError(f"Level {level} not found; available: {sorted(condition['damage_level'].unique())}.")
        return rows.pivot(index="replicate", columns="category", values="differentiation")

    damaged = values_at(damage_level)
    missing = [c for c in categories if c not in damaged.columns]
    if missing:
        raise ValueError(f"Categories {missing} missing at level {damage_level}.")
    damaged = damaged.loc[:, list(categories)]
    if value_mode == "scaled_percent":
        baseline = values_at(baseline_level).loc[:, list(categories)]
        common = damaged.index.intersection(baseline.index)
        baseline = baseline.loc[common]
        damaged = 100.0 * damaged.loc[common] / baseline.where(~np.isclose(baseline, 0.0))
    matched = damaged.replace([np.inf, -np.inf], np.nan).dropna(how="any").sort_index()
    if len(matched) < 2:
        raise ValueError(f"Fewer than two complete simulations at level {damage_level}.")
    return matched.reset_index().rename_axis(columns=None)


# ------------------------- Analysis configs and statistics -------------------------

def load_config_jobs(config_path: str | Path, section_name: str) -> list[dict]:
    """
    Return the jobs of one defaults/jobs section of a YAML analysis config.

    Each job is the section's defaults updated with the job's own keys.

    Parameters:
        config_path (str | Path): YAML file, e.g. configs/stats/cornet_rt5_haupt.yaml.
        section_name (str): Section, e.g. "haupt_fit" or "place_alignment".

    Returns:
        list[dict]: One merged mapping per job.
    """
    with open(config_path, encoding="utf-8-sig") as f:
        document = yaml.safe_load(f)
    section = document.get(section_name) if isinstance(document, Mapping) else None
    if not isinstance(section, Mapping):
        raise ValueError(f"{config_path} has no mapping section {section_name!r}.")
    defaults, jobs = section.get("defaults", {}), section.get("jobs", [])
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"{section_name}.jobs in {config_path} must be a nonempty list.")
    return [{**defaults, **job} for job in jobs]


def namespace_from_mapping(parser: argparse.ArgumentParser, values: Mapping,
                           extra_keys: tuple[str, ...] = ()) -> argparse.Namespace:
    """
    Build an argparse namespace from a config mapping, rejecting misspelled keys.

    Parameters:
        parser (argparse.ArgumentParser): Parser whose options define the allowed keys and defaults.
        values (Mapping): Config keys and values, named like the option destinations.
        extra_keys (tuple[str, ...]): Further allowed keys handled by the caller.

    Returns:
        argparse.Namespace: Parser defaults updated with the config values.
    """
    actions = {a.dest: a for a in parser._actions if a.dest not in {"help", argparse.SUPPRESS}}
    unknown = set(values) - set(actions) - set(extra_keys)
    if unknown:
        raise ValueError(f"Unknown config keys {sorted(unknown)}; allowed: {sorted(actions)}.")
    namespace = {name: action.default for name, action in actions.items()}
    namespace.update({key: value for key, value in values.items() if key in actions})
    return argparse.Namespace(**namespace)


def holm_adjust(p_values) -> np.ndarray:
    """Return Holm-adjusted p-values; non-finite inputs stay NaN."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    running_max = 0.0
    for rank, index in enumerate(finite[np.argsort(p_values[finite])]):
        running_max = max(running_max, (len(finite) - rank) * p_values[index])
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def load_best_fit_levels(csv_path: str | Path) -> dict[tuple[str, str], float]:
    """
    Return the human-matched damage level of each damage type and area.

    Reads the descriptive table written by fit_haupt_damage_levels.py, whose model rows
    hold the level with the overall drop closest to the human drop.

    Example:
        A "Whole model" connections row at 0.15 -> {("connections", "All"): 0.15}

    Parameters:
        csv_path (str | Path): *_haupt-fit_descriptives*.csv.

    Returns:
        dict[tuple[str, str], float]: (damage_type, damage_area) -> level.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run fit_haupt_damage_levels.py first.")
    table = pd.read_csv(path, encoding="utf-8-sig")
    model_rows = table[table["source"] == "Model"]
    if model_rows.empty:
        raise ValueError(f"{path} has no model rows.")
    area_names = {label: area for area, label in DAMAGE_AREA_LABELS.items()}
    return {(row.damage_type, area_names.get(row.damaged_area, row.damaged_area)): float(row.damage_level)
            for row in model_rows.itertuples()}


# ------------------------- Category-selective units -------------------------

def normalize_module_name(name: str) -> str:
    """
    Remove the "_modules" tokens PyTorch inserts into nested module paths.

    Example:
        "module._modules.IT._modules.output" -> "module.IT.output"
    """
    return ".".join(token for token in str(name).split(".") if token != "_modules")


def canonical_layer_name(layer_name: str) -> str:
    """
    Return one canonical name for the equivalent spellings of a readout layer.

    Selectivity tables store module paths, while data directories use file-system safe
    names, so both are mapped to the same key.

    Example:
        "module.IT"     -> "IT"
        "classifier_2"  -> "classifier.2"
        "IT.output"     -> "IT"
    """
    canonical = normalize_module_name(str(layer_name).strip())
    canonical = canonical.removesuffix(".output").removeprefix("module.")
    return re.sub(r"_(\d+)$", r".\1", canonical)


def load_selectivity_table(path: str | Path) -> pd.DataFrame:
    """
    Load a unit-selectivity table written by unit_categ_selectivity.py.

    Parameters:
        path (str | Path): .csv or .csv.gz table with columns layer, unit, and
            mw_animals, mw_faces, mw_objects, mw_places.

    Returns:
        pd.DataFrame: The table.
    """
    table = pd.read_csv(path)
    required = {"layer", "unit", *SELECTIVITY_COLUMNS.values()}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{path} is missing columns {missing}.")
    return table


def select_top_units(selectivity_table: pd.DataFrame, readout_layer: str, category: str,
                     top_fraction: float) -> np.ndarray:
    """
    Return the indices of the most category-selective units in the readout layer.

    Units are ranked by their normalised Mann-Whitney U for the category and the top
    round(n_units * top_fraction) units are kept (at least one). Ties keep table order.

    Parameters:
        selectivity_table (pd.DataFrame): Table from load_selectivity_table().
        readout_layer (str): Layer whose units are selected, e.g. "IT" or "classifier_2".
        category (str): One of CATEGORIES.
        top_fraction (float): Fraction of units to keep, e.g. 0.05 for the top 5%.

    Returns:
        np.ndarray: Unit indices into the flattened readout-layer activations.
    """
    if category not in SELECTIVITY_COLUMNS:
        raise ValueError(f"category must be one of {list(SELECTIVITY_COLUMNS)}; got {category!r}.")
    if not 0 < top_fraction <= 1:
        raise ValueError(f"top_fraction must be in (0, 1]; got {top_fraction!r}.")

    target = canonical_layer_name(readout_layer)
    rows = selectivity_table[selectivity_table["layer"].map(canonical_layer_name) == target]
    if rows.empty:
        available = sorted(selectivity_table["layer"].astype(str).unique())
        raise ValueError(f"No selectivity rows for layer {readout_layer!r}; available layers: {available}.")

    n_selected = max(1, int(round(len(rows) * float(top_fraction))))
    return rows.nlargest(n_selected, SELECTIVITY_COLUMNS[category])["unit"].astype(int).to_numpy()


def finite_correlation_matrix(activations: np.ndarray) -> np.ndarray:
    """
    Return the image-by-image Pearson correlation matrix with undefined values set to 0.

    A unit subset can respond identically to every image after heavy damage, which makes
    Pearson's r undefined. Those pairs are treated as zero correlation, so a collapsed
    representation has zero differentiation instead of dropping out of the analysis.

    Parameters:
        activations (np.ndarray): Image-by-unit activation matrix [N,U].

    Returns:
        np.ndarray: Correlation matrix [N,N] with a diagonal of 1.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation_matrix = np.asarray(np.corrcoef(activations), dtype=float)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation_matrix, 1.0)
    return correlation_matrix


def save_activations(path: str | Path, activations: np.ndarray, image_names) -> None:
    """
    Save one simulation's readout-layer activations as float16, as in the original runs.

    Parameters:
        path (str | Path): Output file, e.g. ".../activations/IT/damaged_0.15/7.npz".
        activations (np.ndarray): Image-by-unit activation matrix [N,U].
        image_names (list[str]): Image names in row order [N].

    Returns:
        None
    """
    if activations.ndim != 2 or activations.shape[0] != len(image_names):
        raise ValueError(f"Activations {activations.shape} do not match {len(image_names)} images.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, activations=activations.astype(np.float16), image_names=np.array(image_names, dtype=str)
    )


def load_activations(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """
    Load activations written by save_activations().

    Parameters:
        path (str | Path): .npz file.

    Returns:
        tuple: (activations [N,U] float16, image_names [N]).
    """
    with np.load(path, allow_pickle=False) as stored:
        return stored["activations"], stored["image_names"].tolist()


def build_selective_rdms(activation_dir: str | Path, rdm_dir: str | Path, selectivity_table: pd.DataFrame,
                         readout_layer: str, top_fraction: float, categories=CATEGORIES) -> None:
    """
    Build category-selective RDMs for every damage level of one simulated condition.

    For each damage level and category, the activations of the category's top units are
    correlated across images in every simulation, and the RDMs of all simulations at
    that level are saved together with save_selective_rdms().

    Parameters:
        activation_dir (str | Path): ".../<area>/activations/<readout>" holding damaged_<level>/<perm>.npz.
        rdm_dir (str | Path): ".../<area>/RDM_<frac>_percentage/<readout>" to write to.
        selectivity_table (pd.DataFrame): Table from load_selectivity_table().
        readout_layer (str): Readout layer name, e.g. "IT".
        top_fraction (float): Fraction of most selective units, e.g. 0.05.
        categories (tuple[str]): Categories to build, default all four.

    Returns:
        None
    """
    activation_dir, rdm_dir = Path(activation_dir), Path(rdm_dir)
    level_dirs = sorted(d for d in activation_dir.glob("damaged_*") if d.is_dir())
    if not level_dirs:
        raise FileNotFoundError(f"No damaged_* folders under {activation_dir}.")
    unit_indices = {c: select_top_units(selectivity_table, readout_layer, c, top_fraction) for c in categories}

    for level_dir in level_dirs:
        # Sorted file-name order pairs the categories of each simulation, as in the original analysis
        files = sorted(level_dir.glob("*.npz"), key=lambda p: p.stem)
        if not files:
            raise FileNotFoundError(f"No activation files in {level_dir}.")
        rdms = {c: [] for c in categories}
        image_names = None
        for file in files:
            activations, names = load_activations(file)
            if image_names is None:
                image_names = names
            elif names != image_names:
                raise ValueError(f"Image order in {file} differs from {files[0]}.")
            for category, indices in unit_indices.items():
                if indices.max() >= activations.shape[1]:
                    raise ValueError(f"Unit index {indices.max()} exceeds the {activations.shape[1]} units in {file}.")
                rdms[category].append(finite_correlation_matrix(activations[:, indices].astype(np.float32)))
        for category in categories:
            save_selective_rdms(
                rdm_dir / f"{category}_selective" / f"{level_dir.name}.npz",
                np.stack(rdms[category]),
                image_names,
                [f.stem for f in files],
            )


# ------------------------- Models and stimuli -------------------------

def load_model(name: str, time_steps: int | None = None, device: str | torch.device | None = None) -> nn.Module:
    """
    Load a pretrained model in evaluation mode.

    CORnet models come from the official CORnet package and are wrapped in
    nn.DataParallel, so their module paths start with "module.". Only CORnet-RT takes a
    number of time steps; CORnet-S has fixed within-block recurrence.

    Parameters:
        name (str): "cornet_rt", "cornet_s", "alexnet", or "vgg16".
        time_steps (int | None): CORnet-RT time steps, e.g. 5.
        device (str | torch.device | None): Target device; defaults to CUDA when available.

    Returns:
        nn.Module: The model on the target device.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if name == "cornet_rt":
        from cornet import cornet_rt
        model = cornet_rt(pretrained=True, map_location=device, times=time_steps or 5)
    elif name == "cornet_s":
        if time_steps is not None:
            raise ValueError("cornet_s has fixed recurrence and takes no time_steps; remove it from the config.")
        from cornet import cornet_s
        model = cornet_s(pretrained=True, map_location=device)
    elif name in TORCHVISION_WEIGHTS:
        from torchvision.models import get_model
        model = get_model(name, weights=TORCHVISION_WEIGHTS[name])
    else:
        raise ValueError(f"name must be one of {['cornet_rt', 'cornet_s', *TORCHVISION_WEIGHTS]}; got {name!r}.")
    if isinstance(model, nn.DataParallel) and device.type == "cpu":
        # Without device ids the CORnet wrapper runs its module directly, even when a GPU exists
        model.device_ids = []
    return model.to(device).eval()


def get_module(model: nn.Module, path: str) -> nn.Module:
    """
    Return the submodule at a dotted path.

    Example:
        "module._modules.V1._modules.conv_input" -> model.module.V1.conv_input
        "classifier.2"                           -> model.classifier[2]

    Parameters:
        model (nn.Module): Model to search.
        path (str): Dotted module path; "_modules" tokens are ignored.

    Returns:
        nn.Module: The submodule.
    """
    module = model
    for step in normalize_module_name(path).split("."):
        if step in module._modules:
            module = module._modules[step]
        else:
            raise KeyError(f"Cannot resolve {step!r} in module path {path!r}.")
    return module


def conv_layers_under(model: nn.Module, area_paths: list[str]) -> list[tuple[str, nn.Conv2d]]:
    """
    Return every Conv2d inside the given model areas, in registration order.

    Parameters:
        model (nn.Module): Model to search.
        area_paths (list[str]): Area module paths, e.g. ["module.V1"] or ["features.0"].

    Returns:
        list[tuple[str, nn.Conv2d]]: (module path, layer) pairs.
    """
    layers = []
    for area_path in area_paths:
        area = get_module(model, area_path)
        found = [(f"{normalize_module_name(area_path)}.{name}".rstrip("."), m)
                 for name, m in area.named_modules() if isinstance(m, nn.Conv2d)]
        if not found:
            raise ValueError(f"No Conv2d layers under {area_path!r}.")
        layers += found
    return layers


def list_stimuli(image_dir: str | Path) -> list[Path]:
    """
    Return the stimulus images sorted by category name, then by image number.

    Example:
        animal1.jpg, animal2.jpg, ..., animal16.jpg, face1.jpg, ..., place16.jpg

    Parameters:
        image_dir (str | Path): Folder of images named <category><number>.<ext>.

    Returns:
        list[Path]: Image paths in RDM order.
    """
    paths = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise FileNotFoundError(f"No images in {image_dir}; run python download_stimuli.py first.")

    def sort_key(path: Path):
        match = re.fullmatch(r"(\D+)(\d+)", path.stem)
        if match is None:
            raise ValueError(f"Stimulus {path.name!r} is not named <category><number>.")
        return match.group(1).lower(), int(match.group(2))

    return sorted(paths, key=sort_key)


def category_images(image_dir: str | Path, category: str) -> list[Path]:
    """
    Return the stimuli of one category in file-name order, the image order of the gradient analysis.

    Example:
        "place" -> place1.jpg, place10.jpg, ..., place16.jpg, place2.jpg, ..., place9.jpg
    """
    paths = sorted((p for p in list_stimuli(image_dir) if re.fullmatch(rf"{category}\d+", p.stem)),
                   key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"No {category} images in {image_dir}.")
    return paths


def preprocess_image(image_path: str | Path, device: str | torch.device | None = None,
                     center_crop: bool = False) -> torch.Tensor:
    """
    Load an image as a normalised 224 x 224 tensor with a batch dimension.

    Parameters:
        image_path (str | Path): Path to the image file.
        device (str | torch.device | None): Optional destination device.
        center_crop (bool): Resize the short side to 256 and crop the central 224 x 224
            (gradient analysis) instead of resizing the whole image to 224 x 224 (simulations).

    Returns:
        torch.Tensor: Tensor [1,3,224,224].
    """
    resize = [transforms.Resize(256), transforms.CenterCrop(224)] if center_crop else [transforms.Resize((224, 224))]
    preprocess = transforms.Compose([
        *resize,
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    input_tensor = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)  # Add batch dimension
    return input_tensor if device is None else input_tensor.to(device)


def record_readout(model: nn.Module, readout_path: str) -> dict:
    """
    Register a hook that keeps the latest output of the readout layer.

    Recurrent blocks call their output layer once per time step, so the stored value is
    the readout at the final time step.

    Parameters:
        model (nn.Module): Model to hook.
        readout_path (str): Module path, e.g. "module.IT.output" or "classifier.2".

    Returns:
        dict: Filled on every forward pass as {"output": np.ndarray}.
    """
    store = {}

    def hook(_module, _inputs, output):
        store["output"] = output.detach().cpu().numpy()

    get_module(model, readout_path).register_forward_hook(hook)
    return store


def simulation_seed(*labels) -> int:
    """
    Return a stable 63-bit seed derived from text labels.

    Example:
        simulation_seed("cornet_rt5_c/connections/V1", 0.15, 7) is the same on every machine.
    """
    digest = hashlib.sha256("|".join(map(str, labels)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


# ------------------------- Damage -------------------------

def sever_connections(layers: list[tuple[str, nn.Conv2d]], fraction: float, generator: torch.Generator) -> None:
    """
    Set a random fraction of the weights of every layer to zero (Methods 2.4.2).

    Each layer loses int(fraction * n_weights) weights, drawn independently per layer.
    Biases are kept.

    Parameters:
        layers (list): (path, Conv2d) pairs from conv_layers_under().
        fraction (float): Fraction of weights to sever, 0 to 1.
        generator (torch.Generator): Random generator on the model's device.

    Returns:
        None
    """
    with torch.no_grad():
        for _, layer in layers:
            n_weights = layer.weight.numel()
            n_severed = int(fraction * n_weights)
            severed = torch.randperm(n_weights, device=layer.weight.device, generator=generator)[:n_severed]
            layer.weight.view(-1)[severed] = 0.0


def delete_units(layers: list[tuple[str, nn.Conv2d]], fraction: float, generator: torch.Generator,
                 include_bias: bool = True) -> None:
    """
    Delete a random fraction of whole filters in every layer (neuron deletion, Methods 2.4.2).

    All weights of each deleted filter, and its bias when include_bias is True, are set
    to zero, so the filter's direct output is zero.

    Parameters:
        layers (list): (path, Conv2d) pairs from conv_layers_under().
        fraction (float): Fraction of filters to delete, 0 to 1.
        generator (torch.Generator): Random generator on the model's device.
        include_bias (bool): Also zero the deleted filters' biases.

    Returns:
        None
    """
    with torch.no_grad():
        for _, layer in layers:
            n_filters = layer.weight.shape[0]
            deleted = torch.randperm(n_filters, device=layer.weight.device, generator=generator)[:int(fraction * n_filters)]
            layer.weight[deleted] = 0.0
            if include_bias and layer.bias is not None:
                layer.bias[deleted] = 0.0


def weight_standard_deviations(layers: list[tuple[str, nn.Conv2d]]) -> dict[str, float]:
    """
    Return the standard deviation of each intact layer's weights (sigma_l in Methods Eq. 7).

    Parameters:
        layers (list): (path, Conv2d) pairs from conv_layers_under() of an intact model.

    Returns:
        dict[str, float]: Layer path -> weight SD.
    """
    return {path: float(layer.weight.detach().std().item()) for path, layer in layers}


def add_weight_noise(layers: list[tuple[str, nn.Conv2d]], noise_level: float, weight_sds: dict[str, float],
                     generator: torch.Generator) -> None:
    """
    Add zero-mean Gaussian noise scaled to each layer's intact weight SD (Methods Eq. 7).

    Parameters:
        layers (list): (path, Conv2d) pairs from conv_layers_under().
        noise_level (float): Scaling factor k; noise SD is k * sigma_l.
        weight_sds (dict[str, float]): Intact weight SDs from weight_standard_deviations().
        generator (torch.Generator): Random generator on the model's device.

    Returns:
        None
    """
    with torch.no_grad():
        for path, layer in layers:
            noise = torch.randn(layer.weight.shape, device=layer.weight.device, generator=generator)
            layer.weight.add_(noise * (weight_sds[path] * noise_level))


def eccentricity_deletion_probability(height: int, width: int, r_start: float, r_end: float, p_max: float,
                                      reverse: bool = False, device=None) -> torch.Tensor:
    """
    Return the deletion probability at each feature-map position (Methods Eq. 11).

    Eccentricity is the distance from the map centre, normalised to 1 at the corners.
    Probability rises linearly from 0 at r_start to p_max at r_end (peripheral damage).
    With reverse=True it falls from p_max at r_start to 0 at r_end (foveal damage).
    Positions outside [r_start, r_end] take the value at the nearest end.

    Example:
        r_start=0.5, r_end=1.0, p_max=0.4: centre and r <= 0.5 -> 0, corners -> 0.4

    Parameters:
        height (int): Feature-map height.
        width (int): Feature-map width.
        r_start (float): Inner eccentricity of the graded band, 0 to 1.
        r_end (float): Outer eccentricity of the graded band, 0 to 1.
        p_max (float): Maximum deletion probability.
        reverse (bool): Put the maximum at r_start instead of r_end.
        device: Torch device.

    Returns:
        torch.Tensor: Probabilities [H,W].
    """
    if not 0.0 <= r_start < r_end <= 1.0:
        raise ValueError(f"Need 0 <= r_start < r_end <= 1; got r_start={r_start}, r_end={r_end}.")
    if not 0.0 <= p_max <= 1.0:
        raise ValueError(f"p_max must be in [0, 1]; got {p_max}.")
    rows = torch.arange(height, device=device).view(height, 1).float()
    cols = torch.arange(width, device=device).view(1, width).float()
    radius = torch.sqrt((rows - (height - 1) / 2.0) ** 2 + (cols - (width - 1) / 2.0) ** 2)
    radius = radius / radius.max()
    progress = torch.clamp((radius - r_start) / max(1e-6, r_end - r_start), 0.0, 1.0)
    return p_max * (1.0 - progress) if reverse else p_max * progress


def delete_activations_by_eccentricity(model: nn.Module, layer_path: str, r_start: float, r_end: float,
                                       p_max: float, generator: torch.Generator, reverse: bool = False,
                                       per_channel: bool = True) -> None:
    """
    Zero V1 unit activations with an eccentricity-dependent probability (Fig S5).

    One Bernoulli mask is drawn on the first forward pass and reused for every image and
    time step, so each simulation deletes a fixed set of units.

    Parameters:
        model (nn.Module): Model to damage.
        layer_path (str): Layer whose output is masked, e.g. "module.V1.conv_input".
        r_start (float), r_end (float), p_max (float), reverse (bool): See eccentricity_deletion_probability().
        generator (torch.Generator): Random generator on the model's device.
        per_channel (bool): Draw a separate mask for every channel instead of one shared spatial mask.

    Returns:
        None
    """
    masks = {}

    def hook(_module, _inputs, output):
        if output.shape not in masks:
            _, n_channels, height, width = output.shape
            p_delete = eccentricity_deletion_probability(height, width, r_start, r_end, p_max, reverse, output.device)
            keep_probability = (1.0 - p_delete).expand(n_channels if per_channel else 1, height, width).contiguous()
            masks[output.shape] = torch.bernoulli(keep_probability, generator=generator).unsqueeze(0)
        return output * masks[output.shape]

    get_module(model, layer_path).register_forward_hook(hook)


def sever_connections_by_eccentricity(model: nn.Module, layer_path: str, r_start: float, r_end: float,
                                      p_max: float, generator: torch.Generator, reverse: bool = False,
                                      per_channel: bool = True, include_bias: bool = False,
                                      location_chunk_size: int = 256) -> None:
    """
    Sever a convolution's connections with an eccentricity-dependent probability (Fig 2e, S6, S8).

    Standard convolution shares one filter across all output positions. Here the
    convolution is recomputed position by position from unfolded input patches, so every
    weight at every output position is kept or set to zero independently, with the
    deletion probability of that position (Methods 2.6.3). The severed weights are drawn
    once and reused for every image and time step.

    Parameters:
        model (nn.Module): Model to damage.
        layer_path (str): Conv2d to damage, e.g. "module.V1.conv_input".
        r_start (float), r_end (float), p_max (float), reverse (bool): See eccentricity_deletion_probability().
        generator (torch.Generator): Random generator on the model's device.
        per_channel (bool): Draw separate deletions for every output channel.
        include_bias (bool): Also delete each output position's bias with the same probability.
        location_chunk_size (int): Output positions computed at once (memory only).

    Returns:
        None
    """
    conv = get_module(model, layer_path)
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"{layer_path} must be an nn.Conv2d; got {type(conv).__name__}.")
    if conv.padding_mode != "zeros":
        raise ValueError(f"{layer_path} uses padding_mode={conv.padding_mode!r}; only 'zeros' is supported.")
    lesions = {}

    def build_lesion(height: int, width: int, device) -> list:
        """Return per-chunk (start, stop, masked weights, bias) for one simulation."""
        groups = conv.groups
        kernel_size = (conv.in_channels // groups) * conv.kernel_size[0] * conv.kernel_size[1]
        weights = conv.weight.reshape(groups, conv.out_channels // groups, kernel_size)
        p_delete = eccentricity_deletion_probability(height, width, r_start, r_end, p_max, reverse, device).reshape(-1)
        chunks = []
        for start in range(0, height * width, location_chunk_size):
            stop = min(start + location_chunk_size, height * width)
            p_chunk = p_delete[start:stop]
            mask_channels = conv.out_channels // groups if per_channel else 1
            keep = torch.rand((groups, mask_channels, kernel_size, stop - start), device=device, generator=generator)
            masked_weights = (weights.unsqueeze(-1) * (keep >= p_chunk.view(1, 1, 1, -1)).to(weights.dtype)).detach()
            bias = None
            if conv.bias is not None:
                bias = conv.bias.view(1, conv.out_channels, 1).detach()
                if include_bias:
                    keep_bias = torch.rand((1, conv.out_channels if per_channel else 1, stop - start),
                                           device=device, generator=generator)
                    bias = bias * (keep_bias >= p_chunk.view(1, 1, -1)).to(bias.dtype)
            chunks.append((start, stop, masked_weights, bias))
        return chunks

    def hook(_module, inputs, output):
        if p_max == 0.0:
            return output
        n_batch, _, height, width = output.shape
        key = (height, width, output.device)
        if key not in lesions:
            lesions[key] = build_lesion(height, width, output.device)
        patches = F.unfold(inputs[0], kernel_size=conv.kernel_size, dilation=conv.dilation,
                           padding=conv.padding, stride=conv.stride)
        patches = patches.reshape(n_batch, conv.groups, -1, height * width)
        outputs = []
        for start, stop, masked_weights, bias in lesions[key]:
            chunk = torch.einsum("ngkl,gokl->ngol", patches[..., start:stop], masked_weights)
            chunk = chunk.reshape(n_batch, conv.out_channels, stop - start)
            outputs.append(chunk if bias is None else chunk + bias)
        return torch.cat(outputs, dim=-1).reshape(n_batch, conv.out_channels, height, width)

    # Prepend so the damaged output is what later hooks and layers see
    conv.register_forward_hook(hook, prepend=True)


# ------------------------- Curves and summaries -------------------------

def level_values(condition: pd.DataFrame, category: str, scaled_percent: bool = False) -> dict[float, np.ndarray]:
    """
    Return each damage level's per-simulation differentiation for one category or the total.

    Simulations are ordered by replicate_order (sorted simulation file names). "total" is
    the mean of the four categories within each simulation. With scaled_percent, every
    simulation is expressed as a percentage of the simulation in the same position at
    the lowest (intact) level, 100 * value / (intact + 1e-12), as in the original figures.

    Parameters:
        condition (pd.DataFrame): Rows of one condition from select_condition().
        category (str): One of CATEGORIES or "total".
        scaled_percent (bool): Scale to the intact level.

    Returns:
        dict[float, np.ndarray]: Damage level -> values [n_simulations], in increasing level order.
    """
    if category != "total" and category not in CATEGORIES:
        raise ValueError(f"category must be one of {[*CATEGORIES, 'total']}; got {category!r}.")
    wide = condition.pivot_table(index=["damage_level", "replicate_order"], columns="category",
                                 values="differentiation", aggfunc="first")
    if category == "total":
        missing = [c for c in CATEGORIES if c not in wide.columns]
        if missing:
            raise ValueError(f"The total needs all four categories; missing {missing}.")
        # Same summation order as the original per-simulation total
        series = pd.Series(np.nanmean(wide.loc[:, list(CATEGORIES)].to_numpy(), axis=1), index=wide.index)
    else:
        series = wide[category]
    values = {float(level): group.sort_index(level="replicate_order").dropna().to_numpy(dtype=float)
              for level, group in series.groupby(level="damage_level")}
    if scaled_percent:
        baseline = values[min(values)]
        values = {level: 100.0 * v[:len(baseline)] / (baseline[:len(v)] + 1e-12) for level, v in values.items()}
    return dict(sorted(values.items()))


def auc_loss(levels, mean_values) -> float:
    """
    Return the normalised area under the relative-loss curve (Methods Eqs. 5-6).

    The loss at each level is 1 - D(x) / D(0), integrated with the trapezoidal rule and
    divided by the width of the level range. Positive values mean an average loss.

    Example:
        levels [0, 0.5, 1], means [0.2, 0.1, 0.0] -> losses [0, 0.5, 1] -> AUC loss 0.5

    Parameters:
        levels (array-like): Damage levels, starting at the intact level 0.
        mean_values (array-like): Differentiation at each level.

    Returns:
        float: AUC loss.
    """
    levels, mean_values = np.asarray(levels, dtype=float), np.asarray(mean_values, dtype=float)
    order = np.argsort(levels)
    levels, mean_values = levels[order], mean_values[order]
    if len(levels) < 2 or not np.isclose(levels[0], 0.0):
        raise ValueError(f"AUC loss needs at least two levels starting at 0; got {levels.tolist()}.")
    if np.isclose(mean_values[0], 0.0):
        raise ValueError("Intact differentiation is zero, so the relative loss is undefined.")
    loss = 1.0 - mean_values / mean_values[0]
    return float(np.trapezoid(loss, levels) / (levels[-1] - levels[0]))


def curve_auc_summary(values_by_level: dict[float, np.ndarray]) -> tuple[float, float]:
    """
    Return the AUC loss of the mean curve and its SD across simulations.

    The SD is taken over the AUC losses of the individual simulations, pairing
    simulations across levels by their position, as in the original side panels.

    Parameters:
        values_by_level (dict[float, np.ndarray]): Output of level_values().

    Returns:
        tuple[float, float]: (AUC loss of the mean curve, SD of per-simulation AUC losses).
    """
    levels = list(values_by_level)
    mean_curve = [float(np.mean(v)) for v in values_by_level.values()]
    n_simulations = min(len(v) for v in values_by_level.values())
    per_simulation = []
    for i in range(n_simulations):
        curve = [values_by_level[level][i] for level in levels]
        if not np.isclose(curve[0], 0.0):
            per_simulation.append(auc_loss(levels, curve))
    sd = float(np.std(per_simulation, ddof=1)) if len(per_simulation) > 1 else 0.0
    return auc_loss(levels, mean_curve), sd


def bootstrap_mean_ci(values, n_bootstrap: int = 10000, seed: int = 0) -> tuple[float, float]:
    """
    Return the percentile 95% bootstrap CI of the mean of independent simulations.

    Parameters:
        values (array-like): Per-simulation values.
        n_bootstrap (int): Number of resamples.
        seed (int): Seed for numpy.random.default_rng.

    Returns:
        tuple[float, float]: (lower, upper).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError(f"A bootstrap CI needs at least two values; got {len(values)}.")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_bootstrap, len(values)))].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def load_human_reference_table(path: str | Path) -> dict[str, tuple[float, float, float]]:
    """
    Return the older-adult relative drops with 95% CIs from the Table S1 human row.

    Parameters:
        path (str | Path): <model>_table-S1_human.csv written by fit_haupt_damage_levels.py.

    Returns:
        dict[str, tuple[float, float, float]]: Target ("overall", "face", "object", "place",
        "pooled") -> (mean, ci95_lower, ci95_upper) in percent.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run fit_haupt_damage_levels.py first.")
    row = pd.read_csv(path, encoding="utf-8-sig").iloc[0]
    targets = [c.removesuffix("_mean") for c in row.index if c.endswith("_mean")]
    return {t: (float(row[f"{t}_mean"]), float(row[f"{t}_ci95_lower"]), float(row[f"{t}_ci95_upper"])) for t in targets}


def resolve_damage_level(level, damage_type: str, damage_area: str, best_fit_table: str | Path | None) -> float:
    """
    Return a numeric damage level, reading "best_fit" from the Haupt fit's descriptive table.

    Example:
        ("best_fit", "connections", "All") -> 0.15 when the whole-model connections fit is at 0.15

    Parameters:
        level (float | str): A level such as 0.15, or "best_fit".
        damage_type (str): e.g. "connections".
        damage_area (str): e.g. "All".
        best_fit_table (str | Path | None): Descriptive table; required for "best_fit".

    Returns:
        float: The damage level.
    """
    if str(level) != BEST_FIT_LEVEL:
        return float(level)
    if best_fit_table is None:
        raise ValueError(f"damage level {BEST_FIT_LEVEL!r} for {damage_type}/{damage_area} needs best_fit_table.")
    levels = load_best_fit_levels(best_fit_table)
    if (damage_type, damage_area) not in levels:
        raise ValueError(f"{best_fit_table} has no row for {damage_type}/{damage_area}; has {sorted(levels)}.")
    return levels[(damage_type, damage_area)]


# ------------------------- Plotting -------------------------

# Serif font to match the manuscript figures
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})

HUMAN_BAND_COLOUR = "#d9d9d9"
HUMAN_DASHES = (0, (3, 2))
MM_PER_INCH = 25.4
# Height / width of one category or bar panel
PANEL_ASPECT = 2.0
# Small-text style of the multi-panel figures (sizes in points)
PANEL_RC = {"font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
            "xtick.major.size": 2.0, "ytick.major.size": 2.0}
# Grid layout in millimetres: margins, and gaps between columns, rows, and a panel and its side panel
PANEL_GRID_MARGINS_MM = {"left": 13.0, "right": 7.0, "top": 8.0, "bottom": 9.0}
PANEL_GRID_GAPS_MM = {"column": 8.0, "row": 11.0, "side": 1.5}
# Side-panel width as a fraction of the main panel width
SIDE_PANEL_RATIO = 0.22
# Categories drawn with filled markers in the category and progress panels
FILLED_CATEGORIES = ("place",)
CATEGORY_LABELS = {"animal": "Animals", "face": "Faces", "object": "Objects", "place": "Places"}
# Early/middle/late split of the CORnet areas and AlexNet/VGG16 blocks (Fig S1)
DAMAGE_GROUPS = {"early": ("V1", "block1"), "middle": ("V2", "V4", "block2", "block3"), "late": ("IT", "block4", "block5")}


def despine(ax) -> None:
    """Hide the top and right axis lines."""
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig, base_path: str | Path, dpi: int = 500, tight: bool = False) -> list[Path]:
    """
    Save a figure as PNG and SVG next to each other.

    Parameters:
        fig (matplotlib.figure.Figure): Figure to save.
        base_path (str | Path): Path without extension, e.g. plots/figure_1c_connections.
        dpi (int): PNG resolution.
        tight (bool): Crop to the drawn content; otherwise keep the exact figure size.

    Returns:
        list[Path]: The saved files.
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    # Append rather than replace: names such as "figure_S8_retained_fovea_0.1" contain a dot
    paths = [base_path.with_name(f"{base_path.name}.png"), base_path.with_name(f"{base_path.name}.svg")]
    bbox = "tight" if tight else None
    fig.savefig(paths[0], dpi=dpi, bbox_inches=bbox)
    fig.savefig(paths[1], bbox_inches=bbox)
    plt.close(fig)
    return paths


def rounded_limits(values, step: float, include_zero: bool = True) -> tuple[float, float]:
    """
    Return (low, high) axis limits containing the finite values, rounded outwards to step.

    Example:
        values [-0.36, 0.07], step 0.1 -> (-0.4, 0.1)
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if include_zero:
        values = np.append(values, 0.0)
    if values.size == 0:
        raise ValueError("Cannot set axis limits without finite values.")
    low, high = math.floor(values.min() / step) * step, math.ceil(values.max() / step) * step
    if math.isclose(low, high):
        high = low + step
    return round(low, 10), round(high, 10)


def format_p_value(p_value: float) -> str:
    """Return a p-value in manuscript style, e.g. 0.0432 -> "p = .043"."""
    if not np.isfinite(p_value):
        return ""
    return "p < .001" if p_value < 0.001 else f"p = {p_value:.3f}".replace("0.", ".", 1)


def human_legend_entry(show_band: bool = True, line_width: float = 0.8) -> tuple:
    """Return the legend handle and label of the older-adult estimate (dashed line) and its 95% CI band."""
    line = Line2D([0], [0], color="black", linewidth=line_width, linestyle=HUMAN_DASHES)
    if not show_band:
        return line, "Human"
    return (Patch(facecolor=HUMAN_BAND_COLOUR, edgecolor="none"), line), "Human (95% CI)"


def category_legend_entries(categories, palette: Mapping[str, str], marker_size: float = 4.5,
                            line_width: float = 0.8) -> tuple[list, list[str]]:
    """Return legend handles and labels for category markers, filled for FILLED_CATEGORIES and open otherwise."""
    handles = [Line2D([0], [0], linestyle="none", marker="o", markersize=marker_size, markeredgewidth=line_width,
                      markerfacecolor=palette[c] if c in FILLED_CATEGORIES else "white", markeredgecolor=palette[c])
               for c in categories]
    return handles, [CATEGORY_LABELS[c] for c in categories]


def panel_grid_axes(n_panels: int, ncols: int, width_mm: float, panel_aspect: float, side_ratio: float) -> tuple:
    """
    Return a figure with a grid of panels, optional narrow side panels, and the free legend cell.

    Parameters:
        n_panels (int): Number of panels, filled row by row.
        ncols (int): Panels per row.
        width_mm (float): Figure width in millimetres.
        panel_aspect (float): Main panel height / width.
        side_ratio (float): Side-panel width / main panel width; 0 draws none.

    Returns:
        tuple: (fig, main axes [n_panels], side axes [n_panels] or Nones, legend cell
        rectangle (x, y, width, height) in figure fractions, or None without an empty cell).
    """
    nrows = math.ceil(n_panels / ncols)
    margins, gaps = PANEL_GRID_MARGINS_MM, PANEL_GRID_GAPS_MM
    side_gap = gaps["side"] if side_ratio > 0 else 0.0
    panel_width = (width_mm - margins["left"] - margins["right"] - (ncols - 1) * gaps["column"]
                   - ncols * side_gap) / (ncols * (1.0 + side_ratio))
    if panel_width <= 0:
        raise ValueError(f"width_mm={width_mm} is too narrow for {ncols} columns.")
    panel_height = panel_width * panel_aspect
    height_mm = margins["top"] + nrows * panel_height + (nrows - 1) * gaps["row"] + margins["bottom"]
    fig = plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH))
    cell_width = panel_width * (1.0 + side_ratio) + side_gap
    main_axes, side_axes, legend_rect = [], [], None
    for cell in range(nrows * ncols):
        row, column = divmod(cell, ncols)
        left = margins["left"] + column * (cell_width + gaps["column"])
        bottom = 1.0 - (margins["top"] + row * (panel_height + gaps["row"]) + panel_height) / height_mm
        if cell >= n_panels:
            legend_rect = legend_rect or (left / width_mm, bottom, cell_width / width_mm, panel_height / height_mm)
            continue
        main_axes.append(fig.add_axes([left / width_mm, bottom, panel_width / width_mm, panel_height / height_mm]))
        side_axes.append(fig.add_axes([(left + panel_width + side_gap) / width_mm, bottom,
                                       side_ratio * panel_width / width_mm, panel_height / height_mm])
                         if side_ratio > 0 else None)
    return fig, main_axes, side_axes, legend_rect


def draw_side_estimate(ax, estimate: float, lower: float, upper: float, colour: str, ylim, label: str,
                       p_text: str | None = None) -> None:
    """
    Draw one estimate with its 95% CI against zero in a narrow side panel.

    The axis is inverted so that more place dedifferentiation, or slower place progress,
    points down as on the main panels.
    """
    ax.axhline(0.0, color="0.55", linewidth=0.6, linestyle=HUMAN_DASHES, zorder=1)
    ax.vlines(0.0, lower, upper, colors=colour, linewidth=0.9, zorder=2)
    ax.plot(0.0, estimate, marker="o", markersize=4.0, color=colour, zorder=3)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(ylim[1], ylim[0])
    ax.set_xticks([])
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", labelsize=5.5, length=1.5, pad=1.0)
    ax.set_title(f"{label}\n{p_text}" if p_text else label, fontsize=5.5, pad=2.0)
    for spine in ("top", "left", "bottom"):
        ax.spines[spine].set_visible(False)


def place_alignment_table(stats_dir: str | Path, model_label: str, damage_type: str, kind: str) -> pd.DataFrame:
    """
    Read one place_alignment_stats.py output table.

    Example:
        ("stats/place_alignment", "cornet-rt", "noise", "progress-trend") ->
        cornet-rt_noise_All-V1-V2-V4-IT_IT_top0.05_place-alignment_progress-trend.csv

    Parameters:
        stats_dir (str | Path): Output folder of place_alignment_stats.py.
        model_label (str): Model label in the file names, e.g. "cornet-rt".
        damage_type (str): e.g. "connections".
        kind (str): "level-curves", "level-index", or "progress-trend".

    Returns:
        pd.DataFrame: The table.
    """
    matches = sorted(Path(stats_dir).glob(f"{model_label}_{damage_type}_*_place-alignment_{kind}.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {kind} table for {model_label}/{damage_type} in {stats_dir}; "
                                f"found {[p.name for p in matches]}. Run place_alignment_stats.py first.")
    return pd.read_csv(matches[0], encoding="utf-8-sig")


def plot_differentiation_curves(model_dir: str, damage_type: str, damage_areas: list[str], readout_layer: str,
                                categories: list[str], colours: Mapping[str, str], markers: Mapping[str, str],
                                file_name: str, plot_dir: str | Path = "plots", top_fraction: float = 0.05,
                                scaled_percent: bool = False, ylim=None, title: str | None = None,
                                xlabel: str = "Damage level", differentiation_dir: str | Path = "data/differentiation") -> pd.DataFrame:
    """
    Plot mean differentiation against damage level with +-1 SD bands and an AUC-loss side panel.

    Draws one line per damaged area (when categories is ["total"]) or per category (for
    one damaged area). Points are means across simulations and bands +-1 SD. The side
    bars show the AUC loss of each mean curve with +-1 SD across simulations
    (Methods 2.3.1). Used for Fig 1c, 2e, S2, S5-S8.

    Parameters:
        model_dir (str): Model folder of the differentiation table, e.g. "cornet_rt5_c".
        damage_type (str): e.g. "connections" or "eccentricity_weights_gradual_band0.50-1.00".
        damage_areas (list[str]): Damaged areas, e.g. ["V1", "V2", "V4", "IT"].
        readout_layer (str): e.g. "IT".
        categories (list[str]): ["total"] (mean of the four categories) or categories, e.g.
            ["animal", "face", "object", "place"].
        colours (Mapping[str, str]): Line colour per area or category.
        markers (Mapping[str, str]): Marker per area or category.
        file_name (str): Output name without extension.
        plot_dir (str | Path): Output folder.
        top_fraction (float): Fraction of selective units, e.g. 0.05.
        scaled_percent (bool): Scale each simulation to its intact value (%).
        ylim (Sequence[float] | None): y limits of the main panel.
        title (str | None): Panel title.
        xlabel (str): x-axis label.
        differentiation_dir (str | Path): Folder of the differentiation tables.

    Returns:
        pd.DataFrame: The plotted values (mean, SD, and n per line and level, and AUC loss
        with its SD), also saved as <file_name>_values.csv.
    """
    if categories != ["total"] and len(damage_areas) != 1:
        raise ValueError(f"Per-category lines need one damaged area; got {damage_areas}.")
    table = load_differentiation_table(model_dir, differentiation_dir)
    lines = [(area, category) for area in damage_areas for category in categories]
    fig, (ax, side_ax) = plt.subplots(1, 2, figsize=(10.0, 5.0), gridspec_kw={"width_ratios": [1.0, 0.25], "wspace": 0.18})
    rows, summaries = [], []
    for area, category in lines:
        key = area if categories == ["total"] else category
        label = area if categories == ["total"] else CATEGORY_LABELS[category]
        values = level_values(select_condition(table, damage_type, area, readout_layer, top_fraction), category, scaled_percent)
        levels = np.array(list(values))
        means = np.array([v.mean() for v in values.values()])
        sds = np.array([v.std(ddof=1) for v in values.values()])
        ax.fill_between(levels, means - sds, means + sds, color=colours[key], alpha=0.2, linewidth=0, zorder=1)
        ax.plot(levels, means, "-", marker=markers[key], color=colours[key], label=label, zorder=2)
        auc, auc_sd = curve_auc_summary(values)
        summaries.append((label, auc, auc_sd, colours[key]))
        rows += [{"damage_area": area, "category": category, "damage_level": level, "n": len(v), "mean": m, "sd": s,
                  "auc_loss": auc, "auc_loss_sd": auc_sd} for level, v, m, s in zip(levels, values.values(), means, sds)]

    positions = np.arange(len(summaries))
    side_ax.barh(positions, [s[1] for s in summaries], xerr=[s[2] for s in summaries], capsize=0,
                 color=[s[3] for s in summaries], alpha=0.9, error_kw={"elinewidth": 2.5, "ecolor": "0.25"})
    side_ax.axvline(0, color="0.35", linewidth=0.8)
    side_ax.set_yticks(positions, [s[0] for s in summaries])
    side_ax.invert_yaxis()
    side_ax.set_xlabel("AUC loss")
    side_ax.tick_params(axis="both", labelsize=8)
    side_ax.grid(axis="x", alpha=0.25)
    if min(s[1] for s in summaries) >= 0:
        side_ax.set_xlim(left=0)
    despine(side_ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Differentiation (scaled %)" if scaled_percent else "Differentiation")
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    despine(ax)
    values_table = pd.DataFrame(rows)
    save_figure(fig, Path(plot_dir) / file_name)
    values_table.to_csv(Path(plot_dir) / f"{file_name}_values.csv", index=False)
    return values_table


def plot_total_differentiation(model_dir: str, damage_area: str, readout_layer: str, conditions: list[dict],
                               best_fit_table: str | Path, human_table: str | Path, file_name: str,
                               plot_dir: str | Path = "plots", top_fraction: float = 0.05, ylim=(0, 110),
                               figsize=(3.0, 6.0), point_width: float = 0.52, point_alpha: float = 0.25,
                               seed: int = 12345, title: str | None = None,
                               differentiation_dir: str | Path = "data/differentiation") -> pd.DataFrame:
    """
    Plot overall scaled differentiation per simulation for a few damage conditions (Fig 1b).

    Each simulation's value is the mean of its face, object, and place differentiation,
    each scaled to the same simulation's intact value (100 - the overall drop of Table
    S1). Animals are left out because the human data have no animal ROI. Points show
    simulations, black lines the means with bootstrapped 95% CIs (10,000 resamples), and
    the dashed line and grey band the older-adult overall value with its 95% CI.

    Parameters:
        model_dir (str): e.g. "cornet_rt5_c".
        damage_area (str): e.g. "All".
        readout_layer (str): e.g. "IT".
        conditions (list[dict]): One mapping per condition with damage_type, damage_level
            (a number or "best_fit"), label (may use {level} and {percent}), colour, and marker.
        best_fit_table (str | Path): *_haupt-fit_descriptives_all-jobs.csv.
        human_table (str | Path): *_table-S1_human.csv.
        file_name (str): Output name without extension.
        plot_dir (str | Path): Output folder.
        top_fraction (float): Fraction of selective units.
        ylim (Sequence[float]): y limits.
        figsize (Sequence[float]): Figure size in inches.
        point_width (float): Horizontal spread of the simulation points in x units.
        point_alpha (float): Opacity of the simulation points.
        seed (int): Seed for the point jitter and the bootstrap.
        title (str | None): Axis title.
        differentiation_dir (str | Path): Folder of the differentiation tables.

    Returns:
        pd.DataFrame: Per condition the level, n, mean, SD, and 95% CI; also saved as <file_name>_values.csv.
    """
    table = load_differentiation_table(model_dir, differentiation_dir)
    human_mean, human_lower, human_upper = load_human_reference_table(human_table)["overall"]
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_box_aspect(PANEL_ASPECT)
    rows, labels = [], []
    for x, spec in enumerate(conditions):
        level = resolve_damage_level(spec["damage_level"], spec["damage_type"], damage_area, best_fit_table)
        labels.append(spec["label"].format(level=level, percent=round(100.0 * level, 6)))
        condition = select_condition(table, spec["damage_type"], damage_area, readout_layer, top_fraction)
        values = matched_category_values(condition, level, ["face", "object", "place"]).loc[:, ["face", "object", "place"]].mean(axis=1).to_numpy()
        lower, upper = (100.0, 100.0) if level == 0 else bootstrap_mean_ci(values, seed=seed)
        mean = float(values.mean())
        jitter = rng.uniform(-point_width / 2, point_width / 2, size=len(values))
        ax.scatter(x + jitter, values, s=12, alpha=point_alpha, color=spec["colour"], marker=spec["marker"], zorder=1.5)
        ax.vlines(x, lower, upper, colors="black", linewidth=1.2, zorder=3)
        ax.hlines(mean, x - 0.195, x + 0.195, colors="black", linewidth=2.0, zorder=3)
        rows.append({"label": labels[-1], "damage_type": spec["damage_type"], "damage_level": level, "n": len(values),
                     "mean": mean, "sd": float(values.std(ddof=1)), "ci95_lower": lower, "ci95_upper": upper})

    ax.set_xlim(-0.5, len(conditions) - 0.5)
    # The human band starts after the intact condition
    intact = [i for i, row in enumerate(rows) if row["damage_level"] == 0]
    band_start = (intact[0] + 0.365) if intact else ax.get_xlim()[0]
    band_end = ax.get_xlim()[1]
    ax.fill_between([band_start, band_end], 100 - human_upper, 100 - human_lower, color=HUMAN_BAND_COLOUR, linewidth=0, zorder=1)
    ax.hlines(100 - human_mean, band_start, band_end, colors="black", linestyles=[HUMAN_DASHES], linewidth=1.2, zorder=2)
    handle, label = human_legend_entry(line_width=1.2)
    ax.legend(handles=[handle], labels=[label], frameon=False, loc="upper right", handlelength=1.8)
    ax.set_xticks(range(len(conditions)), labels, rotation=35, ha="right")
    ax.set_ylabel("Differentiation (scaled %)")
    ax.set_ylim(ylim)
    if title:
        ax.set_title(title)
    despine(ax)
    fig.tight_layout()
    values_table = pd.DataFrame(rows)
    save_figure(fig, Path(plot_dir) / file_name)
    values_table.to_csv(Path(plot_dir) / f"{file_name}_values.csv", index=False)
    return values_table


def plot_category_panels(model_dir: str, readout_layer: str, panels: list[dict], palette: Mapping[str, str],
                         best_fit_table: str | Path, human_table: str | Path, place_index_dir: str | Path,
                         file_name: str, model_label: str = "cornet-rt", plot_dir: str | Path = "plots",
                         categories=("object", "place", "face", "animal"), top_fraction: float = 0.05,
                         ncols: int = 3, width_mm: float = 120.0, ylim=(0, 100), yticks=(0, 25, 50, 75, 100),
                         differentiation_dir: str | Path = "data/differentiation") -> pd.DataFrame:
    """
    Plot category differentiation at the human-matched level with place specificity side panels (Fig 1d).

    Each panel is one damaged area. Markers are the mean scaled differentiation across
    simulations with +-1 SD (places filled); grey bands and dashed lines show the
    older-adult value and 95% CI per category. Each category is matched to its own intact
    simulations, so animals (no human ROI) cannot remove simulations from the others.
    The side panel shows the place specificity index at the panel's level with its 95%
    CI including human variability, read from place_alignment_stats.py.

    Parameters:
        model_dir (str): e.g. "cornet_rt5_c".
        readout_layer (str): e.g. "IT".
        panels (list[dict]): One mapping per panel with damage_type, damage_area,
            damage_level (number or "best_fit"), and title, which may use {level} and
            {percent}, e.g. "V1 {percent:g}%".
        palette (Mapping[str, str]): Colour per category.
        best_fit_table (str | Path): *_haupt-fit_descriptives_all-jobs.csv.
        human_table (str | Path): *_table-S1_human.csv.
        place_index_dir (str | Path): Output folder of place_alignment_stats.py.
        file_name (str): Output name without extension.
        model_label (str): Model label in the statistics file names.
        plot_dir (str | Path): Output folder.
        categories (Sequence[str]): Category order on the x axis.
        top_fraction (float): Fraction of selective units.
        ncols (int): Panels per row; the first empty cell holds the legend.
        width_mm (float): Figure width in millimetres.
        ylim, yticks: Shared y limits and ticks.
        differentiation_dir (str | Path): Folder of the differentiation tables.

    Returns:
        pd.DataFrame: Plotted model and human values per panel and category, also saved as
        <file_name>_values.csv; the place index rows are saved as <file_name>_place-index.csv.
    """
    table = load_differentiation_table(model_dir, differentiation_dir)
    human = load_human_reference_table(human_table)
    rows, index_rows = [], []
    for panel in panels:
        level = resolve_damage_level(panel["damage_level"], panel["damage_type"], panel["damage_area"], best_fit_table)
        panel["resolved_title"] = panel["title"].format(level=level, percent=round(100.0 * level, 6))
        condition = select_condition(table, panel["damage_type"], panel["damage_area"], readout_layer, top_fraction)
        for category in categories:
            values = matched_category_values(condition, level, [category])[category].to_numpy()
            human_drop = human.get(category)
            rows.append({"panel": panel["resolved_title"], "damage_type": panel["damage_type"],
                         "damage_area": panel["damage_area"], "damage_level": level, "category": category,
                         "n": len(values), "model_mean": values.mean(), "model_sd": values.std(ddof=1),
                         "human_mean": np.nan if human_drop is None else 100 - human_drop[0],
                         "human_ci95_lower": np.nan if human_drop is None else 100 - human_drop[2],
                         "human_ci95_upper": np.nan if human_drop is None else 100 - human_drop[1]})
        level_index = place_alignment_table(place_index_dir, model_label, panel["damage_type"], "level-index")
        at_level = level_index[(level_index["damage_layer"] == panel["damage_area"])
                               & np.isclose(level_index["damage_level"], level, atol=1e-6, rtol=0.0)]
        if at_level.empty:
            raise ValueError(f"No place index for {panel['damage_type']}/{panel['damage_area']} at level {level}.")
        index_rows.append(at_level)
    values_table = pd.DataFrame(rows)
    index_table = pd.concat(index_rows, ignore_index=True)
    psi = index_table[index_table["quantity"] == "place_specificity_index"].reset_index(drop=True)
    psi_ylim = rounded_limits(psi[["ci95_lower", "ci95_upper"]].to_numpy(), 10.0)

    with plt.rc_context(PANEL_RC):
        fig, axes, side_axes, legend_rect = panel_grid_axes(len(panels), ncols, width_mm, PANEL_ASPECT, SIDE_PANEL_RATIO)
        x = np.arange(len(categories))
        for position, (ax, side_ax, panel) in enumerate(zip(axes, side_axes, panels)):
            panel_rows = values_table[values_table["panel"] == panel["resolved_title"]].set_index("category")
            for i, category in enumerate(categories):
                row = panel_rows.loc[category]
                if np.isfinite(row["human_mean"]):
                    ax.fill_between([i - 0.3, i + 0.3], row["human_ci95_lower"], row["human_ci95_upper"],
                                    color=HUMAN_BAND_COLOUR, linewidth=0, zorder=1)
                    ax.hlines(row["human_mean"], i - 0.3, i + 0.3, colors="black", linestyles=[HUMAN_DASHES],
                              linewidth=0.8, zorder=2)
                colour = palette[category]
                ax.errorbar(i, row["model_mean"], yerr=row["model_sd"], fmt="o", markersize=4.5,
                            markerfacecolor=colour if category in FILLED_CATEGORIES else "white",
                            markeredgecolor=colour, markeredgewidth=0.8, ecolor=colour, elinewidth=0.8,
                            capsize=1.5, capthick=0.8, zorder=3)
            ax.set_title(panel["resolved_title"])
            ax.set_xticks(x, [CATEGORY_LABELS[c][0] for c in categories])
            ax.set_xlim(-0.6, len(categories) - 0.4)
            ax.set_ylim(ylim)
            ax.set_yticks(yticks)
            despine(ax)
            if position % ncols == 0:
                ax.set_ylabel("Differentiation (scaled %)")
            else:
                ax.tick_params(labelleft=False)
            row = psi.iloc[position]
            draw_side_estimate(side_ax, row["estimate"], row["ci95_lower"], row["ci95_upper"], palette["place"],
                               psi_ylim, "PSI")
        handles, labels = category_legend_entries(categories, palette)
        human_handle, human_label = human_legend_entry()
        handles += [human_handle, Line2D([0], [0], marker="o", markersize=4.0, color=palette["place"], linewidth=0.9)]
        labels += [human_label, "PSI (95% CI)"]
        fig.legend(handles=handles, labels=labels, loc="center", frameon=False, handlelength=1.4, handletextpad=0.4,
                   bbox_to_anchor=(legend_rect[0] + legend_rect[2] / 2, legend_rect[1] + legend_rect[3] / 2))
        save_figure(fig, Path(plot_dir) / file_name, dpi=600)
    values_table.to_csv(Path(plot_dir) / f"{file_name}_values.csv", index=False)
    index_table.to_csv(Path(plot_dir) / f"{file_name}_place-index.csv", index=False)
    return values_table


def plot_place_progress_panels(damage_type: str, panels: list[dict], palette: Mapping[str, str], file_name: str,
                               stats_dir: str | Path = "stats/place_alignment", model_label: str = "cornet-rt",
                               plot_dir: str | Path = "plots", categories=("object", "place", "face"),
                               ncols: int = 3, width_mm: float = 120.0, panel_aspect: float = 1.0) -> pd.DataFrame:
    """
    Plot each category's progress towards its human level against the overall drop (Fig S3).

    Each panel is one damaged area. Points are the mean progress (model drop / human drop,
    1 = human level) at the levels of the trend window and lines the weighted least-squares
    fits from place_alignment_stats.py. The side panel shows the place slope difference
    (place slope minus the mean face and object slope) with its 95% CI including human
    variability and its two-sided p-value. Axes are inverted so that more
    dedifferentiation points down.

    Parameters:
        damage_type (str): e.g. "connections".
        panels (list[dict]): One mapping per panel with damage_area and title.
        palette (Mapping[str, str]): Colour per category.
        file_name (str): Output name without extension.
        stats_dir (str | Path): Output folder of place_alignment_stats.py.
        model_label (str): Model label in its file names.
        plot_dir (str | Path): Output folder.
        categories (Sequence[str]): Categories with a human ROI, in legend order.
        ncols (int): Panels per row; the first empty cell holds the legend.
        width_mm (float): Figure width in millimetres.
        panel_aspect (float): Main panel height / width.

    Returns:
        pd.DataFrame: The progress slopes and place slope differences of the plotted areas,
        also saved as <file_name>_stats.csv.
    """
    curves = place_alignment_table(stats_dir, model_label, damage_type, "level-curves")
    progress = place_alignment_table(stats_dir, model_label, damage_type, "progress-trend")
    areas = [panel["damage_area"] for panel in panels]
    progress = progress[progress["damage_layer"].isin(areas)]
    estimates = progress.set_index(["damage_layer", "quantity"])
    human_drop = {c: float(progress.loc[progress["quantity"] == f"{c}_progress_slope", "human_drop_percent"].iloc[0])
                  for c in categories}
    window = curves[curves["damage_layer"].isin(areas) & curves["in_trend_window"]]
    progress_ylim = rounded_limits(np.concatenate([window[f"{c}_mean"] / human_drop[c] for c in categories]), 0.1)
    interaction = progress[progress["quantity"] == "progress_place_minus_others"]
    interaction_ylim = rounded_limits(interaction[["ci95_lower", "ci95_upper"]].to_numpy(), 0.1)

    with plt.rc_context(PANEL_RC):
        fig, axes, side_axes, legend_rect = panel_grid_axes(len(panels), ncols, width_mm, panel_aspect, SIDE_PANEL_RATIO)
        for position, (ax, side_ax, panel) in enumerate(zip(axes, side_axes, panels)):
            area = panel["damage_area"]
            levels = window[window["damage_layer"] == area]
            x = levels["overall_mean"].to_numpy(dtype=float)
            weights = levels["n_simulations"].to_numpy(dtype=float)
            x_line = np.array([x.min(), x.max()])
            for category in categories:
                y = levels[f"{category}_mean"].to_numpy(dtype=float) / human_drop[category]
                # Slopes are per 10 points of overall drop; the WLS line passes through the weighted means
                slope = estimates.loc[(area, f"{category}_progress_slope"), "estimate"] / 10.0
                y_line = np.average(y, weights=weights) + slope * (x_line - np.average(x, weights=weights))
                colour = palette[category]
                ax.plot(x_line, y_line, color=colour, linewidth=1.0, zorder=2)
                ax.plot(x, y, linestyle="none", marker="o", markersize=3.0, markeredgewidth=0.7, zorder=3,
                        markerfacecolor=colour if category in FILLED_CATEGORIES else "white", markeredgecolor=colour)
            ax.axhline(1.0, color="black", linewidth=0.7, linestyle=HUMAN_DASHES, zorder=1)
            ax.set_ylim(progress_ylim[1], progress_ylim[0])
            ax.set_xlim(0.0, None)
            ax.set_title(panel["title"])
            despine(ax)
            if position % ncols == 0:
                ax.set_ylabel("Progress to human level")
            else:
                ax.tick_params(labelleft=False)
            if position + ncols >= len(panels):
                ax.set_xlabel("Overall drop (%)")
            row = estimates.loc[(area, "progress_place_minus_others")]
            draw_side_estimate(side_ax, row["estimate"], row["ci95_lower"], row["ci95_upper"], palette["place"],
                               interaction_ylim, "P − O/F", format_p_value(row["bootstrap_p_two_sided"]))
        handles = [Line2D([0], [0], color=palette[c], linewidth=1.0, marker="o", markersize=4.5,
                          markerfacecolor=palette[c] if c in FILLED_CATEGORIES else "white", markeredgecolor=palette[c])
                   for c in categories]
        labels = [CATEGORY_LABELS[c] for c in categories]
        handles += [Line2D([0], [0], color="black", linewidth=0.7, linestyle=HUMAN_DASHES),
                    Line2D([0], [0], marker="o", markersize=4.0, color=palette["place"], linewidth=0.9)]
        labels += ["Human level", "Place slope Δ (95% CI)"]
        fig.legend(handles=handles, labels=labels, loc="center", frameon=False, handlelength=1.6, handletextpad=0.4,
                   bbox_to_anchor=(legend_rect[0] + legend_rect[2] / 2, legend_rect[1] + legend_rect[3] / 2))
        save_figure(fig, Path(plot_dir) / file_name, dpi=600)
    progress.to_csv(Path(plot_dir) / f"{file_name}_stats.csv", index=False)
    return progress


def plot_damage_group_auc(models: list[dict], file_name: str, plot_dir: str | Path = "plots",
                          damage_types=("connections", "noise", "units"), colours: Mapping[str, str] | None = None,
                          top_fraction: float = 0.05, width_mm: float = 120.0, bar_width: float = 0.65,
                          differentiation_dir: str | Path = "data/differentiation") -> pd.DataFrame:
    """
    Plot dedifferentiation (AUC loss) after early, middle, and late damage across models (Fig S1).

    For each model, damage type, and damaged area, the AUC loss of the mean total
    differentiation curve is computed (as in the line-plot side panels). Areas are
    averaged within each model into early/middle/late groups (DAMAGE_GROUPS), so each
    model contributes one value per group; bars are the mean across models and markers
    the individual models.

    Parameters:
        models (list[dict]): One mapping per model with model (label), marker, readout_layer,
            damage_areas, and model_dirs (damage type -> model folder), e.g.
            {"model": "AlexNet", "marker": "D", "readout_layer": "classifier_2",
             "damage_areas": ["block1", ..., "block5"],
             "model_dirs": {"connections": "alexnet_c", "noise": "alexnet_c", "units": "alexnet_c+b"}}.
        file_name (str): Output name without extension.
        plot_dir (str | Path): Output folder.
        damage_types (Sequence[str]): One panel per damage type.
        colours (Mapping[str, str] | None): Bar colour per group.
        top_fraction (float): Fraction of selective units.
        width_mm (float): Figure width in millimetres.
        bar_width (float): Bar width in x units.
        differentiation_dir (str | Path): Folder of the differentiation tables.

    Returns:
        pd.DataFrame: Per model, damage type, and group the AUC loss and the across-model
        mean, also saved as <file_name>_values.csv.
    """
    titles = {"connections": "Connection severing", "noise": "Neural noise", "units": "Neuron deletion"}
    colours = colours or {"early": "#4477AA", "middle": "#228833", "late": "#EE6677"}
    area_group = {area: group for group, areas in DAMAGE_GROUPS.items() for area in areas}
    rows = []
    for spec in models:
        for damage_type in damage_types:
            table = load_differentiation_table(spec["model_dirs"][damage_type], differentiation_dir)
            for area in spec["damage_areas"]:
                condition = select_condition(table, damage_type, area, spec["readout_layer"], top_fraction)
                values = level_values(condition, "total")
                rows.append({"model": spec["model"], "damage_type": damage_type, "damage_area": area,
                             "damage_group": area_group[area],
                             "auc_loss": auc_loss(list(values), [v.mean() for v in values.values()])})
    per_area = pd.DataFrame(rows)
    per_model = per_area.groupby(["model", "damage_type", "damage_group"], sort=False, as_index=False)["auc_loss"].mean()
    per_model["group_mean_auc_loss"] = per_model.groupby(["damage_type", "damage_group"])["auc_loss"].transform("mean")

    groups = list(DAMAGE_GROUPS)
    x = np.arange(len(groups))
    offsets = dict(zip([m["model"] for m in models], np.linspace(-0.3, 0.3, len(models)) * bar_width))
    margins = {"left": 14.0, "right": 2.0, "top": 6.0, "bottom": 20.0}
    panel_width = (width_mm - margins["left"] - margins["right"]) / (len(damage_types) + (len(damage_types) - 1) * 0.15)
    height_mm = panel_width * PANEL_ASPECT + margins["top"] + margins["bottom"]
    style = {**PANEL_RC, "font.size": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, len(damage_types), sharey=True, figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH))
        for ax, damage_type in zip(np.atleast_1d(axes), damage_types):
            panel = per_model[per_model["damage_type"] == damage_type]
            means = panel.groupby("damage_group")["auc_loss"].mean().reindex(groups)
            ax.bar(x, means, width=bar_width, color=[colours[g] for g in groups], alpha=0.75, edgecolor="none", zorder=1)
            for spec in models:
                model_values = panel[panel["model"] == spec["model"]].set_index("damage_group")["auc_loss"].reindex(groups)
                ax.plot(x + offsets[spec["model"]], model_values, linestyle="none", marker=spec["marker"], markersize=3.0,
                        markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.6, zorder=2)
            ax.axhline(0.0, color="black", linewidth=0.6, zorder=0)
            ax.set_title(titles[damage_type])
            ax.set_xticks(x, [g.capitalize() for g in groups])
            ax.set_xlim(-0.6, len(groups) - 0.4)
            ax.set_box_aspect(PANEL_ASPECT)
            despine(ax)
        for ax in np.atleast_1d(axes)[1:]:
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False, labelleft=False)
        top = max(per_model["auc_loss"].max(), per_model["group_mean_auc_loss"].max())
        np.atleast_1d(axes)[0].set_ylim(min(0.0, per_model["auc_loss"].min() * 1.1), top * 1.1)
        np.atleast_1d(axes)[0].set_ylabel("Dedifferentiation (AUC loss)")
        left, right = margins["left"] / width_mm, 1.0 - margins["right"] / width_mm
        fig.subplots_adjust(left=left, right=right, bottom=margins["bottom"] / height_mm,
                            top=1.0 - margins["top"] / height_mm, wspace=0.15)
        fig.text((left + right) / 2, 13.0 / height_mm, "Degraded layers", ha="center", va="center", fontsize=7)
        handles = [Line2D([], [], linestyle="none", marker=m["marker"], markersize=3.0, markerfacecolor="white",
                          markeredgecolor="black", markeredgewidth=0.6) for m in models]
        fig.legend(handles=handles, labels=[m["model"] for m in models], loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=((left + right) / 2, 0.0), handlelength=1.4, handletextpad=0.4, columnspacing=1.2)
        save_figure(fig, Path(plot_dir) / file_name, dpi=600)
    per_model.to_csv(Path(plot_dir) / f"{file_name}_values.csv", index=False)
    return per_model


def gradient_analysis_name(units: str, images: str) -> str:
    """Return the file prefix of one gradient analysis, e.g. place_units_place_images."""
    return f"{units}_units_{images}_images"


def plot_gradient_delta_maps(panels: list[dict], file_name: str, plot_dir: str | Path = "plots",
                             stats_dir: str | Path = "stats/it2v1", ncols: int = 2,
                             cmap: str = "plasma") -> pd.DataFrame:
    """
    Plot V1 maps of Cliff's delta between selective and random-unit gradients (Fig 2b left, S4).

    Each panel shows, at every V1 position, the within-image Cliff's delta from
    v1_gradient_analysis.py stats on the full -1 to 1 scale, so panels share one colour scale.

    Parameters:
        panels (list[dict]): One {units, images, title} per panel, e.g.
            {units: place, images: place, title: "Avg. Places"}.
        file_name (str): Output name without extension.
        plot_dir (str | Path): Output folder.
        stats_dir (str | Path): Folder of the <analysis>_position_delta.csv tables.
        ncols (int): Panels per row.
        cmap (str): Colour map.

    Returns:
        pd.DataFrame: The plotted Cliff's delta and its 95% CI per panel and position, also
        saved as <file_name>_values.csv.
    """
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.7 * nrows), squeeze=False, layout="constrained")
    rows = []
    for ax, panel in zip(axes.flat, panels):
        table = pd.read_csv(Path(stats_dir) / f"{gradient_analysis_name(panel['units'], panel['images'])}_position_delta.csv")
        height, width = table["fy"].max() + 1, table["fx"].max() + 1
        delta_map = table["cliffs_delta"].to_numpy(dtype=float).reshape(height, width)
        image = ax.imshow(delta_map, cmap=cmap, vmin=-1.0, vmax=1.0, interpolation="nearest")
        colourbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colourbar.set_label("Cliff's $\\delta$")
        ax.set_title(panel.get("title", ""))
        ax.set_xlabel("W")
        ax.set_ylabel("H", rotation=0, labelpad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        rows.append(table.assign(units=panel["units"], images=panel["images"])[
            ["units", "images", "fy", "fx", "cliffs_delta", "ci_low", "ci_high"]])
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    values_table = pd.concat(rows, ignore_index=True)
    save_figure(fig, Path(plot_dir) / file_name, tight=True)
    values_table.to_csv(Path(plot_dir) / f"{file_name}_values.csv", index=False)
    return values_table


def plot_gradient_overlays(panels: list[dict], file_name: str, plot_dir: str | Path = "plots",
                           stats_dir: str | Path = "stats/it2v1", image_dir: str | Path = "stimuli",
                           ncols: int = 2, alpha: float = 0.65, cmap: str = "viridis") -> pd.DataFrame:
    """
    Overlay one image's V1 Cliff's delta map on the greyscale image (Fig 2b right).

    The map of the image (Cliff's delta between the selective and the random-set values of
    that image at each position) is upsampled bilinearly to the 224 x 224 crop that the
    model saw and scaled to its own range.

    Parameters:
        panels (list[dict]): One {units, images, image} per panel, e.g.
            {units: place, images: place, image: place10.jpg}.
        file_name (str): Output name without extension.
        plot_dir (str | Path): Output folder.
        stats_dir (str | Path): Folder of the <analysis>_image_delta_maps.npy arrays.
        image_dir (str | Path): Stimulus folder.
        ncols (int): Panels per row.
        alpha (float): Opacity of the delta map.
        cmap (str): Colour map.

    Returns:
        pd.DataFrame: The delta range of each overlaid map, also saved as <file_name>_values.csv.
    """
    nrows = math.ceil(len(panels) / ncols)
    crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows), squeeze=False, layout="constrained")
    rows = []
    for ax, panel in zip(axes.flat, panels):
        delta_maps = np.load(Path(stats_dir) / f"{gradient_analysis_name(panel['units'], panel['images'])}_image_delta_maps.npy")
        names = [p.name for p in category_images(image_dir, panel["images"])]
        if panel["image"] not in names:
            raise ValueError(f"image must be one of {names}; got {panel['image']!r}.")
        if delta_maps.shape[0] != len(names):
            raise ValueError(f"{len(names)} {panel['images']} images but delta maps of shape {delta_maps.shape}.")
        delta_map = delta_maps[names.index(panel["image"])].astype(np.float32)
        greyscale = np.asarray(crop(Image.open(Path(image_dir) / panel["image"]).convert("L")))
        upsampled = np.asarray(Image.fromarray(delta_map, mode="F").resize(greyscale.shape[::-1], Image.BILINEAR))
        ax.imshow(greyscale, cmap="gray", vmin=0, vmax=255)
        image = ax.imshow(upsampled, cmap=cmap, alpha=alpha, vmin=delta_map.min(), vmax=delta_map.max())
        ax.set_title(panel.get("title", ""))
        ax.axis("off")
        rows.append({"units": panel["units"], "images": panel["images"], "image": panel["image"],
                     "delta_min": float(delta_map.min()), "delta_max": float(delta_map.max())})
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    colourbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03, ticks=[])
    colourbar.set_label("Cliff's $\\delta$ (each image's range)")
    values_table = pd.DataFrame(rows)
    save_figure(fig, Path(plot_dir) / file_name, tight=True)
    values_table.to_csv(Path(plot_dir) / f"{file_name}_values.csv", index=False)
    return values_table


def plot_eccentricity_bias(categories: list[str], colours: Mapping[str, str], markers: Mapping[str, str],
                           file_name: str, plot_dir: str | Path = "plots", stats_dir: str | Path = "stats/it2v1",
                           xlabel: str = "V1 eccentricity band distance from centre (filters)",
                           ylabel: str = "Cliff's $\\delta$ effect size") -> pd.DataFrame:
    """
    Plot Cliff's delta between selective and random-unit gradient density per eccentricity band (Fig 2c).

    Points are the within-image Cliff's delta per band, averaged over images; shading is the
    pointwise 95% bootstrap CI over images and random sets (v1_gradient_analysis.py stats,
    Methods 2.6.2).

    Parameters:
        categories (list[str]): Categories to draw, e.g. ["animal", "object", "place", "face"].
        colours (Mapping[str, str]): Line colour per category.
        markers (Mapping[str, str]): Marker per category.
        file_name (str): Output name without extension.
        plot_dir (str | Path): Output folder.
        stats_dir (str | Path): Folder of eccentricity_bias.csv.
        xlabel (str), ylabel (str): Axis labels.

    Returns:
        pd.DataFrame: The plotted rows of eccentricity_bias.csv, also saved as <file_name>_values.csv.
    """
    table = pd.read_csv(Path(stats_dir) / "eccentricity_bias.csv")
    missing = sorted(set(categories) - set(table["category"]))
    if missing:
        raise ValueError(f"eccentricity_bias.csv has no rows for {missing}.")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for category in categories:
        rows = table[table["category"] == category].sort_values("band_center")
        ax.fill_between(rows["band_center"], rows["ci_low"], rows["ci_high"], color=colours[category], alpha=0.2,
                        linewidth=0)
        ax.plot(rows["band_center"], rows["cliffs_delta"], "-", marker=markers[category], color=colours[category],
                label=CATEGORY_LABELS[category])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="lower left")
    despine(ax)
    values_table = table[table["category"].isin(categories)]
    save_figure(fig, Path(plot_dir) / file_name)
    values_table.to_csv(Path(plot_dir) / f"{file_name}_values.csv", index=False)
    return values_table

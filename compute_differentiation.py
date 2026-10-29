"""
Compute category differentiation for every simulation from the saved model activations.

Two steps, run in order (the downloadable data release already contains the output of
step 1, so step 2 alone rebuilds the differentiation tables from it):

1. rdms: for each simulated condition of a model, select the top category-selective
   readout units (fixed from the intact model and the localizer images, Methods 2.2)
   and save one image-by-image correlation matrix (RDM) per simulation, category and
   damage level.
2. table: compute within-category minus between-category similarity (Methods Eq. 2)
   for every RDM and write one table per model to data/differentiation/<model_dir>.csv.gz,
   with one row per simulation, category, damage level, damaged area, and readout.

Inputs and outputs live under data/simulations/<model_dir>/<damage_type>/<area>/:
    activations/<readout>/damaged_<level>/<simulation>.npz          (from main.py)
    RDM_<fraction>_percentage/<readout>/<category>_selective/damaged_<level>.npz

Usage:
    python compute_differentiation.py rdms --model-dir cornet_rt5_c --readout IT \
        --selectivity-table unit_selectivity/cornet_rt5_unit_selectivity.csv.gz \
        --top-fractions 0.01 0.05 0.10
    python compute_differentiation.py table --model-dirs cornet_rt5_c cornet_rt5_c+b
    python compute_differentiation.py table          # every model under data/simulations
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from utils import (
    assign_categories,
    build_selective_rdms,
    calc_within_between,
    load_selectivity_table,
    load_selective_rdms,
)

SIMULATIONS_DIR = Path("data/simulations")
DIFFERENTIATION_DIR = Path("data/differentiation")
# e.g. RDM_0.05_percentage -> top fraction 0.05
RDM_DIR_PATTERN = re.compile(r"RDM_(\d+(?:\.\d+)?)_(percentage)$")
TABLE_COLUMNS = [
    "damage_type", "damage_area", "readout_layer", "top_fraction", "selection_mode", "category",
    "damage_level", "replicate", "replicate_order", "within", "between", "differentiation",
]


def build_model_rdms(model_dir: Path, readout_layer: str, selectivity_table_path: Path,
                     top_fractions: list[float]) -> None:
    """
    Build selective RDMs for every simulated condition of one model.

    Parameters:
        model_dir (Path): data/simulations/<model_dir>.
        readout_layer (str): Readout layer directory name, e.g. "IT" or "classifier_2".
        selectivity_table_path (Path): Unit-selectivity table of the intact model.
        top_fractions (list[float]): Fractions of selective units, e.g. [0.05].

    Returns:
        None
    """
    selectivity_table = load_selectivity_table(selectivity_table_path)
    activation_dirs = sorted(model_dir.glob(f"*/*/activations/{readout_layer}"))
    if not activation_dirs:
        raise FileNotFoundError(f"No */*/activations/{readout_layer} folders under {model_dir}.")
    for activation_dir in activation_dirs:
        area_dir = activation_dir.parent.parent
        for top_fraction in top_fractions:
            rdm_dir = area_dir / f"RDM_{top_fraction:.2f}_percentage" / readout_layer
            print(f"Building {rdm_dir}")
            build_selective_rdms(activation_dir, rdm_dir, selectivity_table, readout_layer, top_fraction)


def differentiation_rows(rdm_file: Path, model_dir: Path) -> list[dict]:
    """
    Compute differentiation for every simulation stored in one selective-RDM file.

    Parameters:
        rdm_file (Path): .../<damage_type>/<area>/RDM_<frac>_percentage/<readout>/<category>_selective/damaged_<level>.npz
        model_dir (Path): data/simulations/<model_dir>, used to parse the condition from the path.

    Returns:
        list[dict]: One row per simulation with the TABLE_COLUMNS fields.
    """
    damage_type, damage_area, rdm_dirname, readout_layer, category_dir, _ = rdm_file.relative_to(model_dir).parts
    match = RDM_DIR_PATTERN.match(rdm_dirname)
    if match is None or not category_dir.endswith("_selective"):
        raise ValueError(f"Unexpected selective-RDM path: {rdm_file}")
    category = category_dir.removesuffix("_selective")
    damage_level = float(rdm_file.stem.removeprefix("damaged_"))

    rdms, image_names, replicates = load_selective_rdms(rdm_file)
    categories_array = assign_categories(image_names)
    rows = []
    for order, (replicate, rdm) in enumerate(zip(replicates, rdms)):
        values = calc_within_between(rdm, categories_array, category)
        rows.append({
            "damage_type": damage_type, "damage_area": damage_area, "readout_layer": readout_layer,
            "top_fraction": float(match.group(1)), "selection_mode": match.group(2), "category": category,
            "damage_level": damage_level, "replicate": replicate, "replicate_order": order,
            "within": values["within"], "between": values["between"],
            "differentiation": values["differentiation"],
        })
    return rows


def build_differentiation_table(model_dir: Path) -> pd.DataFrame:
    """
    Collect differentiation for every selective-RDM file of one model.

    Parameters:
        model_dir (Path): data/simulations/<model_dir>.

    Returns:
        pd.DataFrame: Long table with TABLE_COLUMNS, sorted by condition and replicate.
    """
    rdm_files = sorted(model_dir.glob("*/*/RDM_*_percentage/*/*_selective/damaged_*.npz"))
    if not rdm_files:
        raise FileNotFoundError(f"No selective-RDM files under {model_dir}.")
    rows = [row for rdm_file in rdm_files for row in differentiation_rows(rdm_file, model_dir)]
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    return table.sort_values(TABLE_COLUMNS[:7] + ["replicate_order"], ignore_index=True)


def main() -> None:
    """Run the requested step from the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--simulations-dir", type=Path, default=SIMULATIONS_DIR,
                        help="Root folder of the simulation outputs (default: data/simulations).")
    steps = parser.add_subparsers(dest="step", required=True)

    rdms_parser = steps.add_parser("rdms", help="Build selective RDMs from saved activations.")
    rdms_parser.add_argument("--model-dir", required=True, help="Model folder name, e.g. cornet_rt5_c.")
    rdms_parser.add_argument("--readout", required=True, help="Readout layer folder, e.g. IT or classifier_2.")
    rdms_parser.add_argument("--selectivity-table", type=Path, required=True,
                             help="Unit-selectivity table of the intact model (.csv or .csv.gz).")
    rdms_parser.add_argument("--top-fractions", type=float, nargs="+", default=[0.05],
                             help="Fractions of most selective units (default: 0.05).")

    table_parser = steps.add_parser("table", help="Compute differentiation tables from selective RDMs.")
    table_parser.add_argument("--model-dirs", nargs="+", default=None,
                              help="Model folder names (default: every folder under --simulations-dir).")
    table_parser.add_argument("--output-dir", type=Path, default=DIFFERENTIATION_DIR,
                              help="Where to write <model_dir>.csv.gz (default: data/differentiation).")
    args = parser.parse_args()

    if args.step == "rdms":
        build_model_rdms(args.simulations_dir / args.model_dir, args.readout, args.selectivity_table,
                         args.top_fractions)
        return

    model_names = args.model_dirs or sorted(p.name for p in args.simulations_dir.iterdir() if p.is_dir())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_name in model_names:
        table = build_differentiation_table(args.simulations_dir / model_name)
        output_path = args.output_dir / f"{model_name}.csv.gz"
        table.to_csv(output_path, index=False)
        print(f"{output_path}: {len(table)} rows")


if __name__ == "__main__":
    main()

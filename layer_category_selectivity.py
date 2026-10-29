"""
Choose each model's readout layer: the most category-selective candidate layer on the localizer images (Methods 2.2).

The 80 localizer images in categ_images/ are passed through the intact pretrained model.
For every unit of each candidate layer and each category, the Mann-Whitney U of the 20
category images against the 60 other images is normalised to U* = U / (20 x 60) (Methods
Eq. 1). A layer's score is the mean U* of its top 5% units within each category, averaged
across the four categories; the layer with the highest score is the readout layer.

Candidate layers are listed under candidate_layers in the selectivity configs: the block
outputs for the CORnet models and every post-ReLU convolutional and fully connected layer
for AlexNet and VGG16. Units are the flattened layer outputs (each channel at each position),
as in unit_categ_selectivity.py, which then builds the selectivity table of the chosen layer.
The paper's choices are module.IT.output (CORnet-RT, CORnet-S), classifier.2 (AlexNet), and
classifier.4 (VGG16). The criterion picks these for every model except CORnet-RT4, where V4
scores slightly higher than IT (0.677 vs 0.647); RT4 keeps the IT readout so that it matches
CORnet-RT5, and the script reports the difference.

Writes stats/layer_choice/<config name>_layer_scores.csv with one row per candidate layer:
layer, n_units, u_star_<category> (top-5% mean per category), score, and chosen.

Usage:
    python layer_category_selectivity.py configs/selectivity/alexnet.yaml
    python layer_category_selectivity.py configs/selectivity/*.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unit_categ_selectivity import LOCALIZER_DIR, layer_activations, localizer_images, mann_whitney_selectivity
from utils import load_model

OUTPUT_DIR = Path("stats/layer_choice")
TOP_FRACTION = 0.05
# Units per Mann-Whitney call; the early VGG16 layers have about 3.2 million units
CHUNK_SIZE = 200_000


def top_fraction_u_star(activations: np.ndarray, labels: list[str], top_fraction: float = TOP_FRACTION,
                        chunk_size: int = CHUNK_SIZE) -> dict[str, float]:
    """
    Return, per category, the mean normalised U (U*) of the layer's most selective units.

    Parameters:
        activations (np.ndarray): Image-by-unit activations [N,U].
        labels (list[str]): Category of each image [N], e.g. "faces".
        top_fraction (float): Fraction of units kept per category; round(U x fraction) units, at least one,
            as in compute_differentiation.py.
        chunk_size (int): Units per Mann-Whitney call, to bound memory.

    Returns:
        dict[str, float]: Category -> mean U* of its top units.
    """
    if activations.ndim != 2 or activations.shape[0] != len(labels):
        raise ValueError(f"activations must be [N,U] with N = {len(labels)} images; got shape {activations.shape}.")
    chunks = [mann_whitney_selectivity(activations[:, start:start + chunk_size], labels)
              for start in range(0, activations.shape[1], chunk_size)]
    labels = np.asarray(labels)
    n_top = max(1, int(round(activations.shape[1] * top_fraction)))
    u_star = {}
    for category in chunks[0]:
        n_target = int(np.sum(labels == category))
        statistics = np.concatenate([chunk[category] for chunk in chunks])
        u_star[category] = float(np.sort(statistics)[-n_top:].mean() / (n_target * (len(labels) - n_target)))
    return u_star


def score_layers(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Add each layer's score (mean of the per-category U* columns) and mark the highest-scoring layer.

    Parameters:
        scores (pd.DataFrame): One row per layer with layer, n_units, and u_star_<category> columns.

    Returns:
        pd.DataFrame: The table with score and chosen columns added.
    """
    u_star_columns = [c for c in scores.columns if c.startswith("u_star_")]
    if not u_star_columns:
        raise ValueError(f"scores needs u_star_<category> columns; got {list(scores.columns)}.")
    scores = scores.assign(score=scores[u_star_columns].mean(axis=1))
    return scores.assign(chosen=scores.index == scores["score"].idxmax())


def run_config(config_path: Path) -> pd.DataFrame:
    """
    Score every candidate layer of one selectivity config and save the table.

    Parameters:
        config_path (Path): Selectivity YAML with model, optional time_steps, layer_paths, and candidate_layers.

    Returns:
        pd.DataFrame: The layer scores (see score_layers()).
    """
    with open(config_path, encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    missing = [key for key in ("model", "layer_paths", "candidate_layers") if key not in config]
    if missing:
        raise ValueError(f"{config_path} is missing {missing}.")

    model = load_model(config["model"], config.get("time_steps"))
    image_paths, labels = localizer_images(Path(config.get("image_dir", LOCALIZER_DIR)))
    rows = []
    # One layer per pass keeps only one flattened layer in memory
    for layer in config["candidate_layers"]:
        activations = layer_activations(model, [layer], image_paths)[layer]
        u_star = top_fraction_u_star(activations, labels)
        rows.append({"layer": layer, "n_units": activations.shape[1], **{f"u_star_{c}": v for c, v in u_star.items()}})
        print(f"{config_path.stem} {layer}: {activations.shape[1]} units, score {np.mean(list(u_star.values())):.4f}")
    scores = score_layers(pd.DataFrame(rows))

    output_path = OUTPUT_DIR / f"{config_path.stem}_layer_scores.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)
    chosen = scores.loc[scores["chosen"], "layer"].item()
    agreement = "matches" if chosen in config["layer_paths"] else "DIFFERS FROM"
    print(f"{config_path.stem}: chosen layer {chosen} {agreement} the config's layer_paths {config['layer_paths']} "
          f"-> {output_path}")
    return scores


def main() -> None:
    """Run every given selectivity config."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", type=Path, help="Selectivity config YAML file(s).")
    args = parser.parse_args()
    for config_path in args.configs:
        run_config(config_path)


if __name__ == "__main__":
    main()

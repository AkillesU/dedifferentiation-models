"""
Measure the category selectivity of every readout unit on the independent localizer images (Methods 2.2).

The 80 localizer images in categ_images/ (20 animals, faces, objects, places) are passed
through the intact pretrained model. For each unit of the readout layer(s) and each
category, a two-sided Mann-Whitney U test compares the unit's responses to the 20 images
of the category with its responses to the 60 other images. The U statistic of the target
category (larger = stronger response to the category) is saved; dividing it by 20 x 60
gives the normalised U* of Methods Eq. 1, which ranks units identically.
compute_differentiation.py keeps the top 5% of units per category.

Activations are taken after the activation function, at the last time step for recurrent
models. Images are resized to 256 pixels and centre-cropped to 224, as in the original
selectivity analysis.

Writes unit_selectivity/<output_name>.csv.gz with columns layer, unit, mw_animals,
mw_faces, mw_objects, mw_places (units index the flattened layer output).

Usage:
    python unit_categ_selectivity.py configs/selectivity/cornet_rt5.yaml
    python unit_categ_selectivity.py configs/selectivity/*.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from scipy.stats import mannwhitneyu
from torchvision import transforms

from utils import IMAGE_EXTENSIONS, IMAGENET_MEAN, IMAGENET_STD, get_module, load_model

LOCALIZER_DIR = Path("categ_images")
OUTPUT_DIR = Path("unit_selectivity")


def localizer_images(image_dir: Path) -> tuple[list[Path], list[str]]:
    """
    Return the localizer image paths and their category folder names, sorted.

    Parameters:
        image_dir (Path): Folder with one subfolder per category, e.g. categ_images/faces/.

    Returns:
        tuple[list[Path], list[str]]: Image paths and category labels, e.g. "faces".
    """
    paths, labels = [], []
    for category_dir in sorted(p for p in image_dir.iterdir() if p.is_dir()):
        for path in sorted(p for p in category_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
            paths.append(path)
            labels.append(category_dir.name)
    if not paths:
        raise FileNotFoundError(f"No category subfolders with images in {image_dir}.")
    return paths, labels


def layer_activations(model: torch.nn.Module, layer_paths: list[str], image_paths: list[Path]) -> dict[str, np.ndarray]:
    """
    Return each layer's flattened activations for every image.

    Parameters:
        model (torch.nn.Module): Model in evaluation mode.
        layer_paths (list[str]): Module paths, e.g. ["module.IT.output"].
        image_paths (list[Path]): Images in order.

    Returns:
        dict[str, np.ndarray]: Layer path -> image-by-unit activations [N,U].
    """
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    latest = {}
    hooks = [get_module(model, path).register_forward_hook(
        lambda _m, _i, out, path=path: latest.__setitem__(path, (out[0] if isinstance(out, tuple) else out).detach()))
        for path in layer_paths]
    device = next(model.parameters()).device
    rows = {path: [] for path in layer_paths}
    with torch.no_grad():
        for image_path in image_paths:
            model(preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device))
            # Recurrent blocks call their output layer once per time step; the last call is kept
            for path in layer_paths:
                rows[path].append(latest[path].cpu().numpy().ravel())
    for hook in hooks:
        hook.remove()
    return {path: np.stack(values) for path, values in rows.items()}


def mann_whitney_selectivity(activations: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    """
    Return the Mann-Whitney U of every unit for each category against the other categories.

    Parameters:
        activations (np.ndarray): Image-by-unit activations [N,U].
        labels (list[str]): Category of each image [N].

    Returns:
        dict[str, np.ndarray]: Category -> U statistic of the target category per unit [U].
    """
    labels = np.asarray(labels)
    statistics = {}
    for category in sorted(set(labels)):
        in_category = labels == category
        statistics[category] = mannwhitneyu(activations[in_category], activations[~in_category],
                                            alternative="two-sided", axis=0).statistic
    return statistics


def run_config(config_path: Path) -> Path:
    """
    Compute and save the selectivity table described by one config.

    Parameters:
        config_path (Path): YAML with model, optional time_steps, layer_paths, and output_name.

    Returns:
        Path: The written table.
    """
    with open(config_path, encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    missing = [key for key in ("model", "layer_paths", "output_name") if key not in config]
    if missing:
        raise ValueError(f"{config_path} is missing {missing}.")

    model = load_model(config["model"], config.get("time_steps"))
    image_paths, labels = localizer_images(Path(config.get("image_dir", LOCALIZER_DIR)))
    activations = layer_activations(model, config["layer_paths"], image_paths)
    tables = []
    for path, layer_activations_ in activations.items():
        statistics = mann_whitney_selectivity(layer_activations_, labels)
        tables.append(pd.DataFrame({"layer": path, "unit": np.arange(layer_activations_.shape[1]),
                                    **{f"mw_{category}": values for category, values in statistics.items()}}))
    output_path = OUTPUT_DIR / f"{config['output_name']}.csv.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(output_path, index=False)
    print(f"{config_path}: {sum(len(t) for t in tables)} units -> {output_path}")
    return output_path


def main() -> None:
    """Run every given selectivity config."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", type=Path, help="Selectivity config YAML file(s).")
    args = parser.parse_args()
    for config_path in args.configs:
        run_config(config_path)


if __name__ == "__main__":
    main()

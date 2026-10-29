"""
Map which V1 positions drive category-selective IT units in CORnet-RT (Fig 2b, 2c, S4).

For each analysis (one group of top-5% category-selective IT units and one image category),
the response of every selected IT unit is backpropagated to the V1 feature map for each
image, and the absolute gradients are averaged across the units and the V1 channels
(Methods 2.6.1). The same is done for 100 random sets of the same number of IT units.

Two steps, run in order (the data release contains the output of step 1, so step 2 alone
rebuilds the statistics):

1. gradients: model and stimuli -> data/it2v1_gradients/<analysis>/
       maps_selective.npy   [N_img,H,W] mean |d IT / d V1| of the selective units, averaged
                            over the V1 channels
       maps_random.npy      [N_sets,N_img,H,W] the same for every random set
2. stats: maps -> stats/it2v1/
       <analysis>_position_delta.csv    Cliff's delta per V1 position with its 95% CI, and the
                                        mean difference in share of V1 sensitivity (Fig 2b, S4)
       <analysis>_image_delta_maps.npy  [N_img,H,W] Cliff's delta of every image (Fig 2b overlays)
       eccentricity_bias.csv            Cliff's delta per eccentricity band with its 95% CI,
                                        self analyses only (Fig 2c)

Statistics (Methods 2.6.1, 2.6.2):
- Each map is divided by its own sum, so a value is that position's share of the IT
  population's total V1 sensitivity. A population that is uniformly more sensitive across V1
  has the same map as a less sensitive one.
- For image i and position (or band) x, delta_i(x) is Cliff's delta between the selective
  value and the values of the random sets on the same image, i.e. the mean over random sets
  of sign(selective - random set). The reported delta is the mean of delta_i over images.
- Images and random sets are the sampled units, and they are crossed (every random set is
  evaluated on every image). The 95% CI is a percentile bootstrap that resamples images and
  random sets independently. Positions and channels are not treated as samples, so no p
  values are computed.

Analysis choices, as in the original analysis:
- The gradient is taken at V1's input block output (module.V1.nonlin_input) at the first
  time step, the V1 response that all later IT responses depend on.
- Images are resized to 256 and centre-cropped to 224, and kept in file-name order
  (place1.jpg, place10.jpg, ..., place9.jpg).
- Random sets are drawn from all IT units with random.Random(random_seed), one sample per
  set in order, so every analysis uses the same random sets.
- TF32 is disabled on GPUs; it changes the gradients by up to about 10%.
- Eccentricity bands: 12 equal-width bands from the map centre to the corners. Each
  normalised map is averaged within each band; the four corner positions at exactly the
  maximum eccentricity fall outside the last band.

The gradient step backpropagates every IT unit separately (in batches of chunk_size), so
a full analysis (16 images x 101 unit sets x 1254 units) takes 2-3 hours on a laptop GPU.

Usage:
    python v1_gradient_analysis.py gradients --config configs/gradients/cornet_rt5_top5.yaml
    python v1_gradient_analysis.py gradients --config configs/gradients/cornet_rt5_top5.yaml \
        --only place_units_place_images
    python v1_gradient_analysis.py stats --config configs/gradients/cornet_rt5_top5.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils import (
    CATEGORIES,
    canonical_layer_name,
    category_images,
    get_module,
    gradient_analysis_name,
    load_config_jobs,
    load_model,
    load_selectivity_table,
    preprocess_image,
    select_top_units,
    simulation_seed,
)

V1_PATH = "module.V1.nonlin_input"
IT_PATH = "module.IT.output"
IT_SHAPE = (512, 7, 7)
CI_LEVEL = 0.95
REQUIRED_KEYS = ("model", "time_steps", "selectivity_table", "readout_layer", "top_fraction", "n_random_sets",
                 "random_seed", "image_dir", "gradient_dir", "stats_dir", "n_bands", "n_bootstrap", "seed",
                 "chunk_size", "units", "images")


def analysis_name(job: dict) -> str:
    """Return the folder name of one analysis, e.g. place_units_place_images."""
    return gradient_analysis_name(job["units"], job["images"])


def validate_job(job: dict) -> None:
    """Check that one analysis job has every setting and valid categories."""
    missing = [key for key in REQUIRED_KEYS if key not in job]
    if missing:
        raise ValueError(f"Gradient job {job} is missing {missing}.")
    for key in ("units", "images"):
        if job[key] not in CATEGORIES:
            raise ValueError(f"{key} must be one of {list(CATEGORIES)}; got {job[key]!r}.")


# ------------------------- Gradients -------------------------

def mean_abs_v1_gradients(model: torch.nn.Module, image: torch.Tensor, units: np.ndarray, chunk_size: int) -> np.ndarray:
    """
    Return the mean absolute gradient of IT units' responses with respect to V1 for one image.

    The image is repeated once per unit in a batch, and each copy backpropagates only its
    own unit, so one backward pass gives the gradients of a whole chunk of units.
    CORnet-RT normalises every sample separately (GroupNorm), so the copies do not interact.

    Parameters:
        model (torch.nn.Module): CORnet-RT in evaluation mode.
        image (torch.Tensor): Preprocessed image [1,3,224,224] on the model's device.
        units (np.ndarray): Flat IT unit indices into the [512,7,7] output.
        chunk_size (int): Units (image copies) per backward pass.

    Returns:
        np.ndarray: Mean |gradient| over the units, V1 feature map [C,H,W].
    """
    v1_outputs, it_outputs = [], []
    hooks = [get_module(model, V1_PATH).register_forward_hook(lambda _m, _i, out: v1_outputs.append(out)),
             get_module(model, IT_PATH).register_forward_hook(lambda _m, _i, out: it_outputs.append(out))]
    # [3, n_units] channel, row, and column of every unit in the IT output
    positions = torch.as_tensor(np.stack(np.unravel_index(units, IT_SHAPE)), device=image.device)
    total = None
    try:
        for start in range(0, len(units), chunk_size):
            stop = min(start + chunk_size, len(units))
            v1_outputs.clear()
            it_outputs.clear()
            model(image.expand(stop - start, -1, -1, -1).contiguous())
            channel, row, col = positions[:, start:stop]
            # Final time step's IT response; first time step's V1 response
            responses = it_outputs[-1][torch.arange(stop - start, device=image.device), channel, row, col]
            gradient, = torch.autograd.grad(responses.sum(), v1_outputs[0])
            chunk_sum = gradient.abs().sum(dim=0)
            total = chunk_sum if total is None else total + chunk_sum
    finally:
        for hook in hooks:
            hook.remove()
    return (total / len(units)).cpu().numpy()


def gradient_maps(model: torch.nn.Module, images: list[torch.Tensor], units: np.ndarray, chunk_size: int) -> np.ndarray:
    """Return mean_abs_v1_gradients() averaged over the V1 channels for every image, [N_img,H,W]."""
    return np.stack([mean_abs_v1_gradients(model, image, units, chunk_size).mean(axis=0) for image in images])


def run_gradients(job: dict) -> None:
    """
    Compute and save the selective and random-set gradient maps of one analysis.

    Parameters:
        job (dict): Merged analysis settings from the config.

    Returns:
        None
    """
    out_dir = Path(job["gradient_dir"]) / analysis_name(job)
    if (out_dir / "maps_random.npy").exists():
        print(f"{out_dir} is complete; skipped")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(job["model"], job["time_steps"], device)

    table = load_selectivity_table(job["selectivity_table"])
    selective = select_top_units(table, job["readout_layer"], job["units"], job["top_fraction"])
    readout_rows = table["layer"].map(canonical_layer_name) == canonical_layer_name(job["readout_layer"])
    all_units = table.loc[readout_rows, "unit"].astype(int).tolist()
    if len(all_units) != int(np.prod(IT_SHAPE)):
        raise ValueError(f"Expected {int(np.prod(IT_SHAPE))} IT units in {job['selectivity_table']}; got {len(all_units)}.")

    images = [preprocess_image(p, device, center_crop=True) for p in category_images(job["image_dir"], job["images"])]
    print(f"{analysis_name(job)}: {len(selective)} selective units, {len(images)} images, {device}")

    out_dir.mkdir(parents=True, exist_ok=True)
    maps_selective = gradient_maps(model, images, selective, job["chunk_size"])
    np.save(out_dir / "maps_selective.npy", maps_selective)

    sampler = random.Random(job["random_seed"])
    maps_random = np.empty((job["n_random_sets"], *maps_selective.shape), dtype=np.float32)
    for repeat in tqdm(range(job["n_random_sets"]), desc="random sets"):
        random_units = np.asarray(sampler.sample(all_units, len(selective)))
        maps_random[repeat] = gradient_maps(model, images, random_units, job["chunk_size"])
    np.save(out_dir / "maps_random.npy", maps_random)
    print(f"Saved {out_dir}")


# ------------------------- Statistics -------------------------

def normalised_maps(maps: np.ndarray) -> np.ndarray:
    """
    Return gradient maps divided by their own sum, so each [H,W] map sums to one.

    Parameters:
        maps (np.ndarray): Gradient maps [...,H,W], e.g. [N_img,H,W] or [N_sets,N_img,H,W].

    Returns:
        np.ndarray: Float64 maps of the same shape.
    """
    maps = np.asarray(maps, dtype=np.float64)
    if maps.ndim < 2:
        raise ValueError(f"maps must be [...,H,W]; got shape {maps.shape}.")
    totals = maps.sum(axis=(-2, -1), keepdims=True)
    if np.any(totals <= 0):
        empty = np.argwhere(totals[..., 0, 0] <= 0).tolist()
        raise ValueError(f"Gradient maps at leading indices {empty} are all zero and cannot be normalised.")
    return maps / totals


def eccentricity_bands(height: int, width: int, n_bands: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the eccentricity of every feature-map position and equal-width band edges.

    Eccentricity is the distance from the map centre in positions; the edges run from 0 to
    the corner distance.

    Returns:
        tuple[np.ndarray, np.ndarray]: Eccentricity [H,W] and band edges [n_bands+1].
    """
    rows, cols = np.mgrid[0:height, 0:width]
    eccentricity = np.sqrt((rows - (height - 1) / 2) ** 2 + (cols - (width - 1) / 2) ** 2)
    return eccentricity, np.linspace(0, eccentricity.max(), n_bands + 1)


def band_densities(maps: np.ndarray, eccentricity: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Return the area-normalised density of normalised gradient maps in each eccentricity band.

    Parameters:
        maps (np.ndarray): Normalised maps [...,H,W] from normalised_maps().
        eccentricity (np.ndarray): Eccentricity [H,W] from eccentricity_bands().
        edges (np.ndarray): Band edges [n_bands+1].

    Returns:
        np.ndarray: Densities [...,n_bands], the mean map value within each band.
    """
    if maps.ndim < 2 or maps.shape[-2:] != eccentricity.shape:
        raise ValueError(f"maps must be [...,H,W] with H,W = {eccentricity.shape}; got {maps.shape}.")
    flat = maps.reshape(*maps.shape[:-2], -1)
    # digitize puts the corner positions (eccentricity == last edge) past the last band
    band = np.digitize(eccentricity.ravel(), edges) - 1
    empty = [b for b in range(len(edges) - 1) if not np.any(band == b)]
    if empty:
        raise ValueError(f"Bands {empty} contain no positions of the {eccentricity.shape} map; use fewer bands.")
    return np.stack([flat[..., band == b].mean(axis=-1) for b in range(len(edges) - 1)], axis=-1)


def within_image_signs(selective: np.ndarray, random_sets: np.ndarray) -> np.ndarray:
    """
    Return sign(selective - random set) for every image, random set, and position.

    Parameters:
        selective (np.ndarray): Selective values [N_img,...], e.g. maps [N_img,H,W] or
            band densities [N_img,n_bands].
        random_sets (np.ndarray): Random-set values [N_sets,N_img,...] on the same images.

    Returns:
        np.ndarray: int8 signs [N_img,N_sets,X], with X the flattened trailing shape; ties are zero.
    """
    if random_sets.ndim != selective.ndim + 1 or random_sets.shape[1:] != selective.shape:
        raise ValueError(f"random_sets must be [N_sets, *selective.shape] with selective.shape = {selective.shape}; "
                         f"got {random_sets.shape}.")
    n_images = selective.shape[0]
    flat_selective = selective.reshape(n_images, 1, -1)
    flat_random = np.swapaxes(random_sets.reshape(random_sets.shape[0], n_images, -1), 0, 1)
    return np.sign(flat_selective - flat_random).astype(np.int8)


def crossed_bootstrap_delta(signs: np.ndarray, n_bootstrap: int, rng: np.random.Generator,
                            chunk_size: int = 50) -> np.ndarray:
    """
    Return bootstrap draws of the within-image Cliff's delta, resampling images and random sets.

    Each draw resamples the images and the random sets with replacement, independently, and
    averages the signs over the drawn pairs. Resampling uses multinomial counts, so a draw
    is a weighted mean of the signs.

    Parameters:
        signs (np.ndarray): Output of within_image_signs(), [N_img,N_sets,X].
        n_bootstrap (int): Number of draws, e.g. 10000.
        rng (np.random.Generator): Seeded generator.
        chunk_size (int): Draws computed together; memory grows with chunk_size * N_sets * X.

    Returns:
        np.ndarray: Draws [n_bootstrap,X].
    """
    if signs.ndim != 3:
        raise ValueError(f"signs must be [N_img,N_sets,X]; got shape {signs.shape}.")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be at least 1; got {n_bootstrap}.")
    n_images, n_sets, n_positions = signs.shape
    flat = signs.reshape(n_images, n_sets * n_positions).astype(np.float32)
    draws = np.empty((n_bootstrap, n_positions))
    for start in range(0, n_bootstrap, chunk_size):
        size = min(chunk_size, n_bootstrap - start)
        image_counts = rng.multinomial(n_images, np.full(n_images, 1.0 / n_images), size=size).astype(np.float32)
        set_counts = rng.multinomial(n_sets, np.full(n_sets, 1.0 / n_sets), size=size).astype(np.float32)
        # [size, N_sets, X]: signs summed over the drawn images, then weighted by the drawn sets
        by_set = (image_counts @ flat).reshape(size, n_sets, n_positions)
        draws[start:start + size] = np.einsum("brx,br->bx", by_set, set_counts) / (n_images * n_sets)
    return draws


def within_image_delta(selective: np.ndarray, random_sets: np.ndarray, n_bootstrap: int,
                       seed_label: str) -> dict[str, np.ndarray]:
    """
    Return the within-image Cliff's delta per position with its 95% bootstrap CI.

    Parameters:
        selective (np.ndarray): Normalised selective values [N_img,...].
        random_sets (np.ndarray): Normalised random-set values [N_sets,N_img,...].
        n_bootstrap (int): Bootstrap draws for the CI, e.g. 10000.
        seed_label (str): Label from which a stable seed is derived, e.g. "0|place_units_place_images".

    Returns:
        dict[str, np.ndarray]: With the trailing shape of ``selective``: cliffs_delta (mean
        over images of delta_i), ci_low, ci_high, mean_diff (selective minus mean random
        value, averaged over images), and image_delta [N_img,...] (delta_i of every image).
    """
    signs = within_image_signs(selective, random_sets)
    position_shape = selective.shape[1:]
    rng = np.random.default_rng(simulation_seed(seed_label))
    tail = (1.0 - CI_LEVEL) / 2.0
    ci_low, ci_high = np.quantile(crossed_bootstrap_delta(signs, n_bootstrap, rng), [tail, 1.0 - tail], axis=0)
    image_delta = signs.mean(axis=1, dtype=np.float64)
    return {
        "cliffs_delta": image_delta.mean(axis=0).reshape(position_shape),
        "ci_low": ci_low.reshape(position_shape),
        "ci_high": ci_high.reshape(position_shape),
        "mean_diff": (selective - random_sets.mean(axis=0)).mean(axis=0).reshape(position_shape),
        "image_delta": image_delta.reshape(selective.shape),
    }


def position_delta_table(result: dict[str, np.ndarray]) -> pd.DataFrame:
    """Return within_image_delta() of [N_img,H,W] maps as one row per V1 position (fy, fx)."""
    height, width = result["cliffs_delta"].shape
    table = pd.DataFrame({"fy": np.repeat(np.arange(height), width), "fx": np.tile(np.arange(width), height)})
    for column in ("mean_diff", "cliffs_delta", "ci_low", "ci_high"):
        table[column] = result[column].ravel()
    return table


def load_maps(gradient_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the normalised selective and random-set maps of one analysis.

    Parameters:
        gradient_dir (Path): Folder with maps_selective.npy and maps_random.npy.

    Returns:
        tuple[np.ndarray, np.ndarray]: Selective [N_img,H,W] and random [N_sets,N_img,H,W] maps.
    """
    selective = np.load(gradient_dir / "maps_selective.npy")
    random_sets = np.load(gradient_dir / "maps_random.npy")
    if selective.ndim != 3 or random_sets.shape[1:] != selective.shape:
        raise ValueError(f"{gradient_dir}: expected maps [N_img,H,W] and [N_sets,N_img,H,W]; "
                         f"got {selective.shape} and {random_sets.shape}.")
    return normalised_maps(selective), normalised_maps(random_sets)


def run_stats(jobs: list[dict]) -> None:
    """
    Write the position statistics of every analysis and the eccentricity bias table.

    Parameters:
        jobs (list[dict]): Analysis settings from the config.

    Returns:
        None
    """
    bias_rows = []
    for job in jobs:
        name = analysis_name(job)
        stats_dir = Path(job["stats_dir"])
        stats_dir.mkdir(parents=True, exist_ok=True)
        selective, random_sets = load_maps(Path(job["gradient_dir"]) / name)
        result = within_image_delta(selective, random_sets, job["n_bootstrap"], f"{job['seed']}|{name}")
        position_delta_table(result).to_csv(stats_dir / f"{name}_position_delta.csv", index=False)
        np.save(stats_dir / f"{name}_image_delta_maps.npy", result["image_delta"])

        if job["units"] == job["images"]:
            # Fig 2c compares each category's selective units on their own images
            eccentricity, edges = eccentricity_bands(*selective.shape[1:], job["n_bands"])
            bands = within_image_delta(band_densities(selective, eccentricity, edges),
                                       band_densities(random_sets, eccentricity, edges),
                                       job["n_bootstrap"], f"{job['seed']}|{job['units']}|bands")
            bias_rows.append(pd.DataFrame({
                "category": job["units"], "band": np.arange(job["n_bands"]),
                "band_center": (edges[:-1] + edges[1:]) / 2, "cliffs_delta": bands["cliffs_delta"],
                "ci_low": bands["ci_low"], "ci_high": bands["ci_high"], "mean_diff": bands["mean_diff"],
            }))
        print(f"Saved statistics of {name}")
    if bias_rows:
        pd.concat(bias_rows).to_csv(Path(jobs[0]["stats_dir"]) / "eccentricity_bias.csv", index=False)


def main() -> None:
    """Run the gradient or statistics step from the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["gradients", "stats"], help="Which step to run.")
    parser.add_argument("--config", required=True, help="YAML with a v1_gradients defaults/jobs section.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Gradient step only: analyses to run, e.g. place_units_place_images; default all.")
    args = parser.parse_args()

    jobs = load_config_jobs(args.config, "v1_gradients")
    for job in jobs:
        validate_job(job)
    if args.step == "stats":
        if args.only:
            raise ValueError("--only applies to the gradients step; stats always uses every analysis.")
        run_stats(jobs)
        return
    names = [analysis_name(job) for job in jobs]
    unknown = sorted(set(args.only or []) - set(names))
    if unknown:
        raise ValueError(f"--only {unknown} not in {args.config}; analyses are {names}.")
    for job, name in zip(jobs, names):
        if not args.only or name in args.only:
            run_gradients(job)


if __name__ == "__main__":
    main()

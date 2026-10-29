"""
Run damage simulations and save the readout-layer activations of every simulation.

One YAML config describes one simulated condition: a model, a type of damage applied to
one model area (or the whole model), a range of damage levels, and the number of
independent simulations per level (Methods 2.4). For every level and simulation, a fresh
copy of the intact model is damaged with a newly drawn random lesion or noise sample,
the 64 experimental stimuli are passed through it, and the readout-layer activations are
saved as float16 to

    data/simulations/<model_dir>/<damage_type>/<damaged_area>/activations/<readout_name>/damaged_<level>/<simulation>.npz

Damage types:
    connections                    sever a fraction of conv weights in each layer
    units                          delete a fraction of conv filters (weights and bias)
    noise                          add Gaussian noise scaled to each layer's weight SD
    eccentricity_weights_gradual   sever V1 connections with eccentricity-graded probability
    eccentricity_gradual           delete V1 activations with eccentricity-graded probability
The eccentricity types are saved as <type>_band<start>-<end>, e.g.
eccentricity_weights_gradual_band0.50-1.00.

Each simulation uses a seed derived from the condition (e.g. cornet_rt5_c/connections/V1),
damage level, and simulation index, so a rerun reproduces it exactly and conditions never
share lesions. Existing outputs are skipped, so an interrupted run
can be restarted. Afterwards, run compute_differentiation.py.

Usage:
    python main.py configs/simulations/cornet_rt5/connections_V1.yaml
    python main.py configs/simulations/cornet_rt5/connections_V1.yaml --simulations 2   # quick test
"""

import argparse
import copy
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from utils import (
    add_weight_noise,
    conv_layers_under,
    delete_activations_by_eccentricity,
    delete_units,
    list_stimuli,
    load_model,
    preprocess_image,
    record_readout,
    save_activations,
    sever_connections,
    sever_connections_by_eccentricity,
    simulation_seed,
    weight_standard_deviations,
)

SIMULATIONS_DIR = Path("data/simulations")
UNIFORM_DAMAGE = ("connections", "units", "noise")
ECCENTRICITY_DAMAGE = ("eccentricity_weights_gradual", "eccentricity_gradual")
REQUIRED_KEYS = ("model", "model_dir", "damage_type", "damaged_area", "levels", "simulations",
                 "readout_path", "readout_name")


def load_config(config_path: str | Path) -> dict:
    """
    Read and validate a simulation config.

    Parameters:
        config_path (str | Path): YAML file.

    Returns:
        dict: The config with defaults filled in.
    """
    with open(config_path, encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"{config_path} is missing {missing}.")
    allowed = UNIFORM_DAMAGE + ECCENTRICITY_DAMAGE
    if config["damage_type"] not in allowed:
        raise ValueError(f"damage_type must be one of {allowed}; got {config['damage_type']!r}.")
    if config["damage_type"] in UNIFORM_DAMAGE and not config.get("damage_paths"):
        raise ValueError(f"{config_path}: damage_type {config['damage_type']!r} needs damage_paths.")
    if config["damage_type"] in ECCENTRICITY_DAMAGE and "eccentricity" not in config:
        raise ValueError(f"{config_path}: damage_type {config['damage_type']!r} needs an eccentricity block.")
    if len(config["levels"]) != 3:
        raise ValueError(f"levels must be [start, count, step]; got {config['levels']!r}.")
    config.setdefault("image_dir", "stimuli")
    config.setdefault("include_bias", config["damage_type"] == "units")
    config.setdefault("seed_label", output_damage_dir(config).relative_to(SIMULATIONS_DIR).as_posix())
    return config


def damage_levels(levels: list) -> list[float]:
    """
    Return the damage levels of a [start, count, step] specification.

    Levels are applied unrounded, as in the original runs (int(fraction * n) can change
    at exact boundaries); folder names use the level rounded to three decimals.

    Example:
        [0, 21, 0.05] -> 0.0, 0.05, 0.1, 0.15000000000000002, ..., 1.0
    """
    start, count, step = levels
    return [start + i * step for i in range(int(count))]


def output_damage_dir(config: dict) -> Path:
    """Return data/simulations/<model_dir>/<damage_type folder>/<damaged_area>."""
    damage_folder = config["damage_type"]
    if config["damage_type"] in ECCENTRICITY_DAMAGE:
        r_start, r_end = config["eccentricity"]["band"]
        damage_folder = f"{damage_folder}_band{r_start:.2f}-{r_end:.2f}"
    return SIMULATIONS_DIR / config["model_dir"] / damage_folder / config["damaged_area"]


def apply_damage(model: torch.nn.Module, config: dict, level: float, generator: torch.Generator,
                 weight_sds: dict[str, float] | None) -> None:
    """
    Damage a model copy in place at one damage level.

    Parameters:
        model (torch.nn.Module): Intact model copy.
        config (dict): Simulation config.
        level (float): Damage level (fraction, noise SD multiple, or maximum deletion probability).
        generator (torch.Generator): Seeded generator on the model's device.
        weight_sds (dict[str, float] | None): Intact weight SDs, needed for noise.

    Returns:
        None
    """
    damage_type = config["damage_type"]
    if damage_type in UNIFORM_DAMAGE:
        layers = conv_layers_under(model, config["damage_paths"])
        if damage_type == "connections":
            sever_connections(layers, level, generator)
        elif damage_type == "units":
            delete_units(layers, level, generator, include_bias=config["include_bias"])
        else:
            add_weight_noise(layers, level, weight_sds, generator)
        return

    eccentricity = config["eccentricity"]
    r_start, r_end = eccentricity["band"]
    damage = (sever_connections_by_eccentricity if damage_type == "eccentricity_weights_gradual"
              else delete_activations_by_eccentricity)
    extra = {"include_bias": config["include_bias"]} if damage_type == "eccentricity_weights_gradual" else {}
    damage(model, eccentricity["layer_path"], r_start, r_end, level, generator,
           reverse=eccentricity.get("reverse", False), per_channel=eccentricity.get("per_channel", True), **extra)


def run_simulations(config: dict, n_simulations: int | None = None) -> None:
    """
    Run every damage level and simulation of one config and save the activations.

    Parameters:
        config (dict): Validated config from load_config().
        n_simulations (int | None): Override the number of simulations (e.g. for a quick test).

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    intact = load_model(config["model"], config.get("time_steps"), device)
    weight_sds = None
    if config["damage_type"] == "noise":
        weight_sds = weight_standard_deviations(conv_layers_under(intact, config["damage_paths"]))

    image_paths = list_stimuli(config["image_dir"])
    image_names = [p.name for p in image_paths]
    images = torch.cat([preprocess_image(p) for p in image_paths]).to(device)
    activation_root = output_damage_dir(config) / "activations" / config["readout_name"]
    n_simulations = n_simulations or int(config["simulations"])

    plan = [(level, i) for level in damage_levels(config["levels"]) for i in range(n_simulations)
            if not (activation_root / f"damaged_{round(level, 3)}" / f"{i}.npz").exists()]
    print(f"{activation_root}: {len(plan)} simulations to run")
    for level, simulation in tqdm(plan):
        generator = torch.Generator(device=device)
        generator.manual_seed(simulation_seed(config["seed_label"], round(level, 3), simulation))
        model = copy.deepcopy(intact)
        apply_damage(model, config, level, generator, weight_sds)
        readout = record_readout(model, config["readout_path"])
        with torch.no_grad():
            model(images)
        activations = readout["output"].reshape(len(image_names), -1)
        save_activations(activation_root / f"damaged_{round(level, 3)}" / f"{simulation}.npz", activations, image_names)
        del model


def main() -> None:
    """Run the simulations of one config from the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Simulation config YAML.")
    parser.add_argument("--simulations", type=int, default=None,
                        help="Run this many simulations per level instead of the config's value.")
    args = parser.parse_args()
    run_simulations(load_config(args.config), args.simulations)


if __name__ == "__main__":
    main()

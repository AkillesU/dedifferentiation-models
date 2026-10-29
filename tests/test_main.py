"""Tests for main.py: config validation, damage levels, and output naming (the simulations themselves are not run)."""

import pytest
import yaml

import main

BASE_CONFIG = {
    "model": "cornet_rt", "time_steps": 5, "model_dir": "cornet_rt5_c", "damage_type": "connections",
    "damaged_area": "V1", "damage_paths": ["module.V1"], "levels": [0, 21, 0.05], "simulations": 200,
    "readout_path": "module.IT.output", "readout_name": "IT",
}


def write_config(tmp_path, **changes):
    """Write BASE_CONFIG with changes (None removes a key) and return its path."""
    config = {**BASE_CONFIG, **changes}
    config = {k: v for k, v in config.items() if v is not None}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_load_config_fills_defaults(tmp_path):
    config = main.load_config(write_config(tmp_path))
    assert config["image_dir"] == "stimuli" and config["include_bias"] is False
    assert config["seed_label"] == "cornet_rt5_c/connections/V1"
    assert main.load_config(write_config(tmp_path, damage_type="units"))["include_bias"] is True


@pytest.mark.parametrize("changes, message", [
    ({"readout_path": None}, "missing"),
    ({"damage_type": "lesion"}, "damage_type must be one of"),
    ({"damage_paths": None}, "needs damage_paths"),
    ({"damage_type": "eccentricity_gradual"}, "needs an eccentricity block"),
    ({"levels": [0, 21]}, r"levels must be \[start, count, step\]"),
])
def test_load_config_rejects_invalid(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        main.load_config(write_config(tmp_path, **changes))


def test_damage_levels_are_unrounded():
    levels = main.damage_levels([0, 21, 0.05])
    assert len(levels) == 21 and levels[3] == 3 * 0.05 and round(levels[-1], 3) == 1.0


def test_output_damage_dir_names_eccentricity_bands():
    config = {**BASE_CONFIG, "model_dir": "cornet_rt5_all", "damage_type": "eccentricity_weights_gradual",
              "eccentricity": {"band": [0.5, 1.0]}}
    assert main.output_damage_dir(config).as_posix() == \
        "data/simulations/cornet_rt5_all/eccentricity_weights_gradual_band0.50-1.00/V1"


def test_simulation_configs_are_valid():
    from pathlib import Path
    paths = sorted(Path("configs/simulations").rglob("*.yaml"))
    for path in paths:
        main.load_config(path)

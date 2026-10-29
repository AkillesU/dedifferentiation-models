"""Tests for unit_categ_selectivity.py: localizer images, Mann-Whitney selectivity, activations, and configs (no pretrained models)."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

import unit_categ_selectivity as selectivity

CONFIGS = sorted(Path("configs/selectivity").glob("*.yaml"))
LABELS = ["animals"] * 20 + ["faces"] * 20 + ["objects"] * 20 + ["places"] * 20


def test_localizer_images_has_20_images_per_category():
    paths, labels = selectivity.localizer_images(Path("categ_images"))
    assert len(paths) == 80
    assert labels == LABELS


def test_localizer_images_skips_non_images_and_rejects_empty_folder(tmp_path):
    (tmp_path / "faces").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "faces" / "b.jpg")
    Image.new("RGB", (8, 8)).save(tmp_path / "faces" / "a.png")
    (tmp_path / "faces" / "notes.txt").write_text("not an image")
    paths, labels = selectivity.localizer_images(tmp_path)
    assert [p.name for p in paths] == ["a.png", "b.jpg"] and labels == ["faces", "faces"]
    with pytest.raises(FileNotFoundError):
        selectivity.localizer_images(tmp_path / "faces")


def test_mann_whitney_selectivity_scores_a_face_unit_and_a_flat_unit():
    activations = np.zeros((80, 2))
    activations[20:40, 0] = 1.0  # unit 0 responds only to faces; unit 1 is constant
    statistics = selectivity.mann_whitney_selectivity(activations, LABELS)
    assert set(statistics) == {"animals", "faces", "objects", "places"}
    # U of 20 target images against 60 others: 20 x 60 when every target response is larger, half that for ties
    assert statistics["faces"][0] == 1200
    assert statistics["places"][1] == 600


def test_layer_activations_flattens_the_hooked_layer(tmp_path):
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 2, kernel_size=1), torch.nn.ReLU()).eval()
    image_paths = []
    for i in range(3):
        image_paths.append(tmp_path / f"{i}.png")
        Image.new("RGB", (300, 260), color=(40 * i, 0, 0)).save(image_paths[-1])
    activations = selectivity.layer_activations(model, ["1"], image_paths)
    # Resize(256) + CenterCrop(224) -> 2 channels x 224 x 224 units per image
    assert activations["1"].shape == (3, 2 * 224 * 224)
    assert (activations["1"] >= 0).all()


def test_run_config_rejects_missing_keys(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("model: alexnet\n")
    with pytest.raises(ValueError, match="missing"):
        selectivity.run_config(config_path)


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.stem)
def test_selectivity_configs_name_their_table_after_the_config(config_path):
    with open(config_path, encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    assert {"model", "layer_paths", "output_name"} <= set(config)
    assert config["output_name"] == f"{config_path.stem}_unit_selectivity"

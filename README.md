# dedifferentiation-models

Code for the manuscript *A Mechanistic Account of Dedifferentiation in the Aging Visual System*.

Pretrained deep neural networks (CORnet-RT, CORnet-S, AlexNet, VGG16) are damaged by
connection severing, neuron deletion, or weight noise, and the category differentiation of
their category-selective readout units is compared with the dedifferentiation measured in
older adults by Haupt et al. (2025). The simulation outputs are on Zenodo
([10.5281/zenodo.22864335](https://doi.org/10.5281/zenodo.22864335)), so every figure and
table can be redrawn without rerunning the simulations.

## Repository structure

```
main.py                         damage simulations -> readout activations (Methods 2.4)
compute_differentiation.py      activations -> selective RDMs -> differentiation tables
unit_categ_selectivity.py       category selectivity of every readout unit (Methods 2.2)
layer_category_selectivity.py   choice of the readout layer (Methods 2.2)
fit_haupt_damage_levels.py      comparison with older adults, human-matched levels (Table S1)
place_alignment_stats.py        place specificity index and progress slopes (Table S2, Fig 1d, S3)
v1_gradient_analysis.py         IT-to-V1 gradient maps and eccentricity bias (Fig 2b, 2c, S4)
run_plots.py                    draws the figures listed in a figure config
utils.py                        shared functions (RDMs, models, damage, statistics, plotting)
download_stimuli.py             fetches the 64 experimental stimuli from OSF
download_data.py                fetches the data release from Zenodo
configs/
    simulations/<model>/        one config per simulated condition (73)
    selectivity/                localizer settings and candidate readout layers per model
    stats/                      Haupt comparison (Tables S1, S2)
    gradients/                  IT-to-V1 gradient analysis
    figures/                    figure_1.yaml, figure_2.yaml, supplementary.yaml
categ_images/                   80 localizer images (Unsplash), 20 per category
stimuli/                        MANIFEST.csv; the stimuli themselves are downloaded
data/reference/haupt_2025/      human reference data from Haupt et al. (2025)
tests/                          unit tests (pytest)
```

Created by the scripts or the downloads (not in git): `data/simulations/`,
`data/differentiation/`, `data/it2v1_gradients/`, `unit_selectivity/`, `stats/`, `plots/`,
`downloads/`.

## Installation

```bash
conda env create -f environment.yml
conda activate dedifferentiation
```

This installs Python 3.10 and the pinned packages in `requirements.txt`, including the
official CORnet implementation. On Linux, pip installs the CUDA build of PyTorch. On Windows
and macOS it installs the CPU build, which is enough for the statistics and figures. For a
GPU on Windows, replace it afterwards:

```bash
pip install --force-reinstall torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

Run the tests with `python -m pytest -q`.

All commands below are run from the repository root.

## Quick start: figures and tables from the data release

```bash
python download_stimuli.py      # 64 Haupt et al. stimuli from OSF (Fig 2b overlays)
python download_data.py         # core data release (~0.4 GB), checked against its SHA-256 hashes
python run_plots.py configs/figures/figure_1.yaml
python run_plots.py configs/figures/figure_2.yaml
python run_plots.py configs/figures/supplementary.yaml
```

The core release already contains the statistics outputs in `stats/`. To recompute them from
the differentiation tables and gradient arrays (about 15-20 minutes in total):

```bash
python fit_haupt_damage_levels.py --config configs/stats/cornet_rt5_haupt.yaml
python place_alignment_stats.py --config configs/stats/cornet_rt5_haupt.yaml
python v1_gradient_analysis.py stats --config configs/gradients/cornet_rt5_top5.yaml
```

Figures are saved to `plots/<config name>/` as PNG and SVG, each with a CSV of the plotted values.

## Figures and tables

| Item | Command | Reads |
|---|---|---|
| Fig 1b, 1c, 1d | `run_plots.py configs/figures/figure_1.yaml` | `data/differentiation/cornet_rt5_c`, `stats/haupt_fits`, `stats/place_alignment` |
| Fig 2b, 2c | `v1_gradient_analysis.py stats`, then `run_plots.py configs/figures/figure_2.yaml` | `data/it2v1_gradients`, `stats/it2v1`, `stimuli/` |
| Fig 2e | `run_plots.py configs/figures/figure_2.yaml` | `data/differentiation/cornet_rt5_all` |
| Fig S1 | `run_plots.py configs/figures/supplementary.yaml --only figure_S1_damage_groups` | CORnet-RT (4 steps), CORnet-S, AlexNet, VGG16 tables |
| Fig S2, S7 | `run_plots.py configs/figures/supplementary.yaml` | `cornet_rt5_c`, `cornet_rt5_c+b` |
| Fig S3 | `place_alignment_stats.py`, then `run_plots.py configs/figures/supplementary.yaml` | `stats/place_alignment` |
| Fig S4 | `v1_gradient_analysis.py stats`, then `run_plots.py configs/figures/supplementary.yaml` | `stats/it2v1` |
| Fig S5, S6, S8 | `run_plots.py configs/figures/supplementary.yaml` | `cornet_rt5_all` |
| Table S1 | `fit_haupt_damage_levels.py --config configs/stats/cornet_rt5_haupt.yaml` | `stats/haupt_fits/cornet-rt_table-S1_levels.csv`, `..._human.csv` |
| Table S2 | `place_alignment_stats.py --config configs/stats/cornet_rt5_haupt.yaml` | `stats/place_alignment/cornet-rt_table-S2.csv` |
| Readout layers (Methods 2.2) | `layer_category_selectivity.py configs/selectivity/*.yaml` | `categ_images/` -> `stats/layer_choice/` |
| Selective units (Methods 2.2) | `unit_categ_selectivity.py configs/selectivity/*.yaml` | `categ_images/` -> `unit_selectivity/` |

Figures 1a, 2a and 2d are schematics drawn by hand.

## Full pipeline: from the pretrained models

The simulations (73 conditions x 21 or 31 damage levels x 200 simulations) need a GPU or substantial CPU compute time.

1. **Readout layer and selective units** (CPU is enough). `layer_category_selectivity.py`
   scores every candidate layer on the 80 localizer images; `unit_categ_selectivity.py`
   writes the selectivity table of the chosen layer.

   ```bash
   python layer_category_selectivity.py configs/selectivity/*.yaml
   python unit_categ_selectivity.py configs/selectivity/*.yaml
   ```

   The data release contains the selectivity tables that the published simulations used
   (see Notes); `download_data.py` puts them in `unit_selectivity/`.

2. **Simulations.** One config per condition; 200 simulations per damage level.

   ```bash
   python main.py configs/simulations/cornet_rt5/connections_V1.yaml --simulations 2   # quick test
   for config in configs/simulations/*/*.yaml; do python main.py "$config"; done
   ```

3. **Selective RDMs and differentiation tables.**

   ```bash
   python compute_differentiation.py rdms --model-dir cornet_rt5_c --readout IT \
       --selectivity-table unit_selectivity/cornet_rt5_unit_selectivity.csv.gz --top-fractions 0.01 0.05 0.10
   python compute_differentiation.py table
   ```

   | `--model-dir` | `--readout` | `--selectivity-table` | `--top-fractions` |
   |---|---|---|---|
   | `cornet_rt5_c`, `cornet_rt5_c+b` | `IT` | `cornet_rt5_unit_selectivity` | `0.01 0.05 0.10` |
   | `cornet_rt5_all` | `IT` | `cornet_rt5_unit_selectivity` | `0.05` |
   | `cornet_rt4_c`, `cornet_rt4_c+b` | `IT` | `cornet_rt4_unit_selectivity` | `0.05` |
   | `cornet_s4_c`, `cornet_s4_c+b` | `IT` | `cornet_s_unit_selectivity` | `0.05` |
   | `alexnet_c`, `alexnet_c+b` | `classifier_2` | `alexnet_unit_selectivity` | `0.05` |
   | `vgg16_c`, `vgg16_c+b` | `classifier_4` | `vgg16_unit_selectivity` | `0.05` |

   To start from the published RDMs instead, download the RDM parts
   (`python download_data.py --rdms all`, ~27 GB, or only the parts a figure needs; see
   `python download_data.py --list`) and run only the `table` step.

4. **IT-to-V1 gradients**, then the statistics and figures
   as in the quick start.

   ```bash
   python v1_gradient_analysis.py gradients --config configs/gradients/cornet_rt5_top5.yaml
   ```

## Notes

- **Model folders.** `cornet_rt5_*` is CORnet-RT with 5 time steps, the main model;
  `cornet_rt4_*` CORnet-RT with 4 time steps; `cornet_s4_*` the standard CORnet-S. The
  suffix `_c` holds connection severing and weight noise, `_c+b` neuron deletion (weights
  and biases), and `cornet_rt5_all` the eccentricity-dependent V1 damage.
- **CORnet-S.** Its within-block recurrence is fixed by the architecture, so the
  simulation configs take no time-step setting. Damage is applied to the block
  convolutions only; the learned skip projections are left intact.
- **Readout of CORnet-RT (4 time steps).** The layer-choice criterion scores V4 slightly
  higher than IT for this model (0.677 vs 0.647). IT is used, as for the 5-time-step model;
  `layer_category_selectivity.py` reports the difference.
- **Hardware rounding.** Rerunning on different hardware (e.g., GPU vs. CPU) can change RDM values slightly.
- **Long paths on Windows.** The RDM files sit up to 134 characters below the repository root.
  Unless Windows long paths are enabled, keep the path of the repository folder under about
  115 characters (e.g., `C:\Users\<name>\Documents\dedifferentiation-models`), or unpacking
  the RDM zips fails with "The filename or extension is too long".

## Licence and citation

Code: MIT (`LICENSE`). Data release: CC-BY 4.0. The localizer images in `categ_images/` are
from Unsplash (Unsplash licence). The experimental stimuli are not redistributed; they are
downloaded from the OSF project of Haupt et al. (2025), <https://osf.io/xeukw/>.

Haupt, M., Garrett, D. D., & Cichy, R. M. (2025). Healthy aging delays and dedifferentiates
high-level visual representations. *Current Biology*, 35, 2112-2127.e6.
https://doi.org/10.1016/j.cub.2025.03.062

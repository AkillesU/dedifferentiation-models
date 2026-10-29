# Haupt et al. (2025) Table S5 reference data

These files transcribe and derive the pattern-similarity values in supplementary Table S5 of:

Haupt, M., Garrett, D. D., & Cichy, R. M. (2025). *Healthy aging delays and dedifferentiates high-level visual representations*. Current Biology, 35, 2112-2127.e6. https://doi.org/10.1016/j.cub.2025.03.062

Source: https://pure.mpg.de/rest/items/item_3621884/component/file_3662071/content

## Files

Read by the code (`fit_haupt_damage_levels.py`, `place_alignment_stats.py`):

- `haupt2025_fig6B_derived.csv`: participant-level differentiation for each ROI comparison and age group, digitised from Figure 6B. `value_moment_matched` rescales the digitised values towards the Table S5 means. `dot_rank` orders the dots within one comparison and group; it does not identify participants. This is the human input to the comparison with the models.
- `table_s5_raw.csv`: all 27 visible Table S5 rows. The `*_reported` fields preserve the printed values; the final numeric fields are analysis-ready parses. Used to centre each digitised comparison-by-group cell on its published mean (`center_on_table_s5`).

Reference only (the derivations behind the human targets quoted below):

- `table_s5_derived_comparisons.csv`: the same 27 rows with younger-minus-older, retained proportion, and relative drop added.
- `table_s5_summary_all_variants.csv`: category/ROI and whole-model summaries for all three analysis variants.
- `table_s5_primary_analysis_targets.csv`: the four rows of the main comparison: face/FFA, object/LOC, place/PPA, and the whole-model aggregate.
- `metadata.csv`: source, definition, analysis selection, and sample metadata.

## Primary analysis choice

Use `normalized_all_runs`, corresponding to normalized-space fMRI patterns averaged across all 20 task runs. This is the analysis shown in the paper's main Figure 6B. The authors state that all 20 runs were used there to maximize signal-to-noise ratio. The two-run normalized and all-run native-space analyses are robustness checks shown in supplementary Figure S4 and detailed in Table S5.

## Measures and aggregation

Table S5 reports distinctiveness as within-category minus between-category pattern similarity for each ROI's preferred category against each nonpreferred category. The mapping is FFA to face, LOC to object, and PPA to place.

Three whole-model percentage summaries are retained because they answer different questions:

- `relative_drop_percent_ratio_of_means`: pool the nine absolute D values within each age group, then calculate the percentage drop. For the primary analysis this is 43.249%.
- `mean_pairwise_relative_drop_percent`: calculate the retained/drop percentage for every published comparison and average those percentages. For the primary analysis this is 47.418%.
- `mean_category_relative_drop_percent_ratio_of_means`: calculate one ratio-of-means percentage within each of the three ROIs/categories and average those three percentages. For the primary analysis this is 47.133%.

For direct comparison with the models' scaled category differentiation (percentage of the intact value), use the three category rows and their `retained_percent_ratio_of_means`. If an overall model statistic is an equal mean of the three category retained percentages, compare it with the equal mean of the three primary category targets (52.867% retained, 47.133% drop). That equal-category target is not identical to either nine-row Table S5 aggregate and should be labeled explicitly when used.

The Table S5 p-values test age-group differences for individual ROI/category comparisons. They do not provide a p-value for any derived category or whole-model aggregate.

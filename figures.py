import os
from constants import (
    _individual_variables,
    _individual_nb_variables,
    _group_variables,
    GENERATED_FIGURES_PATH,
)
from stats import PAIRS_OF_GROUPS
from permutation_stats import permutation_test
import numpy as np

TR_INDIVS_BOXPLOT_MEAN_STAT_MEAN = {
    "save_path": os.path.join(
        GENERATED_FIGURES_PATH, "tr_indivs_boxplot_mean"
    ),
    "extensions": ["pdf", "png"],
    "data_variables_group": "tr_indivs",
    "data_filters": [
        lambda x: x["experiment_type"] == 1,
        lambda x: ~x["gene"].str.contains("srrm"),
    ],
    "variables": [
        "distance_to_origin",
        "speed",
        "acceleration",
        "abs_normal_acceleration",
        "abs_tg_acceleration",
        "distance_travelled",
    ],
    "agg_rule": {
        var_["name"]: "mean" if var_["name"] != "distance_travelled" else "max"
        for var_ in _individual_variables
    },
    "groupby_cols": [
        "trial_uid",
        "identity",
        "genotype_group",
        "genotype",
        "line",
        "line_replicate",
    ],
    "partition_column": "line_replicate",
    "boxplot_kwargs": {"x": "genotype_group", "hue": "genotype"},
    "pairs_of_groups_for_stats": PAIRS_OF_GROUPS,
    "stats_kwargs": {
        "test_func": permutation_test,
        "test_func_kwargs": {
            "repetitions": 10000,
            "stat_func": np.mean,
            "paired": False,
            "two_sided": False,
        },
    },
}
# TR_INDIVS_NB_BOXPLOT_MEAN_STAT_MEAN = {
#     "tr_indivs_nb": {
#         "variables_for_boxplots": [
#             "nb_position_x",
#             "nb_position_y",
#             "nb_angle",
#             "nb_cos_angle",
#         ],
#         "groupby_cols": [
#             "trial_uid",
#             "identity",
#             "genotype_group",
#             "genotype",
#             "line",
#             "line_replicate",
#         ],
#         "agg_rule": {
#             var_["name"]: "mean" for var_ in _individual_nb_variables
#         },
#     },
# }
# TR_GROUP_BOXPLOT_MEAN_STAT_MEAN = {
#     "tr_group": {
#         "groupby_cols": [
#             "trial_uid",
#             "genotype_group",
#             "line",
#             "line_replicate",
#         ],
#         "agg_rule": {var_["name"]: "mean" for var_ in _group_variables},
#     },
# }

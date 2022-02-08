import os
from constants import (
    _individual_variables,
    _individual_nb_variables,
    _group_variables,
    GENERATED_FIGURES_PATH,
)
from stats import PAIRS_OF_GROUPS

# Consider using http://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/
from permutation_stats import permutation_test
import numpy as np

TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEAN_STAT_MEAN = {
    "save_path": os.path.join(
        GENERATED_FIGURES_PATH, "boxplot_line_replicate_mean_stat_tr_indivs"
    ),
    "file_name": "plot",
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
    "rows_partitioned_by": "line_replicate",
    "boxplot_kwargs": {
        "x": "genotype_group",
        "hue": "genotype",
        "whis": 1.5,
        "palette": {"WT": "g", "HET": "b", "DEL": "r"},
    },
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

TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEAN_STAT_MEAN_BL = (
    TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEAN_STAT_MEAN.copy()
)
TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEAN_STAT_MEAN_BL.update(
    {
        "save_path": os.path.join(
            GENERATED_FIGURES_PATH,
            "boxplot_line_replicate_mean_stat_tr_indivs_bl",
        ),
        "data_variables_group": "tr_indivs_bl",
    }
)

TR_INDIVS_BOXPLOT_LINE_MEAN_STAT_MEAN = (
    TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEAN_STAT_MEAN.copy()
)
TR_INDIVS_BOXPLOT_LINE_MEAN_STAT_MEAN.update(
    {
        "save_path": os.path.join(
            GENERATED_FIGURES_PATH,
            "boxplot_line_mean_stat_tr_indivs",
        ),
        "rows_partitioned_by": "line",
    }
)


TR_INDIVS_BOXPLOT_LINE_MEAN_STAT_MEAN_BL = (
    TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEAN_STAT_MEAN.copy()
)
TR_INDIVS_BOXPLOT_LINE_MEAN_STAT_MEAN_BL.update(
    {
        "save_path": os.path.join(
            GENERATED_FIGURES_PATH,
            "boxplot_line_mean_stat_tr_indivs_bl",
        ),
        "data_variables_group": "tr_indivs_bl",
        "rows_partitioned_by": "line",
    }
)

TR_INDIVS_BOXPLOT_LINE_REPLICATE_STAT_MEAN_ALL = {
    "save_path": os.path.join(
        GENERATED_FIGURES_PATH,
        "boxplot_line_replicate_mean_stat_tr_indivs_all",
    ),
    "file_name": "plot",
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
    "rows_partitioned_by": "variables",
    "boxplot_kwargs": {
        "x": "genotype_group",
        "hue": "genotype_group_genotype",
        "whis": 1.5,
        "palette": {
            "HET_HET-HET": "b",
            "HET_DEL-HET": "g",
            "HET_DEL-DEL": "y",
            "DEL_DEL-DEL": "r",
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

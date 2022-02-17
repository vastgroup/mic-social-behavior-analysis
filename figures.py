import os

import numpy as np
from mlxtend.evaluate import permutation_test

from constants import (
    GENERATED_FIGURES_PATH,
    _group_variables,
    _individual_nb_variables,
    _individual_variables,
)
from stats import MEAN_STATS_KWARGS, MEDIAN_STATS_KWARGS, PAIRS_OF_GROUPS

# Agg rule
mean_agg_rule_tr_indivs = {
    var_["name"]: "mean" if var_["name"] != "distance_travelled" else "max"
    for var_ in _individual_variables
}
mean_agg_rule_tr_group = {var_["name"]: "mean" for var_ in _group_variables}
mean_agg_rule_tr_indiv_nb = {
    var_["name"]: "mean" for var_ in _individual_nb_variables
}  # Add circmean and circabs
median_agg_rule_tr_indivs = {
    var_["name"]: "median" if var_["name"] != "distance_travelled" else "max"
    for var_ in _individual_variables
}

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
        "normed_distance_to_origin",
        "speed",
        # "acceleration",
        # "abs_normal_acceleration",
        # "abs_tg_acceleration",
        # "distance_travelled",
    ],
    "agg_rule": mean_agg_rule_tr_indivs,
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
        "test_func_kwargs": MEAN_STATS_KWARGS,
    },
}


TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEDIAN_STAT_MEDIAN = (
    TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEAN_STAT_MEAN.copy()
)
TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEDIAN_STAT_MEDIAN.update(
    {
        "save_path": os.path.join(
            GENERATED_FIGURES_PATH,
            "boxplot_line_replicate_median_stat_tr_indivs",
        ),
        "agg_rule": median_agg_rule_tr_indivs,
        "stats_kwargs": {
            "test_func": permutation_test,
            "test_func_kwargs": MEDIAN_STATS_KWARGS,
        },
    }
)

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

TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEDIAN_STAT_MEDIAN_BL = (
    TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEAN_STAT_MEAN_BL.copy()
)
TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEDIAN_STAT_MEDIAN_BL.update(
    {
        "save_path": os.path.join(
            GENERATED_FIGURES_PATH,
            "boxplot_line_replicate_median_stat_tr_indivs_bl",
        ),
        "agg_rule": median_agg_rule_tr_indivs,
        "stats_kwargs": {
            "test_func": permutation_test,
            "test_func_kwargs": MEDIAN_STATS_KWARGS,
        },
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


TR_INDIVS_BOXPLOT_LINE_MEDIAN_STAT_MEDIAN = (
    TR_INDIVS_BOXPLOT_LINE_MEAN_STAT_MEAN.copy()
)
TR_INDIVS_BOXPLOT_LINE_MEDIAN_STAT_MEDIAN.update(
    {
        "save_path": os.path.join(
            GENERATED_FIGURES_PATH,
            "boxplot_line_median_stat_tr_indivs",
        ),
        "agg_rule": median_agg_rule_tr_indivs,
        "stats_kwargs": {
            "test_func": permutation_test,
            "test_func_kwargs": MEDIAN_STATS_KWARGS,
        },
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

TR_INDIVS_BOXPLOT_LINE_MEDIAN_STAT_MEDIAN_BL = (
    TR_INDIVS_BOXPLOT_LINE_MEAN_STAT_MEAN_BL.copy()
)
TR_INDIVS_BOXPLOT_LINE_MEDIAN_STAT_MEDIAN_BL.update(
    {
        "save_path": os.path.join(
            GENERATED_FIGURES_PATH,
            "boxplot_line_median_stat_tr_indivs_bl",
        ),
        "agg_rule": median_agg_rule_tr_indivs,
        "stats_kwargs": {
            "test_func": permutation_test,
            "test_func_kwargs": MEDIAN_STATS_KWARGS,
        },
    }
)

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
TR_GROUP_BOXPLOT_MEAN_STAT_MEAN = {
    "tr_group": {
        "groupby_cols": [
            "trial_uid",
            "genotype_group",
            "line",
            "line_replicate",
        ],
        "agg_rule": {var_["name"]: "mean" for var_ in _group_variables},
    },
}

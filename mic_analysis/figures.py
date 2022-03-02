import os

import numpy as np
from mlxtend.evaluate import permutation_test
from confapp import conf

from .constants import _group_varialbes_enhanced_names,

from .stats import conf.MEAN_STATS_KWARGS, conf.PAIRS_OF_GROUPS




TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEAN_STAT_MEAN_BL = {
    "save_path": os.path.join(
        conf.conf.GENERATED_FIGURES_PATH, "boxplot_line_replicate_mean_stat_tr_indivs_bl"
    ),
    "file_name": "plot",
    "extensions": ["pdf", "png"],
    "data_variables_group": "tr_indivs_bl",
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
        "x": "genotype_group_genotype",
        # "hue": "genotype_group_genotype",
        "whis": 1.5,
        "palette": conf.conf.COLORS,
    },
    "pairs_of_groups_for_stats": conf.PAIRS_OF_GROUPS,
    "stats_kwargs": {
        "test_func": permutation_test,
        "test_func_kwargs": conf.MEAN_STATS_KWARGS,
    },
}


TR_INDIVS_BOXPLOT_LINE_MEAN_STAT_MEAN_BL = (
    TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEAN_STAT_MEAN_BL.copy()
)
TR_INDIVS_BOXPLOT_LINE_MEAN_STAT_MEAN_BL.update(
    {
        "save_path": os.path.join(
            conf.conf.GENERATED_FIGURES_PATH,
            "boxplot_line_mean_stat_tr_indivs_bl",
        ),
        "rows_partitioned_by": "line",
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
        "agg_rule": {var_: "mean" for var_ in _group_varialbes_enhanced_names},
    },
}

import logging

import numpy as np
import pandas as pd

from .utils import circmean, circstd, ratio_in_back, ratio_in_front
from .variables import (
    _group_varialbes_enhanced_names,
    _individual_nb_variables_enhanced_names,
    _individual_variables_enhanced_names,
    all_variables_names,
)

logger = logging.getLogger(__name__)

# Agg rules for stats
mean_agg_rule_tr_indivs = {
    var_: ["median", "mean", "std"]
    if not "distance_travelled" in var_
    else "max"
    for var_ in _individual_variables_enhanced_names
}
mean_agg_rule_tr_group = {
    var_: ["median", "mean", "std"] for var_ in _group_varialbes_enhanced_names
}
mean_agg_rule_tr_indiv_nb = {
    var_: ["median", "mean", "std"]
    if not "nb_angle" in var_
    else [circmean, circstd, ratio_in_front]
    for var_ in _individual_nb_variables_enhanced_names
}


# Groupby stats constants

indiv_agg_stasts_kwargs = {
    "groupby": [
        "trial_uid",
        "identity",
        "genotype_group",
        "genotype",
        "line",
        "line_replicate",
        "line_experiment",
        "line_replicate_experiment",
        "experiment_type",
    ],
    "agg_rule": mean_agg_rule_tr_indivs,
}

group_agg_stasts_kwargs = {
    "groupby": [
        "trial_uid",
        "genotype_group",
        "line",
        "line_replicate",
        "line_experiment",
        "line_replicate_experiment",
        "experiment_type",
    ],
    "agg_rule": mean_agg_rule_tr_group,
}


indiv_nb_agg_stats_kwargs = {
    "groupby": [
        "trial_uid",
        "identity",
        "identity_nb",
        "genotype_group",
        "genotype",
        "genotype_nb",
        "line",
        "line_replicate",
        "line_experiment",
        "line_replicate_experiment",
        "experiment_type",
        "focal_nb_genotype",
    ],
    "agg_rule": mean_agg_rule_tr_indiv_nb,
}

# Stats
PAIRS_OF_GROUPS = [
    {"pair": (("WT_HET-WT"), ("WT_HET-HET")), "level": 0},  # individual
    {"pair": (("HET_DEL-HET"), ("HET_DEL-DEL")), "level": 0},
    {"pair": (("WT_DEL-WT"), ("WT_DEL-DEL")), "level": 1},
    {"pair": (("WT_WT-WT"), ("WT_HET-WT")), "level": 1},
    {"pair": (("WT_HET-HET"), ("HET_HET-HET")), "level": 2},
    {"pair": (("HET_HET-HET"), ("HET_DEL-HET")), "level": 3},
    {"pair": (("HET_DEL-DEL"), ("DEL_DEL-DEL")), "level": 3},
    {"pair": (("DEL_DEL-DEL"), ("WT_DEL-DEL")), "level": 4},
    {"pair": (("WT_WT-WT"), ("HET_HET-HET")), "level": 4},
    {"pair": (("HET_HET-HET"), ("DEL_DEL-DEL")), "level": 5},
    {"pair": (("WT_WT-WT"), ("DEL_DEL-DEL")), "level": 6},
    {"pair": (("WT_WT-WT"), ("WT_DEL-WT")), "level": 7},
    {"pair": (("WT_WT_WT_WT_WT-WT"), ("DEL_DEL_DEL_DEL_DEL-DEL")), "level": 8},
    {"pair": ("HET_HET", "HET_DEL"), "level": 0},  # group
    {"pair": ("WT_WT", "WT_HET"), "level": 0},
    {"pair": ("DEL_DEL", "WT_DEL"), "level": 0},
    {"pair": ("WT_HET", "HET_HET"), "level": 1},
    {"pair": ("HET_DEL", "DEL_DEL"), "level": 1},
    {"pair": ("WT_WT", "HET_HET"), "level": 2},
    {"pair": ("HET_HET", "DEL_DEL"), "level": 3},
    {"pair": ("WT_WT", "DEL_DEL"), "level": 3},
    {"pair": ("WT_WT_WT_WT_WT", "DEL_DEL_DEL_DEL_DEL"), "level": 4},
    {"pair": ("WT-WT", "WT-HET"), "level": 0},  # focal nb
    {"pair": ("WT-HET", "WT-DEL"), "level": 1},
    {"pair": ("WT-WT", "WT-DEL"), "level": 2},
    {"pair": ("HET-WT", "HET-HET"), "level": 0},
    {"pair": ("HET-HET", "HET-DEL"), "level": 1},
    {"pair": ("HET-WT", "HET-DEL"), "level": 2},
    {"pair": ("DEL-WT", "DEL-HET"), "level": 0},
    {"pair": ("DEL-HET", "DEL-DEL"), "level": 1},
    {"pair": ("DEL-WT", "DEL-DEL"), "level": 2},
    {"pair": ("WT-HET", "HT-WT"), "level": 3},
    {"pair": ("HET-DEL", "DEL-HET"), "level": 3},
    {"pair": ("WT-DEL", "DEL-WT"), "level": 4},
    {"pair": ("WT-WT", "DEL-DEL"), "level": 5},
    {"pair": ("HET-HET", "DEL-DEL"), "level": 6},
]


MEAN_STATS_KWARGS = {
    "method": "approximate",
    "num_rounds": 10000,
    "func": "mean",
    "paired": False,
}
MEDIAN_STATS_KWARGS = {
    "method": "approximate",
    "num_rounds": 10000,
    "func": "median",
    "paired": False,
}


def _get_outliers(data, whis):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    outliers = (data < q1 - whis * iqr) | (data > q3 + whis * iqr)
    return outliers


def _compute_groups_stats(
    grouped_data,
    pairs_of_groups,
    variable,
    whis,
    test_func,
    test_func_kwargs,
):
    stats = []
    outliers = []
    for pair_group in pairs_of_groups:
        group_a, group_b = pair_group["pair"]
        group_a_in_data = group_a in grouped_data.groups.keys()
        group_b_in_data = group_b in grouped_data.groups.keys()
        if group_a_in_data and group_b_in_data:
            stat = {}

            grouped_data_a = grouped_data.get_group(group_a)
            grouped_data_b = grouped_data.get_group(group_b)

            var_data_a = grouped_data_a[variable]
            var_data_b = grouped_data_b[variable]

            outliers_a = _get_outliers(var_data_a, whis)
            outliers_b = _get_outliers(var_data_b, whis)

            if "identity" in grouped_data:
                cols = ["trial_uid", "identity"]
            else:
                # is a group variable
                cols = ["trial_uid"]

            outliers_a_info = grouped_data_a[outliers_a][cols]
            outliers_b_info = grouped_data_b[outliers_b][cols]
            outliers_ = pd.concat([outliers_a_info, outliers_b_info])
            outliers_["variable"] = [variable] * len(outliers_)

            possible_paired = len(group_a.split("-")[0]) > 3
            if possible_paired and (
                group_a.split("-")[0] == group_b.split("-")[0]
            ):
                test_kwargs_updated = test_func_kwargs.copy()
                test_kwargs_updated["paired"] = True

                if ~outliers_.empty:
                    # logger.info(f"Removing outliers paired data")
                    # logger.info(f"{outliers_}")
                    outliers_trials = outliers_.trial_uid.unique()
                    var_data_a = grouped_data_a[
                        ~grouped_data_a.trial_uid.isin(outliers_trials)
                    ][variable]
                    var_data_b = grouped_data_b[
                        ~grouped_data_b.trial_uid.isin(outliers_trials)
                    ][variable]

            else:
                test_kwargs_updated = test_func_kwargs.copy()

                if ~outliers_.empty:
                    # logger.info(f"Removing outliers non paired data")
                    # logger.info(f"{outliers_}")
                    var_data_a = var_data_a[~outliers_a]
                    var_data_b = var_data_b[~outliers_b]

            if test_kwargs_updated["func"] == "median":
                func = lambda x, y: np.abs(np.median(x) - np.median(y))
                stat_func = np.median
            elif test_kwargs_updated["func"] == "mean":
                func = lambda x, y: np.abs(np.mean(x) - np.mean(y))
                stat_func = np.mean
            test_kwargs_updated["func"] = func

            p_value = test_func(
                var_data_a.values, var_data_b.values, **test_kwargs_updated
            )
            stat_value = test_kwargs_updated["func"](
                var_data_a.values, var_data_b.values
            )
            stat["test"] = test_func.__name__
            stat["variable"] = variable
            stat["group_a"] = group_a
            stat["group_b"] = group_b
            stat["stat_a"] = stat_func(var_data_a.values)
            stat["stat_b"] = stat_func(var_data_b.values)
            stat["plot_level"] = pair_group["level"]
            stat["p_value"] = p_value
            stat["value"] = stat_value
            stat.update(
                {
                    f"test_kwarg_{key}": value
                    for key, value in test_func_kwargs.items()
                }
            )
            outliers.append(outliers_)
            stats.append(stat)

    return stats, outliers


def _get_num_data_points(data, boxplot_kwargs):
    num_data_points = data[boxplot_kwargs["x"]].value_counts()
    return num_data_points


def _compute_agg_stat(data, groupby, agg_rule):

    try:
        logger.info("Computting stats per individual")
        individual_stats = data.groupby(groupby).agg(agg_rule).reset_index()
        logger.info("Computed stats per individual")
        return individual_stats
    except KeyError as e:
        print(groupby)
        print(data.columns)
        print(e)
        raise KeyError


def standardize_replicate_data_wrt_het(
    data, normalizing_genotype_group="HET_HET"
):
    logger.info(
        "Normalizing replicates stats with "
        f"respect to {normalizing_genotype_group}"
    )
    variables = [col for col in data.columns if col in all_variables_names]

    data_line_replicate = data.groupby("line_replicate", as_index=False)
    data_enhanced = []
    for idx, data_replicate in data_line_replicate:
        # mean_agg_rule = {
        #     var_: "mean" if not "angle" in var_ else circmean
        #     for var_ in variables
        # }
        # std_agg_rule = {
        #     var_: "mean" if not "angle" in var_ else circstd
        #     for var_ in variables
        # }
        means = data_replicate[
            (data.genotype_group == normalizing_genotype_group)
        ][variables].mean()
        stds = data_replicate[
            (data.genotype_group == normalizing_genotype_group)
        ][variables].std()

        diff_variables = [f"{variable}_diff" for variable in variables]
        data_replicate[diff_variables] = data_replicate[variables] - means

        diff_variables = [f"{variable}_standardized" for variable in variables]
        data_replicate[diff_variables] = (
            data_replicate[variables] - means
        ) / stds

        data_enhanced.append(data_replicate)
    data_enhanced = pd.concat(data_enhanced)
    logger.info("Standardized")
    return data_enhanced

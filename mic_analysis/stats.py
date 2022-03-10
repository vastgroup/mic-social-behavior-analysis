import logging

import numpy as np
import pandas as pd
from confapp import conf

from .variables import (
    _group_varialbes_enhanced_names,
    _individual_nb_variables_enhanced_names,
    _individual_variables_enhanced_names,
    all_variables_names,
)

logger = logging.getLogger(__name__)


def _get_agg_rule_dictionary(variables_names):
    return {
        var_: conf.AGGREGATION_STATS["default"]
        if not var_ in conf.AGGREGATION_STATS.keys()
        else conf.AGGREGATION_STATS[var_]
        for var_ in variables_names
    }


# Agg rules for stats
mean_agg_rule_tr_indivs = _get_agg_rule_dictionary(
    _individual_variables_enhanced_names
)
mean_agg_rule_tr_group = _get_agg_rule_dictionary(
    _group_varialbes_enhanced_names
)
mean_agg_rule_tr_indiv_nb = _get_agg_rule_dictionary(
    _individual_nb_variables_enhanced_names
)


# Groupby stats constants

indiv_agg_stasts_kwargs = {
    "groupby": conf.AGGREGATION_COLUMNS["indiv"],
    "agg_rule": mean_agg_rule_tr_indivs,
}

group_agg_stasts_kwargs = {
    "groupby": conf.AGGREGATION_COLUMNS["group"],
    "agg_rule": mean_agg_rule_tr_group,
}


indiv_nb_agg_stats_kwargs = {
    "groupby": conf.AGGREGATION_COLUMNS["indiv_nb"],
    "agg_rule": mean_agg_rule_tr_indiv_nb,
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

            if (
                "identity" in grouped_data_a.columns
                and "identity_nb" in grouped_data_a.columns
            ):
                cols = ["trial_uid", "identity", "identity_nb"]
            elif "identity" in grouped_data_a.columns:
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
        # TODO: standardize angles by circmean circstd
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

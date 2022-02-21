import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from constants import (
    COLORS,
    GENOTYPE_GROUP_GENOTYPE_ORDER,
    TRAJECTORYTOOLS_DATASETS_INFO,
    all_variables_names,
    all_variables_names_enhanced,
)
from logger import setup_logs
from utils import circmean, circstd, data_filter

logger = setup_logs("plotters")


def _add_num_data_points(
    ax,
    data,
    num_data_points,
    boxplot_kwargs,
    y_offset=None,
    y_lim=None,
):
    if y_lim is None:
        y_min, y_max = ax.get_ylim()
    else:
        y_min, y_max = y_lim
    for i, order_ in enumerate(boxplot_kwargs["order"]):
        if order_ in num_data_points:
            num_data_points_group = num_data_points[order_]
            x = i
            min_y_group = data[data[boxplot_kwargs["x"]] == order_][
                boxplot_kwargs["y"]
            ].min()
            y = min_y_group - y_offset
            str_ = f"n={num_data_points_group}"
            ax.text(
                x,
                y,
                str_,
                ha="center",
                va="top",
            )
            if y - y_offset < y_min:
                ax.set_ylim([y - y_offset, y_max])
            y_min, y_max = ax.get_ylim()


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


def _boxplot_one_variable(ax, data, boxplot_kwargs):
    sns.boxplot(
        ax=ax,
        data=data,
        boxprops=dict(facecolor="w"),
        showmeans=True,
        meanline=True,
        meanprops=dict(linestyle="--", linewidth=1, color="k"),
        **boxplot_kwargs,
    )
    strip_kwargs = boxplot_kwargs.copy()
    strip_kwargs.pop("whis", None)
    sns.stripplot(
        ax=ax,
        data=data,
        dodge=True,
        alpha=0.5,
        **strip_kwargs,
    )
    sns.despine(ax=ax)
    if "order" in boxplot_kwargs:
        ax.set_xticklabels(boxplot_kwargs["order"], rotation=45, ha="right")
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")


def _boxplot_axes_one_variable(ax, data, variable, how="h", add_text=False):
    median = data[variable].median()
    mean = data[variable].mean()
    q1 = data[variable].quantile(0.25)
    q3 = data[variable].quantile(0.75)
    iqr = q3 - q1
    outliers_low = q1 - 1.5 * iqr
    outliers_high = q3 + 1.5 * iqr
    whis_low = np.max(
        [data[data[variable] > outliers_low][variable].min(), outliers_low]
    )
    whis_high = np.min(
        [data[data[variable] < outliers_high][variable].max(), outliers_high]
    )
    if how == "h":
        func = ax.axhline
    elif how == "v":
        func = ax.axvline
    else:
        raise

    func(median, c="k")
    func(mean, ls="--", c="k")
    func(q1, c=".25")
    func(q3, c=".25")
    func(whis_low, c=".25")
    func(whis_high, c=".25")

    if add_text:
        str_ = (
            f"mean={mean:.3f}, median={median:.3f}, q1={q1:.3f}, q3={q3:.3f}"
        )
        ax.set_title(str_)


def _get_x_group(group, order):
    x_value = group
    x = order.index(x_value)
    return x


def _plot_var_stat(ax, var_stat, y_line_stat, order, y_offset, y_lim):

    x_group_a = _get_x_group(
        var_stat["group_a"],
        order,
    )
    x_group_b = _get_x_group(
        var_stat["group_b"],
        order,
    )
    alpha = 0.5 if var_stat["p_value"] > 0.05 else 1
    ax.plot([x_group_a, x_group_b], [y_line_stat] * 2, "k-", alpha=alpha)
    x_text = np.mean([x_group_a, x_group_b])
    ax.text(
        x_text,
        y_line_stat + 0.1 * y_offset,
        f"{var_stat['value']:.2f}, {var_stat['p_value']:.4f}",
        ha="center",
        va="bottom",
        alpha=alpha,
    )
    y_min, y_max = y_lim
    ax.plot([x_group_a], [var_stat["stat_a"]], "ok")
    ax.plot([x_group_b], [var_stat["stat_b"]], "ok")
    if y_line_stat + y_offset > y_max:
        ax.set_ylim([y_min, y_line_stat + 2 * y_offset])


def _plot_var_stats(
    ax,
    data,
    var_stats,
    y_var,
    order,
    y_lim=None,
    y_offset=None,
):
    if y_lim is None:
        y_lim = ax.get_ylim()

    # y_offset = _get_y_offset(ax)
    y_start = data[y_var].max()

    actual_plot_level = 0
    last_level = 0
    for i, pair_stat in enumerate(var_stats):
        if pair_stat["plot_level"] != last_level:
            actual_plot_level += 1
            last_level = pair_stat["plot_level"]
        y_line_stat = y_start + (actual_plot_level + 1) * y_offset
        _plot_var_stat(
            ax,
            pair_stat,
            y_line_stat,
            order,
            y_offset,
            y_lim,
        )
        y_lim = ax.get_ylim()


def _boxplots_one_variable_with_stats(
    ax,
    data,
    variable,
    num_data_points=None,
    boxplot_kwargs=None,
    stats_kwargs=None,
    pairs_of_groups_for_stats=None,
    variable_ylim=None,
    variable_y_offset=None,
):
    boxplot_kwargs["y"] = variable
    _boxplot_one_variable(ax, data, boxplot_kwargs)
    if variable_ylim is not None:
        ax.set_ylim(variable_ylim)
    if num_data_points is not None:
        # logger.info("Adding number of data points")
        _add_num_data_points(
            ax,
            data,
            num_data_points,
            boxplot_kwargs,
            y_lim=variable_ylim,
            y_offset=variable_y_offset,
        )
    if stats_kwargs is not None:
        # logger.info("Computing stats")
        grouped_data = data.groupby([boxplot_kwargs["x"]])
        var_stats, outliers = _compute_groups_stats(
            grouped_data,
            pairs_of_groups_for_stats,
            variable,
            whis=boxplot_kwargs["whis"],
            **stats_kwargs,
        )
        # logger.info("Plotting stats")
        _plot_var_stats(
            ax,
            data,
            var_stats,
            variable,
            boxplot_kwargs["order"],
            y_lim=ax.get_ylim(),
            y_offset=variable_y_offset,
        )
    else:
        var_stats = None
        outliers = None
    return var_stats, outliers, ax.get_ylim()


def _get_num_data_points(data, boxplot_kwargs):
    num_data_points = data[boxplot_kwargs["x"]].value_counts()
    return num_data_points


def _update_order(data, boxplot_kwargs, valid_x_values):
    boxplot_kwargs.update(
        {
            "order": [
                gg
                for gg in valid_x_values
                if gg in data[boxplot_kwargs["x"]].unique()
            ],
        }
    )


def boxplot_variables_partition(
    axs,
    data,
    variables,
    title,
    boxplot_kwargs,
    pairs_of_groups_for_stats,
    stats_kwargs,
    valid_x_values=GENOTYPE_GROUP_GENOTYPE_ORDER,
    variables_ylims=None,
    varialbes_y_offsets=None,
):
    assert len(axs) == len(variables), f"{len(variables)} {len(axs)}"
    num_data_points = _get_num_data_points(data, boxplot_kwargs)
    _update_order(data, boxplot_kwargs, valid_x_values)

    all_var_stats = []
    all_outliers = []

    for i, (ax, variable) in enumerate(zip(axs, variables)):
        print(f"variable {variable}:")
        boxplot_kwargs.update({"y": variable})
        var_stats, outliers, var_ylim = _boxplots_one_variable_with_stats(
            ax,
            data,
            variable,
            num_data_points,
            boxplot_kwargs,
            stats_kwargs,
            pairs_of_groups_for_stats,
            variable_ylim=variables_ylims[variable],
            variable_y_offset=varialbes_y_offsets[variable],
        )
        if i == 0:
            ax.set_title(title)
        variables_ylims[variable] = var_ylim
        all_var_stats.extend(var_stats)
        all_outliers.extend(outliers)
    all_var_stats = pd.DataFrame(all_var_stats)
    all_outliers = pd.concat(all_outliers)
    return all_var_stats, all_outliers, variables_ylims


def _compute_per_individual_stat(data, groupby, agg_rule):
    try:
        return data.groupby(groupby).agg(agg_rule).reset_index()
    except KeyError as e:
        print(groupby)
        print(data.columns)
        print(e)
        raise KeyError


def standardize_replicate_data_wrt_het(
    data, normalizing_genotype_group="HET_HET"
):
    logger.info(
        f"Normalizing replicates stats with respect to {normalizing_genotype_group}"
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
    return data_enhanced


def _get_data(path, data_filters, per_indiv_stats_kwargs):
    logger.info("Getting data")
    data = pd.read_pickle(path)

    if "nb_angle" in data.columns:
        data["nb_angle"] = np.arctan2(
            data["nb_position_x"], data["nb_position_y"]
        )

    data = standardize_replicate_data_wrt_het(data)

    if "indiv" in path:
        data["genotype_group_genotype"] = (
            data["genotype_group"] + "-" + data["genotype"]
        )
        data["trial_uid_id"] = (
            data["trial_uid"] + "_" + data["identity"].astype(str)
        )
    if "indiv_nb" in path:
        data["genotype_group_genotype_nb"] = (
            data["genotype_group"] + "-" + data["genotype_nb"]
        )
        data["trial_uid_id_nb"] = (
            data["trial_uid"] + "_" + data["identity_nb"].astype(str)
        )
        data["focal_nb_genotype"] = (
            data["genotype"] + "-" + data["genotype_nb"]
        )
    logger.info("Filtering data")
    data_filtered = data_filter(data, data_filters)
    logger.info("Groupping data")
    data_stats = _compute_per_individual_stat(
        data=data_filtered, **per_indiv_stats_kwargs
    )
    if "indiv" in path:
        data_stats["genotype_group_genotype"] = (
            data_stats["genotype_group"] + "-" + data_stats["genotype"]
        )
    if "indiv_nb" in path:
        data_stats["genotype_group_genotype_nb"] = (
            data_stats["genotype_group"] + "-" + data_stats["genotype_nb"]
        )
    return data_filtered, data_stats


def _get_variables_ylims_and_offsets(data_stats, variables=None):
    if variables is None:
        if isinstance(data_stats.columns, pd.MultiIndex):
            variables = [
                c
                for c in data_stats.columns
                if c[0] in all_variables_names_enhanced
            ]
        else:
            variables = [
                c
                for c in data_stats.columns
                if c in all_variables_names_enhanced
            ]
    variables_ylims = {}
    varialbes_y_offsets = {}
    for variable in variables:
        y_min = data_stats[variable].min()
        y_max = data_stats[variable].max()
        variables_ylims[variable] = (y_min, y_max)
        varialbes_y_offsets[variable] = np.abs(y_max - y_min) * 0.1
    return variables_ylims, varialbes_y_offsets


def plot(config_dict):
    logger.info(f"Plotting with {config_dict}")
    data_info = TRAJECTORYTOOLS_DATASETS_INFO[
        config_dict["data_variables_group"]
    ]
    data, data_stats = _get_data(
        data_info["file_path"],
        config_dict["data_filters"],
        {
            "groupby": config_dict["groupby_cols"],
            "agg_rule": config_dict["agg_rule"],
        },
    )
    assert config_dict["rows_partitioned_by"] in data_stats
    partitions = data_stats[config_dict["rows_partitioned_by"]].unique()
    variables = config_dict["variables"]
    logger.info("Preparing figgure")

    fig, axs = plt.subplots(
        len(variables),
        len(partitions),
        figsize=(5 * len(partitions), 6 * len(variables)),
        sharey="row",
    )
    plt.subplots_adjust(wspace=0.4, hspace=0.5)

    variables_ylims, variables_y_offsets = _get_variables_ylims_and_offsets(
        data_stats
    )
    all_var_stats = []
    all_outliers = []
    for axs_col, partition in tqdm(zip(axs.T, partitions), desc="Plotting..."):
        logger.info(f"Plotting partition {partition}")
        partition_data = data_stats[
            data_stats[config_dict["rows_partitioned_by"]] == partition
        ]
        (
            all_var_stats_,
            all_outliers_,
            variables_ylims,
        ) = boxplot_variables_partition(
            axs_col,
            partition_data,
            variables,
            partition,
            config_dict["boxplot_kwargs"],
            config_dict["pairs_of_groups_for_stats"],
            config_dict["stats_kwargs"],
            variables_ylims=variables_ylims,
            varialbes_y_offsets=variables_y_offsets,
        )
        all_var_stats_["partition"] = [partition] * len(all_var_stats_)
        all_var_stats.append(all_var_stats_)
        all_outliers.append(all_outliers_)

    all_var_stats = pd.concat(all_var_stats)
    all_outliers = pd.concat(all_outliers)

    if not os.path.isdir(config_dict["save_path"]):
        os.makedirs(config_dict["save_path"])

    for extension in config_dict["extensions"]:
        fig.savefig(
            os.path.join(
                config_dict["save_path"],
                config_dict["file_name"] + f".{extension}",
            )
        )

    # TODO: Factorize this into another plot
    fig2, axs = plt.subplots(
        len(variables), 1, figsize=(30, 10 * len(variables))
    )

    for ax, variable in zip(axs, variables):
        _boxplot_one_variable(
            ax,
            data_stats,
            {
                "x": config_dict["rows_partitioned_by"],
                "y": variable,
                "hue": "genotype_group_genotype",
                "palette": COLORS,
                "order": data_stats[
                    config_dict["rows_partitioned_by"]
                ].unique(),
                "whis": 1.5,
            },
        )
        _boxplot_axes_one_variable(ax, data_stats, variable)
    for extension in config_dict["extensions"]:
        fig2.savefig(
            os.path.join(
                config_dict["save_path"],
                "vars_dist_summary" + f".{extension}",
            )
        )

    data_stats.to_csv(
        os.path.join(config_dict["save_path"], "data.csv"), index=False
    )
    all_var_stats.to_csv(
        os.path.join(config_dict["save_path"], "stats.csv"), index=False
    )
    all_outliers.to_csv(
        os.path.join(config_dict["save_path"], "outliers.csv"), index=False
    )

    return data_stats, all_var_stats, all_outliers


variables = [
    "normed_distance_to_origin",
    "speed",
    "acceleration",
    "abs_normal_acceleration",
    "abs_tg_acceleration",
    "distance_travelled",
]


# def plot_normalized_variables(path_with_data, variables=variables):
#     data = pd.read_csv(os.path.join(path_with_data, "data.csv"))

#     normalized_data = standardized_data

#     datasets = {
#         "line_replicate_data": {"data": data, "x": "line_replicate"},
#         "line_data": {"data": data, "x": "line"},
#         "line_replicate_diff_data": {
#             "data": diff_data,
#             "x": "line_replicate",
#         },
#         "line_diff_data": {"data": diff_data, "x": "line"},
#         "line_replicate_standardized_data": {
#             "data": standardized_data,
#             "x": "line_replicate",
#         },
#         "line_standardized_data": {
#             "data": standardized_data,
#             "x": "line",
#         },
#     }
#     fig, axs = plt.subplots(
#         len(datasets),
#         len(variables),
#         figsize=(30 * len(variables), 10 * len(datasets)),
#     )
#     for axs_col, variable in zip(axs.T, variables):
#         for dataset_name, ax in zip(datasets.keys(), axs_col):
#             data = datasets[dataset_name]["data"]
#             x = datasets[dataset_name]["x"]
#             ax.set_title(dataset_name)
#             _boxplot_one_variable(
#                 ax,
#                 data,
#                 {
#                     "x": x,
#                     "y": variable,
#                     "hue": "genotype_group_genotype",
#                     "palette": COLORS,
#                     "order": normalized_data[x].unique(),
#                     "whis": 1.5,
#                 },
#             )
#             _boxplot_axes_one_variable(ax, data, variable)
#     fig.savefig(os.path.join(path_with_data, "normalization.png"))


# def _plot_time_line_variable_partition(ax, data, variable):
#     # sns.lineplot(
#     #     ax=ax,
#     #     data=data,
#     #     x="frame",
#     #     y=variable,
#     #     hue="genotype_group_genotype",
#     #     units="trial_uid_id",
#     #     estimator=None,
#     #     ci=None,
#     #     alpha=0.1,
#     #     palette=colors,
#     #     legend=False,
#     # )
#     sns.lineplot(
#         ax=ax,
#         data=data,
#         x="frame",
#         y=variable,
#         hue="genotype_group_genotype",
#         estimator="median",
#         ci=None,
#         alpha=1,
#         palette=COLORS,
#     )
#     sns.lineplot(
#         ax=ax,
#         data=data,
#         x="frame",
#         y=variable,
#         hue="genotype_group_genotype",
#         estimator="mean",
#         ci=None,
#         alpha=1,
#         ls="--",
#         palette=COLORS,
#         legend=False,
#     )
#     sns.despine(ax=ax)


# def _plot_time_line_variable_partitions(
#     axs, data, variable, partition_by, partitions
# ):
#     for ax, partition in zip(axs, partitions):
#         logger.info(f"{partition}")
#         subdata = data[data[partition_by] == partition]
#         _plot_time_line_variable_partition(ax, subdata, variable)


# def _plot_variable_summary(
#     data,
#     data_stats,
#     variable,
#     partition_col,
#     partitions,
#     boxplot_kwargs,
#     pairs_of_groups_for_stats,
#     stats_kwargs,
# ):
#     fig = plt.figure(
#         constrained_layout=True, figsize=(40, 10 * len(partitions) + 1)
#     )
#     gs = GridSpec(len(partitions) + 1, 12, figure=fig)
#     axs_time_lines = []
#     axs_distributions = []
#     axs_boxplots = []
#     logger.info("Preparing figure")
#     for i, partition in enumerate(partitions):
#         axs_time_lines.append(fig.add_subplot(gs[i, :8]))
#         axs_distributions.append(fig.add_subplot(gs[i, 8:10]))
#         axs_boxplots.append(fig.add_subplot(gs[i, 10:]))

#     logger.info(f"Plotting {variable} along time")
#     _plot_time_line_variable_partitions(
#         axs_time_lines,
#         data,
#         variable,
#         partition_col,
#         partitions,
#     )

#     ax_boxplot_all = fig.add_subplot(gs[i + 1, :8])
#     colors = {
#         "HET_HET-HET": "b",
#         "HET_DEL-HET": "g",
#         "HET_DEL-DEL": "y",
#         "DEL_DEL-DEL": "r",
#     }
#     logger.info(f"Plotting {variable} boxplot summary")
#     _boxplot_one_variable(
#         ax_boxplot_all,
#         data_stats,
#         {
#             "x": partition_col,
#             "y": variable,
#             "hue": "genotype_group_genotype",
#             "palette": colors,
#             "order": data_stats[partition_col].unique(),
#             "whis": 1.5,
#         },
#     )
#     _boxplot_axes_one_variable(ax_boxplot_all, data_stats, variable)

#     # Plot distributions
#     logger.info(f"Plotting {variable} distributions")
#     for ax, partition in zip(axs_distributions, partitions):
#         logger.info(f"{partition}")
#         ax.set_title(partition)
#         subdata = data[data[partition_col] == partition]
#         # indivs = subdata.trial_uid_id.unique()
#         # for indiv in indivs:
#         #     sns.histplot(
#         #         ax=ax,
#         #         data=subdata[subdata.trial_uid_id == indiv],
#         #         x=variable,
#         #         hue="genotype_group_genotype",
#         #         stat="probability",
#         #         bins=500,
#         #         binrange=(subdata[variable].min(), subdata[variable].max()),
#         #         multiple="stack",
#         #         element="step",
#         #         palette={
#         #             "HET_HET-HET": "b",
#         #             "HET_DEL-HET": "g",
#         #             "HET_DEL-DEL": "y",
#         #             "DEL_DEL-DEL": "r",
#         #         },
#         #         alpha=0.1,
#         #         legend=False,
#         #         fill=False
#         #     )
#         sns.histplot(
#             ax=ax,
#             data=subdata,
#             x=variable,
#             hue="genotype_group_genotype",
#             stat="density",
#             bins=100,
#             binrange=(subdata[variable].min(), subdata[variable].max()),
#             multiple="stack",
#             element="step",
#             palette={
#                 "HET_HET-HET": "b",
#                 "HET_DEL-HET": "g",
#                 "HET_DEL-DEL": "y",
#                 "DEL_DEL-DEL": "r",
#             },
#             alpha=1,
#             legend=True,
#             fill=False,
#         )
#         sns.despine(ax=ax)

#     logger.info(f"Plotting {variable} boxxplots")
#     variables_ylims, variables_y_offsets = _get_variables_ylims_and_offsets(
#         data_stats
#     )
#     for ax, partition in zip(axs_boxplots, partitions):
#         logger.info(f"{partition}")
#         boxplot_kwargs.update({"y": variable})
#         partition_data = data_stats[data_stats[partition_col] == partition]
#         num_data_points = _get_num_data_points(partition_data, boxplot_kwargs)
#         var_stats, outliers = _boxplots_one_variable_with_stats(
#             ax,
#             partition_data,
#             variable,
#             num_data_points,
#             boxplot_kwargs,
#             stats_kwargs,
#             pairs_of_groups_for_stats,
#             variable_ylim=variables_ylims[variable],
#             variable_y_offset=variables_y_offsets[variable],
#         )
#     return fig


# def plot_variables_summary(config_dict):
#     """
#     Plots the time evolution and the boxplots of a varible for all
#     partitions (line or line_replicate)
#     """

#     pairs_of_groups_for_stats = config_dict["pairs_of_groups_for_stats"].copy()
#     stats_kwargs = config_dict["stats_kwargs"].copy()
#     boxplot_kwargs = config_dict["boxplot_kwargs"].copy()

#     logger.info(f"Plotting with {config_dict}")
#     data_info = TRAJECTORYTOOLS_DATASETS_INFO[
#         config_dict["data_variables_group"]
#     ]
#     logger.info("Loading data")
#     data, data_stats = _get_data(
#         data_info["file_path"],
#         config_dict["data_filters"],
#         {
#             "groupby": config_dict["groupby_cols"],
#             "agg_rule": config_dict["agg_rule"],
#         },
#     )
#     logger.info("Data loaded")

#     valid_x_values = GENOTYPE_GROUP_GENOTYPE_ORDER
#     valid_hue_values = GENOTYPE_GROUP_GENOTYPE_ORDER
#     _update_order(data_stats, boxplot_kwargs, valid_x_values, valid_hue_values)
#     assert config_dict["rows_partitioned_by"] in data_stats
#     partitions = data_stats[config_dict["rows_partitioned_by"]].unique()
#     variables = config_dict["variables"]

#     for variable in variables:
#         fig = _plot_variable_summary(
#             data,
#             data_stats,
#             variable,
#             config_dict["rows_partitioned_by"],
#             partitions,
#             boxplot_kwargs,
#             pairs_of_groups_for_stats,
#             stats_kwargs,
#         )
#         save_path = os.path.join(
#             config_dict["save_path"], f"{variable}_summary.png"
#         )
#         logger.info("Saving")
#         fig.savefig(save_path)
#         logger.info(f"Saved at {save_path}")

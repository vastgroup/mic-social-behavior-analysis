import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mlxtend.evaluate import permutation_test
from natsort import natsorted
from tqdm import tqdm
from trajectorytools.plot import plot_polar_histogram, polar_histogram

from .constants import (
    COLORS,
    FOCAL_NB_GENOTYPE_ORDER,
    GENOTYPE_GROUP_GENOTYPE_ORDER,
    GENOTYPE_GROUP_ORDER,
)
from .datasets import TRAJECTORYTOOLS_DATASETS_INFO
from .stats import (
    MEAN_STATS_KWARGS,
    PAIRS_OF_GROUPS,
    _compute_groups_stats,
    _get_num_data_points,
)
from .string_infos import (
    get_animal_info_str,
    get_focal_nb_info,
    get_partition_info_str,
    get_video_info_str,
)
from .utils import _select_partition_from_datasets, circmean, circstd
from .variables import all_variables_names, all_variables_names_enhanced

logger = logging.getLogger(__name__)


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


# def plot(config_dict):
#     logger.info(f"Plotting with {config_dict}")
#     data_info = TRAJECTORYTOOLS_DATASETS_INFO[
#         config_dict["data_variables_group"]
#     ]
#     data, data_stats = get_data(
#         data_info["file_path"],
#         config_dict["data_filters"],
#         {
#             "groupby": config_dict["groupby_cols"],
#             "agg_rule": config_dict["agg_rule"],
#         },
#     )
#     assert config_dict["rows_partitioned_by"] in data_stats
#     partitions = data_stats[config_dict["rows_partitioned_by"]].unique()
#     variables = config_dict["variables"]
#     logger.info("Preparing figgure")

#     fig, axs = plt.subplots(
#         len(variables),
#         len(partitions),
#         figsize=(5 * len(partitions), 6 * len(variables)),
#         sharey="row",
#     )
#     plt.subplots_adjust(wspace=0.4, hspace=0.5)

#     variables_ylims, variables_y_offsets = _get_variables_ylims_and_offsets(
#         data_stats
#     )
#     all_var_stats = []
#     all_outliers = []
#     for axs_col, partition in tqdm(zip(axs.T, partitions), desc="Plotting..."):
#         logger.info(f"Plotting partition {partition}")
#         partition_data = data_stats[
#             data_stats[config_dict["rows_partitioned_by"]] == partition
#         ]
#         (
#             all_var_stats_,
#             all_outliers_,
#             variables_ylims,
#         ) = boxplot_variables_partition(
#             axs_col,
#             partition_data,
#             variables,
#             partition,
#             config_dict["boxplot_kwargs"],
#             config_dict["pairs_of_groups_for_stats"],
#             config_dict["stats_kwargs"],
#             variables_ylims=variables_ylims,
#             varialbes_y_offsets=variables_y_offsets,
#         )
#         all_var_stats_["partition"] = [partition] * len(all_var_stats_)
#         all_var_stats.append(all_var_stats_)
#         all_outliers.append(all_outliers_)

#     all_var_stats = pd.concat(all_var_stats)
#     all_outliers = pd.concat(all_outliers)

#     if not os.path.isdir(config_dict["save_path"]):
#         os.makedirs(config_dict["save_path"])

#     for extension in config_dict["extensions"]:
#         fig.savefig(
#             os.path.join(
#                 config_dict["save_path"],
#                 config_dict["file_name"] + f".{extension}",
#             )
#         )

#     # TODO: Factorize this into another plot
#     fig2, axs = plt.subplots(
#         len(variables), 1, figsize=(30, 10 * len(variables))
#     )

#     for ax, variable in zip(axs, variables):
#         _boxplot_one_variable(
#             ax,
#             data_stats,
#             {
#                 "x": config_dict["rows_partitioned_by"],
#                 "y": variable,
#                 "hue": "genotype_group_genotype",
#                 "palette": COLORS,
#                 "order": data_stats[
#                     config_dict["rows_partitioned_by"]
#                 ].unique(),
#                 "whis": 1.5,
#             },
#         )
#         _boxplot_axes_one_variable(ax, data_stats, variable)
#     for extension in config_dict["extensions"]:
#         fig2.savefig(
#             os.path.join(
#                 config_dict["save_path"],
#                 "vars_dist_summary" + f".{extension}",
#             )
#         )

#     data_stats.to_csv(
#         os.path.join(config_dict["save_path"], "data.csv"), index=False
#     )
#     all_var_stats.to_csv(
#         os.path.join(config_dict["save_path"], "stats.csv"), index=False
#     )
#     all_outliers.to_csv(
#         os.path.join(config_dict["save_path"], "outliers.csv"), index=False
#     )

#     return data_stats, all_var_stats, all_outliers


variables = [
    "normed_distance_to_origin",
    "speed",
    "acceleration",
    "abs_normal_acceleration",
    "abs_tg_acceleration",
    "distance_travelled",
]


def _plot_time_line_variable_partition(ax, data, variable):
    # sns.lineplot(
    #     ax=ax,
    #     data=data,
    #     x="frame",
    #     y=variable,
    #     hue="genotype_group_genotype",
    #     units="trial_uid_id",
    #     estimator=None,
    #     ci=None,
    #     alpha=0.1,
    #     palette=colors,
    #     legend=False,
    # )
    sns.lineplot(
        ax=ax,
        data=data,
        x="frame",
        y=variable,
        hue="genotype_group_genotype",
        estimator="median",
        ci=None,
        alpha=1,
        palette=COLORS,
    )
    sns.lineplot(
        ax=ax,
        data=data,
        x="frame",
        y=variable,
        hue="genotype_group_genotype",
        estimator="mean",
        ci=None,
        alpha=1,
        ls="--",
        palette=COLORS,
        legend=False,
    )
    sns.despine(ax=ax)


def _plot_time_line_variable_partitions(
    axs, data, variable, partition_by, partitions
):
    for ax, partition in zip(axs, partitions):
        logger.info(f"{partition}")
        subdata = data[data[partition_by] == partition]
        _plot_time_line_variable_partition(ax, subdata, variable)


def _plot_variable_summary(
    data,
    data_stats,
    variable,
    partition_col,
    partitions,
    boxplot_kwargs,
    pairs_of_groups_for_stats,
    stats_kwargs,
):
    fig = plt.figure(
        constrained_layout=True, figsize=(40, 10 * len(partitions) + 1)
    )
    gs = GridSpec(len(partitions) + 1, 12, figure=fig)
    axs_time_lines = []
    axs_distributions = []
    axs_boxplots = []
    logger.info("Preparing figure")
    for i, partition in enumerate(partitions):
        axs_time_lines.append(fig.add_subplot(gs[i, :8]))
        axs_distributions.append(fig.add_subplot(gs[i, 8:10]))
        axs_boxplots.append(fig.add_subplot(gs[i, 10:]))

    logger.info(f"Plotting {variable} along time")
    _plot_time_line_variable_partitions(
        axs_time_lines,
        data,
        variable,
        partition_col,
        partitions,
    )

    ax_boxplot_all = fig.add_subplot(gs[i + 1, :8])
    colors = {
        "HET_HET-HET": "b",
        "HET_DEL-HET": "g",
        "HET_DEL-DEL": "y",
        "DEL_DEL-DEL": "r",
    }
    logger.info(f"Plotting {variable} boxplot summary")
    _boxplot_one_variable(
        ax_boxplot_all,
        data_stats,
        {
            "x": partition_col,
            "y": variable,
            "hue": "genotype_group_genotype",
            "palette": colors,
            "order": data_stats[partition_col].unique(),
            "whis": 1.5,
        },
    )
    _boxplot_axes_one_variable(ax_boxplot_all, data_stats, variable)

    # Plot distributions
    logger.info(f"Plotting {variable} distributions")
    for ax, partition in zip(axs_distributions, partitions):
        logger.info(f"{partition}")
        ax.set_title(partition)
        subdata = data[data[partition_col] == partition]
        # indivs = subdata.trial_uid_id.unique()
        # for indiv in indivs:
        #     sns.histplot(
        #         ax=ax,
        #         data=subdata[subdata.trial_uid_id == indiv],
        #         x=variable,
        #         hue="genotype_group_genotype",
        #         stat="probability",
        #         bins=500,
        #         binrange=(subdata[variable].min(), subdata[variable].max()),
        #         multiple="stack",
        #         element="step",
        #         palette={
        #             "HET_HET-HET": "b",
        #             "HET_DEL-HET": "g",
        #             "HET_DEL-DEL": "y",
        #             "DEL_DEL-DEL": "r",
        #         },
        #         alpha=0.1,
        #         legend=False,
        #         fill=False
        #     )
        sns.histplot(
            ax=ax,
            data=subdata,
            x=variable,
            hue="genotype_group_genotype",
            stat="density",
            bins=100,
            binrange=(subdata[variable].min(), subdata[variable].max()),
            multiple="stack",
            element="step",
            palette={
                "HET_HET-HET": "b",
                "HET_DEL-HET": "g",
                "HET_DEL-DEL": "y",
                "DEL_DEL-DEL": "r",
            },
            alpha=1,
            legend=True,
            fill=False,
        )
        sns.despine(ax=ax)

    logger.info(f"Plotting {variable} boxxplots")
    variables_ylims, variables_y_offsets = _get_variables_ylims_and_offsets(
        data_stats
    )
    for ax, partition in zip(axs_boxplots, partitions):
        logger.info(f"{partition}")
        boxplot_kwargs.update({"y": variable})
        partition_data = data_stats[data_stats[partition_col] == partition]
        num_data_points = _get_num_data_points(partition_data, boxplot_kwargs)
        var_stats, outliers = _boxplots_one_variable_with_stats(
            ax,
            partition_data,
            variable,
            num_data_points,
            boxplot_kwargs,
            stats_kwargs,
            pairs_of_groups_for_stats,
            variable_ylim=variables_ylims[variable],
            variable_y_offset=variables_y_offsets[variable],
        )
    return fig


def plot_variables_summary(datasets):
    """
    Plots the time evolution and the boxplots of a varible for all
    partitions (line or line_replicate)
    """

    pairs_of_groups_for_stats = config_dict["pairs_of_groups_for_stats"].copy()
    stats_kwargs = config_dict["stats_kwargs"].copy()
    boxplot_kwargs = config_dict["boxplot_kwargs"].copy()

    logger.info(f"Plotting with {config_dict}")
    data_info = TRAJECTORYTOOLS_DATASETS_INFO[
        config_dict["data_variables_group"]
    ]
    logger.info("Loading data")
    data, data_stats = get_data(
        data_info["file_path"],
        config_dict["data_filters"],
        {
            "groupby": config_dict["groupby_cols"],
            "agg_rule": config_dict["agg_rule"],
        },
    )
    logger.info("Data loaded")

    valid_x_values = GENOTYPE_GROUP_GENOTYPE_ORDER
    valid_hue_values = GENOTYPE_GROUP_GENOTYPE_ORDER
    _update_order(data_stats, boxplot_kwargs, valid_x_values, valid_hue_values)
    assert config_dict["rows_partitioned_by"] in data_stats
    partitions = data_stats[config_dict["rows_partitioned_by"]].unique()
    variables = config_dict["variables"]

    for variable in variables:
        fig = _plot_variable_summary(
            data,
            data_stats,
            variable,
            config_dict["rows_partitioned_by"],
            partitions,
            boxplot_kwargs,
            pairs_of_groups_for_stats,
            stats_kwargs,
        )
        save_path = os.path.join(
            config_dict["save_path"], f"{variable}_summary.png"
        )
        logger.info("Saving")
        fig.savefig(save_path)
        logger.info(f"Saved at {save_path}")


### SOME CONSTANTS FOR SUMMARY FIGURES THAT SHOULD NOT BE HERE
INDIVIDUAL_VARIABLES_TO_PLOT = ["normed_distance_to_origin", "speed"]
GROUP_VARIABLES_TO_PLOT = [
    "mean_distance_to_center_of_group",
    "polarization_order_parameter",
    "rotation_order_parameter",
]
INDIVIDUAL_NB_VARIABLES_TO_PLOT = ["nb_angle", "nb_distance"]

INDIVIDUAL_VARIABLES_STATS_TO_PLOT = [
    ("normed_distance_to_origin", "mean"),
    ("speed", "mean"),
]
GROUP_VARIABLES_STATS_TO_PLOT = [
    ("mean_distance_to_center_of_group", "mean"),
    ("polarization_order_parameter", "mean"),
    ("rotation_order_parameter", "mean"),
]
INDIVIDUAL_NB_VARIALBES_STATS_TO_PLOT = [
    ("nb_angle", "ratio_in_front"),
    ("nb_distance", "mean"),
]

boxplot_kwargs = {
    "x": "genotype_group_genotype",
    "palette": COLORS,
    "whis": 100,
}
boxplot_kwargs_group = {
    "x": "genotype_group",
    "palette": COLORS,
    "whis": 100,
}
boxplot_kwargs_indiv_nb = {
    "x": "focal_nb_genotype",
    "whis": 100,
}
stats_kwargs = {
    "test_func": permutation_test,
    "test_func_kwargs": MEAN_STATS_KWARGS,
}


##### FIGURES PREPARATION FUNCTIONS


def _prepare_animal_indiv_vars_fig(num_variables):
    fig = plt.figure(constrained_layout=True, figsize=(30, 10))
    num_cols = num_variables * 3
    num_rows = num_variables
    gs = GridSpec(num_rows, num_cols, figure=fig)

    ax_trajectories = fig.add_subplot(gs[:num_rows, :num_rows])
    axs_variables = []
    axs_distributions = []
    for i in range(num_variables):
        axs_variables.append(
            fig.add_subplot(gs[i : i + 1, num_rows : num_cols - 1])
        )
        axs_distributions.append(
            fig.add_subplot(gs[i : i + 1, num_cols - 1 :])
        )
    return fig, ax_trajectories, axs_variables, axs_distributions


def _prepare_video_group_fig(num_variables):
    fig = plt.figure(constrained_layout=True, figsize=(30, 10))
    num_cols = num_variables * 3
    num_rows = num_variables
    gs = GridSpec(num_rows, num_cols, figure=fig)

    ax_order_params = fig.add_subplot(gs[:num_rows, :num_rows])
    axs_variables = []
    axs_distributions = []
    for i in range(num_variables):
        axs_variables.append(
            fig.add_subplot(gs[i : i + 1, num_rows : num_cols - 1])
        )
        axs_distributions.append(
            fig.add_subplot(gs[i : i + 1, num_cols - 1 :])
        )
    return fig, ax_order_params, axs_variables, axs_distributions


def _prepare_video_indiv_nb_fig(num_variables):
    fig = plt.figure(constrained_layout=True, figsize=(30, 10))
    num_cols = num_variables * 3
    num_rows = num_variables
    gs = GridSpec(num_rows, num_cols, figure=fig)

    ax_relative_position = fig.add_subplot(gs[:num_rows, :num_rows])
    axs_variables = []
    axs_distributions = []
    for i in range(num_variables):
        axs_variables.append(
            fig.add_subplot(gs[i : i + 1, num_rows : num_cols - 1])
        )
        axs_distributions.append(
            fig.add_subplot(gs[i : i + 1, num_cols - 1 :])
        )
    return fig, ax_relative_position, axs_variables, axs_distributions


def _prepare_partition_indiv_vars_summary_fig(
    num_variables, num_genotype_groups
):
    assert num_variables == 2
    fig = plt.figure(constrained_layout=True, figsize=(30, 10))
    num_columns_position_hist = np.ceil(
        (num_genotype_groups + 1) / num_variables
    ).astype(int)
    num_cols = 4 + num_columns_position_hist
    num_rows = num_variables
    gs = GridSpec(num_rows, num_cols, figure=fig)

    axs_order_params_dist = []
    for row in range(num_rows):
        for col in range(num_columns_position_hist):
            axs_order_params_dist.append(
                fig.add_subplot(gs[row : row + 1, col : col + 1])
            )
    # axs_variables = []
    axs_distributions = []
    axs_boxplots_raw = []
    axs_boxplots_diff = []
    axs_boxplots_standardized = []
    for i in range(num_variables):
        # axs_variables.append(fig.add_subplot(gs[i:i+1, num_rows:num_cols-4]))
        axs_distributions.append(
            fig.add_subplot(gs[i : i + 1, num_cols - 4 : num_cols - 3])
        )
        axs_boxplots_raw.append(
            fig.add_subplot(gs[i : i + 1, num_cols - 3 : num_cols - 2])
        )
        axs_boxplots_diff.append(
            fig.add_subplot(gs[i : i + 1, num_cols - 2 : num_cols - 1])
        )
        axs_boxplots_standardized.append(
            fig.add_subplot(gs[i : i + 1, num_cols - 1 :])
        )
    return (
        fig,
        axs_order_params_dist,
        axs_distributions,
        axs_boxplots_raw,
        axs_boxplots_diff,
        axs_boxplots_standardized,
    )


def _prepare_partition_group_vars_summary_fig(
    num_variables, num_genotype_groups
):
    assert num_variables == 3
    fig = plt.figure(constrained_layout=True, figsize=(30, 10))
    num_columns_position_hist = np.ceil(
        (num_genotype_groups + 1) / num_variables
    ).astype(int)
    num_cols = 4 + num_columns_position_hist
    num_rows = num_variables
    gs = GridSpec(num_rows, num_cols, figure=fig)

    axs_positions_dist = []
    for row in range(num_rows):
        for col in range(num_columns_position_hist):
            axs_positions_dist.append(
                fig.add_subplot(gs[row : row + 1, col : col + 1])
            )
    # axs_variables = []
    axs_distributions = []
    axs_boxplots_raw = []
    axs_boxplots_diff = []
    axs_boxplots_standardized = []
    for i in range(num_variables):
        # axs_variables.append(fig.add_subplot(gs[i:i+1, num_rows:num_cols-4]))
        axs_distributions.append(
            fig.add_subplot(gs[i : i + 1, num_cols - 4 : num_cols - 3])
        )
        axs_boxplots_raw.append(
            fig.add_subplot(gs[i : i + 1, num_cols - 3 : num_cols - 2])
        )
        axs_boxplots_diff.append(
            fig.add_subplot(gs[i : i + 1, num_cols - 2 : num_cols - 1])
        )
        axs_boxplots_standardized.append(
            fig.add_subplot(gs[i : i + 1, num_cols - 1 :])
        )
    return (
        fig,
        axs_positions_dist,
        axs_distributions,
        axs_boxplots_raw,
        axs_boxplots_diff,
        axs_boxplots_standardized,
    )


def _prepare_partition_indiv_nb_vars_summary_fig(
    num_variables, num_focal_nb_genotype_groups
):
    assert num_variables == 2
    fig = plt.figure(constrained_layout=True, figsize=(30, 10))
    num_columns_position_hist = np.ceil(
        num_focal_nb_genotype_groups / num_variables
    ).astype(int)
    num_cols = 4 + num_columns_position_hist
    num_rows = num_variables
    gs = GridSpec(num_rows, num_cols, figure=fig)

    axs_positions_dist = []
    for row in range(num_rows):
        for col in range(num_columns_position_hist):
            axs_positions_dist.append(
                fig.add_subplot(gs[row : row + 1, col : col + 1], polar=True)
            )
    # axs_variables = []
    axs_distributions = []
    axs_boxplots_raw = []
    axs_boxplots_diff = []
    axs_boxplots_standardized = []
    for i in range(num_variables):
        # axs_variables.append(fig.add_subplot(gs[i:i+1, num_rows:num_cols-4]))
        axs_distributions.append(
            fig.add_subplot(gs[i : i + 1, num_cols - 4 : num_cols - 3])
        )
        axs_boxplots_raw.append(
            fig.add_subplot(gs[i : i + 1, num_cols - 3 : num_cols - 2])
        )
        # axs_boxplots_diff.append(
        #     fig.add_subplot(gs[i : i + 1, num_cols - 2 : num_cols - 1])
        # )
        # axs_boxplots_standardized.append(
        #     fig.add_subplot(gs[i : i + 1, num_cols - 1 :])
        # )
    return (
        fig,
        axs_positions_dist,
        axs_distributions,
        axs_boxplots_raw
        # axs_boxplots_diff,
        # axs_boxplots_standardized,
    )


##### AXES PLOTTERS
def plot_order_parameter_dist(data, ax=None):
    x_var = "rotation_order_parameter"
    y_var = "polarization_order_parameter"
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    assert x_var in data
    assert y_var in data
    sns.histplot(
        ax=ax,
        data=data,
        x=x_var,
        y=y_var,
        bins=(10, 10),
        binrange=((0, 1), (0, 1)),
    )
    ax.set_aspect("equal")
    ax.set_ylabel(y_var)
    ax.set_xlabel(x_var)
    sns.despine(ax=ax)


def plot_relative_position_dist(data, ax=None):
    x_var = "nb_position_x"
    y_var = "nb_position_y"
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    assert x_var in data
    assert y_var in data
    sns.histplot(
        ax=ax,
        data=data,
        x=x_var,
        y=y_var,
        cbar=False,
        binrange=((-5, 5), (-5, 5)),
        bins=(50, 50),
    )
    ax.set_aspect("equal")
    ax.set_ylabel(y_var)
    ax.set_xlabel(x_var)
    ax.axhline(0, c="k", ls=":")
    ax.axvline(0, c="k", ls=":")
    focal_nb_str = f"focal: {data['genotype'].unique()[0]} - neighbour: {data['genotype_nb'].unique()[0]}"
    ax.set_title(focal_nb_str)
    sns.despine(ax=ax)


def plot_trajectory(
    data, ax=None, hue=None, show_trajectories=True, x_var="s_x", y_var="s_y"
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    assert x_var in data
    assert y_var in data
    if hue is not None:
        cbar = False
    else:
        cbar = True
    sns.histplot(
        ax=ax, data=data, x=x_var, y=y_var, cbar=cbar, hue=hue, bins=(20, 20)
    )
    if show_trajectories:
        sns.lineplot(
            ax=ax,
            data=data,
            x=x_var,
            y=y_var,
            sort=False,
            hue=hue,
            alpha=0.5,
            units=hue,
            estimator=None,
        )
    ax.set_aspect("equal")
    ax.set_ylabel(y_var)
    ax.set_xlabel(x_var)
    sns.despine(ax=ax)


def plot_variable_along_time(
    data, variable, ax=None, hue=None, units=None, estimator=None, legend=True
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(30, 10))
    assert "frame" in data
    assert variable in data
    if units is None:
        units = hue
    _boxplot_axes_one_variable(ax, data, variable, how="h", add_text=True)
    if estimator is None:
        sns.lineplot(
            ax=ax,
            data=data,
            x="frame",
            y=variable,
            alpha=0.5,
            hue=hue,
            units=units,
            legend=legend,
            estimator=estimator,
        )
    else:
        if hue == "genotype_group_genotype":
            sns.lineplot(
                ax=ax,
                data=data,
                x="frame",
                y=variable,
                alpha=0.25,
                hue=hue,
                estimator=estimator,
                ci=None,
                legend=legend,
                palette=COLORS,
            )
        else:
            sns.lineplot(
                ax=ax,
                data=data,
                x="frame",
                y=variable,
                alpha=0.25,
                hue=hue,
                estimator=estimator,
                ci=None,
                legend=legend,
            )
    sns.despine(ax=ax)


def plot_variable_1d_distribution(
    data, variable, variables_ranges, ax=None, hue=None, legend=None, how="h"
):
    bin_range = (
        variables_ranges[variable]["min"],
        variables_ranges[variable]["max"],
    )
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(30, 10))
    assert variable in data
    _boxplot_axes_one_variable(ax, data, variable, how=how)
    if how == "h":
        x_var = None
        y_var = variable
    elif how == "v":
        x_var = variable
        y_var = None
    if hue == "genotype_group_genotype":
        sns.histplot(
            ax=ax,
            data=data,
            x=x_var,
            y=y_var,
            stat="probability",
            common_norm=False,
            element="poly",
            fill=False,
            alpha=0.5,
            hue=hue,
            bins=100,
            binrange=bin_range,
            legend=legend,
            palette=COLORS,
        )
    else:
        sns.histplot(
            ax=ax,
            data=data,
            x=x_var,
            y=y_var,
            stat="probability",
            common_norm=False,
            element="poly",
            fill=False,
            alpha=0.5,
            hue=hue,
            bins=100,
            binrange=bin_range,
            legend=legend,
        )
    sns.despine(ax=ax)


def plot_positions_dist_per_genotype_group(data, axs):
    genotype_groups = data["genotype_group"].unique()
    for i, (genotype_group, ax) in enumerate(zip(genotype_groups, axs)):
        sub_data = data[data.genotype_group == genotype_group]
        plot_trajectory(
            sub_data,
            ax,
            show_trajectories=False,
            x_var="s_x_normed",
            y_var="s_y_normed",
        )
        ax.set_title(genotype_group)
    plot_trajectory(
        data,
        axs[i + 1],
        show_trajectories=False,
        x_var="s_x_normed",
        y_var="s_y_normed",
    )
    axs[i + 1].set_title("all")


def plot_order_parameter_dist_per_genotype_group(data, axs):
    genotype_groups = data["genotype_group"].unique()
    for i, (genotype_group, ax) in enumerate(zip(genotype_groups, axs)):
        sub_data = data[data.genotype_group == genotype_group]
        plot_order_parameter_dist(sub_data, ax)
        ax.set_title(genotype_group)
    plot_order_parameter_dist(data, axs[i + 1])
    axs[i + 1].set_title("all")
    if i + 1 < len(axs) - 1:
        # is not the last axes
        [ax.set_visible(False) for ax in axs[i + 2 :]]


def _plot_polar_dist_relative_positions(data, axs_polar_plots):
    valid_focal_nb_genotype = data.focal_nb_genotype.unique()
    focal_nb_genotype_order = [
        fng
        for fng in FOCAL_NB_GENOTYPE_ORDER
        if fng in valid_focal_nb_genotype
    ]
    pos_hists = {
        focal_nb_genotype: [] for focal_nb_genotype in focal_nb_genotype_order
    }
    for trial_uid_id in data.trial_uid_id.unique():
        data_focal = data[data.trial_uid_id == trial_uid_id]
        pos_hist, r_edges, theta_edges = polar_histogram(
            data_focal.nb_distance.values,
            data_focal.nb_angle.values,
            density=True,
            range_r=4,
            bins=(10, 12),
        )
        assert len(data_focal.focal_nb_genotype.unique()) == 1
        pos_hists[data_focal.focal_nb_genotype.unique()[0]].append(pos_hist)

    pos_hists_arrs = {
        focal_nb_genotype: np.asarray(pos_hists[focal_nb_genotype])
        for focal_nb_genotype in focal_nb_genotype_order
    }

    # Plot polar histogram/maps for relative neighbor positions, turning and acceleration
    vmin = 0
    vmax = 0
    for i, (focal_nb_genotype, pos_hist) in enumerate(pos_hists_arrs.items()):
        mean_pos_hist = np.mean(pos_hist, axis=0)
        vmin = np.min([vmin, np.min(mean_pos_hist)])
        vmax = np.max([vmax, np.max(mean_pos_hist)])

    for i, (focal_nb_genotype, pos_hist) in enumerate(pos_hists_arrs.items()):
        ax = axs_polar_plots[i]
        mean_pos_hist = np.mean(pos_hist, axis=0)
        plot_polar_histogram(
            mean_pos_hist,
            r_edges,
            theta_edges,
            ax,
            vmin=vmin,
            vmax=vmax,
            symmetric_color_limits=False,
        )
        ax.set_title(focal_nb_genotype)


##### SUMMARY FIGURES


def _plot_animal_indiv_vars_summary(
    data, variables, variables_ranges, hue=None
):
    (
        fig,
        ax_trajectories,
        axs_variables,
        axs_distributions,
    ) = _prepare_animal_indiv_vars_fig(len(variables))
    plot_trajectory(data, ax=ax_trajectories, hue=hue)
    for variable, ax_time, ax_dist in zip(
        variables, axs_variables, axs_distributions
    ):
        plot_variable_along_time(data, variable, ax=ax_time, hue=hue)
        plot_variable_1d_distribution(
            data, variable, variables_ranges, ax=ax_dist, hue=hue
        )
    return fig


def _plot_video_indiv_vars_summary(
    data, variables, variables_ranges, hue=None
):
    (
        fig,
        ax_trajectories,
        axs_variables,
        axs_distributions,
    ) = _prepare_animal_indiv_vars_fig(len(variables))
    plot_trajectory(data, ax=ax_trajectories, hue=hue)
    for variable, ax_time, ax_dist in zip(
        variables, axs_variables, axs_distributions
    ):
        plot_variable_along_time(data, variable, ax=ax_time, hue=hue)
        plot_variable_1d_distribution(
            data, variable, variables_ranges, ax=ax_dist, hue=hue
        )
    return fig


def _plot_group_variables_summary(data, variables, variables_ranges):
    (
        fig,
        ax_order_params,
        axs_variables,
        axs_distributions,
    ) = _prepare_video_group_fig(len(variables))
    plot_order_parameter_dist(data, ax=ax_order_params)
    for variable, ax_time, ax_dist in zip(
        variables, axs_variables, axs_distributions
    ):
        plot_variable_along_time(data, variable, ax=ax_time)
        plot_variable_1d_distribution(
            data, variable, variables_ranges, ax=ax_dist
        )
    return fig


def _plot_video_indiv_nb_variables_summary(data, variables, variables_ranges):
    (
        fig,
        ax_order_params,
        axs_variables,
        axs_distributions,
    ) = _prepare_video_indiv_nb_fig(len(variables))
    plot_relative_position_dist(data, ax=ax_order_params)
    for variable, ax_time, ax_dist in zip(
        variables, axs_variables, axs_distributions
    ):
        plot_variable_along_time(
            data, variable, ax=ax_time, hue="genotype_nb", units="identity_nb"
        )
        plot_variable_1d_distribution(
            data, variable, variables_ranges, ax=ax_dist, hue="genotype_nb"
        )
    return fig


def _plot_partition_indiv_vars_summary(
    data, data_stats, variables, variables_ranges, variables_stats, hue=None
):
    num_genotype_groups = len(data["genotype_group"].unique())
    (
        fig,
        axs_positions_dist,
        axs_distributions,
        axs_boxplots_raw,
        axs_boxplots_diff,
        axs_boxplots_standardized,
    ) = _prepare_partition_indiv_vars_summary_fig(
        len(variables_stats), num_genotype_groups
    )
    plot_positions_dist_per_genotype_group(data, axs=axs_positions_dist)
    num_data_points = _get_num_data_points(data_stats, boxplot_kwargs)
    for i, (variable, ax_dist) in enumerate(zip(variables, axs_distributions)):
        if i == 0:
            legend = True
        else:
            legend = False
        plot_variable_1d_distribution(
            data,
            variable,
            variables_ranges,
            ax=ax_dist,
            hue=hue,
            legend=legend,
            how="v",
        )
    for i, (
        variable,
        ax_boxplot,
        ax_boxplot_diff,
        ax_boxplot_standardized,
    ) in enumerate(
        zip(
            variables_stats,
            axs_boxplots_raw,
            axs_boxplots_diff,
            axs_boxplots_standardized,
        )
    ):
        if i == 0:
            legend = True
        else:
            legend = False
        (
            variables_ylims,
            variables_y_offsets,
        ) = _get_variables_ylims_and_offsets(data_stats)
        _update_order(data, boxplot_kwargs, GENOTYPE_GROUP_GENOTYPE_ORDER)
        _boxplots_one_variable_with_stats(
            ax_boxplot,
            data_stats,
            variable,
            num_data_points=num_data_points,
            boxplot_kwargs=boxplot_kwargs,
            stats_kwargs=stats_kwargs,
            pairs_of_groups_for_stats=PAIRS_OF_GROUPS,
            variable_ylim=None,
            variable_y_offset=variables_y_offsets[variable],
        )
        _boxplots_one_variable_with_stats(
            ax_boxplot_diff,
            data_stats,
            (f"{variable[0]}_diff", variable[1]),
            num_data_points=num_data_points,
            boxplot_kwargs=boxplot_kwargs,
            stats_kwargs=stats_kwargs,
            pairs_of_groups_for_stats=PAIRS_OF_GROUPS,
            variable_ylim=None,
            variable_y_offset=variables_y_offsets[
                (f"{variable[0]}_diff", variable[1])
            ],
        )
        _boxplots_one_variable_with_stats(
            ax_boxplot_standardized,
            data_stats,
            (f"{variable[0]}_standardized", variable[1]),
            num_data_points=num_data_points,
            boxplot_kwargs=boxplot_kwargs,
            stats_kwargs=stats_kwargs,
            pairs_of_groups_for_stats=PAIRS_OF_GROUPS,
            variable_ylim=None,
            variable_y_offset=variables_y_offsets[
                (f"{variable[0]}_standardized", variable[1])
            ],
        )
    return fig


def _plot_partition_group_vars_summary(
    data, data_stats, variables, variables_ranges, variables_stats, hue=None
):
    num_genotype_groups = len(data["genotype_group"].unique())
    (
        fig,
        axs_order_params_dist,
        axs_distributions,
        axs_boxplots_raw,
        axs_boxplots_diff,
        axs_boxplots_standardized,
    ) = _prepare_partition_group_vars_summary_fig(
        len(variables_stats), num_genotype_groups
    )
    plot_order_parameter_dist_per_genotype_group(data, axs_order_params_dist)
    num_data_points = _get_num_data_points(data_stats, boxplot_kwargs_group)
    for i, (variable, ax_dist) in enumerate(zip(variables, axs_distributions)):
        if i == 0:
            legend = True
        else:
            legend = False
        plot_variable_1d_distribution(
            data,
            variable,
            variables_ranges,
            ax=ax_dist,
            hue=hue,
            legend=legend,
            how="v",
        )
    for i, (
        variable,
        ax_boxplot,
        ax_boxplot_diff,
        ax_boxplot_standardized,
    ) in enumerate(
        zip(
            variables_stats,
            axs_boxplots_raw,
            axs_boxplots_diff,
            axs_boxplots_standardized,
        )
    ):
        if i == 0:
            legend = True
        else:
            legend = False
        (
            variables_ylims,
            variables_y_offsets,
        ) = _get_variables_ylims_and_offsets(data_stats)
        _update_order(data, boxplot_kwargs_group, GENOTYPE_GROUP_ORDER)
        _boxplots_one_variable_with_stats(
            ax_boxplot,
            data_stats,
            variable,
            num_data_points=num_data_points,
            boxplot_kwargs=boxplot_kwargs_group,
            stats_kwargs=stats_kwargs,
            pairs_of_groups_for_stats=PAIRS_OF_GROUPS,
            variable_ylim=None,
            variable_y_offset=variables_y_offsets[variable],
        )
        _boxplots_one_variable_with_stats(
            ax_boxplot_diff,
            data_stats,
            (f"{variable[0]}_diff", variable[1]),
            num_data_points=num_data_points,
            boxplot_kwargs=boxplot_kwargs_group,
            stats_kwargs=stats_kwargs,
            pairs_of_groups_for_stats=PAIRS_OF_GROUPS,
            variable_ylim=None,
            variable_y_offset=variables_y_offsets[
                (f"{variable[0]}_diff", variable[1])
            ],
        )
        _boxplots_one_variable_with_stats(
            ax_boxplot_standardized,
            data_stats,
            (f"{variable[0]}_standardized", variable[1]),
            num_data_points=num_data_points,
            boxplot_kwargs=boxplot_kwargs_group,
            stats_kwargs=stats_kwargs,
            pairs_of_groups_for_stats=PAIRS_OF_GROUPS,
            variable_ylim=None,
            variable_y_offset=variables_y_offsets[
                (f"{variable[0]}_standardized", variable[1])
            ],
        )
    return fig


def _plot_partition_indiv_nb_summary(
    data, data_stats, variables, variables_ranges, variables_stats, hue=None
):
    num_focal_nb_genotype_groups = len(data["focal_nb_genotype"].unique())
    (
        fig,
        axs_polar_plots,
        axs_distributions,
        axs_boxplots_raw,
        # axs_boxplots_diff,
        # axs_boxplots_standardized,
    ) = _prepare_partition_indiv_nb_vars_summary_fig(
        len(variables_stats), num_focal_nb_genotype_groups
    )
    _plot_polar_dist_relative_positions(data, axs_polar_plots)
    num_data_points = _get_num_data_points(data_stats, boxplot_kwargs_indiv_nb)
    for i, (variable, ax_dist) in enumerate(zip(variables, axs_distributions)):
        if i == 0:
            legend = True
        else:
            legend = False
        plot_variable_1d_distribution(
            data,
            variable,
            variables_ranges,
            ax=ax_dist,
            hue=hue,
            legend=legend,
            how="v",
        )
    for i, (
        variable,
        ax_boxplot,
        # ax_boxplot_diff,
        # ax_boxplot_standardized,
    ) in enumerate(
        zip(
            variables_stats,
            axs_boxplots_raw,
            # axs_boxplots_diff,
            # axs_boxplots_standardized,
        )
    ):
        if i == 0:
            legend = True
        else:
            legend = False
        (
            variables_ylims,
            variables_y_offsets,
        ) = _get_variables_ylims_and_offsets(data_stats)
        _update_order(data, boxplot_kwargs_indiv_nb, FOCAL_NB_GENOTYPE_ORDER)
        _boxplots_one_variable_with_stats(
            ax_boxplot,
            data_stats,
            variable,
            num_data_points=num_data_points,
            boxplot_kwargs=boxplot_kwargs_indiv_nb,
            stats_kwargs=stats_kwargs,
            pairs_of_groups_for_stats=PAIRS_OF_GROUPS,
            variable_ylim=None,
            variable_y_offset=variables_y_offsets[variable],
        )
        # _boxplots_one_variable_with_stats(
        #     ax_boxplot_diff,
        #     data_stats,
        #     (f"{variable[0]}_diff", variable[1]),
        #     num_data_points=num_data_points,
        #     boxplot_kwargs=boxplot_kwargs_indiv_nb,
        #     stats_kwargs=stats_kwargs,
        #     pairs_of_groups_for_stats=PAIRS_OF_GROUPS,
        #     variable_ylim=None,
        #     variable_y_offset=variables_y_offsets[(f"{variable[0]}_diff", variable[1])],
        # )
        # _boxplots_one_variable_with_stats(
        #     ax_boxplot_standardized,
        #     data_stats,
        #     (f"{variable[0]}_standardized", variable[1]),
        #     num_data_points=num_data_points,
        #     boxplot_kwargs=boxplot_kwargs_indiv_nb,
        #     stats_kwargs=stats_kwargs,
        #     pairs_of_groups_for_stats=PAIRS_OF_GROUPS,
        #     variable_ylim=None,
        #     variable_y_offset=variables_y_offsets[(f"{variable[0]}_standardized", variable[1])],
        # )
    return fig


###### SUMMARY PLOTS


def plot_summary_animal(
    datasets, animal_col, animal_uid, variables_ranges, save=False
):
    datasets_partition = _select_partition_from_datasets(
        datasets, ["data_indiv"], animal_col, animal_uid
    )
    animal_info_str = get_animal_info_str(datasets_partition["data_indiv"])

    fig = _plot_animal_indiv_vars_summary(
        datasets_partition["data_indiv"],
        INDIVIDUAL_VARIABLES_TO_PLOT,
        variables_ranges,
    )
    fig.suptitle(animal_info_str)
    if save:
        fig.savefig(f"{animal_uid}.png")
        fig.savefig(f"{animal_uid}.pdf")


def plot_summary_video(
    datasets, video_col, video_uid, animal_col, variables_ranges, save=False
):
    datasets_partition = _select_partition_from_datasets(
        datasets,
        ["data_indiv", "data_group", "data_indiv_nb"],
        video_col,
        video_uid,
    )

    video_info_str = get_video_info_str(datasets_partition["data_indiv"])
    print(video_info_str)
    fig = _plot_video_indiv_vars_summary(
        datasets_partition["data_indiv"],
        INDIVIDUAL_VARIABLES_TO_PLOT,
        variables_ranges,
        hue="identity",
    )
    fig.suptitle(video_info_str)
    if save:
        fig.savefig(f"{video_uid}_indiv.png")
        fig.savefig(f"{video_uid}_indiv.pdf")

    fig = _plot_group_variables_summary(
        datasets_partition["data_group"], GROUP_VARIABLES_TO_PLOT, variables_ranges
    )
    fig.suptitle(video_info_str)
    if save:
        fig.savefig(f"{video_uid}_group.png")
        fig.savefig(f"{video_uid}_group.pdf")

    for animal_uid in datasets_partition["data_indiv_nb"][animal_col].unique():
        animal_nb_data = datasets_partition["data_indiv_nb"][
            datasets_partition["data_indiv_nb"][animal_col] == animal_uid
        ]
        fig = _plot_video_indiv_nb_variables_summary(
            animal_nb_data, INDIVIDUAL_NB_VARIABLES_TO_PLOT, variables_ranges
        )
        focal_nb_info_str = get_focal_nb_info(animal_nb_data)
        fig.suptitle(focal_nb_info_str)
        if save:
            fig.savefig(f"{animal_uid}_indiv_nb.png")
            fig.savefig(f"{animal_uid}_indiv_nb.pdf")


def plot_summary_partition(
    datasets,
    partition_col,
    partition_uid,
    variables_ranges,
    save=False,
):
    datasets_partition = _select_partition_from_datasets(
        datasets,
        [
            "data_indiv",
            "data_group",
            "data_indiv_nb",
            "data_indiv_stats",
            "data_group_stats",
            "data_indiv_nb_stats",
        ],
        partition_col,
        partition_uid,
    )

    line_replicate_info_str = get_partition_info_str(
        datasets_partition["data_indiv"], partition_col
    )
    fig = _plot_partition_indiv_vars_summary(
        datasets_partition["data_indiv"],
        datasets_partition["data_indiv_stats"],
        INDIVIDUAL_VARIABLES_TO_PLOT,
        variables_ranges,
        INDIVIDUAL_VARIABLES_STATS_TO_PLOT,
        hue="genotype_group_genotype",
    )
    fig.suptitle(line_replicate_info_str)
    if save:
        fig.savefig(f"{partition_uid}_indiv.png")
        fig.savefig(f"{partition_uid}_indiv.pdf")

    fig = _plot_partition_group_vars_summary(
        datasets_partition["data_group"],
        datasets_partition["data_group_stats"],
        GROUP_VARIABLES_TO_PLOT,
        variables_ranges,
        GROUP_VARIABLES_STATS_TO_PLOT,
        hue="genotype_group",
    )
    fig.suptitle(line_replicate_info_str)
    if save:
        fig.savefig(f"{partition_uid}_group.png")
        fig.savefig(f"{partition_uid}_group.pdf")

    fig = _plot_partition_indiv_nb_summary(
        datasets_partition["data_indiv_nb"],
        datasets_partition["data_indiv_nb_stats"],
        INDIVIDUAL_NB_VARIABLES_TO_PLOT,
        variables_ranges,
        INDIVIDUAL_NB_VARIALBES_STATS_TO_PLOT,
        hue="focal_nb_genotype",
    )
    fig.suptitle(line_replicate_info_str)
    if save:
        fig.savefig(f"{partition_uid}_indiv_nb.png")
        fig.savefig(f"{partition_uid}_indiv_nb.pdf")



def plot_variables_partition_summary(path_with_data, variables=variables):
    data = pd.read_csv(os.path.join(path_with_data, "data.csv"))

    normalized_data = standardized_data

    datasets = {
        "line_replicate_data": {"data": data, "x": "line_replicate"},
        "line_data": {"data": data, "x": "line"},
        "line_replicate_diff_data": {
            "data": diff_data,
            "x": "line_replicate",
        },
        "line_diff_data": {"data": diff_data, "x": "line"},
        "line_replicate_standardized_data": {
            "data": standardized_data,
            "x": "line_replicate",
        },
        "line_standardized_data": {
            "data": standardized_data,
            "x": "line",
        },
    }
    fig, axs = plt.subplots(
        len(datasets),
        len(variables),
        figsize=(30 * len(variables), 10 * len(datasets)),
    )
    for axs_col, variable in zip(axs.T, variables):
        for dataset_name, ax in zip(datasets.keys(), axs_col):
            data = datasets[dataset_name]["data"]
            x = datasets[dataset_name]["x"]
            ax.set_title(dataset_name)
            _boxplot_one_variable(
                ax,
                data,
                {
                    "x": x,
                    "y": variable,
                    "hue": "genotype_group_genotype",
                    "palette": COLORS,
                    "order": normalized_data[x].unique(),
                    "whis": 1.5,
                },
            )
            _boxplot_axes_one_variable(ax, data, variable)
    fig.savefig(os.path.join(path_with_data, "normalization.png"))

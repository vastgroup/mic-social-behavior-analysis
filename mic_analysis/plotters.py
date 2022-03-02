import logging
import os
from pprint import pprint

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import trajectorytools as tt
from confapp import conf
from matplotlib.gridspec import GridSpec
from natsort import natsorted
from tqdm import tqdm
from trajectorytools.export.variables import GROUP_VARIABLES
from trajectorytools.plot import plot_polar_histogram, polar_histogram

from .datasets import TRAJECTORYTOOLS_DATASETS_INFO
from .stats import _compute_groups_stats, _get_num_data_points
from .string_infos import (get_animal_info_str, get_focal_nb_info,
                           get_partition_info_str, get_video_info_str)
from .utils import _select_partition_from_datasets, circmean, circstd
from .variables import all_variables_names, all_variables_names_enhanced

logger = logging.getLogger(__name__)

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.size"] = "14"


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
            y = min_y_group - y_offset * conf.RATIO_Y_OFFSET
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
        y_line_stat + conf.RATIO_Y_OFFSET * y_offset,
        f"{var_stat['value']:.2f}, {var_stat['p_value']:.4f}",
        ha="center",
        va="bottom",
        alpha=alpha,
    )
    y_min, y_max = y_lim
    ax.plot([x_group_a], [var_stat["stat_a"]], "ok")
    ax.plot([x_group_b], [var_stat["stat_b"]], "ok")
    if y_line_stat + y_offset > y_max:
        ax.set_ylim([y_min, y_line_stat + y_offset])


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
        y_line_stat = y_start + actual_plot_level * y_offset
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
        var_test_stats, outliers = _compute_groups_stats(
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
            var_test_stats,
            variable,
            boxplot_kwargs["order"],
            y_lim=ax.get_ylim(),
            y_offset=variable_y_offset,
        )
    else:
        var_test_stats = None
        outliers = None
    return var_test_stats, outliers, ax.get_ylim()


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
    valid_x_values=conf.GENOTYPE_GROUP_GENOTYPE_ORDER,
    variables_ylims=None,
    varialbes_y_offsets=None,
):
    assert len(axs) == len(variables), f"{len(variables)} {len(axs)}"
    num_data_points = _get_num_data_points(data, boxplot_kwargs)
    _update_order(data, boxplot_kwargs, valid_x_values)

    all_var_stats = []
    all_outliers = []

    for i, (ax, variable) in enumerate(zip(axs, variables)):
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
    all_outliers.drop_duplicates(inplace=True)
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
        varialbes_y_offsets[variable] = (
            np.abs(y_max - y_min) * conf.VARIABLE_RATIO_Y_OFFSET
        )
    return variables_ylims, varialbes_y_offsets


def plot(config_dict):
    logger.info(f"Plotting with {config_dict}")
    data_info = TRAJECTORYTOOLS_DATASETS_INFO[
        config_dict["data_variables_group"]
    ]
    data, data_stats = get_data(
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
    all_outliers.drop_duplicates(inplace=True)

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
                "palette": conf.COLORS,
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
    fig = plt.figure(constrained_layout=True, figsize=(30, 5 * num_variables))
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
    fig = plt.figure(constrained_layout=True, figsize=(30, 5 * num_variables))
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
    fig = plt.figure(constrained_layout=True, figsize=(30, 5 * num_variables))
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


##### AXES PLOTTERS
def _plot_order_parameter_dist(data, ax=None):
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


def _plot_relative_position_dist(data, ax=None):
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


def _plot_trajectory(
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


def _plot_variable_along_time(
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
                palette=conf.COLORS,
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


def _plot_variable_1d_dist(
    data, variable, variables_ranges, ax=None, hue=None, legend=None, how="h"
):
    bin_range = (
        variables_ranges[variables_ranges.variable == variable]["min"].values[
            0
        ],
        variables_ranges[variables_ranges.variable == variable]["max"].values[
            0
        ],
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
            palette=conf.COLORS,
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


def _plot_positions_dist_per_genotype_group(data, axs):
    genotype_groups = data["genotype_group"].unique()
    for i, (genotype_group, ax) in enumerate(zip(genotype_groups, axs)):
        sub_data = data[data.genotype_group == genotype_group]
        _plot_trajectory(
            sub_data,
            ax,
            show_trajectories=False,
            x_var="s_x_normed",
            y_var="s_y_normed",
        )
        ax.set_title(genotype_group)
    _plot_trajectory(
        data,
        axs[i + 1],
        show_trajectories=False,
        x_var="s_x_normed",
        y_var="s_y_normed",
    )
    axs[i + 1].set_title("all")
    [ax.set_visible(False) for ax in axs[i + 2 :]]


def _plot_order_parameter_dist_per_genotype_group(data, axs):
    genotype_groups = data["genotype_group"].unique()
    for i, (genotype_group, ax) in enumerate(zip(genotype_groups, axs)):
        sub_data = data[data.genotype_group == genotype_group]
        _plot_order_parameter_dist(sub_data, ax)
        ax.set_title(genotype_group)
    _plot_order_parameter_dist(data, axs[i + 1])
    axs[i + 1].set_title("all")
    if i + 1 < len(axs) - 1:
        # is not the last axes
        [ax.set_visible(False) for ax in axs[i + 2 :]]


def _plot_polar_dist_relative_positions(data, axs_polar_plots):
    valid_focal_nb_genotype = data.focal_nb_genotype.unique()
    focal_nb_genotype_order = [
        fng
        for fng in conf.FOCAL_NB_GENOTYPE_ORDER
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
    if i + 1 < len(axs_polar_plots) - 1:
        # is not the last axes
        [ax.set_visible(False) for ax in axs_polar_plots[i + 2 :]]


##### SUMMARY FIGURES


def _plot_animal_indiv_vars_summary(
    data, variables, variables_ranges, hue="genotype"
):
    (
        fig,
        ax_trajectories,
        axs_variables,
        axs_distributions,
    ) = _prepare_animal_indiv_vars_fig(len(variables))
    _plot_trajectory(data, ax=ax_trajectories, hue=hue)
    for variable, ax_time, ax_dist in zip(
        variables, axs_variables, axs_distributions
    ):
        _plot_variable_along_time(data, variable, ax=ax_time, hue=hue)
        _plot_variable_1d_dist(
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
    _plot_trajectory(data, ax=ax_trajectories, hue=hue)
    for variable, ax_time, ax_dist in zip(
        variables, axs_variables, axs_distributions
    ):
        _plot_variable_along_time(data, variable, ax=ax_time, hue=hue)
        _plot_variable_1d_dist(
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
    _plot_order_parameter_dist(data, ax=ax_order_params)
    for variable, ax_time, ax_dist in zip(
        variables, axs_variables, axs_distributions
    ):
        _plot_variable_along_time(data, variable, ax=ax_time)
        _plot_variable_1d_dist(data, variable, variables_ranges, ax=ax_dist)
    return fig


def _plot_video_indiv_nb_variables_summary(data, variables, variables_ranges):
    (
        fig,
        ax_order_params,
        axs_variables,
        axs_distributions,
    ) = _prepare_video_indiv_nb_fig(len(variables))
    _plot_relative_position_dist(data, ax=ax_order_params)
    for variable, ax_time, ax_dist in zip(
        variables, axs_variables, axs_distributions
    ):
        _plot_variable_along_time(
            data, variable, ax=ax_time, hue="genotype_nb", units="identity_nb"
        )
        _plot_variable_1d_dist(
            data, variable, variables_ranges, ax=ax_dist, hue="genotype_nb"
        )
    return fig


def _plot_partition_indiv_vars_summary(
    data,
    data_stats,
    variables,
    variables_ranges,
    variables_stats,
    boxplot_kwargs,
    stats_kwargs,
    pairs_of_groups_for_stats=conf.PAIRS_OF_GROUPS,
    hue=None,
):
    num_genotype_groups = len(data["genotype_group"].unique())
    # TODO: Need to add more axes for boxplots if we consider different
    # mean and median stats
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
    _plot_positions_dist_per_genotype_group(data, axs=axs_positions_dist)
    num_data_points = _get_num_data_points(data_stats, boxplot_kwargs)
    for i, (variable, ax_dist) in enumerate(zip(variables, axs_distributions)):
        if i == 0:
            legend = True
        else:
            legend = False
        _plot_variable_1d_dist(
            data,
            variable,
            variables_ranges,
            ax=ax_dist,
            hue=hue,
            legend=legend,
            how="v",
        )
    [ax.set_visible(False) for ax in axs_distributions[i + 1 :]]
    outliers = []
    test_stats = []
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
        _update_order(data, boxplot_kwargs, conf.GENOTYPE_GROUP_GENOTYPE_ORDER)
        variable_stats = [
            variable,
            (f"{variable[0]}_diff", variable[1]),
            (f"{variable[0]}_standardized", variable[1]),
        ]
        axs = [ax_boxplot, ax_boxplot_diff, ax_boxplot_standardized]
        for variable_stat, ax in zip(variable_stats, axs):
            # TODO: Here loop for mean/median stats and whis=100 vs 1.5
            var_test_stats, outliers_, _ = _boxplots_one_variable_with_stats(
                ax,
                data_stats,
                variable_stat,
                num_data_points,
                boxplot_kwargs,
                stats_kwargs,
                pairs_of_groups_for_stats,
                variable_ylim=None,
                variable_y_offset=variables_y_offsets[variable_stat],
            )
            outliers.extend(outliers_)
            test_stats.extend(var_test_stats)
    outliers = pd.concat(outliers)
    outliers.drop_duplicates(inplace=True)
    test_stats = pd.DataFrame(test_stats)
    return fig, outliers, test_stats


def _plot_partition_group_vars_summary(
    data,
    data_stats,
    variables,
    variables_ranges,
    variables_stats,
    boxplot_kwargs,
    stats_kwargs,
    pairs_of_groups_for_stats=conf.PAIRS_OF_GROUPS,
    hue=None,
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
    _plot_order_parameter_dist_per_genotype_group(data, axs_order_params_dist)
    num_data_points = _get_num_data_points(data_stats, boxplot_kwargs)
    outliers = []
    test_stats = []
    for i, (variable, ax_dist) in enumerate(zip(variables, axs_distributions)):
        if i == 0:
            legend = True
        else:
            legend = False
        _plot_variable_1d_dist(
            data,
            variable,
            variables_ranges,
            ax=ax_dist,
            hue=hue,
            legend=legend,
            how="v",
        )
    [ax.set_visible(False) for ax in axs_distributions[i + 1 :]]
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
        _update_order(data, boxplot_kwargs, conf.GENOTYPE_GROUP_ORDER)
        variable_stats = [
            variable,
            (f"{variable[0]}_diff", variable[1]),
            (f"{variable[0]}_standardized", variable[1]),
        ]
        axs = [ax_boxplot, ax_boxplot_diff, ax_boxplot_standardized]
        for variable_stat, ax in zip(variable_stats, axs):
            var_test_stats, outliers_, _ = _boxplots_one_variable_with_stats(
                ax,
                data_stats,
                variable_stat,
                num_data_points,
                boxplot_kwargs,
                stats_kwargs,
                pairs_of_groups_for_stats,
                variable_ylim=None,
                variable_y_offset=variables_y_offsets[variable_stat],
            )
            outliers.extend(outliers_)
            test_stats.extend(var_test_stats)
    outliers = pd.concat(outliers)
    outliers.drop_duplicates(inplace=True)
    test_stats = pd.DataFrame(test_stats)
    return fig, outliers, test_stats


def _plot_partition_indiv_nb_summary(
    data,
    data_stats,
    variables,
    variables_ranges,
    variables_stats,
    boxplot_kwargs,
    stats_kwargs,
    pairs_of_groups_for_stats=conf.PAIRS_OF_GROUPS,
    hue=None,
):
    num_focal_nb_genotype_groups = len(data["focal_nb_genotype"].unique())
    (
        fig,
        axs_polar_plots,
        axs_distributions,
        axs_boxplots_raw,
        axs_boxplots_diff,
        axs_boxplots_standardized,
    ) = _prepare_partition_indiv_nb_vars_summary_fig(
        len(variables_stats), num_focal_nb_genotype_groups
    )
    _plot_polar_dist_relative_positions(data, axs_polar_plots)
    num_data_points = _get_num_data_points(data_stats, boxplot_kwargs)
    for i, (variable, ax_dist) in enumerate(zip(variables, axs_distributions)):
        if i == 0:
            legend = True
        else:
            legend = False
        _plot_variable_1d_dist(
            data,
            variable,
            variables_ranges,
            ax=ax_dist,
            hue=hue,
            legend=legend,
            how="v",
        )
    [ax.set_visible(False) for ax in axs_distributions[i + 1 :]]
    outliers = []
    test_stats = []
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
        _update_order(data, boxplot_kwargs, conf.FOCAL_NB_GENOTYPE_ORDER)
        variable_stats = [
            variable,
            (f"{variable[0]}_diff", variable[1]),
            (f"{variable[0]}_standardized", variable[1]),
        ]
        axs = [ax_boxplot, ax_boxplot_diff, ax_boxplot_standardized]

        for variable_stat, ax in zip(variable_stats, axs):
            if variable_stat in data_stats.columns:
                (
                    var_test_stats,
                    outliers_,
                    _,
                ) = _boxplots_one_variable_with_stats(
                    ax,
                    data_stats,
                    variable_stat,
                    num_data_points,
                    boxplot_kwargs,
                    stats_kwargs,
                    pairs_of_groups_for_stats,
                    variable_ylim=None,
                    variable_y_offset=variables_y_offsets[variable_stat],
                )
                outliers.extend(outliers_)
                test_stats.extend(var_test_stats)
            else:
                ax.set_visible(False)
    outliers = pd.concat(outliers)
    outliers.drop_duplicates(inplace=True)
    test_stats = pd.DataFrame(test_stats)
    return fig, outliers, test_stats


###### SUMMARY PLOTS


def plot_summary_animal(
    datasets,
    animal_col,
    animal_uid,
    variables_ranges,
    save=False,
    save_path=".",
):
    datasets_partition = _select_partition_from_datasets(
        datasets, ["data_indiv"], animal_col, animal_uid
    )
    animal_info_str = get_animal_info_str(datasets_partition["data_indiv"])

    fig = _plot_animal_indiv_vars_summary(
        datasets_partition["data_indiv"],
        conf.INDIVIDUAL_VARIABLES_TO_PLOT,
        variables_ranges,
    )
    fig.suptitle(animal_info_str)
    if save:
        fig.savefig(os.path.join(save_path, f"{animal_uid}.png"))
        fig.savefig(os.path.join(save_path, f"{animal_uid}.pdf"))


def plot_summary_video(
    datasets,
    video_col,
    video_uid,
    animal_col,
    variables_ranges,
    save=False,
    save_path=".",
):
    datasets_partition = _select_partition_from_datasets(
        datasets,
        ["data_indiv", "data_group", "data_indiv_nb"],
        video_col,
        video_uid,
    )

    video_info_str = get_video_info_str(datasets_partition["data_indiv"])
    fig = _plot_video_indiv_vars_summary(
        datasets_partition["data_indiv"],
        conf.INDIVIDUAL_VARIABLES_TO_PLOT,
        variables_ranges,
        hue="identity",
    )
    fig.suptitle(video_info_str)
    if save:
        fig.savefig(os.path.join(save_path, f"{video_uid}_indiv.png"))
        fig.savefig(os.path.join(save_path, f"{video_uid}_indiv.pdf"))

    fig = _plot_group_variables_summary(
        datasets_partition["data_group"],
        conf.GROUP_VARIABLES_TO_PLOT,
        variables_ranges,
    )
    fig.suptitle(video_info_str)
    if save:
        fig.savefig(os.path.join(save_path, f"{video_uid}_group.png"))
        fig.savefig(os.path.join(save_path, f"{video_uid}_group.pdf"))

    for animal_uid in datasets_partition["data_indiv_nb"][animal_col].unique():
        animal_nb_data = datasets_partition["data_indiv_nb"][
            datasets_partition["data_indiv_nb"][animal_col] == animal_uid
        ]
        fig = _plot_video_indiv_nb_variables_summary(
            animal_nb_data,
            conf.INDIVIDUAL_NB_VARIABLES_TO_PLOT,
            variables_ranges,
        )
        focal_nb_info_str = get_focal_nb_info(animal_nb_data)
        fig.suptitle(focal_nb_info_str)
        if save:
            fig.savefig(os.path.join(save_path, f"{animal_uid}_indiv_nb.png"))
            fig.savefig(os.path.join(save_path, f"{animal_uid}_indiv_nb.pdf"))


def plot_summary_partition(
    datasets,
    partition_col,
    partition_uid,
    variables_ranges,
    indiv_boxplot_kwargs=conf.INDIV_BOXPLOT_KWARGS,
    group_boxplot_kwargs=conf.GROUP_BOXPLOT_KWARGS,
    indiv_nb_boxplot_kwargs=conf.INDIV_NB_BOXPLOT_KWARGS,
    stats_kwargs=conf.MEAN_STATS_CONFIG,
    save=False,
    save_path=".",
):
    save_path = os.path.join(save_path, partition_uid)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
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

    all_outliers = []
    all_test_stats = []

    line_replicate_info_str = get_partition_info_str(
        datasets_partition["data_indiv"], partition_col
    )
    fig, outliers, test_stats = _plot_partition_indiv_vars_summary(
        datasets_partition["data_indiv"],
        datasets_partition["data_indiv_stats"],
        conf.INDIVIDUAL_VARIABLES_TO_PLOT,
        variables_ranges,
        conf.INDIVIDUAL_VARIABLES_STATS_TO_PLOT,
        indiv_boxplot_kwargs,
        stats_kwargs,
        hue="genotype_group_genotype",
    )
    outliers["var_type"] = ["indiv"] * len(outliers)
    test_stats["var_type"] = ["indiv"] * len(test_stats)
    all_outliers.append(outliers)
    all_test_stats.append(test_stats)
    fig.suptitle(line_replicate_info_str)
    if save:
        fig.savefig(os.path.join(save_path, f"indiv_vars.png"))
        fig.savefig(os.path.join(save_path, f"indiv_vars.pdf"))
        outliers.to_csv(
            os.path.join(save_path, "indiv_outliers.csv"), index=False
        )
        test_stats.to_csv(
            os.path.join(save_path, "indiv_test_stats.csv"), index=False
        )

    fig, outliers, test_stats = _plot_partition_group_vars_summary(
        datasets_partition["data_group"],
        datasets_partition["data_group_stats"],
        conf.GROUP_VARIABLES_TO_PLOT,
        variables_ranges,
        conf.GROUP_VARIABLES_STATS_TO_PLOT,
        group_boxplot_kwargs,
        stats_kwargs,
        hue="genotype_group",
    )
    outliers["var_type"] = ["group"] * len(outliers)
    test_stats["var_type"] = ["group"] * len(test_stats)
    all_outliers.append(outliers)
    all_test_stats.append(test_stats)

    fig.suptitle(line_replicate_info_str)
    if save:
        fig.savefig(os.path.join(save_path, f"group_vars.png"))
        fig.savefig(os.path.join(save_path, f"group_vars.pdf"))
        outliers.to_csv(
            os.path.join(save_path, "group_outliers.csv"), index=False
        )
        test_stats.to_csv(
            os.path.join(save_path, "group_test_stats.csv"), index=False
        )

    fig, outliers, test_stats = _plot_partition_indiv_nb_summary(
        datasets_partition["data_indiv_nb"],
        datasets_partition["data_indiv_nb_stats"],
        conf.INDIVIDUAL_NB_VARIABLES_TO_PLOT,
        variables_ranges,
        conf.INDIVIDUAL_NB_VARIALBES_STATS_TO_PLOT,
        indiv_nb_boxplot_kwargs,
        stats_kwargs,
        hue="focal_nb_genotype",
    )
    outliers["var_type"] = ["indiv_nb"] * len(outliers)
    test_stats["var_type"] = ["indiv_nb"] * len(test_stats)
    all_outliers.append(outliers)
    all_test_stats.append(test_stats)

    fig.suptitle(line_replicate_info_str)
    if save:
        fig.savefig(os.path.join(save_path, f"indiv_nb_vars.png"))
        fig.savefig(os.path.join(save_path, f"indiv_nb_vars.pdf"))
        outliers.to_csv(
            os.path.join(save_path, "indiv_nb_outliers.csv"), index=False
        )
        test_stats.to_csv(
            os.path.join(save_path, "indiv_nb_test_stats.csv"), index=False
        )

    all_outliers = pd.concat(all_outliers)
    all_test_stats = pd.concat(all_test_stats)
    if save:
        all_outliers.to_csv(
            os.path.join(save_path, "all_outliers.csv"), index=False
        )
        all_test_stats.to_csv(
            os.path.join(save_path, "all_test_stats.csv"), index=False
        )
    return all_outliers, all_test_stats


def plot_summary_all_partitions_with_outliers(
    datasets, variables_ranges, partition_col
):

    possible_partition_uids = natsorted(
        datasets["data_group"][partition_col].unique()
    )

    save_path = os.path.join(
        conf.GENERATED_FIGURES_PATH, f"summary_{partition_col}"
    )
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for partition in possible_partition_uids:
        logger.info(f"Plotting {partition} summary")
        plot_summary_partition(
            datasets,
            partition_col,
            partition,
            variables_ranges,
            save=True,
            save_path=save_path,
        )


### Video sanity check summary
def count_nan_intervals(indiv_df, id_):
    indiv_df = indiv_df[indiv_df.identity == id_][["s_x"]]
    df = (
        indiv_df.s_x.isnull()
        .astype(int)
        .groupby(indiv_df.s_x.notnull().astype(int).cumsum())
        .sum()
    )
    return df.value_counts


def get_nans_array(tr):
    nans_bool_array = np.isnan(tr.s[..., 0].T)
    return nans_bool_array


def visualize_nans(ax, tr, max_num_frames):
    tx = get_nans_array(tr)
    video = (
        np.zeros((tr.number_of_individuals, max_num_frames + 2000)) * np.nan
    )
    video[:, : tx.shape[1]] = tx
    ax.imshow(video, interpolation="None", origin="lower")
    ax.set_aspect("auto")


def plot_video_tracking_states(videos_table, state="for_analysis_state"):
    videos_table.sort_values("folder_name_track", inplace=True)
    folder_name_tracks = videos_table.folder_name_track.unique()

    if state == "for_analysis_state":
        title = (
            f"{state}: \n"
            + "   ".join(conf.TRACKING_STATE_COLUMNS)
            + "\n"
            + "   ".join(conf.ID_LAST_FISH_STATE_COLUMNS)
        )
    elif state == "tracking_state":
        title = f"{state}: \n" + "   ".join(conf.TRACKING_STATE_COLUMNS)
    elif state == "id_last_fish_state":
        title = f"{state}: \n" + "   ".join(conf.ID_LAST_FISH_STATE_COLUMNS)
    else:
        raise Exception(f"No valid state {state}")

    fig, axs = plt.subplots(6, 6, figsize=(30, 30))
    fig.suptitle(title)
    plt.subplots_adjust(
        left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.4
    )

    videos_table.sort_values(
        [state, "genotype_group"], inplace=True, ascending=False
    )
    for folder_name_track, ax in zip(folder_name_tracks, axs.flatten()):
        sub_videos = videos_table[
            videos_table.folder_name_track == folder_name_track
        ]
        sns.countplot(
            ax=ax,
            data=sub_videos,
            x=state,
            hue="genotype_group",
            order=videos_table[state].unique(),
        )
        ax.set_title(folder_name_track)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    for extension in [".png", ".pdf"]:
        fig.savefig(
            os.path.join(conf.GENERATED_FIGURES_PATH, f"{state}{extension}")
        )


def visualize_speed_jumps(ax, speed_jumps):
    if not "[]" in speed_jumps:
        speed_jumps = eval(speed_jumps.replace("array", "np.array"))
        x = speed_jumps[0]
        y = speed_jumps[1]
        ax.plot(x, y, "r.", markersize=3)


def plot_tracked_videos_summary(videos_table):

    videos_table = videos_table.sort_values(["trial_uid"]).reset_index(
        drop=True
    )

    n_trajectories = len(videos_table)
    max_num_frames = int(videos_table.number_of_frames.max())

    fig, axs = plt.subplots(
        n_trajectories, 1, figsize=(30, 0.2 * n_trajectories)
    )
    plt.subplots_adjust(
        left=0.01, bottom=0.01, right=0.7, top=0.99, wspace=0.01, hspace=0.01
    )

    for idx, video_info in tqdm(
        videos_table.iterrows(), desc="Plotting nans..."
    ):
        ax = axs[idx]
        tr = tt.trajectories.FishTrajectories.from_idtrackerai(
            video_info.abs_trajectory_path, interpolate_nans=False
        )

        visualize_nans(ax, tr, max_num_frames)
        visualize_speed_jumps(ax, video_info.speed_jumps)
        ax.set_aspect("auto")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        if not video_info.valid_genotype_id:
            ax.plot([max_num_frames + 200], [1], "rs", ms=3)
        if not video_info.valid_tracking:
            ax.plot([max_num_frames + 300], [1], "ro", ms=3)
        if video_info.valid_for_analysis:
            ax.plot([max_num_frames + 100], [1], "g^", ms=3)

        ax.text(
            max_num_frames + 2000,
            0,
            f"{video_info.for_analysis_state}",
            ha="right",
        )
        ax.text(
            max_num_frames + 2100,
            0,
            f"trial: {video_info.trial_uid} - "
            f"tracked: {video_info.ratio_frames_tracked:.3f} - "
            f"id_probs: {video_info.mean_id_probabilities:.3f} - "
            f"estimated_accuracy: {video_info.estimated_accuracy:.3f} - "
            f"speed jumps: {video_info.num_impossible_speed_jumps:.0f} - "
            f"id: {video_info.id_last_fish:.0f} - "
            f"certainty: {video_info.certainty_id_last_fish:.3f}",
            ha="left",
            c="k",
        )
    code_legend = " - ".join(
        [
            "   ".join(conf.TRACKING_STATE_COLUMNS),
            "   ".join(conf.ID_LAST_FISH_STATE_COLUMNS),
        ]
    )
    fig.suptitle("Ready for analysis state code: \n" + code_legend, y=0.995)
    for extension in [".png", ".pdf"]:
        fig.savefig(
            os.path.join(
                conf.GENERATED_FIGURES_PATH, f"videos_summary{extension}"
            )
        )

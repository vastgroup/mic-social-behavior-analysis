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
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec
from natsort import natsorted
from tqdm import tqdm
from trajectorytools.export.variables import GROUP_VARIABLES
from trajectorytools.plot import plot_polar_histogram, polar_histogram

from .datasets import TRAJECTORYTOOLS_DATASETS_INFO, get_partition_datasets
from .stats import _compute_groups_stats, _get_num_data_points
from .string_infos import (
    get_animal_info_str,
    get_focal_nb_info,
    get_partition_info_str,
    get_video_info_str,
)
from .utils import data_filter
from .variables import all_variables_names_enhanced

logger = logging.getLogger(__name__)

# So that text in PDF can be editted
mpl.rcParams["pdf.fonttype"] = 42
# General font size
mpl.rcParams["font.size"] = "14"
# trigger core fonts for PDF backend
mpl.rcParams["pdf.use14corefonts"] = True
# trigger core fonts for PS backend
mpl.rcParams["ps.useafm"] = True


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


def _get_x_group(group, order):
    x_value = group
    x = order.index(x_value)
    return x


def _plot_var_stat(
    ax, var_stat, y_line_stat, order, y_offset, y_lim, comparison_stat
):
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
    if comparison_stat in ["mean", "median"]:
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
    comparison_stat=None,
):
    if y_lim is None:
        y_lim = ax.get_ylim()

    y_start = data[y_var].max() + y_offset * 0.5

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
            comparison_stat,
        )
        y_lim = ax.get_ylim()


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
            comparison_stat=stats_kwargs["test_func_kwargs"]["func"],
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


def __prepare_fig_axes(vars_type, num_variables, num_groups):
    if vars_type == "indiv_nb":
        polar = True
    else:
        polar = False
    fig = plt.figure(constrained_layout=True, figsize=(30, 7 * num_variables))
    num_columns_position_hist = np.ceil(
        (num_groups + 1) / num_variables
    ).astype(int)
    num_cols = 3 + num_columns_position_hist
    num_rows = num_variables
    gs = GridSpec(num_rows, num_cols, figure=fig)

    axs_2d_dists = []
    for row in range(num_rows):
        for col in range(num_columns_position_hist):
            axs_2d_dists.append(
                fig.add_subplot(gs[row : row + 1, col : col + 1], polar=polar)
            )
    axs_1d_dists = []
    axs_boxplots_raw = []
    axs_boxplots_standardized = []
    for i in range(num_variables):
        axs_1d_dists.append(
            fig.add_subplot(gs[i : i + 1, num_cols - 3 : num_cols - 2])
        )
        axs_boxplots_raw.append(
            fig.add_subplot(gs[i : i + 1, num_cols - 2 : num_cols - 1])
        )
        axs_boxplots_standardized.append(
            fig.add_subplot(gs[i : i + 1, num_cols - 1 :])
        )
    return (
        fig,
        axs_2d_dists,
        axs_1d_dists,
        axs_boxplots_raw,
        axs_boxplots_standardized,
    )


def _prepare_partition_indiv_vars_summary_fig(
    num_variables, num_genotype_groups
):
    return __prepare_fig_axes("indiv", num_variables, num_genotype_groups)


def _prepare_partition_group_vars_summary_fig(
    num_variables, num_genotype_groups
):
    return __prepare_fig_axes("group", num_variables, num_genotype_groups)


def _prepare_partition_indiv_nb_vars_summary_fig(
    num_variables, num_focal_nb_genotype_groups
):
    return __prepare_fig_axes(
        "indiv_nb", num_variables, num_focal_nb_genotype_groups
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
    # import pdb
    # pdb.set_trace()
    assert variable in variables_ranges.variable.values, (
        variable,
        variables_ranges,
    )
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


def _plot_polar_dist_relative_positions(data, axs):
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
            data_focal.nb_angle_degrees.values * np.pi / 180,
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

    for i, (ax, (focal_nb_genotype, pos_hist)) in enumerate(
        zip(axs, pos_hists_arrs.items())
    ):
        print(focal_nb_genotype)
        print(len(axs), len(pos_hists_arrs))
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
    if i + 1 < len(axs) - 1:
        # is not the last axes
        [ax.set_visible(False) for ax in axs[i + 2 :]]


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


axes_preparation_functions = {
    "indiv": _prepare_partition_indiv_vars_summary_fig,
    "group": _prepare_partition_group_vars_summary_fig,
    "indiv_nb": _prepare_partition_indiv_nb_vars_summary_fig,
}

plot_2d_dist_functions = {
    "indiv": _plot_positions_dist_per_genotype_group,
    "group": _plot_order_parameter_dist_per_genotype_group,
    "indiv_nb": _plot_polar_dist_relative_positions,
}


plot_1d_dist_functions = {
    "indiv": _plot_variable_1d_dist,
    "group": _plot_variable_1d_dist,
    "indiv_nb": _plot_variable_1d_dist,
}


def __plot_partition_vars_summary(
    vars_type,
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
    logger.info(f"Plotting partition_{vars_type}_vars_summary")
    num_genotype_groups = len(data["genotype_group"].unique())

    # Pepare figure axes
    logger.info("Preparing axes")
    axes_preparation_func = axes_preparation_functions[vars_type]
    (
        fig,
        axs_2d_dist,
        axs_distributions,
        axs_boxplots_raw,
        axs_boxplots_standardized,
    ) = axes_preparation_func(len(variables_stats), num_genotype_groups)

    # Plot 2d distributions
    logger.info("Plotting 2d distributions")
    plot_2d_dist_func = plot_2d_dist_functions[vars_type]
    plot_2d_dist_func(data, axs=axs_2d_dist)

    # Plot 1d distributions
    logger.info("Plotting 1d distributions")
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
    for ax in axs_distributions[i + 1 :]:
        ax.set_visible(False)

    # Get num data points
    logger.info("Getting num datapoints")
    num_data_points = _get_num_data_points(data_stats, boxplot_kwargs)

    # Plot boxplots
    logger.info("Plotting boxplots")
    outliers = []
    test_stats = []
    # import pdb
    # pdb.set_trace()
    for i, (variable, ax_boxplot, ax_boxplot_standardized,) in enumerate(
        zip(
            variables_stats,
            axs_boxplots_raw,
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
        if vars_type == "indiv":
            order = conf.GENOTYPE_GROUP_GENOTYPE_ORDER
        elif vars_type == "group":
            order = conf.GENOTYPE_GROUP_ORDER
        elif vars_type == "indiv_nb":
            order = conf.FOCAL_NB_GENOTYPE_ORDER
        else:
            raise Exception(f"No valid vars_type {vars_type}")
        _update_order(data, boxplot_kwargs, order)
        variable_stats = [
            variable,
            (f"{variable[0]}_standardized", variable[1]),
        ]
        axs = [ax_boxplot, ax_boxplot_standardized]
        for variable_stat, ax in zip(variable_stats, axs):
            # TODO: Here loop for mean/median stats and whis=100 vs 1.5
            # TODO: Compute stats for median and stats
            # import pdb
            # pdb.set_trace()
            try:
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
            except KeyError:
                # Variables aggregators "frames_moving" and
                # "frames_from_periphery" are computed from the raw variables
                # without standardization because thresholds (speed and
                # distance) are set on the raw variable.
                # In this cases the plot fails with a KeyError
                logging.warning(f"Cannot plot for variable {variable_stat}")
                ax.set_visible(False)
    outliers = pd.concat(outliers)
    outliers.drop_duplicates(inplace=True)
    test_stats = pd.DataFrame(test_stats)
    return fig, outliers, test_stats


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
    # import pdb
    # pdb.set_trace()
    return __plot_partition_vars_summary(
        "indiv",
        data,
        data_stats,
        variables,
        variables_ranges,
        variables_stats,
        boxplot_kwargs,
        stats_kwargs,
        pairs_of_groups_for_stats,
        hue,
    )


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
    return __plot_partition_vars_summary(
        "group",
        data,
        data_stats,
        variables,
        variables_ranges,
        variables_stats,
        boxplot_kwargs,
        stats_kwargs,
        pairs_of_groups_for_stats,
        hue,
    )


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
    return __plot_partition_vars_summary(
        "indiv_nb",
        data,
        data_stats,
        variables,
        variables_ranges,
        variables_stats,
        boxplot_kwargs,
        stats_kwargs,
        pairs_of_groups_for_stats,
        hue,
    )


###### SUMMARY PLOTS


def plot_summary_animal(
    datasets_partition,
    animal_col,
    animal_uid,
    variables_ranges,
    save=False,
    save_path=".",
):
    # datasets_partition = _select_partition_from_datasets(
    #     datasets, ["data_indiv"], animal_col, animal_uid
    # )
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
    datasets_partition,
    video_uid,
    animal_col,
    variables_ranges,
    save=False,
    save_path=".",
    reuse_plots=False,
):

    video_info_str = get_video_info_str(datasets_partition["data_indiv"])

    fig_save_path_png = os.path.join(save_path, f"{video_uid}_indiv.png")
    fig_save_path_pdf = os.path.join(save_path, f"{video_uid}_indiv.pdf")
    files_exists = os.path.isfile(fig_save_path_png) and os.path.isfile(
        fig_save_path_pdf
    )
    if not files_exists or not reuse_plots:
        fig = _plot_video_indiv_vars_summary(
            datasets_partition["data_indiv"],
            conf.INDIVIDUAL_VARIABLES_TO_PLOT,
            variables_ranges,
            hue="identity",
        )
        fig.suptitle(video_info_str)
        if save:
            fig.savefig(fig_save_path_png)
            fig.savefig(fig_save_path_pdf)
            plt.close()

    fig_save_path_png = os.path.join(save_path, f"{video_uid}_group.png")
    fig_save_path_pdf = os.path.join(save_path, f"{video_uid}_group.pdf")
    files_exists = os.path.isfile(fig_save_path_png) and os.path.isfile(
        fig_save_path_pdf
    )
    if not files_exists or not reuse_plots:
        fig = _plot_group_variables_summary(
            datasets_partition["data_group"],
            conf.GROUP_VARIABLES_TO_PLOT,
            variables_ranges,
        )
        fig.suptitle(video_info_str)
        if save:
            fig.savefig(fig_save_path_png)
            fig.savefig(fig_save_path_pdf)
            plt.close()

    for animal_uid in datasets_partition["data_indiv_nb"][animal_col].unique():
        fig_save_path_png = os.path.join(
            save_path, f"{animal_uid}_indiv_nb.png"
        )
        fig_save_path_pdf = os.path.join(
            save_path, f"{animal_uid}_indiv_nb.pdf"
        )
        files_exists = os.path.isfile(fig_save_path_png) and os.path.isfile(
            fig_save_path_pdf
        )
        if not files_exists or reuse_plots:
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
                fig.savefig(fig_save_path_png)
                fig.savefig(fig_save_path_pdf)
                plt.close()


plot_partition_funcs = {
    "indiv": _plot_partition_indiv_vars_summary,
    "group": _plot_partition_group_vars_summary,
    "indiv_nb": _plot_partition_indiv_nb_summary,
}


def _plot_vars_summary_partition(
    vars_type,
    datasets_partition,
    partition_col,
    variables_to_plot,
    variables_stats_to_plot,
    variables_ranges,
    boxplots_kwargs,
    stats_kwargs,
    hue,
    save,
    save_path,
):
    line_replicate_info_str = get_partition_info_str(
        datasets_partition[f"data_{vars_type}"], partition_col
    )
    fig, outliers, test_stats = plot_partition_funcs[vars_type](
        datasets_partition[f"data_{vars_type}"],
        datasets_partition[f"data_{vars_type}_stats"],
        variables_to_plot,
        variables_ranges,
        variables_stats_to_plot,
        boxplots_kwargs,
        stats_kwargs,
        hue=hue,
    )
    outliers["var_type"] = [vars_type] * len(outliers)
    test_stats["var_type"] = [vars_type] * len(test_stats)
    fig.suptitle(line_replicate_info_str)
    if save:
        fig_save_path_png = os.path.join(save_path, f"{vars_type}_vars.png")
        fig_save_path_pdf = os.path.join(save_path, f"{vars_type}_vars.pdf")
        logger.info("Saving figures outliers and stats")
        fig.savefig(fig_save_path_png, format="png")
        fig.savefig(fig_save_path_pdf, format="pdf")
        outliers.to_csv(
            os.path.join(save_path, f"{vars_type}_outliers.csv"), index=False
        )
        test_stats.to_csv(
            os.path.join(save_path, f"{vars_type}_test_stats.csv"), index=False
        )
    return outliers, test_stats


def plot_summary_partition(
    datasets_info,
    videos_table,
    partition_col,
    partition_uid,
    variables_ranges,
    indiv_boxplot_kwargs=conf.INDIV_BOXPLOT_KWARGS,
    group_boxplot_kwargs=conf.GROUP_BOXPLOT_KWARGS,
    indiv_nb_boxplot_kwargs=conf.INDIV_NB_BOXPLOT_KWARGS,
    test_stats_func=conf.TEST_STATS_FUNC,
    test_stats_kwargs=conf.TEST_STATS_KWARGS,
    save=False,
    save_path=".",
):
    stats_kwargs = {
        "test_func": test_stats_func,
        "test_func_kwargs": test_stats_kwargs,
    }
    trials_uids = videos_table[
        videos_table[partition_col] == partition_uid
    ].trial_uid.unique()
    logger.info(f"*********   Plotting {partition_uid} summary   ******")
    datasets_partition, no_data = get_partition_datasets(
        datasets_info, trials_uids
    )
    if not no_data:
        save_path = os.path.join(save_path, partition_uid)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        vars_summary_plot_info = {
            "indiv": {
                "variables_to_plot": conf.INDIVIDUAL_VARIABLES_TO_PLOT,
                "variables_stats_to_plot": conf.INDIVIDUAL_VARIABLES_STATS_TO_PLOT,
                "boxplot_kwargs": indiv_boxplot_kwargs,
                "hue": "genotype_group_genotype",
            },
            "group": {
                "variables_to_plot": conf.GROUP_VARIABLES_TO_PLOT,
                "variables_stats_to_plot": conf.GROUP_VARIABLES_STATS_TO_PLOT,
                "boxplot_kwargs": group_boxplot_kwargs,
                "hue": "genotype_group",
            },
            "indiv_nb": {
                "variables_to_plot": conf.INDIVIDUAL_NB_VARIABLES_TO_PLOT,
                "variables_stats_to_plot": conf.INDIVIDUAL_NB_VARIALBES_STATS_TO_PLOT,
                "boxplot_kwargs": indiv_nb_boxplot_kwargs,
                "hue": "focal_nb_genotype",
            },
        }

        all_outliers = []
        all_test_stats = []

        for vars_type, plot_info in vars_summary_plot_info.items():
            logger.info(f"vars_type {vars_type}")
            outliers, test_stats = _plot_vars_summary_partition(
                vars_type,
                datasets_partition,
                partition_col,
                plot_info["variables_to_plot"],
                plot_info["variables_stats_to_plot"],
                variables_ranges,
                plot_info["boxplot_kwargs"],
                stats_kwargs,
                hue=plot_info["hue"],
                save=save,
                save_path=save_path,
            )
            all_outliers.append(outliers)
            all_test_stats.append(test_stats)

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

    else:
        logger.info(f"There is not data for {partition_uid}")
        return None, None


# TODO: pass list of outliers to ignore and replot in a new folder
def plot_summary_all_partitions(
    datasets_info,
    videos_table,
    variables_ranges,
    partition_col,
    folder_suffix,
):

    possible_partition_uids = natsorted(videos_table[partition_col].unique())

    if folder_suffix and not folder_suffix.startswith("_"):
        folder_suffix = f"_{folder_suffix}"
    save_path = os.path.join(
        conf.GENERATED_FIGURES_PATH, f"summary_{partition_col}{folder_suffix}"
    )
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    Parallel(n_jobs=conf.NUM_JOBS_PARALLELIZATION)(
        delayed(plot_summary_partition)(
            datasets_info,
            videos_table,
            partition_col,
            partition,
            variables_ranges,
            save=True,
            save_path=save_path,
        )
        for partition in possible_partition_uids
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
        n_trajectories, 1, figsize=(50, 0.2 * n_trajectories)
    )
    plt.subplots_adjust(
        left=0.01, bottom=0.01, right=0.6, top=0.99, wspace=0.01, hspace=0.01
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


def filter_datasets(datasets, filters):
    filtered_datasets = {}
    for name, dataset in datasets.items():
        filtered_datasets[name] = data_filter(dataset, filters)
    return filtered_datasets


def plot_partition_videos_summaries(
    datasets,
    variables_ranges,
    videos_table,
    partition,
    partition_col,
    path_to_summary_folder,
    video_column="trial_uid",
    animal_column="trial_uid_id",
    reuse_plots=False,
):
    partition_videos = videos_table[videos_table[partition_col] == partition]
    logger.info(f"Plotting video summaries for partition {partition}")
    datasets, no_data = get_partition_datasets(
        TRAJECTORYTOOLS_DATASETS_INFO, partition_videos[video_column].values
    )
    partition_folder = os.path.join(path_to_summary_folder, partition)
    if not no_data:
        logger.info(f"Plotting outliers for {partition}")
        save_path = partition_folder
        for trial_uid in partition_videos.trial_uid:

            video_datasets = filter_datasets(
                datasets, [lambda x: x[video_column] == trial_uid]
            )
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            plot_summary_video(
                video_datasets,
                trial_uid,
                animal_column,
                variables_ranges,
                save=True,
                save_path=save_path,
                reuse_plots=reuse_plots,
            )
    else:
        logger.info(f"Partition {partition} has no data")


# TODO: Plot variables by data and replicate
# TODO: Plot variables by hour mixing pooling data of all HETs

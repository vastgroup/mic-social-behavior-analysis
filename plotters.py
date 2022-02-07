import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from constants import GENOTYPE_GROUP_ORDER, GENOTYPE_ORDER

colors = {"WT": "g", "HET": "b", "DEL": "r"}


def _get_hue_offsets(width, hue_order):
    each_width = width / len(hue_order)
    offsets = np.linspace(0, width - each_width, len(hue_order))
    offsets -= offsets.mean()
    return offsets


def _add_num_data_points(
    ax,
    data,
    num_data_points,
    boxplot_kwargs,
    width=0.8,
    y_offset_ratio=0.001,
):
    hue_offsets = _get_hue_offsets(width, boxplot_kwargs["hue_order"])
    y_min, y_max = ax.get_ylim()
    y_offset = np.abs(y_max - y_min) * y_offset_ratio
    for i, order_ in enumerate(boxplot_kwargs["order"]):
        for j, hue_order_ in enumerate(boxplot_kwargs["hue_order"]):
            if (order_, hue_order_) in num_data_points:
                num_data_points_group = num_data_points[(order_, hue_order_)]
                x = (
                    i
                    + hue_offsets[
                        boxplot_kwargs["hue_order"].index(hue_order_)
                    ]
                )
                min_y_group = data[
                    (data[boxplot_kwargs["x"]] == order_)
                    & (data[boxplot_kwargs["hue"]] == hue_order_)
                ][boxplot_kwargs["y"]].min()
                y = min_y_group - y_offset
                str_ = f"n={num_data_points_group}"
                ax.text(
                    x,
                    y,
                    str_,
                    ha="center",
                    va="top",
                )


def _compute_groups_stats(
    grouped_data,
    pairs_of_groups,
    variable,
    test_func,
    test_func_kwargs,
):
    stats = []
    for pair_group in pairs_of_groups:
        group_a, group_b = pair_group["pair"]
        group_a_in_data = group_a in grouped_data.groups.keys()
        group_b_in_data = group_b in grouped_data.groups.keys()
        if group_a_in_data and group_b_in_data:
            stat = {}
            a = grouped_data.get_group(group_a)[variable].values
            b = grouped_data.get_group(group_b)[variable].values
            if group_a[0] == group_b[0]:
                test_kwargs_updated = test_func_kwargs.copy()
                test_kwargs_updated["paired"] = True
            else:
                test_kwargs_updated = test_func_kwargs.copy()
            p_value, stat_value = test_func(a, b, **test_kwargs_updated)
            stat["test"] = test_func.__name__
            stat["group_a"] = group_a
            stat["group_b"] = group_b
            stat["plot_level"] = pair_group["level"]
            stat["p_value"] = p_value
            stat["value"] = stat_value
            stat.update(
                {
                    f"test_kwarg_{key}": value
                    for key, value in test_func_kwargs.items()
                }
            )
            stats.append(stat)

    return stats


def _boxplot_one_variable(ax, data, boxplot_kwargs):
    sns.boxplot(
        ax=ax,
        data=data,
        boxprops=dict(facecolor="w"),
        whis=100,
        showmeans=True,
        meanline=True,
        meanprops=dict(linestyle="--", linewidth=1, color="k"),
        palette=colors,
        **boxplot_kwargs,
    )
    sns.stripplot(
        ax=ax,
        data=data,
        dodge=True,
        alpha=0.5,
        palette=colors,
        **boxplot_kwargs,
    )
    sns.despine(ax=ax)
    ax.set_xticklabels(boxplot_kwargs["order"], rotation=45, ha="right")


def _set_legend(ax, col_number, num_cols):
    if col_number != num_cols - 1:
        ax.get_legend().remove()
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))


def _set_title(ax, column, title, title_in_colum=0):
    if column == title_in_colum:
        ax.set_title(title)


def _get_y_offset(ax, y_offset_ratio=0.1):
    y_min, y_max = ax.get_ylim()
    y_offset = np.abs(y_max - y_min) * y_offset_ratio
    return y_offset


def _get_x_group(group, order, hue_order, width):
    hue_offsets = _get_hue_offsets(width, hue_order)
    x_value = group[0]
    x_hue_valuesubvalue = group[1]
    x = (
        order.index(x_value)
        + hue_offsets[hue_order.index(x_hue_valuesubvalue)]
    )
    return x


def _plot_var_stat(
    ax,
    var_stat,
    y_line_stat,
    order,
    hue_order,
    width,
    y_offset,
):

    x_group_a = _get_x_group(
        var_stat["group_a"],
        order,
        hue_order,
        width,
    )
    x_group_b = _get_x_group(
        var_stat["group_b"],
        order,
        hue_order,
        width,
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
    y_min, y_max = ax.get_ylim()
    ax.set_ylim([y_min, y_line_stat + y_offset])


def _plot_var_stats(ax, data, var_stats, y_var, order, hue_order, width=0.8):

    y_offset = _get_y_offset(ax)
    y_start = data[y_var].max()

    actual_plot_level = 0
    last_level = 0
    for i, pair_stat in enumerate(var_stats):
        if pair_stat["plot_level"] != last_level:
            actual_plot_level += 1
            last_level = pair_stat["plot_level"]
        y_line_stat = y_start + (actual_plot_level + 1) * y_offset
        _plot_var_stat(
            ax, pair_stat, y_line_stat, order, hue_order, width, y_offset
        )


def boxplot_variables(
    axs,
    data,
    variables,
    title,
    boxplot_kwargs,
    pairs_of_groups_for_stats,
    stats_kwargs,
    valid_x_values=GENOTYPE_GROUP_ORDER,
    valid_hue_values=GENOTYPE_ORDER,
):
    assert len(axs) == len(variables), f"{len(variables)} {len(axs)}"
    if "hue" in boxplot_kwargs:
        num_data_points = data[
            [boxplot_kwargs["x"], boxplot_kwargs["hue"]]
        ].value_counts()
    else:
        num_data_points = data[boxplot_kwargs["x"]].value_counts()
    boxplot_kwargs.update(
        {
            "order": [
                gg
                for gg in valid_x_values
                if gg in data[boxplot_kwargs["x"]].unique()
            ],
        }
    )
    if "hue" in boxplot_kwargs:
        boxplot_kwargs.update(
            {
                "hue_order": [
                    g
                    for g in valid_hue_values
                    if g in data[boxplot_kwargs["hue"]].unique()
                ],
            }
        )
    for i, (ax, variable) in enumerate(zip(axs, variables)):
        boxplot_kwargs.update({"y": variable})
        _boxplot_one_variable(ax, data, boxplot_kwargs)
        _set_legend(ax, i, len(variables))
        _set_title(ax, i, f"{title}")
        # add num data points
        _add_num_data_points(ax, data, num_data_points, boxplot_kwargs)
        grouped_data = data.groupby(
            [boxplot_kwargs["x"], boxplot_kwargs["hue"]]
        )
        var_stats = _compute_groups_stats(
            grouped_data, pairs_of_groups_for_stats, variable, **stats_kwargs
        )
        _plot_var_stats(
            ax,
            data,
            var_stats,
            variable,
            boxplot_kwargs["order"],
            boxplot_kwargs["hue_order"],
        )

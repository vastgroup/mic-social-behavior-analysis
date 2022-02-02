from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from permutation_stats import permutation_test

from constants import GENOTYPE_GROUP_ORDER, GENOTYPE_ORDER


def _add_num_data_points(
    ax,
    x_var,
    y_var,
    hue_var,
    data,
    order,
    hue_order,
    num_data_points,
    width=0.8,
    y_offset_ratio=0.001,
):
    each_width = width / len(hue_order)
    offsets = np.linspace(0, width - each_width, len(hue_order))
    offsets -= offsets.mean()
    y_min, y_max = ax.get_ylim()
    y_offset = np.abs(y_max - y_min) * y_offset_ratio
    for i, order_ in enumerate(order):
        for j, hue_order_ in enumerate(hue_order):
            if (order_, hue_order_) in num_data_points:
                num_data_points_group = num_data_points[(order_, hue_order_)]
                x = i + offsets[hue_order.index(hue_order_)]
                min_y_group = data[
                    (data[x_var] == order_) & (data[hue_var] == hue_order_)
                ][y_var].min()
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
    test_kwargs,
):
    print(test_func)
    stats = []
    for group_a, group_b in pairs_of_groups:
        stat = {}
        print(group_a, group_b)
        a = grouped_data.get_group(group_a)[variable].values
        b = grouped_data.get_group(group_b)[variable].values
        p_value, stat_value = test_func(a, b, **test_kwargs)
        stat["test"] = test_func.__name__
        stat["group_a"] = group_a
        stat["group_b"] = group_b
        stat["p_value"] = p_value
        stat["value"] = stat_value
        stat.update(
            {f"test_kwarg_{key}": value for key, value in test_kwargs.items()}
        )
        stats.append(stat)

    return stats


def _boxplot_one_variable(ax, data, x_var, y_var, hue_var, order, hue_order):
    sns.boxplot(
        ax=ax,
        data=data,
        x=x_var,
        order=order,
        y=y_var,
        hue=hue_var,
        hue_order=hue_order,
        boxprops=dict(facecolor="w"),
        whis=100,
        showmeans=True,
        meanline=True,
        meanprops=dict(linestyle="--", linewidth=1, color="k"),
    )
    sns.stripplot(
        ax=ax,
        data=data,
        x=x_var,
        order=order,
        y=y_var,
        hue=hue_var,
        hue_order=hue_order,
        dodge=True,
        alpha=0.5,
    )
    sns.despine(ax=ax)
    ax.set_xticklabels(order, rotation=45, ha="right")


def _set_legend(ax, col_number, num_cols):
    if col_number != num_cols - 1:
        ax.get_legend().remove()
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))


def _set_title(ax, column, title, title_in_colum=0):
    if column == title_in_colum:
        ax.set_title(title)


def _plot_line_variables(axs, data, line, variables):
    assert len(axs) == len(variables)
    line_data = data[data.line == line]
    order = [
        gg
        for gg in GENOTYPE_GROUP_ORDER
        if gg in line_data.genotype_group.unique()
    ]
    hue_order = [g for g in GENOTYPE_ORDER if g in line_data.genotype.unique()]
    num_data_points = line_data[["genotype_group", "genotype"]].value_counts()
    x_var = "genotype_group"
    hue_var = "genotype"
    for i, (ax, variable) in enumerate(zip(axs, variables)):
        _boxplot_one_variable(
            ax, line_data, x_var, variable, hue_var, order, hue_order
        )
        _set_legend(ax, i, len(variables))
        _set_title(ax, i, f"{line}")
        # add num data points
        _add_num_data_points(
            ax,
            x_var,
            variable,
            hue_var,
            line_data,
            order,
            hue_order,
            num_data_points,
        )
        # Compute stats
        pairs_of_groups = [
            (("HET_HET", "HET"), ("HET_DEL", "HET")),
            (("HET_DEL", "DEL"), ("DEL_DEL", "DEL")),
            (("HET_HET", "HET"), ("DEL_DEL", "DEL")),
        ]
        grouped_data = line_data.groupby(["genotype_group", "genotype"])
        var_stats = _compute_groups_stats(
            grouped_data,
            pairs_of_groups,
            variable,
            permutation_test,
            dict(
                repetitions=10000,
                stat_func=np.mean,
                paired=False,
                two_sided=False,
            ),
        )

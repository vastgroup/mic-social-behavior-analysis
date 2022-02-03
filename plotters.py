import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from permutation_stats import permutation_test

from constants import GENOTYPE_GROUP_ORDER, GENOTYPE_ORDER


def _get_hue_offsets(width, hue_order):
    each_width = width / len(hue_order)
    offsets = np.linspace(0, width - each_width, len(hue_order))
    offsets -= offsets.mean()
    return offsets


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
    hue_offsets = _get_hue_offsets(width, hue_order)
    y_min, y_max = ax.get_ylim()
    y_offset = np.abs(y_max - y_min) * y_offset_ratio
    for i, order_ in enumerate(order):
        for j, hue_order_ in enumerate(hue_order):
            if (order_, hue_order_) in num_data_points:
                num_data_points_group = num_data_points[(order_, hue_order_)]
                x = i + hue_offsets[hue_order.index(hue_order_)]
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
    stats = []
    for group_a, group_b in pairs_of_groups:
        stat = {}
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
    ax.plot([x_group_a, x_group_b], [y_line_stat] * 2, "k-")
    x_text = np.mean([x_group_a, x_group_b])
    ax.text(
        x_text,
        y_line_stat + 0.1 * y_offset,
        f"{var_stat['value']:.2f}, {var_stat['p_value']:.4f}",
        ha="center",
        va="bottom",
    )
    y_min, y_max = ax.get_ylim()
    ax.set_ylim([y_min, y_line_stat + y_offset])


def _plot_var_stats(ax, data, var_stats, y_var, order, hue_order, width=0.8):

    y_offset = _get_y_offset(ax)
    y_start = data[y_var].max()

    for i, var_stat in enumerate(var_stats):

        y_line_stat = y_start + (i + 1) * y_offset
        _plot_var_stat(
            ax, var_stat, y_line_stat, order, hue_order, width, y_offset
        )


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
            (("HET_DEL", "HET"), ("HET_DEL", "DEL")),
            (("HET_HET", "HET"), ("HET_DEL", "HET")),
            (("HET_DEL", "DEL"), ("DEL_DEL", "DEL")),
            (("HET_HET", "HET"), ("DEL_DEL", "DEL")),
        ]
        working_groups = [
            ("HET_HET", "HET"),
            ("HET_DEL", "DEL"),
            ("HET_DEL", "HET"),
            ("DEL_DEL", "DEL"),
        ]
        grouped_data = line_data.groupby(["genotype_group", "genotype"])
        possible_groups = grouped_data.groups.keys()
        if all(
            [
                possible_group in working_groups
                for possible_group in possible_groups
            ]
        ):
            try:
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
                _plot_var_stats(
                    ax, line_data, var_stats, variable, order, hue_order
                )
            except Exception as err:
                print(Exception, err)
                print("This one failed")

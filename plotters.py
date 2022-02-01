import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from constants import GENOTYPE_GROUP_ORDER, GENOTYPE_ORDER


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
        sns.boxplot(
            ax=ax,
            data=line_data,
            x=x_var,
            order=order,
            y=variable,
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
            data=line_data,
            x=x_var,
            order=order,
            y=variable,
            hue=hue_var,
            hue_order=hue_order,
            dodge=True,
            alpha=0.5,
        )
        sns.despine(ax=ax)
        ax.set_xticklabels(order, rotation=45, ha="right")
        if i != 0:
            ax.get_legend().remove()
        if i == 0:
            ax.set_title(f"{line}")

            # add num data points
            width = 0.8
            each_width = width / len(hue_order)
            offsets = np.linspace(0, width - each_width, len(hue_order))
            offsets -= offsets.mean()
            y_min, y_max = ax.get_ylim()
            y_offset = np.abs(y_max - y_min) * 0.001
            for i, order_ in enumerate(order):
                for j, hue_order_ in enumerate(hue_order):
                    if (order_, hue_order_) in num_data_points:
                        num_data_points_group = num_data_points[
                            (order_, hue_order_)
                        ]
                        x = i + offsets[hue_order.index(hue_order_)]
                        max_y_group = line_data[
                            (line_data[x_var] == order_)
                            & (line_data[hue_var] == hue_order_)
                        ][variable].max()
                        y = max_y_group + y_offset
                        str_ = f"n={num_data_points_group}"
                        ax.text(
                            x,
                            y,
                            str_,
                            ha="center",
                            va="bottom",
                        )

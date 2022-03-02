import matplotlib.pyplot as plt
import seaborn as sns
from mic_analysis.datasets import data_filter, get_datasets
from mic_analysis.plotters import (
    _boxplot_axes_one_variable,
    _boxplot_one_variable,
)
from mic_analysis.variables import (
    all_variables_names_enhanced,
    compute_variables_ranges,
)
from natsort import nasorted

filters = [
    lambda x: x.experiment_type == 1,
    lambda x: ~x.line_experiment.str.contains("srrm"),
]
x = "line"
variable_name = "speed"
stat = "mean"
whis = 1.5

data_filters = []
datasets = get_datasets(data_filters)
variables_ranges = compute_variables_ranges(datasets)


# Get variables for each dataset
variables_to_datasets_names = {}
for dataset_name in [
    "data_indiv_stats",
    "data_indiv_nb_stats",
    "data_group_stats",
]:
    variables_to_dataset_name = {
        var_[0]: dataset_name
        for var_ in datasets[dataset_name].columns
        if var_[0] in all_variables_names_enhanced
    }
    variables_to_datasets_names.update(variables_to_dataset_name)

# Get hue for each dataset
dataset_to_hue = {
    "data_indiv_stats": "genotype_group_genotype",
    "data_group_stats": "genotype_group",
    "data_indiv_nb_stats": "focal_nb_genotype",
}

stats = ["median", "mean", "std"]
suffixes = ["", "_diff", "_standardized"]

dataset_name = variables_to_datasets_names[variable_name]
subdata = data_filter(datasets[dataset_name], filters)

fig, axs = plt.subplots(len(stats), len(suffixes), figsize=(60, 30))
fig.suptitle(
    f"Variable: {variable_name}. Data: {dataset_name}. Partition: {x}"
)
for i, stat in enumerate(stats):
    for j, suffix in enumerate(suffixes):
        ax = axs[i, j]
        if i == 0:
            if suffix == "":
                treatment = "raw"
            else:
                treatment = suffix
            ax.set_title(f"replicate data treatment {treatment}")
        variable = (f"{variable_name}{suffix}", stat)
        _boxplot_one_variable(
            ax,
            subdata,
            {
                "x": x,
                "y": variable,
                "hue": dataset_to_hue[dataset_name],
                "order": natsorted(subdata[x].unique()),
                "whis": whis,
            },
        )
        _boxplot_axes_one_variable(ax, subdata, variable)

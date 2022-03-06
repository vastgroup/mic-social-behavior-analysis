import os

import matplotlib.pyplot as plt
import pandas as pd
from confapp import conf
from mic_analysis.datasets import TRAJECTORYTOOLS_DATASETS_INFO, data_filter
from mic_analysis.logger import setup_logs
from mic_analysis.plotters import (
    _boxplot_axes_one_variable,
    _boxplot_one_variable,
)
from mic_analysis.variables import (
    all_variables_names,
    all_variables_names_enhanced,
)
from natsort import natsorted

logger = setup_logs("variables_summary")

if __name__ == "__main__":

    datasets = {}
    for name, dataset_info in TRAJECTORYTOOLS_DATASETS_INFO.items():
        logger.info(f"Loading dataset {name}")
        datasets[name] = pd.read_pickle(
            os.path.join(dataset_info["dir_path"], "per_animal_stats.pkl")
        )

    filters = [
        lambda x: x.experiment_type == 1,
        lambda x: ~x.line_experiment.str.contains("srrm"),
    ]
    partition = "line"
    whis = 1.5

    save_folder = os.path.join(
        conf.GENERATED_FIGURES_PATH, f"variables_summary_per_{partition}"
    )
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # Get variables for each dataset
    variables_to_datasets_names = {}
    for dataset_name in TRAJECTORYTOOLS_DATASETS_INFO.keys():
        variables_to_dataset_name = {
            var_[0]: dataset_name
            for var_ in datasets[dataset_name].columns
            if var_[0] in all_variables_names
        }
        variables_to_datasets_names.update(variables_to_dataset_name)

    # Get hue for each dataset
    dataset_to_hue = {
        "tr_indiv_bl": "genotype_group_genotype",
        "tr_group_bl": "genotype_group",
        "tr_indiv_nb_bl": "focal_nb_genotype",
    }

    stats = ["median", "mean", "std"]
    suffixes = ["", "_diff", "_standardized"]

    for variable_name in (
        conf.INDIVIDUAL_VARIABLES_TO_PLOT
        + conf.GROUP_VARIABLES_TO_PLOT
        + conf.INDIVIDUAL_NB_VARIABLES_TO_PLOT
    ):
        logger.info(f"Plotting variable summary {variable_name}")
        dataset_name = variables_to_datasets_names[variable_name]
        subdata = data_filter(datasets[dataset_name], filters)

        variables_stats = [c for c in subdata.columns if c[0] == variable_name]
        fig, axs = plt.subplots(len(variables_stats), 3, figsize=(90, 30))
        fig.suptitle(
            f"Variable: {variable_name}. Data: {dataset_name}. Partition: {partition}"
        )

        for i, variable_stat in enumerate(variables_stats):
            for j, treatment in enumerate(["", "_diff", "_standardized"]):
                ax = axs[i, j]
                variable_to_plot = (
                    f"{variable_stat[0]}{treatment}",
                    variable_stat[1],
                )
                if variable_to_plot in subdata.columns:
                    ax.set_title(f"replicate data treatment {treatment}")
                    _boxplot_one_variable(
                        ax,
                        subdata,
                        {
                            "x": partition,
                            "y": variable_to_plot,
                            "hue": dataset_to_hue[dataset_name],
                            "order": natsorted(subdata[partition].unique()),
                            "whis": whis,
                        },
                    )
                    _boxplot_axes_one_variable(ax, subdata, variable_to_plot)

        fig.savefig(
            os.path.join(
                save_folder,
                f"{variable_name}.png",
            )
        )
        fig.savefig(
            os.path.join(
                save_folder,
                f"{variable_name}.pdf",
            )
        )

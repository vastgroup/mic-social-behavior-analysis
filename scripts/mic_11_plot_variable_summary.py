import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from confapp import conf
from mic_analysis.datasets import TRAJECTORYTOOLS_DATASETS_INFO, data_filter
from mic_analysis.logger import setup_logs
from mic_analysis.plotters import (_boxplot_axes_one_variable,
                                   _boxplot_one_variable)
from mic_analysis.variables import (all_variables_names,
                                    all_variables_names_enhanced)
from natsort import natsorted

logger = setup_logs("variables_summary")

if __name__ == "__main__":

    logger = setup_logs("plot_summary_figures")

    parser = argparse.ArgumentParser(
        description="Generates dataframes using trajectorytools for each video"
        "that that has been tracked and is valid for analysis"
    )
    # parser.add_argument(
    #     "-rp",
    #     "--replot",
    #     action="store_true",
    #     default=False,
    #     help="Replots figures previously plotted",
    # )
    parser.add_argument(
        "-pc",
        "--partition_col",
        type=str,
        default="line_experiment",
        choices=["line_experiment", "line_replicate_experiment"],
        help="Partition column to select data to plot each figure",
    )
    parser.add_argument(
        "-fs",
        "--folder_suffix",
        type=str,
        default="",
        help="A suffix to be added to the name of the folder where "
        "figures are stored",
    )
    parser.add_argument(
        "-df",
        "--data_filters",
        type=str,
        default="",
        choices=conf.DATA_FILTERS.keys(),
        nargs="+",
    )
    args = parser.parse_args()

    datasets = {}
    for name, dataset_info in TRAJECTORYTOOLS_DATASETS_INFO.items():
        logger.info(f"Loading dataset {name}")
        datasets[name] = pd.read_pickle(
            os.path.join(dataset_info["dir_path"], "per_animal_stats.pkl")
        )

    # TODO: externalize filters with in argparse
    filters_to_apply = []
    for filter_name in args.data_filters:
        filters_to_apply.extend(conf.DATA_FILTERS[filter_name])

    whis = 1.5

    folder_suffix = args.folder_suffix
    if folder_suffix and not folder_suffix.startswith("_"):
        folder_suffix = f"_{folder_suffix}"

    save_folder = os.path.join(
        conf.GENERATED_FIGURES_PATH,
        f"variables_summary_per_{args.partition_col}{folder_suffix}",
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
        subdata = data_filter(datasets[dataset_name], filters_to_apply)

        variables_stats = [
            c for c in subdata.columns if c[0].startswith(variable_name)
        ]
        treatments_to_raw_vars = sorted(
            set([c[0].split(variable_name)[-1] for c in variables_stats])
        )
        stat_computed_over_treated_var = set([v[1] for v in variables_stats])

        fig, axs = plt.subplots(
            len(stat_computed_over_treated_var),
            len(treatments_to_raw_vars),
            figsize=(
                30 * len(treatments_to_raw_vars),
                10 * len(stat_computed_over_treated_var),
            ),
        )
        fig.suptitle(
            f"Variable: {variable_name}. Data: {dataset_name}. Partition: {args.partition_col}"
        )

        for i, stat in enumerate(stat_computed_over_treated_var):
            treatmes = []
            for j, treatment in enumerate(treatments_to_raw_vars):
                if isinstance(axs, np.ndarray): 
                    if axs.ndim == 2:
                        ax = axs[i, j]
                    elif axs.ndim == 1:
                        if len(stat_computed_over_treated_var) > 1:
                            ax = axs[i]
                        else:
                            ax = axs[j]
                else:
                    ax = axs
                variable_to_plot = (
                    f"{variable_name}{treatment}",
                    stat,
                )
                if variable_to_plot in subdata.columns:
                    ax.set_title(f"replicate data treatment {treatment}")
                    _boxplot_one_variable(
                        ax,
                        subdata,
                        {
                            "x": args.partition_col,
                            "y": variable_to_plot,
                            "hue": dataset_to_hue[dataset_name],
                            "order": natsorted(
                                subdata[args.partition_col].unique()
                            ),
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

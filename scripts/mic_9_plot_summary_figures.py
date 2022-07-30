import argparse

import pandas as pd
from confapp import conf
from ipywidgets import interact
from mic_analysis.datasets import (
    TRAJECTORYTOOLS_DATASETS_INFO,
    data_filter,
    get_datasets,
)
from mic_analysis.logger import setup_logs
from mic_analysis.plotters import plot_summary_all_partitions
from mic_analysis.variables import get_variables_ranges

logger = setup_logs("plot_summary_figures")

parser = argparse.ArgumentParser(
    description="Generates dataframes using trajectorytools for each video"
    "that that has been tracked and is valid for analysis"
)

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

# TODO: add filters as external variables
filters_to_apply = []
for filter_name in args.data_filters:
    filters_to_apply.extend(conf.DATA_FILTERS[filter_name])

videos_table = pd.read_csv(conf.VIDEOS_INDEX_FILE_NAME)
variables_ranges = get_variables_ranges(TRAJECTORYTOOLS_DATASETS_INFO)

videos_table = data_filter(videos_table, filters_to_apply)

folder_suffix = args.folder_suffix
if folder_suffix and not folder_suffix.startswith("_"):
    folder_suffix = f"_{folder_suffix}"

folder_suffix = (
    folder_suffix + f"_{conf.TEST_STATS_KWARGS['func']}"
)

plot_summary_all_partitions(
    TRAJECTORYTOOLS_DATASETS_INFO,
    videos_table,
    variables_ranges,
    partition_col=args.partition_col,
    folder_suffix=folder_suffix,
)

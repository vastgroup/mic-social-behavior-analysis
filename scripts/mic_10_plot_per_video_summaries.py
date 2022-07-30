import argparse
import os

import pandas as pd
from confapp import conf
from joblib import Parallel, delayed
from mic_analysis.datasets import TRAJECTORYTOOLS_DATASETS_INFO, data_filter
from mic_analysis.logger import setup_logs
from mic_analysis.plotters import plot_partition_videos_summaries
from mic_analysis.variables import get_variables_ranges
from natsort import natsorted

logger = setup_logs("plot_summary_figures")

parser = argparse.ArgumentParser(
    description="Generates dataframes using trajectorytools for each video"
    "that that has been tracked and is valid for analysis"
)
parser.add_argument(
    "-rp",
    "--replot",
    action="store_true",
    default=False,
    help="Replots figures previously plotted",
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
variables_ranges = get_variables_ranges(TRAJECTORYTOOLS_DATASETS_INFO)

# TODO: externilize partition_col as a argument, add suffix to folder, ...
partition_col = args.partition_col
logger.info("Loading videos table")
videos_table = pd.read_csv(conf.VIDEOS_INDEX_FILE_NAME)
videos_table = data_filter(videos_table, filters_to_apply)

possible_partitions = natsorted(videos_table[partition_col].unique())

folder_suffix = args.folder_suffix
if folder_suffix and not folder_suffix.startswith("_"):
    folder_suffix = f"_{folder_suffix}"

path_to_summary_folder = os.path.join(
    conf.GENERATED_FIGURES_PATH, f"summary_per_video_{partition_col}{folder_suffix}"
)
video_column = "trial_uid"
animal_column = "trial_uid_id"


Parallel(n_jobs=conf.NUM_JOBS_PARALLELIZATION)(
    delayed(plot_partition_videos_summaries)(
        TRAJECTORYTOOLS_DATASETS_INFO,
        variables_ranges,
        videos_table,
        partition,
        partition_col,
        path_to_summary_folder,
        video_column,
        animal_column,
        args.replot,
    )
    for partition in possible_partitions
)

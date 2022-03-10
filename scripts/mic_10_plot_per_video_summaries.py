import argparse
import os

import pandas as pd
from confapp import conf
from joblib import Parallel, delayed
from mic_analysis.datasets import TRAJECTORYTOOLS_DATASETS_INFO
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
args = parser.parse_args()

data_filters = []
variables_ranges = get_variables_ranges(TRAJECTORYTOOLS_DATASETS_INFO)
# TODO: externilize partition_col as a argument, add suffix to folder, ...
partition_col = args.partition_col
logger.info("Loading videos table")
videos_table = pd.read_csv(conf.VIDEOS_INDEX_FILE_NAME)

possible_partitions = natsorted(videos_table[partition_col].unique())

path_to_summary_folder = os.path.join(
    conf.GENERATED_FIGURES_PATH, f"summary_{partition_col}"
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

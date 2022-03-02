import os
from glob import glob

import pandas as pd
from confapp import conf
from mic_analysis.datasets import get_datasets
from mic_analysis.logger import setup_logs
from mic_analysis.plotters import plot_summary_video
from mic_analysis.variables import compute_variables_ranges
from natsort import natsorted

logger = setup_logs("plot_summary_figures")

data_filters = []
datasets = get_datasets(data_filters)
variables_ranges = compute_variables_ranges(datasets)
partition_col = "line_experiment"
logger.info("Loading videos table")
videos_table = pd.read_csv(conf.VIDEOS_INDEX_FILE_NAME)

possible_partitions = natsorted(videos_table[partition_col].unique())

path_to_summary_folder = os.path.join(
    conf.GENERATED_FIGURES_PATH, f"summary_{partition_col}"
)
video_column = "trial_uid"
animal_column = "trial_uid_id"

for partition in possible_partitions:
    logger.info(f"Plotting outliers for {partition}")
    partition_folder = os.path.join(path_to_summary_folder, partition)
    outliers_path = os.path.join(partition_folder, "all_outliers.csv")
    if os.path.isfile(outliers_path):
        outliers = pd.read_csv(outliers_path)
        outliers_uids = natsorted(outliers[video_column].unique())
        for outlier_uid in outliers_uids:
            logger.info(f"Plotting outlier {outlier_uid}")
            save_path = os.path.join(partition_folder, "outliers_summaries")
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            plot_summary_video(
                datasets,
                video_column,
                outlier_uid,
                animal_column,
                variables_ranges,
                save=True,
                save_path=save_path,
            )

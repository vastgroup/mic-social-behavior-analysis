import argparse

import pandas as pd
from confapp import conf
from ipywidgets import interact
from mic_analysis.datasets import TRAJECTORYTOOLS_DATASETS_INFO, get_datasets, data_filter
from mic_analysis.logger import setup_logs
from mic_analysis.plotters import plot_summary_all_partitions
from mic_analysis.variables import get_variables_ranges

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
    help="Replots figures previously plotted",
)
args = parser.parse_args()

data_filters = [
    lambda x: x["experiment_type"] == 1,
    lambda x: x["line"] == "srrm3_17",
    lambda x: ~x['genotype_group'].str.contains('WT')
]
videos_table = pd.read_csv(conf.VIDEOS_INDEX_FILE_NAME)
variables_ranges = get_variables_ranges(TRAJECTORYTOOLS_DATASETS_INFO)

videos_table = data_filter(videos_table, data_filters)

plot_summary_all_partitions(
    TRAJECTORYTOOLS_DATASETS_INFO,
    videos_table,
    variables_ranges,
    partition_col=args.partition_col,
    replot=args.replot,
)

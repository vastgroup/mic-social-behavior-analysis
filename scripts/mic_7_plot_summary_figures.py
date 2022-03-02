import os

import numpy as np
import pandas as pd
from confapp import conf
from ipywidgets import interact
from mic_analysis.datasets import TRAJECTORYTOOLS_DATASETS_INFO, get_datasets
from mic_analysis.logger import setup_logs
from mic_analysis.plotters import plot_summary_all_partitions_with_outliers
from mic_analysis.variables import compute_variables_ranges
from natsort import natsorted

logger = setup_logs("plot_summary_figures")

data_filters = []
datasets = get_datasets(data_filters)
variables_ranges = compute_variables_ranges(datasets)
plot_summary_all_partitions_with_outliers(
    datasets,
    variables_ranges,
    partition_col="line_experiment",
)

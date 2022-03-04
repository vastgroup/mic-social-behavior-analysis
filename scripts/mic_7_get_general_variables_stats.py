import os

import numpy as np
import pandas as pd
from confapp import conf
from mic_analysis.datasets import TRAJECTORYTOOLS_DATASETS_INFO, load_dataset
from mic_analysis.logger import setup_logs
from mic_analysis.variables import compute_variables_ranges
from natsort import natsorted

logger = setup_logs("get_variables_tables")

videos_table = pd.read_csv(conf.VIDEOS_INDEX_FILE_NAME)
partition_col = "line_experiment"
possible_partitions = natsorted(videos_table[partition_col].unique())


for name, dataset_info in TRAJECTORYTOOLS_DATASETS_INFO.items():
    variables_ranges = None
    for partition in possible_partitions:
        filters = [lambda x: x[partition_col] == partition]
        logger.info(f"Computing variables_ranges for {name} {partition}")
        data = load_dataset(dataset_info["dir_path"], video_filters=filters)
        if not data.empty:
            other_variable_ranges = compute_variables_ranges(data)
            if variables_ranges is not None:
                variables_ranges["min"] = np.min(
                    [variables_ranges["min"], other_variable_ranges["min"]],
                    axis=0,
                )
                variables_ranges["max"] = np.max(
                    [variables_ranges["max"], other_variable_ranges["max"]],
                    axis=0,
                )
            else:
                variables_ranges = other_variable_ranges

    save_path = os.path.join(
        dataset_info["dir_path"], conf.VARIABLES_RANGES_FILE_NAME
    )
    logger.info(f"Saving at {save_path}")
    variables_ranges.to_pickle(save_path)

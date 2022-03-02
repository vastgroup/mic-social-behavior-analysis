import os

import numpy as np
import pandas as pd
from confapp import conf

from mic_analysis.datasets import TRAJECTORYTOOLS_DATASETS_INFO
from mic_analysis.logger import setup_logs
from mic_analysis.table_generators_utils import (
    add_line_and_genotype_info,
    generate_variables_table,
)

logger = setup_logs("get_variables_tables")

videos_table = pd.read_csv(conf.VIDEOS_INDEX_FILE_NAME)
videos_table["abs_trajectory_path"] = videos_table["trajectory_path"].apply(
    lambda x: os.path.join(conf.TRACKING_DATA_FOLDER_PATH, x)
    if isinstance(x, str)
    else np.nan
)
animals_table = pd.read_csv(conf.ANIMALS_INDEX_FILE_PATH)

for name, tt_dataset_info in TRAJECTORYTOOLS_DATASETS_INFO.items():
    if not os.path.isfile(tt_dataset_info["file_path"]):
        tr_vars_df = generate_variables_table(
            videos_table,
            animals_table,
            tt_dataset_info["variables_to_compute"],
            scale_to_body_length=tt_dataset_info["scale_to_body_length"],
        )
        tr_vars_df = add_line_and_genotype_info(
            tr_vars_df, videos_table, animals_table
        )
        tr_vars_df.to_pickle(tt_dataset_info["file_path"])
    else:
        tr_vars_df = pd.read_pickle(tt_dataset_info["file_path"])

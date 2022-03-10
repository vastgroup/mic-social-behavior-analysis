import argparse
import os

import numpy as np
import pandas as pd
from confapp import conf
from mic_analysis.datasets import (
    TRAJECTORYTOOLS_DATASETS_INFO,
    generate_variables_dataset,
)
from mic_analysis.logger import setup_logs

parser = argparse.ArgumentParser(
    description="Generates dataframes using trajectorytools for each video"
    "that that has been tracked and is valid for analysis"
)
parser.add_argument(
    "-rg",
    "--regenerate",
    action="store_true",
    default=False,
    help="Regenerates previously generated dataframes of a given dataset",
)
args = parser.parse_args()

logger = setup_logs("get_variables_tables")

videos_table = pd.read_csv(conf.VIDEOS_INDEX_FILE_NAME)
videos_table["abs_trajectory_path"] = videos_table["trajectory_path"].apply(
    lambda x: os.path.join(conf.TRACKING_DATA_FOLDER_PATH, x)
    if isinstance(x, str)
    else np.nan
)
videos_table.set_index("trial_uid", drop=True, inplace=True)

animals_table = pd.read_csv(conf.ANIMALS_INDEX_FILE_PATH)

# TODO: Save trajectorytools data as .csv (not sure enough memory to store it)
for name, tt_dataset_info in TRAJECTORYTOOLS_DATASETS_INFO.items():
    index_path = os.path.join(tt_dataset_info["dir_path"], "index.pkl")
    new_tr_vars_index = generate_variables_dataset(
        index_path,
        videos_table,
        animals_table,
        tt_dataset_info["variables_to_compute"],
        scale_to_body_length=tt_dataset_info["scale_to_body_length"],
        save_dir=tt_dataset_info["dir_path"],
        regenerate=args.regenerate,
    )

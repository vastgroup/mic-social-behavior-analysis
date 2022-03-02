import os
from glob import glob

import numpy as np
import pandas as pd
from confapp import conf

import mic_analysis
from mic_analysis.logger import setup_logs
from mic_analysis.table_generators_utils import (
    add_extra_columns,
    correct_data_types,
    drop_unnecessary_rows_columns,
)
from mic_analysis.utils import get_files_with_pattern, read_csv

logger = setup_logs("animals_index")

# Read all files in the directory.
# Files with a pattern different than 20*.xlsx won't be read
experiment_table_paths = get_files_with_pattern(
    os.path.join(conf.EXPERIMENTS_DATA_FOLDER_PATH, "20*.csv")
)

# Some columns added after loading each table
animals_tables = []
animals_tables_reports = []
for animals_table_path in experiment_table_paths:
    # Read csv
    df = read_csv(animals_table_path)
    animals_table_report = {"file": os.path.split(animals_table_path)[1]}
    if not df.empty:
        animals_table_report["has_data"] = True
    else:
        animals_table_report["has_data"] = False
    # Drop unnecesary columns
    df, drop_unnecessary_report = drop_unnecessary_rows_columns(df)
    animals_table_report.update(drop_unnecessary_report)
    # Correct data types
    df, correct_data_types_report = correct_data_types(df)
    animals_table_report.update(correct_data_types_report)
    # Rename
    df.rename(conf.RENAMING_SINGLE_ANIMALS_TABLE, axis=1, inplace=True)
    # Add extra columns
    df = add_extra_columns(df, animals_table_path)
    animals_tables.append(df)
    animals_tables_reports.append(animals_table_report)

animals_table = pd.concat(animals_tables).reset_index(drop=True)
animals_table_report = pd.DataFrame(animals_tables_reports)
animals_table_report.to_csv(conf.ANIMALS_INDEX_REPORT_FILE_PATH, index=False)

# Read the conversion table to have the reference to the old names
# and some other data like controls or the hard drive name where the
# video is stored
conversion_table = read_csv(conf.CONVERSIONS_TABLE_PATH)
conversion_table.rename(conf.RENAMING_CONVERSION_TABLE, axis=1, inplace=True)

# Merge tables
animals_table = pd.merge(
    animals_table,
    conversion_table,
    on=["folder_name_track"],
)
animals_table.to_csv(conf.ANIMALS_INDEX_FILE_PATH, index=False)

import argparse
import os
from glob import glob
from logging import Logger

import numpy as np
import pandas as pd
from pandas._libs import missing

from mic_analysis.constants import (
    ANIMALS_COUNT_FILE_PATH,
    ANIMALS_INDEX_FILE_PATH,
    ANIMALS_INDEX_REPORT_FILE_PATH,
    CONVERSIONS_TABLE_PATH,
    EXPECTED_COLUMNS_IN_SINGLE_ANIMALS_TABLE,
    EXPERIMENTS_DATA_FOLDER_PATH,
    GENOTYPE_CONVERSION_DICT,
    RENAMING_CONVERSION_TABLE,
    RENAMING_SINGLE_ANIMALS_TABLE,
)
from mic_analysis.utils import get_files_with_pattern, read_csv


def drop_unnecessary_rows_columns(df: pd.DataFrame):
    # some tables have empty rows
    logger.info(f"Dropping unnecessary rows and columns")
    df_wo_na_rows = df.dropna(how="all")
    num_empty_rows = len(df) - len(df_wo_na_rows)
    logger.warning(f"Dropped {num_empty_rows} empty rows")
    df = df_wo_na_rows

    # Expected columns
    expected_columns = set(EXPECTED_COLUMNS_IN_SINGLE_ANIMALS_TABLE.keys())

    # Checking expected columns are in table
    missing_columns = expected_columns - set(df.columns)
    extra_columns = set(df.columns) - expected_columns
    if missing_columns or extra_columns:
        logger.info(f"Columns are: {df.columns}")
        logger.info(f"Expected columsn are: {expected_columns}")
    if missing_columns:
        logger.warning(f"Missing columns {missing_columns}")
    if extra_columns:
        df.drop(extra_columns, axis=1, inplace=True)
        logger.warning(f"Dropped extra columns {extra_columns}")

    report = {
        "num_empty_rows": num_empty_rows,
        "num_missing_columns": len(missing_columns),
        "num_extra_columns": len(extra_columns),
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
    }

    return df, report


def correct_data_types(df: pd.DataFrame):
    # Define column types
    logger.info(f"Correcting data types of columns")
    columns_with_correct_data_types = []
    columns_with_incorrect_data_types = []
    for column, dtype in EXPECTED_COLUMNS_IN_SINGLE_ANIMALS_TABLE.items():
        logger.info(f"For column: {column}")
        logger.info(f"the expected data type is {dtype}")
        logger.info(f"the actual data type is {df[column].dtype}")
        if not df[column].dtype == dtype:
            columns_with_incorrect_data_types.append(
                (column, df[column].dtype, dtype)
            )
            logger.warning(f"Data type does not match")
            logger.info("Trying to correct data type")
            if column in ["size_cm", "accuracy", "px_cm"]:
                df[column] = df[column].apply(
                    lambda x: float(x.replace(",", "."))
                    if isinstance(x, str) and x[-1].isnumeric()
                    else np.nan
                )
                logger.info(
                    "Corrected by replacing . with , in numeric "
                    "strings and casting to float. None numeric "
                    "strings or missing values are casted as with NaN"
                )
            elif column == "fish_ID_track":
                df[column] = df[column].apply(
                    lambda x: float(x)
                    if (isinstance(x, str) and x[-1].isnumeric())
                    or isinstance(x, (int, float, complex))
                    else np.nan
                )
                logger.info(
                    "Corrected by casting to float numeric strings"
                    "and casting with NaN non numeric or empty values"
                )
            else:
                try:
                    df[column] = df[column].astype(dtype)
                except:
                    logger.error("Cannot")
                    raise

                logger.info("Corrected by assiging default data type")
        else:
            columns_with_correct_data_types.append(
                (column, df[column].dtype, dtype)
            )
    report = {
        "num_correct_data_type": len(columns_with_correct_data_types),
        "num_incorrect_data_type": len(columns_with_incorrect_data_types),
        "correct_data_type": columns_with_correct_data_types,
        "incorrect_data_type": columns_with_incorrect_data_types,
    }
    return df, report


def add_genotype_and_groupsize(df):
    animals_same_trajectory = df.groupby(["folder_name_track", "trial"])
    all_animals = []
    for idx, animals in animals_same_trajectory:
        if not animals.genotype.isna().sum():
            genotype_group = "_".join(animals.genotype)
        else:
            genotype_group = "-_-"
        animals["group_size"] = [len(animals)] * len(animals)
        animals["genotype_group"] = [genotype_group] * len(animals)
        all_animals.append(animals)

        if len(animals) < 2:
            print(animals[["folder_name_track", "trial"]])
            raise Exception(
                "A group with less than 2 animals, "
                "probabliy a trial number badly set"
            )

    df = pd.concat(all_animals)
    df["genotype_group"] = [
        *map(
            lambda x: x
            if x not in GENOTYPE_CONVERSION_DICT.keys()
            else GENOTYPE_CONVERSION_DICT[x],
            df["genotype_group"],
        )
    ]
    df["genotype_group"] = df["genotype_group"].fillna("-_-")
    return df


def add_trial_uid_and_folder_name_track(df, file_name):
    file_name_no_ext = os.path.splitext(file_name)[0]
    folder_name_track = file_name_no_ext[9:]
    df["folder_name_track"] = [folder_name_track] * len(df)
    df["trial_uid"] = (
        df["folder_name_track"] + "_" + df["trial"].astype(int).astype(str)
    )


def add_extra_columns(df, animals_table_path):
    file_name = os.path.split(animals_table_path)[1]
    df["file_name"] = [file_name] * len(df)
    add_trial_uid_and_folder_name_track(df, file_name)
    df = add_genotype_and_groupsize(df)
    return df


if __name__ == "__main__":
    from mic_analysis.logger import setup_logs

    logger = setup_logs("animals_index")

    # Read all files in the directory.
    # Files with a pattern different than 20*.xlsx won't be read
    experiment_table_paths = get_files_with_pattern(
        os.path.join(EXPERIMENTS_DATA_FOLDER_PATH, "20*.csv")
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
        df.rename(RENAMING_SINGLE_ANIMALS_TABLE, axis=1, inplace=True)
        # Add extra columns
        df = add_extra_columns(df, animals_table_path)
        animals_tables.append(df)
        animals_tables_reports.append(animals_table_report)

    animals_table = pd.concat(animals_tables).reset_index(drop=True)
    animals_table_report = pd.DataFrame(animals_tables_reports)
    animals_table_report.to_csv(ANIMALS_INDEX_REPORT_FILE_PATH, index=False)

    # Read the conversion table to have the reference to the old names
    # and some other data like controls or the hard drive name where the
    # video is stored
    conversion_table = read_csv(CONVERSIONS_TABLE_PATH)
    conversion_table.rename(RENAMING_CONVERSION_TABLE, axis=1, inplace=True)

    # Merge tables
    animals_table = pd.merge(
        animals_table,
        conversion_table,
        on=["folder_name_track"],
    )
    animals_table.to_csv(ANIMALS_INDEX_FILE_PATH, index=False)

import argparse
import copy
import os
from re import S
import time
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import trajectorytools as tt
from constants import (
    ANIMALS_INDEX_FILE_PATH,
    NUM_FRAMES_FOR_ANALYSIS,
    TRACKING_DATA_FOLDER_PATH,
    TRAJECTORIES_INDEX_FILE_NAME,
    TRAJECTORYTOOLS_DATASETS_INFO,
    VIDEOS_INDEX_FILE_NAME,
    SIGMA,
    PX_CM,
    FRAME_RATE,
)
from trajectorytools.export import tr_variables_to_df
from utils import clean_impossible_speed_jumps


def _speed(trajectories):
    vel = np.diff(trajectories, axis=0)
    speed = np.sqrt(vel[..., 0] ** 2 + vel[..., 1] ** 2)
    return speed


def get_trajectories(
    trajectories_path,
    center=None,
    interpolate_nans=None,
    smooth_params=None,
    length_unit_dict=None,
    time_unit_dict=None,
    tracking_interval=None,
):
    tr_dict = np.load(trajectories_path, allow_pickle=True).item()

    tr_dict, _ = clean_impossible_speed_jumps(tr_dict, tracking_interval)

    tr = tt.trajectories.FishTrajectories.from_idtracker_(
        tr_dict,
        center=center,
        interpolate_nans=interpolate_nans,
        smooth_params=smooth_params,
        dtype=np.float64,
    )

    # Select tracked frames
    if tracking_interval is not None:
        tr = tr[tracking_interval[0] : tracking_interval[1]]

    # Change units
    if length_unit_dict is not None:
        tr.new_length_unit(
            length_unit_dict["length_unit"],
            length_unit_dict["length_unit_name"],
        )
    if time_unit_dict is not None:
        tr.new_time_unit(
            time_unit_dict["time_unit"], time_unit_dict["time_unit_name"]
        )

    return tr


def _add_identity_info(
    tr_vars_df, videos_table, animals_table, identity_column_name="identity"
):
    # Add `identity` column with numbers 0 and 1 to
    videos_table[identity_column_name] = videos_table.id_last_fish - 1
    # Add `fish_id_exp` with number 2
    # Each row in the video_table has information about the identity of the
    # last fish. Such fish is the fish_id_exp 2
    if "nb" in identity_column_name:
        suffix = "_nb"
    else:
        suffix = ""
    videos_table[f"fish_id_exp{suffix}"] = 2
    # We merge tr_indivs with video_table so that tr_indivs has columns
    # fish_id_exp that we will use to get the genotype.
    # Note that since we are merging using the identity the fish_id_exp
    # will be allocated to the correct fish, which later will allow us
    # to get the fish genotype
    tr_vars_df = pd.merge(
        tr_vars_df,
        videos_table[
            ["trial_uid", identity_column_name, f"fish_id_exp{suffix}"]
        ],
        on=["trial_uid", identity_column_name],
        how="outer",
    )
    # All fish that are not the last fish (i.e. the second fish) will
    # have NaN in the fish_id_exp column. So we fill the NaN with 1
    # to indicate that they are the first fish
    tr_vars_df[f"fish_id_exp{suffix}"].fillna(1, inplace=True)

    # We merge the tr_indivs dataframe with the animals
    # table to get information about the genotype of each fish, the age and
    # the size
    # To do so we use the fish_id_exp column
    extra_columns = ["genotype", "dpf", "size_cm"]
    if "nb" in identity_column_name:
        for extra_column in extra_columns + ["fish_id_exp"]:
            animals_table[extra_column + suffix] = animals_table[extra_column]
        extra_columns = [
            extra_column + suffix for extra_column in extra_columns
        ]
    tr_vars_df = pd.merge(
        tr_vars_df,
        animals_table[["trial_uid", f"fish_id_exp{suffix}"] + extra_columns],
        on=["trial_uid", f"fish_id_exp{suffix}"],
        how="outer",
    )
    return tr_vars_df


def _add_line_and_genotype_info(tr_vars_df, videos_table, animals_table):
    if "identity" in tr_vars_df.columns:
        tr_vars_df = _add_identity_info(
            tr_vars_df,
            videos_table,
            animals_table,
            identity_column_name="identity",
        )

    if "identity_nb" in tr_vars_df.columns:
        tr_vars_df = _add_identity_info(
            tr_vars_df,
            videos_table,
            animals_table,
            identity_column_name="identity_nb",
        )

    # We merge again tr_indivs with video_table to get extra features
    # of the videos that are important for the analysis
    tr_vars_df = pd.merge(
        tr_vars_df,
        videos_table[
            [
                "trial_uid",
                "line",
                "line_replicate",
                "gene",
                "founder",
                "replicate",
                "experiment_type",
                "genotype_group",
            ]
        ],
        on="trial_uid",
        how="outer",
    )

    # Finally we clean the rows that have no frames or identity
    # as they are rows of animals that have not been tracked

    if "identity_nb" in tr_vars_df.columns:
        columns_to_drop_nan = [
            "frame",
            "identity",
            "identity_nb",
            "nb_position_x",
        ]
    elif "identity" in tr_vars_df.columns:
        columns_to_drop_nan = ["frame", "identity"]
    else:
        columns_to_drop_nan = ["frame"]
    tr_vars_df.dropna(subset=columns_to_drop_nan, inplace=True, how="any")
    return tr_vars_df


def _generate_variables_table(
    videos_table, animals_table, variables_list, scale_to_body_length=False
):
    tr_vars_dfs = []
    for idx, tr_row in tqdm(
        videos_table.iterrows(),
        desc="generating_dataframes",
        total=len(videos_table),
    ):
        if tr_row.tracked and tr_row.valid_for_analysis:
            animals = animals_table[
                animals_table.trial_uid == tr_row.trial_uid
            ]
            mean_size = np.mean(animals.size_cm)
            if np.isnan(mean_size):
                logger.info(
                    "Getting mean size_cm from all videos of same gene"
                )
                mean_size = animals_table[
                    (animals_table.gene == tr_row.gene)
                    & (animals_table.founder == tr_row.founder)
                    & (animals_table.replicate == tr_row.replicate)
                ].size_cm.mean()
            if scale_to_body_length:
                length_unit = PX_CM * mean_size
                length_unit_name = "BL"
            else:
                length_unit = PX_CM
                length_unit_name = "cm"
            tr = get_trajectories(
                tr_row.abs_trajectory_path,
                center=tr_row.roi_center,
                interpolate_nans=True,
                smooth_params={"sigma": SIGMA},
                length_unit_dict={
                    "length_unit": length_unit,
                    "length_unit_name": length_unit_name,
                },
                time_unit_dict={
                    "time_unit": FRAME_RATE,
                    "time_unit_name": "seconds",
                },
                tracking_interval=[0, NUM_FRAMES_FOR_ANALYSIS],
            )
            tr_vars_df = tr_variables_to_df(tr, variables_list)
            tr_vars_df["trial_uid"] = [tr_row.trial_uid] * len(tr_vars_df)
            tr_vars_dfs.append(tr_vars_df)
    tr_vars_df = pd.concat(tr_vars_dfs).reset_index()
    return tr_vars_df


if __name__ == "__main__":

    from logger import setup_logs

    logger = setup_logs("get_variables_tables")

    videos_table = pd.read_csv(VIDEOS_INDEX_FILE_NAME)
    videos_table["abs_trajectory_path"] = videos_table[
        "trajectory_path"
    ].apply(
        lambda x: os.path.join(TRACKING_DATA_FOLDER_PATH, x)
        if isinstance(x, str)
        else np.nan
    )
    animals_table = pd.read_csv(ANIMALS_INDEX_FILE_PATH)

    for name, tt_dataset_info in TRAJECTORYTOOLS_DATASETS_INFO.items():
        if not os.path.isfile(tt_dataset_info["file_path"]):
            tr_vars_df = _generate_variables_table(
                videos_table,
                animals_table,
                tt_dataset_info["variables"],
                scale_to_body_length=tt_dataset_info["scale_to_body_length"],
            )
            tr_vars_df = _add_line_and_genotype_info(
                tr_vars_df, videos_table, animals_table
            )
            tr_vars_df.to_pickle(tt_dataset_info["file_path"])
        else:
            tr_vars_df = pd.read_pickle(tt_dataset_info["file_path"])

    # trajectories_table = pd.read_csv(TRAJECTORIES_INDEX_FILE_NAME)
    # animals_table = pd.read_csv(ANIMALS_INDEX_FILE_PATH)

    # for idx, trajectory in trajectories_table.iterrows():
    #     animals = animals_table[
    #         animals_table.trial_uid == trajectory.trial_uid
    #     ]
    #     variables = tr_indivs[tr_indivs.trial_uid == trajectory.trial_uid]
    #     if not variables.empty:
    #         id_last_fish = trajectory.id_last_fish - 1
    #         id_first_fish = 0 if id_last_fish == 1 else 1
    #         genotype_last_fish = variables[
    #             variables.identity == id_last_fish
    #         ].genotype.unique()[0]
    #         genotype_first_fish = variables[
    #             variables.identity == id_first_fish
    #         ].genotype.unique()[0]
    #         genotype_last_fish_ = animals[
    #             animals.fish_id_exp == 2
    #         ].genotype.values[0]
    #         genotype_first_fish_ = animals[
    #             animals.fish_id_exp == 1
    #         ].genotype.values[0]
    #         assert genotype_first_fish == genotype_first_fish_
    #         assert genotype_last_fish == genotype_last_fish_

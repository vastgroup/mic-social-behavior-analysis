import argparse
import copy
import os
import time
from typing import List

import numpy as np
import pandas as pd
import trajectorytools as tt
from tqdm import tqdm
from trajectorytools.export import (
    GROUP_VARIABLES,
    INDIVIDUAL_NEIGHBOUR_VARIABLES,
    INDIVIDUAL_VARIALBES,
    tr_variables_to_df,
)
from trajectorytools.export.variables import local_polarization

from constants import (
    ANIMALS_INDEX_FILE_PATH,
    NUM_FRAMES_FOR_ANALYSIS,
    TRACKING_DATA_FOLDER_PATH,
    TRAJECTORYTOOLS_INDIV_VARS_FILE_PATH,
    VIDEOS_INDEX_FILE_NAME,
)
from utils import clean_impossible_speed_jumps

INDIVIDUAL_VARIALBES = [
    var_
    for var_ in INDIVIDUAL_VARIALBES
    if var_["name"] != "local_polarization"
]
INDIVIDUAL_VARIALBES.append(
    {
        "name": "local_polarization",
        "func": local_polarization,
        "kwargs": {"number_of_neighbours": 1},
    }
)


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


if __name__ == "__main__":

    videos_table = pd.read_csv(VIDEOS_INDEX_FILE_NAME)
    videos_table["abs_trajectory_path"] = videos_table[
        "trajectory_path"
    ].apply(
        lambda x: os.path.join(TRACKING_DATA_FOLDER_PATH, x)
        if isinstance(x, str)
        else np.nan
    )
    animals_table = pd.read_csv(ANIMALS_INDEX_FILE_PATH)

    if not os.path.isfile(TRAJECTORYTOOLS_INDIV_VARS_FILE_PATH):
        tr_indivs = []
        for idx, tr_row in tqdm(
            videos_table.iterrows(),
            desc="generating_dataframes",
            total=len(videos_table),
        ):
            if tr_row.tracked and tr_row.valid_for_analysis:
                tr = get_trajectories(
                    tr_row.abs_trajectory_path,
                    center=tr_row.roi_center,
                    interpolate_nans=True,
                    smooth_params={"sigma": 1.0},
                    length_unit_dict={
                        "length_unit": 54,
                        "length_unit_name": "cm",
                    },
                    time_unit_dict={
                        "time_unit": 30,
                        "time_unit_name": "seconds",
                    },
                    tracking_interval=[0, NUM_FRAMES_FOR_ANALYSIS],
                )
                tr_indiv = tr_variables_to_df(tr, INDIVIDUAL_VARIALBES)
                tr_indiv["trial_uid"] = [tr_row.trial_uid] * len(tr_indiv)
                tr_indivs.append(tr_indiv)
        tr_indivs = pd.concat(tr_indivs).reset_index()
        print(f"tr_indivs shape: {tr_indivs.shape}")
        print(f"{tr_indivs.identity.unique()}")
        videos_table["identity"] = videos_table.id_last_fish - 1
        videos_table["fish_id_exp"] = 2
        tr_indivs = pd.merge(
            tr_indivs,
            videos_table[["trial_uid", "identity", "fish_id_exp"]],
            on=["trial_uid", "identity"],
            how="outer",
        )
        tr_indivs.fish_id_exp.fillna(1, inplace=True)
        print(f"tr_indivs shape: {tr_indivs.shape}")
        print(f"{tr_indivs.identity.unique()}")
        tr_indivs = pd.merge(
            tr_indivs,
            videos_table[
                [
                    "trial_uid",
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
        tr_indivs["line"] = tr_indivs[["gene", "founder"]].agg("_".join)
        tr_indivs = pd.merge(
            tr_indivs,
            animals_table[
                ["trial_uid", "fish_id_exp", "genotype", "dpf", "size_cm"]
            ],
            on=["trial_uid", "fish_id_exp"],
            how="outer",
        )
        print(f"tr_indivs shape: {tr_indivs.shape}")
        print(f"{tr_indivs.identity.unique()}")
        print("Dropping rows with NaN in frames or identity")
        tr_indivs.dropna(subset=["frame", "identity"], inplace=True)
        print(f"tr_indivs shape: {tr_indivs.shape}")
        print("Saving pkl")
        tr_indivs.to_pickle(TRAJECTORYTOOLS_INDIV_VARS_FILE_PATH)
    else:
        tr_indivs = pd.read_pickle(TRAJECTORYTOOLS_INDIV_VARS_FILE_PATH)

import argparse
import os

import numpy as np
import pandas as pd

from constants import (
    ANIMALS_COUNT_FILE_PATH,
    ANIMALS_INDEX_FILE_PATH,
    GENERATED_TABLES_PATH,
    ID_LAST_FISH_STATE_COLUMNS,
    NO_ID_LAST_FISH_FILL_VALUE,
    PER_VIDEO_COLUMNS,
    THRESHOLD_ACCURACY,
    THRESHOLD_CERTAINTY_ID_LAST_FISH,
    THRESHOLD_MEAN_ID_PROBABILITIES,
    THRESHOLD_NUM_IMPOSSIBLE_SPEED_JUMPS,
    THRESHOLD_RATIO_TRACKED,
    TRACKING_STATE_COLUMNS,
    TRAJECTORIES_INDEX_FILE_NAME,
    VALID_GENOTYPES,
    VIDEOS_INDEX_FILE_NAME,
    VIDEOS_TRACKING_STATE_FILE_NAME,
    VIDEOS_VALID_FOR_ANALYSIS_FILE_PATH,
)


def _add_manually_labeled_id_last_fish(videos_table, animals_table):
    # Laura manually labeled the id of the last fish by looking at the
    # trajectories. We recover this info to compare with the automatic label
    # obtained by looking at the number of NaN in the last chunck of the video.
    logger.info("Extracting manually labeled id of the last fish in the video")
    videos_table.set_index("trial_uid", inplace=True)
    videos = animals_table.groupby("trial_uid")
    experiments_with_same_animal_exp_num = []
    manual_id_last_animal = []
    for idx, video in videos:
        # The experimental animal 2 is the last animal to be extracted from the
        # arena i.e. the last animal in the video
        last_animal = video[video.fish_id_exp == 2]
        if len(last_animal) != 1:
            # There were some typing errors and in some experiments
            # the experimental id of the fish was the same for both
            # fish of the video
            experiments_with_same_animal_exp_num.append(
                video.trial_uid.values[0]
            )
        else:
            manual_id_last_animal.append(
                last_animal[["trial_uid", "fish_id_track"]]
            )
    manual_id_last_animal = pd.concat(manual_id_last_animal)
    # We rename the column to be more clear
    manual_id_last_animal.rename(
        {"fish_id_track": "manual_id_last_fish"}, axis=1, inplace=True
    )
    videos_table = pd.merge(
        videos_table, manual_id_last_animal, on="trial_uid"
    )
    videos_table["manual_id_last_fish"] = videos_table[
        "manual_id_last_fish"
    ].fillna(NO_ID_LAST_FISH_FILL_VALUE)
    videos_table["id_last_fish"] = videos_table["id_last_fish"].fillna(
        NO_ID_LAST_FISH_FILL_VALUE
    )
    videos_table["same_id_last_fish"] = (
        videos_table["id_last_fish"] == videos_table["manual_id_last_fish"]
    )
    return videos_table


def _tracking_state_code(
    row,
    keys=TRACKING_STATE_COLUMNS,
):
    return " ".join(["1" if row[key] else "0" for key in keys])


def _id_last_fish_state_code(
    row,
    keys=ID_LAST_FISH_STATE_COLUMNS,
):
    return " ".join(["1" if row[key] else "0" for key in keys])


def _add_video_quality_state_columns(videos_table):
    # Adding videos labels
    logger.info("Adding column `tracked`")
    videos_table["tracked"] = ~videos_table["trajectory_path"].isna()
    videos_table["valid_genotype_group"] = videos_table["genotype_group"].isin(
        VALID_GENOTYPES
    )
    videos_table["valid_ratio_frames_tracked"] = (
        videos_table["ratio_frames_tracked"] > THRESHOLD_RATIO_TRACKED
    )
    videos_table["valid_estimated_accuracy"] = (
        videos_table["estimated_accuracy"] > THRESHOLD_ACCURACY
    )
    videos_table["valid_mean_id_probabilities"] = (
        videos_table["mean_id_probabilities"] > THRESHOLD_MEAN_ID_PROBABILITIES
    )
    videos_table["valid_num_impossible_speed_jumps"] = (
        videos_table["num_impossible_speed_jumps"]
        < THRESHOLD_NUM_IMPOSSIBLE_SPEED_JUMPS
    )
    videos_table["valid_num_unsolvable_impossible_speed_jumps"] = (
        videos_table["num_unsolvable_impossible_jumps"] == 0
    )
    videos_table["valid_id_last_fish"] = videos_table["id_last_fish"] > 0
    videos_table["valid_certainty_id_last_fish"] = (
        videos_table["certainty_id_last_fish"]
        > THRESHOLD_CERTAINTY_ID_LAST_FISH
    )
    videos_table["tracking_state"] = videos_table.apply(
        _tracking_state_code, axis=1
    )
    videos_table["id_last_fish_state"] = videos_table.apply(
        _id_last_fish_state_code, axis=1
    )
    videos_table["for_analysis_state"] = videos_table[
        ["tracking_state", "id_last_fish_state"]
    ].agg("-".join, axis=1)
    videos_table["valid_tracking"] = (
        videos_table.valid_mean_id_probabilities
        & videos_table.valid_ratio_frames_tracked
        & videos_table.valid_num_unsolvable_impossible_speed_jumps
    )
    videos_table["valid_genotype_id"] = (
        videos_table.valid_id_last_fish
        & videos_table.valid_certainty_id_last_fish
    )
    videos_table["valid_for_analysis"] = (
        videos_table.tracked
        & videos_table.valid_genotype_group
        & videos_table.valid_tracking
        & videos_table.valid_genotype_id
    )
    return videos_table


def generate_videos_table(trajectories_table, animals_table):
    # `animals_table` has information about each animal used in an experiment
    # regardless of whether the videos was tracked or not
    # `trajectories_table` has information about each trajectory of a video
    # tracked, but it has no information about the genotyping.
    # We create a table that has info about each experiment with genotyping
    # and trajectories data
    videos_table = pd.merge(
        animals_table[PER_VIDEO_COLUMNS].drop_duplicates(),
        trajectories_table,
        left_on=["trial_uid", "folder_name_track", "trial", "gene"],
        right_on=["trial_uid", "folder_name_track", "trial", "gene"],
        how="left",
    )
    print(videos_table.columns)
    videos_table.drop(
        ["founder_x", "founder_y", "replicate"], axis=1, inplace=True
    )
    print(videos_table.columns)
    videos_table["gene"] = videos_table.trial_uid.apply(
        lambda x: x.split("_")[0]
    )
    videos_table["founder"] = videos_table.trial_uid.apply(
        lambda x: x.split("_")[1]
    )
    videos_table["line"] = videos_table.gene + "_" + videos_table.founder
    videos_table["replicate"] = videos_table.trial_uid.apply(
        lambda x: x.split("_")[2]
    )
    videos_table["line_replicate"] = (
        videos_table.line + "_" + videos_table.replicate.astype(str)
    )
    videos_table["experiment_type"] = videos_table.trial_uid.apply(
        lambda x: x.split("_")[3]
    )
    videos_table["line_experiment"] = (
        videos_table["line"] + "_" + videos_table["experiment_type"]
    )
    videos_table["line_replicate_experiment"] = (
        videos_table["line_replicate"] + "_" + videos_table["experiment_type"]
    )
    videos_table = _add_manually_labeled_id_last_fish(
        videos_table, animals_table
    )
    videos_table = _add_video_quality_state_columns(videos_table)
    return videos_table


def get_tracking_state_table(videos_table):

    videos_tracking_state = videos_table[
        ["trial_uid", "genotype_group"]
        + [
            "ratio_frames_tracked",
            "certainty_id_last_fish",
            "accuracy",
            "mean_id_probabilities",
            "ratio_impossible_speed_jumps",
            "num_impossible_speed_jumps",
            "num_unsolvable_impossible_jumps",
            "id_last_fish",
            "manual_id_last_fish",
        ]
        + TRACKING_STATE_COLUMNS
        + ["valid_tracking", "valid_for_analysis"]
    ]
    return videos_tracking_state


def print_summary_tracking_state(videos_table):

    for column in (
        TRACKING_STATE_COLUMNS
        + ID_LAST_FISH_STATE_COLUMNS
        + ["valid_tracking", "valid_for_analysis"]
    ):

        if column != "tracked":
            logger.info(f"\n*** Videos with {column}")
            logger.info(
                (videos_table[videos_table.tracked][column])
                .value_counts()
                .to_string()
            )
        else:
            logger.info(f"\n*** Videos {column}")
            logger.info((videos_table[column]).value_counts().to_string())


def generate_videos_valid_for_analysis_table(
    videos_table,
    main_columns=[
        "folder_name_track",
        "genotype_group",
        "valid_for_analysis",
    ],
):
    logger.info("Generating videos valid for analysis table")
    columns_to_count = (
        main_columns + TRACKING_STATE_COLUMNS + ID_LAST_FISH_STATE_COLUMNS
    )
    videos_valid_for_analysis = (
        videos_table[columns_to_count]
        .value_counts()
        .to_frame()
        .reset_index()
        .sort_values(columns_to_count)
        .rename({0: "num_videos"}, axis=1)
        .set_index(columns_to_count)
        .unstack("valid_for_analysis")
        .fillna(0)
    )
    return videos_valid_for_analysis


if __name__ == "__main__":
    from logger import setup_logs

    logger = setup_logs("experiments_summary")

    logger.info(f"Loading {ANIMALS_INDEX_FILE_PATH}")
    animals_table = pd.read_csv(ANIMALS_INDEX_FILE_PATH)
    logger.info(f"Loading {TRAJECTORIES_INDEX_FILE_NAME}")
    trajectories_table = pd.read_csv(TRAJECTORIES_INDEX_FILE_NAME)

    # A table for all experiments
    logger.info("Creating table of videos from animals and trajectories table")
    videos_table = generate_videos_table(trajectories_table, animals_table)
    videos_table.to_csv(VIDEOS_INDEX_FILE_NAME, index=False)

    # A table with only the information that defines the tracking state
    videos_tracking_state = get_tracking_state_table(videos_table)
    videos_tracking_state.to_csv(VIDEOS_TRACKING_STATE_FILE_NAME, index=False)

    # Print some info about the number of videos in each state
    print_summary_tracking_state(videos_table)

    # A table counting the number of videos valid for analysis in gene
    # and genotype group
    videos_valid_for_analysis = generate_videos_valid_for_analysis_table(
        videos_table
    )
    videos_valid_for_analysis.to_csv(VIDEOS_VALID_FOR_ANALYSIS_FILE_PATH)

    videos_valid_for_analysis_total = generate_videos_valid_for_analysis_table(
        videos_table,
        main_columns=[
            "valid_for_analysis",
        ],
    )
    videos_valid_for_analysis_total.to_csv(
        VIDEOS_VALID_FOR_ANALYSIS_FILE_PATH.replace(".csv", "_total.csv")
    )

    # Videos to retrack
    videos_to_retrack = videos_table[
        (~videos_table.valid_tracking)
        & (videos_table.valid_genotype_group)
        & (videos_table.tracked)
    ][
        ["trial_uid", "folder_name_track", "disk_name_video", "genotype_group"]
        + [
            "ratio_frames_tracked",
            "certainty_id_last_fish",
            "accuracy",
            "estimated_accuracy",
            "mean_id_probabilities",
            "ratio_impossible_speed_jumps",
            "num_impossible_speed_jumps",
            "num_unsolvable_impossible_jumps",
            "id_last_fish",
            "manual_id_last_fish",
        ]
        + TRACKING_STATE_COLUMNS
        + ["valid_tracking", "valid_for_analysis"]
    ].sort_values(
        [
            "folder_name_track",
            "ratio_frames_tracked",
            "estimated_accuracy",
            "accuracy",  # accuracy in animals_table
            "num_impossible_speed_jumps",
            "num_unsolvable_impossible_jumps",
        ],
        ascending=True,
    )

    videos_to_retrack.to_csv(
        os.path.join(GENERATED_TABLES_PATH, "videos_to_retrack.csv"),
        index=False,
    )

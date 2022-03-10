import os

import pandas as pd
from confapp import conf
from mic_analysis.logger import setup_logs
from mic_analysis.table_generators_utils import (
    generate_videos_table,
    generate_videos_valid_for_analysis_table,
    get_tracking_state_table,
    print_summary_tracking_state,
)

logger = setup_logs("experiments_summary")

logger.info(f"Loading {conf.ANIMALS_INDEX_FILE_PATH}")
animals_table = pd.read_csv(conf.ANIMALS_INDEX_FILE_PATH)
logger.info(f"Loading {conf.TRAJECTORIES_INDEX_FILE_NAME}")
trajectories_table = pd.read_csv(conf.TRAJECTORIES_INDEX_FILE_NAME)
logger.info(f"Loading {conf.EXPERIMENTS_INFO_TABLE}")
experiments_info_table = pd.read_csv(conf.EXPERIMENTS_INFO_TABLE)

# A table for all experiments
logger.info("Creating table of videos from animals and trajectories table")
videos_table = generate_videos_table(
    trajectories_table, animals_table, experiments_info_table
)
videos_table.to_csv(conf.VIDEOS_INDEX_FILE_NAME, index=False)

# A table with only the information that defines the tracking state
videos_tracking_state = get_tracking_state_table(videos_table)
videos_tracking_state.to_csv(conf.VIDEOS_TRACKING_STATE_FILE_NAME, index=False)

# Print some info about the number of videos in each state
print_summary_tracking_state(videos_table)

# A table counting the number of videos valid for analysis in gene
# and genotype group
videos_valid_for_analysis = generate_videos_valid_for_analysis_table(
    videos_table
)
videos_valid_for_analysis.to_csv(conf.VIDEOS_VALID_FOR_ANALYSIS_FILE_PATH)

videos_valid_for_analysis_total = generate_videos_valid_for_analysis_table(
    videos_table,
    main_columns=[
        "valid_for_analysis",
    ],
)
videos_valid_for_analysis_total.to_csv(
    conf.VIDEOS_VALID_FOR_ANALYSIS_FILE_PATH.replace(".csv", "_total.csv")
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
        "automatic_id_last_fish",
        "manual_id_last_fish",
    ]
    + conf.TRACKING_STATE_COLUMNS
    + conf.ID_LAST_FISH_STATE_COLUMNS
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
    os.path.join(conf.GENERATED_TABLES_PATH, "videos_to_retrack.csv"),
    index=False,
)

# TODO: Report of value counts per sanity validity variable for videos

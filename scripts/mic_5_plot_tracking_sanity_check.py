import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from confapp import conf
from natsort import natsorted

import trajectorytools as tt
from mic_analysis.logger import setup_logs
from mic_analysis.plotters import (
    plot_tracked_videos_summary,
    plot_video_tracking_states,
)

logger = setup_logs("tracking_sanity_check")

logger.info("Loading videos table")
videos_table = pd.read_csv(conf.VIDEOS_INDEX_FILE_NAME)

logger.info("Selecting only tracked videos")
videos_table = videos_table[videos_table.tracked]
logger.info("Adding `abs_trajectory_path`")
videos_table["abs_trajectory_path"] = videos_table["trajectory_path"].apply(
    lambda x: os.path.join(conf.TRACKING_DATA_FOLDER_PATH, x)
)

logger.info("Plotting tracking states")
plot_video_tracking_states(videos_table, state="tracking_state")
plot_video_tracking_states(videos_table, state="id_last_fish_state")
plot_video_tracking_states(videos_table, state="for_analysis_state")

plot_tracked_videos_summary(videos_table)

sns.pairplot(
    data=videos_table[
        [
            "ratio_frames_tracked",
            "estimated_accuracy",
            "accuracy",
            "certainty_id_last_fish",
            "mean_id_probabilities",
            "num_impossible_speed_jumps",
        ]
    ],
)
for extension in [".png", ".pdf"]:
    plt.savefig(
        os.path.join(conf.GENERATED_FIGURES_PATH, f"correlation{extension}")
    )

# TODO: Check if old figures are worth to keep plotting them
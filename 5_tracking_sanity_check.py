import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import trajectorytools as tt
from natsort import natsorted
from tqdm import tqdm

from constants import (
    GENERATED_FIGURES_PATH,
    ID_LAST_FISH_STATE_COLUMNS,
    NUM_FRAMES_FOR_ANALYSIS,
    TRACKING_DATA_FOLDER_PATH,
    TRACKING_STATE_COLUMNS,
    VIDEOS_INDEX_FILE_NAME,
)


def count_nan_intervals(indiv_df, id_):
    indiv_df = indiv_df[indiv_df.identity == id_][["s_x"]]
    df = (
        indiv_df.s_x.isnull()
        .astype(int)
        .groupby(indiv_df.s_x.notnull().astype(int).cumsum())
        .sum()
    )
    return df.value_counts


def get_nans_array(tr):
    nans_bool_array = np.isnan(tr.s[..., 0].T)
    return nans_bool_array


def visualize_nans(ax, tr, max_num_frames):
    tx = get_nans_array(tr)
    video = (
        np.zeros((tr.number_of_individuals, max_num_frames + 2000)) * np.nan
    )
    video[:, : tx.shape[1]] = tx
    ax.imshow(video, interpolation="None", origin="lower")
    ax.set_aspect("auto")


def plot_video_tracking_states(videos_table, state="for_analysis_state"):
    videos_table.sort_values("folder_name_track", inplace=True)
    folder_name_tracks = videos_table.folder_name_track.unique()

    if state == "for_analysis_state":
        title = (
            f"{state}: \n"
            + "   ".join(TRACKING_STATE_COLUMNS)
            + "\n"
            + "   ".join(ID_LAST_FISH_STATE_COLUMNS)
        )
    elif state == "tracking_state":
        title = f"{state}: \n" + "   ".join(TRACKING_STATE_COLUMNS)
    elif state == "id_last_fish_state":
        title = f"{state}: \n" + "   ".join(ID_LAST_FISH_STATE_COLUMNS)
    else:
        raise Exception(f"No valid state {state}")

    fig, axs = plt.subplots(6, 6, figsize=(30, 30))
    fig.suptitle(title)
    plt.subplots_adjust(
        left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.4
    )

    videos_table.sort_values(
        [state, "genotype_group"], inplace=True, ascending=False
    )
    for folder_name_track, ax in zip(folder_name_tracks, axs.flatten()):
        sub_videos = videos_table[
            videos_table.folder_name_track == folder_name_track
        ]
        sns.countplot(
            ax=ax,
            data=sub_videos,
            x=state,
            hue="genotype_group",
            order=videos_table[state].unique(),
        )
        ax.set_title(folder_name_track)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    for extension in [".png", ".pdf"]:
        fig.savefig(
            os.path.join(GENERATED_FIGURES_PATH, f"{state}{extension}")
        )


def visualize_speed_jumps(ax, speed_jumps):
    if not "[]" in speed_jumps:
        speed_jumps = eval(speed_jumps.replace("array", "np.array"))
        x = speed_jumps[0]
        y = speed_jumps[1]
        ax.plot(x, y, "r.", markersize=3)


def plot_tracked_videos_summary(videos_table):

    videos_table = videos_table.sort_values(["trial_uid"]).reset_index(
        drop=True
    )

    n_trajectories = len(videos_table)
    max_num_frames = int(videos_table.number_of_frames.max())

    fig, axs = plt.subplots(
        n_trajectories, 1, figsize=(30, 0.2 * n_trajectories)
    )
    plt.subplots_adjust(
        left=0.01, bottom=0.01, right=0.7, top=0.99, wspace=0.01, hspace=0.01
    )

    for idx, video_info in tqdm(
        videos_table.iterrows(), desc="Plotting nans..."
    ):
        ax = axs[idx]
        tr = tt.trajectories.FishTrajectories.from_idtrackerai(
            video_info.abs_trajectory_path, interpolate_nans=False
        )

        visualize_nans(ax, tr, max_num_frames)
        visualize_speed_jumps(ax, video_info.speed_jumps)
        ax.set_aspect("auto")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        if not video_info.valid_genotype_id:
            ax.plot([max_num_frames + 200], [1], "rs", ms=3)
        if not video_info.valid_tracking:
            ax.plot([max_num_frames + 300], [1], "ro", ms=3)
        if video_info.valid_for_analysis:
            ax.plot([max_num_frames + 100], [1], "g^", ms=3)

        ax.text(
            max_num_frames + 2000,
            0,
            f"{video_info.for_analysis_state}",
            ha="right",
        )
        ax.text(
            max_num_frames + 2100,
            0,
            f"trial: {video_info.trial_uid} - "
            f"tracked: {video_info.ratio_frames_tracked:.3f} - "
            f"id_probs: {video_info.mean_id_probabilities:.3f} - "
            f"estimated_accuracy: {video_info.estimated_accuracy:.3f} - "
            f"speed jumps: {video_info.num_impossible_speed_jumps:.0f} - "
            f"id: {video_info.id_last_fish:.0f} - "
            f"certainty: {video_info.certainty_id_last_fish:.3f}",
            ha="left",
            c="k",
        )
    code_legend = " - ".join(
        [
            "   ".join(TRACKING_STATE_COLUMNS),
            "   ".join(ID_LAST_FISH_STATE_COLUMNS),
        ]
    )
    fig.suptitle("Ready for analysis state code: \n" + code_legend, y=0.995)
    for extension in [".png", ".pdf"]:
        fig.savefig(
            os.path.join(GENERATED_FIGURES_PATH, f"videos_summary{extension}")
        )


if __name__ == "__main__":
    from logger import setup_logs

    logger = setup_logs("tracking_sanity_check")

    logger.info("Loading videos table")
    videos_table = pd.read_csv(VIDEOS_INDEX_FILE_NAME)

    logger.info("Selecting only tracked videos")
    videos_table = videos_table[videos_table.tracked]
    logger.info("Adding `abs_trajectory_path`")
    videos_table["abs_trajectory_path"] = videos_table[
        "trajectory_path"
    ].apply(lambda x: os.path.join(TRACKING_DATA_FOLDER_PATH, x))

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
            os.path.join(GENERATED_FIGURES_PATH, f"correlation{extension}")
        )

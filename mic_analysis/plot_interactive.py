import os

import numpy as np
import pandas as pd
from confapp import conf
from IPython.display import display
from ipywidgets import interact
from natsort import natsorted
from PIL import Image


def display_partition_figures(load_path, var_type):
    figure_path = os.path.join(load_path, f"{var_type}_vars.png")
    if os.path.isfile(figure_path):
        image = Image.open(figure_path)
        factor = 1
        display(
            image.resize(
                (int(factor * image.size[0]), int(factor * image.size[1]))
            )
        )
    else:
        print(f"image {figure_path} not found")


def get_possible_variables(outliers):
    variables = list(outliers["variable"].unique())
    if np.nan in variables:
        variables.remove(np.nan)
    return variables


def get_possible_outliers(outliers, variable, video_uid_col):
    outliers_of_variable = outliers[outliers["variable"] == variable]
    outliers_uids = list(outliers_of_variable[video_uid_col].unique())
    if np.nan in outliers_uids:
        outliers_uids.remove(np.nan)
    return outliers_uids


def visualize_outlier(
    outliers, outlier, outliers_fig_path, variable, video_uid_col
):
    var_type = outliers[
        (outliers[video_uid_col] == outlier)
        & (outliers["variable"] == variable)
    ]["var_type"]
    assert len(set(var_type.values)) == 1, var_type
    var_type = var_type.values[0]
    visualize_video(
        outliers_fig_path, outliers, video_uid_col, outlier, var_type
    )


def visualize_video(
    videos_fig_path, videos_table, video_uid_col, video_uid, var_type
):
    if var_type != "indiv_nb":
        fig_path = os.path.join(videos_fig_path, f"{video_uid}_{var_type}.png")
        if os.path.isfile(fig_path):
            image = Image.open(fig_path)
            factor = 1
            display(
                image.resize(
                    (int(factor * image.size[0]), int(factor * image.size[1]))
                )
            )
        else:
            print(f"image {fig_path} not found")
    else:
        if "identity" in videos_table:
            ids = list(
                videos_table[
                    videos_table[video_uid_col] == video_uid
                ].identity.unique()
            )
        else:
            group_size = (
                videos_table[videos_table[video_uid_col] == video_uid]
                .iloc[0]
                .group_size
            )
            ids = np.arange(group_size).astype(float)
        for id_ in ids:
            fig_path = os.path.join(
                videos_fig_path, f"{video_uid}_{id_}_{var_type}.png"
            )
            if os.path.isfile(fig_path):
                image = Image.open(fig_path)
                factor = 1
                display(
                    image.resize(
                        (
                            int(factor * image.size[0]),
                            int(factor * image.size[1]),
                        )
                    )
                )
            else:
                print(f"image {fig_path} not found")

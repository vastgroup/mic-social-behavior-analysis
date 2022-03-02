import os

from ipywidgets import interact
from confapp import conf
import pandas as pd
import numpy as np

from PIL import Image
from IPython.display import display
from natsort import natsorted

def display_partition_figures(load_path, var_type):
    figure_path = os.path.join(load_path, f"{var_type}_vars.png")
    if os.path.isfile(figure_path):
        image = Image.open(figure_path)
        factor = 1
        display(image.resize((int(factor * image.size[0]), int(factor * image.size[1]))))
    else:
        print(f"image {figure_path} not found")
            
def get_possible_variables(outliers):
    variables = list(outliers['variable'].unique())
    variables.remove(np.nan)
    return variables

def get_possible_outliers(outliers, variable, video_uid_col):
    outliers_of_variable = outliers[outliers['variable'] == variable]
    outliers_uids = list(outliers_of_variable[video_uid_col].unique())
    if np.nan in outliers_uids:
        outliers_uids.remove(np.nan)
    return outliers_uids


def visualize_outlier(outliers, outlier, outliers_fig_path, variable, video_uid_col):
    var_type = outliers[(outliers[video_uid_col] == outlier) & (outliers['variable'] == variable)]['var_type']
    assert len(set(var_type.values)) == 1, var_type
    var_type = var_type.values[0]
    if var_type != "indiv_nb":
        fig_path = os.path.join(outliers_fig_path, f"{outlier}_{var_type}.png")
        if os.path.isfile(fig_path):
            image = Image.open(fig_path)
            factor = 1
            display(image.resize((int(factor * image.size[0]), int(factor * image.size[1]))))
        else:
            print(f"image {fig_path} not found")
    else:
        ids = list(outliers[outliers[video_uid_col] == outlier].identity.unique())
        print(ids)
        for id_ in ids:
            fig_path = os.path.join(outliers_fig_path, f"{outlier}_{id_}_{var_type}.png")
            if os.path.isfile(fig_path):
                image = Image.open(fig_path)
                factor = 1
                display(image.resize((int(factor * image.size[0]), int(factor * image.size[1]))))
            else:
                print(f"image {fig_path} not found")
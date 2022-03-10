import logging
import os
from glob import glob

import numpy as np
import pandas as pd
import tqdm
import trajectorytools as tt
from confapp import conf
from pandas_split.pandas_split import read_all
from tqdm import tqdm
from trajectorytools.export import tr_variables_to_df

from mic_analysis.table_generators_utils import add_line_and_genotype_info

from .stats import (
    _compute_agg_stat,
    group_agg_stasts_kwargs,
    indiv_agg_stasts_kwargs,
    indiv_nb_agg_stats_kwargs,
    standardize_replicate_data_wrt_het,
)
from .utils import clean_impossible_speed_jumps, data_filter
from .variables import (
    _group_variables,
    _individual_nb_variables,
    _individual_variables,
)

logger = logging.getLogger(__name__)


TRAJECTORYTOOLS_DATASETS_INFO = {
    "tr_indiv_bl": {
        "var_type": "indiv",
        "dir_path": os.path.join(
            conf.GENERATED_TABLES_PATH, conf.TR_INDIV_BL_DIR_NAME
        ),
        "variables_to_compute": _individual_variables,
        "scale_to_body_length": True,
        "variables_names": [var_["name"] for var_ in _individual_variables],
        "agg_stats_kwargs": indiv_agg_stasts_kwargs,
    },
    "tr_indiv_nb_bl": {
        "var_type": "indiv_nb",
        "dir_path": os.path.join(
            conf.GENERATED_TABLES_PATH, conf.TR_INDIV_NB_BL_DIR_NAME
        ),
        "scale_to_body_length": True,
        "variables_to_compute": _individual_nb_variables,
        "agg_stats_kwargs": indiv_nb_agg_stats_kwargs,
    },
    "tr_group_bl": {
        "var_type": "group",
        "dir_path": os.path.join(
            conf.GENERATED_TABLES_PATH, conf.TR_GROUP_BL_DIR_NAME
        ),
        "scale_to_body_length": True,
        "variables_to_compute": _group_variables,
        "agg_stats_kwargs": group_agg_stasts_kwargs,
    },
}


def get_data(path, data_filters, agg_stats_kwargs):
    logger.info("Loading data")
    data = load_dataset(path, video_filters=data_filters)
    logger.info("Loaded")

    if "nb_angle" in data.columns:
        data["nb_angle"] = np.arctan2(
            data["nb_position_x"], data["nb_position_y"]
        )

    data = standardize_replicate_data_wrt_het(data)

    logger.info("Adding info columns")
    if "indiv" in path:
        data["genotype_group_genotype"] = (
            data["genotype_group"] + "-" + data["genotype"]
        )
        data["trial_uid_id"] = (
            data["trial_uid"] + "_" + data["identity"].astype(str)
        )
    if "indiv_nb" in path:
        data["genotype_group_genotype_nb"] = (
            data["genotype_group"] + "-" + data["genotype_nb"]
        )
        data["trial_uid_id_nb"] = (
            data["trial_uid"] + "_" + data["identity_nb"].astype(str)
        )
        data["focal_nb_genotype"] = (
            data["genotype"] + "-" + data["genotype_nb"]
        )
    logger.info("Added")

    data_stats = _compute_agg_stat(data=data, **agg_stats_kwargs)
    if "indiv" in path:
        data_stats["genotype_group_genotype"] = (
            data_stats["genotype_group"] + "-" + data_stats["genotype"]
        )
    if "indiv_nb" in path:
        data_stats["genotype_group_genotype_nb"] = (
            data_stats["genotype_group"] + "-" + data_stats["genotype_nb"]
        )
    return data, data_stats


def get_partition_datasets(datasets_info, trial_uids):
    video_filters = [lambda x: x["trial_uid"].isin(trial_uids)]
    datasets = {}
    no_data = False
    for name, dataset_info in datasets_info.items():
        logger.info(f"Getting dataset {name}")
        data = load_dataset(dataset_info["dir_path"], video_filters)
        if data.empty:
            no_data = True
        data_stats = pd.read_pickle(
            os.path.join(
                dataset_info["dir_path"], conf.PER_ANIMAL_STATS_FILE_NAME
            )
        )
        data_stats = data_filter(data_stats, filters=video_filters)
        datasets[f"data_{dataset_info['var_type']}"] = data
        datasets[f"data_{dataset_info['var_type']}_stats"] = data_stats
    return datasets, no_data


def get_datasets(data_filters):
    logger.info("Getting datasets")

    logger.info("Getting dataset data_indiv")
    data_indiv, data_indiv_stats = get_data(
        TRAJECTORYTOOLS_DATASETS_INFO["tr_indiv_bl"]["dir_path"],
        data_filters=data_filters,
        agg_stats_kwargs=TRAJECTORYTOOLS_DATASETS_INFO["tr_indiv_bl"][
            "agg_stats_kwargs"
        ],
    )
    logger.info("done")

    logger.info("Getting dataset data_group")
    data_group, data_group_stats = get_data(
        TRAJECTORYTOOLS_DATASETS_INFO["tr_group_bl"]["dir_path"],
        data_filters=data_filters,
        agg_stats_kwargs=TRAJECTORYTOOLS_DATASETS_INFO["tr_group_bl"][
            "agg_stats_kwargs"
        ],
    )
    logger.info("done")

    logger.info("Getting dataset data_indiv_nb")
    data_indiv_nb, data_indiv_nb_stats = get_data(
        TRAJECTORYTOOLS_DATASETS_INFO["tr_indiv_nb_bl"]["dir_path"],
        data_filters=data_filters,
        agg_stats_kwargs=TRAJECTORYTOOLS_DATASETS_INFO["tr_indiv_nb_bl"][
            "agg_stats_kwargs"
        ],
    )
    logger.info("done")

    datasets = {
        "data_indiv": data_indiv,
        "data_indiv_stats": data_indiv_stats,
        "data_group": data_group,
        "data_group_stats": data_group_stats,
        "data_indiv_nb": data_indiv_nb,
        "data_indiv_nb_stats": data_indiv_nb_stats,
    }

    return datasets


## VARIABLES TABLES
def _speed(trajectories):
    vel = np.diff(trajectories, axis=0)
    speed = np.sqrt(vel[..., 0] ** 2 + vel[..., 1] ** 2)
    return speed


def _get_trajectories(
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


def _get_mean_size_cm(tr_row, animals, animals_table):
    mean_size_cm = np.mean(animals.size_cm)
    if np.isnan(mean_size_cm):
        logger.info("Getting mean size_cm from all videos of same gene")
        mean_size_cm = animals_table[
            (animals_table.gene == tr_row.gene)
            & (animals_table.founder == tr_row.founder)
            & (animals_table.replicate == tr_row.replicate)
            & (animals_table.experiment_type == tr_row.experiment_type)
        ].size_cm.mean()
        if np.isnan(mean_size_cm):
            logger.info(
                "Getting mean size_cm from body_length " "idtracker.ai info"
            )
            mean_size_cm = tr_row.body_length / conf.PX_CM
    return mean_size_cm


def _get_length_unit(scale_to_body_length, mean_size_cm):
    if scale_to_body_length:
        length_unit = conf.PX_CM * mean_size_cm
        length_unit_name = "BL"
    else:
        length_unit = conf.PX_CM
        length_unit_name = "cm"
    return length_unit, length_unit_name


def _add_normed_positions(tr_vars_df):
    if "s_x" in tr_vars_df.columns:
        tr_vars_df["s_x_normed"] = (
            tr_vars_df["s_x"] / tr_vars_df["s_x"].abs().max()
        )
        tr_vars_df["s_y_normed"] = (
            tr_vars_df["s_y"] / tr_vars_df["s_y"].abs().max()
        )


def generate_variables_dataset(
    index_path,
    videos_table,
    animals_table,
    variables_list,
    scale_to_body_length,
    save_dir,
    regenerate=False,
):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    files_in_save_dir = glob(os.path.join(save_dir, "*.pkl"))
    file_names = [os.path.split(f)[1] for f in files_in_save_dir]

    if regenerate:
        for file in files_in_save_dir:
            os.remove(file)

    if os.path.isfile(index_path):
        current_index = pd.read_pickle(index_path)
    else:
        current_index = pd.DataFrame({"trial_uid": [], "file": []}).set_index(
            "trial_uid"
        )

    for i, (trial_uid, tr_row) in tqdm(
        enumerate(videos_table.iterrows()),
        desc="generating_dataset",
        total=len(videos_table),
    ):
        logger.info(f"** Video {i}/{len(videos_table)}: {trial_uid}")
        file_name = f"{trial_uid}.pkl"
        already_generated = file_name in file_names
        already_in_index = trial_uid in current_index.index
        if regenerate or not already_generated or not already_in_index:

            if tr_row.tracked and tr_row.valid_for_analysis:
                logger.info(f"Generating dataframe")
                animals = animals_table[animals_table.trial_uid == trial_uid]
                # TODO: Consider background subtraction for BL computation from idtracker.ai
                mean_size_cm = _get_mean_size_cm(
                    tr_row, animals, animals_table
                )
                length_unit, length_unit_name = _get_length_unit(
                    scale_to_body_length, mean_size_cm
                )
                # TODO: Save metadata trajectories preprocessing next to data
                tr = _get_trajectories(
                    tr_row.abs_trajectory_path,
                    center=tr_row.roi_center,
                    interpolate_nans=True,
                    smooth_params={"sigma": conf.SIGMA},
                    length_unit_dict={
                        "length_unit": length_unit,
                        "length_unit_name": length_unit_name,
                    },
                    time_unit_dict={
                        "time_unit": conf.FRAME_RATE,
                        "time_unit_name": "seconds",
                    },
                    tracking_interval=[0, conf.NUM_FRAMES_FOR_ANALYSIS],
                )

                tr_vars_df = tr_variables_to_df(tr, variables_list)

                _add_normed_positions(tr_vars_df)

                logger.info(f"Saving at {save_dir}")
                tr_vars_df.to_pickle(os.path.join(save_dir, file_name))
                tr_row["file"] = file_name
                current_index = current_index.append(tr_row[["file"]])

                logger.info(f"Updating index {index_path}")
                current_index.to_pickle(index_path)
            else:
                logger.info(f"Video not tracked or not valid for analysis")
        else:
            logger.info(f"Vars dataframe already generated")

    return current_index


def load_dataset(
    dataset_path, video_filters=None, videos_table=None, animals_table=None
):

    if videos_table is None:
        videos_table = pd.read_csv(conf.VIDEOS_INDEX_FILE_NAME)

    if animals_table is None:
        animals_table = pd.read_csv(conf.ANIMALS_INDEX_FILE_PATH)

    if videos_table is not None and video_filters is not None:
        valid_trial_uids = data_filter(videos_table, video_filters).trial_uid
        filter_common = lambda x: x["trial_uid"].isin(valid_trial_uids)
    else:
        filter_common = None

    data = read_all(dataset_path, filter_common)
    if not data.empty:
        data = add_line_and_genotype_info(
            data.reset_index(), videos_table, animals_table
        )
        logger.info("Adding info columns")
        if "indiv" in dataset_path:
            data["genotype_group_genotype"] = (
                data["genotype_group"] + "-" + data["genotype"]
            )
            data["trial_uid_id"] = (
                data["trial_uid"] + "_" + data["identity"].astype(str)
            )
        if "indiv_nb" in dataset_path:
            data["genotype_group_genotype_nb"] = (
                data["genotype_group"] + "-" + data["genotype_nb"]
            )
            data["trial_uid_id_nb"] = (
                data["trial_uid"] + "_" + data["identity_nb"].astype(str)
            )
            data["focal_nb_genotype"] = (
                data["genotype"] + "-" + data["genotype_nb"]
            )
        logger.info("Added")
    return data

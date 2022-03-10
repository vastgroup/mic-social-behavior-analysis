import copy
import logging
from glob import glob

import numpy as np
import pandas as pd
import scipy.stats as ss
import trajectorytools as tt

logger = logging.getLogger(__name__)


def read_csv(csv_path: str):
    logger.info(f"Reading {csv_path}")
    return pd.read_csv(csv_path, delimiter=";")


def get_files_with_pattern(pattern: str):
    logger.info(f"Getting files with pattern {pattern}")
    return glob(pattern)


def _speed(trajectories):
    vel = np.diff(trajectories, axis=0)
    speed = np.sqrt(vel[..., 0] ** 2 + vel[..., 1] ** 2)
    return speed


def clean_impossible_speed_jumps(tr_dict, tracking_interval, num_vels=2):
    trajectories = tr_dict["trajectories"][
        tracking_interval[0] : tracking_interval[1]
    ]
    # Try fixing the speed jumps
    trajectories_interpolated1 = copy.deepcopy(trajectories)
    tt.interpolate_nans(trajectories_interpolated1)
    speed = _speed(trajectories_interpolated1)
    typical_max_speed = np.nanpercentile(speed, 99)
    bad_speeds = np.where(speed > num_vels * typical_max_speed)
    trajectories[bad_speeds[0], bad_speeds[1]] = np.nan
    trajectories[bad_speeds[0] - 1, bad_speeds[1]] = np.nan
    trajectories[bad_speeds[0] + 1, bad_speeds[1]] = np.nan
    trajectories[bad_speeds[0] - 2, bad_speeds[1]] = np.nan
    trajectories[bad_speeds[0] + 2, bad_speeds[1]] = np.nan

    # Check speed jumps are solved
    trajectories_interpolated2 = copy.deepcopy(trajectories)
    tt.interpolate_nans(trajectories_interpolated2)
    speed_clean = _speed(trajectories_interpolated2)
    unsolvable_impossible_speed_jumps = len(
        np.where(speed_clean > num_vels * typical_max_speed)[0]
    )
    tr_dict["trajectories"] = trajectories
    return tr_dict, unsolvable_impossible_speed_jumps


def data_filter(data, filters):
    logger.info("Filtering data")
    logger.info(f"original shape {data.shape}")
    for filter in filters:
        data = data[filter(data)]
        logger.info(data.shape)
    logger.info("Filtered")
    return data


# def _select_partition_data(data, partition_col, partition_uid):
#     return data_filter(data, [lambda x: x[partition_col] == partition_uid])


# def _select_partition_from_datasets(
#     datasets, datasets_names, partition_col, partition_uid
# ):
#     datasets_partition = {}
#     for dataset_name in datasets_names:
#         datasets_partition[dataset_name] = _select_partition_data(
#             datasets[dataset_name], partition_col, partition_uid
#         )
#     return datasets_partition


def circmean(x):
    return ss.circmean(x, high=np.pi, low=-np.pi)


def circstd(x):
    return ss.circstd(x, high=np.pi, low=-np.pi)


def ratio_in_front(x, angle=90):
    # TODO: Add filter for speed and interindividual distance
    angle_rad = angle / 180 * np.pi
    front_angle = angle_rad
    front = (x.abs() <= front_angle).sum()
    return front / len(x)


def ratio_in_back(x, angle=90):
    angle_rad = angle / 180 * np.pi
    back_angle = np.pi - angle_rad
    back = (x.abs() >= back_angle).sum()
    return back / len(x)

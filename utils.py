import copy
import logging
from glob import glob

import numpy as np
import pandas as pd
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
    tr_dict['trajectories'] = trajectories
    return tr_dict, unsolvable_impossible_speed_jumps


def data_filter(data, filters):
    for filter in filters:
        data = data[filter(data)]
    return data
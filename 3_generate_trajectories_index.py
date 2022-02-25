import copy
import json
import os
from ast import arg
from glob import glob

import matplotlib.pyplot as plt
import miniball
import numpy as np
import pandas as pd
import seaborn as sns
import trajectorytools as tt

from mic_analysis.constants import (
    NUM_FRAMES_FOR_ANALYSIS,
    TRACKING_DATA_FOLDER_PATH,
    TRAJECTORIES_INDEX_FILE_NAME,
)
from mic_analysis.utils import clean_impossible_speed_jumps


def info_from_video_path(
    video_path: str,
):  # only file name without directory and with extension
    video_file_name = os.path.split(video_path)[1]
    video_file_name = os.path.splitext(video_file_name)[0]

    # All video file names finish with "comp" of compression
    # we remove that
    info_string = video_file_name[:-4]
    info = info_string.split("_")
    if len(info) == 6:
        # the standard pattern
        trial, date, gene, founder, age, hour = info
    elif len(info) == 5:
        # missing founder, or only one
        trial, date, gene, age, hour = info
        founder = None
    elif len(info) == 7:
        # srmm3_srmm4
        trial, date, gene1, gene2, founder, age, hour = info
        gene = "-".join([gene1, gene2])
    elif len(info) == 9:
        (
            trial,
            genotype,
            group_size,
            date,
            gene,
            founder,
            fish_type,
            age,
            hour,
        ) = info
    else:
        print(f"video_file_name {video_file_name} does not match pattern")
        trial, date, gene, founder, age, hour = (
            "trial-1",
            "00000000",
            "xxxx",
            "-1",
            "-1",
            "-1",
        )

    return {
        "video_path": video_path,
        "trial": int(trial[5:]),
        "date": date,
        "gene": gene,
        "founder": founder,
        "age": age,
        "hour": hour,
    }


def get_info_id_last_animal(
    trajectories, starting_frame=NUM_FRAMES_FOR_ANALYSIS
):
    num_frames_to_detect_last_id = trajectories.shape[0] - starting_frame
    visible_fish = ~np.isnan(trajectories[starting_frame:, :, 0])
    num_frames_with_visible_fish = np.sum(np.any(visible_fish, axis=1))
    num_frames_with_both_fish_visible = np.sum(np.all(visible_fish, axis=1))
    num_frames_with_visible_fish_alone = np.sum(
        np.logical_xor(visible_fish[..., 0], visible_fish[..., 1])
    )
    info_id_last_animal = {
        "num_frames_to_detect_last_id": num_frames_to_detect_last_id,
        "num_frames_with_visible_fish": num_frames_with_visible_fish,
        "num_frames_with_both_fish_visible": num_frames_with_both_fish_visible,
        "num_frames_with_visible_fish_alone": num_frames_with_visible_fish_alone,
    }
    num_frames_ids_visible = []
    num_frames_ids_visible_alone = []
    ratio_frames_ids_visible_alone = []
    for id_ in range(trajectories.shape[1]):
        num_frames_id_visible = np.sum(
            ~np.isnan(trajectories[starting_frame:, id_, 0])
        )
        num_frames_ids_visible.append(num_frames_id_visible)
        num_frames_id_visible_alone = (
            num_frames_id_visible - num_frames_with_both_fish_visible
        )
        num_frames_ids_visible_alone.append(num_frames_id_visible_alone)
        ratio_frames_id_visible_alone = (
            num_frames_id_visible_alone / num_frames_with_visible_fish_alone
        )
        ratio_frames_ids_visible_alone.append(ratio_frames_id_visible_alone)
    info_id_last_animal.update(
        {
            "num_frames_ids_visible": tuple(num_frames_ids_visible),
            "num_frames_ids_visible_alone": tuple(
                num_frames_ids_visible_alone
            ),
            "ratio_frames_ids_visible_alone": tuple(
                ratio_frames_ids_visible_alone
            ),
        }
    )
    if np.all(~np.isnan(ratio_frames_ids_visible_alone)):
        assert np.sum(ratio_frames_ids_visible_alone) == 1
        index_last_id = np.argmax(ratio_frames_ids_visible_alone)
        info_id_last_animal["automatic_id_last_fish"] = index_last_id + 1
        info_id_last_animal[
            "certainty_id_last_fish"
        ] = ratio_frames_ids_visible_alone[index_last_id]
    else:
        info_id_last_animal["automatic_id_last_fish"] = 0

    return info_id_last_animal


def get_info_tracked_fish(trajectories, last_frame=NUM_FRAMES_FOR_ANALYSIS):
    visible_fish = ~np.isnan(trajectories[:last_frame, :, 0])
    num_frames_visible_fish = np.sum(visible_fish)
    info_tracked_fish = {
        "ratio_frames_tracked": num_frames_visible_fish
        / (last_frame * trajectories.shape[1])
    }
    return info_tracked_fish


def info_from_trajectories(trajectories):
    trajectories_info = {}
    trajectories_info["number_of_frames"] = trajectories.shape[0]
    trajectories_info["number_of_animals"] = trajectories.shape[1]
    trajectories_info.update(get_info_tracked_fish(trajectories))
    if trajectories_info["number_of_animals"] == 2:
        trajectories_info.update(get_info_id_last_animal(trajectories))
    return trajectories_info


def info_from_id_probabilities(id_probabilities):
    probabilities_info = {
        "min_id_probabilities": np.nanmin(id_probabilities),
        "mean_id_probabilities": np.nanmean(id_probabilities),
        "mean_id_probabilities_per_animal": tuple(
            np.nanmean(id_probabilities, axis=0)[:, 0]
        ),
    }
    return probabilities_info


def get_info_from_trajectory_dict(trajectory_dict):

    trajectory_dict_info = {
        "frames_per_second": trajectory_dict["frames_per_second"],
        "body_length": trajectory_dict["body_length"],
    }
    trajectory_dict_info.update(
        info_from_video_path(trajectory_dict["video_path"])
    )
    trajectory_dict_info.update(
        info_from_trajectories(trajectory_dict["trajectories"])
    )
    trajectory_dict_info.update(
        info_from_id_probabilities(trajectory_dict["id_probabilities"])
    )
    return trajectory_dict_info


def get_roi_center_radius(roi):
    mb = miniball.Miniball(roi)
    center_x, center_y = mb.center()
    radius = np.sqrt(mb.squared_radius())
    return (center_x, center_y), radius


def get_info_from_params_json(params_json_path):
    with open(params_json_path) as json_file:
        data = json.load(json_file)
    json_dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            if key == "_roi":
                roi = [tuple(p) for p in eval(value["value"][0][0])]
                json_dict[f"params{key}_value"] = roi
            else:
                for key2, value2 in value.items():
                    json_dict[f"params{key}_{key2}"] = value2
        else:
            json_dict[f"params_{key}"] = value
    return json_dict


def get_unsolvable_impossible_speed_jumps(tr_dict, tracking_interval):
    _, num_unsolvable_impossible_jumps = clean_impossible_speed_jumps(
        tr_dict, tracking_interval, num_vels=2
    )
    return {"num_unsolvable_impossible_jumps": num_unsolvable_impossible_jumps}


def get_info_from_trajectory_object(tr):

    trajectory_object_info = {
        "mean_speed": np.nanmean(tr.speed),
        "max_speed": np.nanmax(tr.speed),
        "std_speed": np.nanstd(tr.speed),
        "typical_speed": np.nanpercentile(tr.speed, 99),
    }
    threshold_speed = np.nanpercentile(tr.speed, 99) * 2
    frames_bad_speed = np.where(tr.speed > threshold_speed)
    trajectory_object_info["speed_jumps"] = frames_bad_speed
    num_impossible_speed_jumps = len(frames_bad_speed[0])
    ratio_bad_speeds = num_impossible_speed_jumps / (
        tr.number_of_frames * tr.number_of_individuals
    )
    trajectory_object_info[f"ratio_impossible_speed_jumps"] = ratio_bad_speeds
    trajectory_object_info[
        f"num_impossible_speed_jumps"
    ] = num_impossible_speed_jumps
    return trajectory_object_info


if __name__ == "__main__":
    from mic_analysis.logger import setup_logs

    logger = setup_logs("trajectories_index")

    trajectories_info = []
    for root, dirs, files in os.walk(TRACKING_DATA_FOLDER_PATH):
        if "trajectories_wo_gaps.npy" in files:
            trajectory_path = os.path.join(root, "trajectories_wo_gaps.npy")
            logger.info(f"Getting info for trajectories in {root}")
            trajectory_path_relative = os.path.relpath(
                trajectory_path, TRACKING_DATA_FOLDER_PATH
            )
            trajectory_info = {
                "trajectory_path": trajectory_path_relative,
                "folder_name_track": os.path.split(
                    os.path.split(trajectory_path_relative)[0]
                )[0],
            }
            trajectory_dict = np.load(
                trajectory_path, allow_pickle=True
            ).item()
            trajectory_info.update(
                get_info_from_trajectory_dict(trajectory_dict)
            )
            trajectory_info.update(
                get_unsolvable_impossible_speed_jumps(
                    copy.deepcopy(trajectory_dict),
                    [0, NUM_FRAMES_FOR_ANALYSIS],
                )
            )
            trajectory_info["trial_uid"] = (
                trajectory_info["folder_name_track"]
                + "_"
                + str(trajectory_info["trial"])
            )

            logger.info("Getting trajectory with trajectorytools")
            tr = tt.trajectories.FishTrajectories.from_idtrackerai(
                trajectory_path, interpolate_nans=False
            )
            tr = tr[:NUM_FRAMES_FOR_ANALYSIS]
            trajectory_object_info = get_info_from_trajectory_object(tr)

            trajectory_info.update(trajectory_object_info)

            if "params.json" in files:
                params_path = os.path.join(root, "params.json")
                if os.path.exists(params_path):
                    trajectory_info.update(
                        {
                            "params_path": os.path.relpath(
                                params_path, TRACKING_DATA_FOLDER_PATH
                            )
                        }
                    )
                    params_info = get_info_from_params_json(params_path)
                    trajectory_info.update(params_info)

                    # Add roi radius and center
                    if trajectory_info["params_roi_value"]:
                        center, radius = get_roi_center_radius(
                            trajectory_info["params_roi_value"]
                        )
                        trajectory_info["roi_center"] = center
                        trajectory_info["roi_radius"] = radius
                        trajectory_dict["setup_points"] = {
                            "border": np.asarray(
                                trajectory_info["params_roi_value"]
                            )
                        }

                    else:
                        trajectory_info["roi_center"] = None
                        trajectory_info["roi_radius"] = None
                        trajectory_info.update(params_info)
                        trajectory_dict["setup_points"] = {}
                    np.save(trajectory_path, trajectory_dict)

                    # Copy ROI as setup points border into trajectories
                    # This is usfeul to center the trajectories afterwards
                    # using the ROI. This can be important if some fish
                    # never make it into the border of the arena
                    # This also allows to center automatically the trajectories
                    # to the ROI (setup_points['border]) in trajectorytools
                    # instead of using miniball from the locations array
                if "video_object.npy" in files:
                    video_object_path = os.path.join(root, "video_object.npy")
                    video_object = np.load(
                        video_object_path, allow_pickle=True
                    ).item()
                    trajectory_info[
                        "estimated_accuracy"
                    ] = video_object.overall_P2

            trajectories_info.append(trajectory_info)
    trajectories_table = pd.DataFrame(trajectories_info)
    trajectories_table.to_csv(TRAJECTORIES_INDEX_FILE_NAME, index=False)

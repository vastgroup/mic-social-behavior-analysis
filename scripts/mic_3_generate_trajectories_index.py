import copy
import os

import numpy as np
import pandas as pd
import trajectorytools as tt
from confapp import conf
from mic_analysis.logger import setup_logs
from mic_analysis.table_generators_utils import (
    get_info_from_params_json,
    get_info_from_trajectory_dict,
    get_info_from_trajectory_object,
    get_roi_center_radius,
    get_unsolvable_impossible_speed_jumps,
)

logger = setup_logs("trajectories_index")

trajectories_info = []
for root, dirs, files in os.walk(conf.TRACKING_DATA_FOLDER_PATH):

    if "trajectories_wo_gaps.npy" in files:
        trajectory_path = os.path.join(root, "trajectories_wo_gaps.npy")
        logger.info(f"Getting info for trajectories in {root}")
        trajectory_path_relative = os.path.relpath(
            trajectory_path, conf.TRACKING_DATA_FOLDER_PATH
        )
        (
            gene,
            founder,
            replicate,
            experiment_type,
        ) = trajectory_path_relative.split("/")[0].split("_")
        trajectory_info = {
            "trajectory_path": trajectory_path_relative,
            "folder_name_track": os.path.split(
                os.path.split(trajectory_path_relative)[0]
            )[0],
            "gene": gene,
            "founder": founder,
            "replicate": replicate,
            "experiment_type": experiment_type,
        }

        trajectory_dict = np.load(trajectory_path, allow_pickle=True).item()
        trajectory_info.update(get_info_from_trajectory_dict(trajectory_dict))
        trajectory_info.update(
            get_unsolvable_impossible_speed_jumps(
                copy.deepcopy(trajectory_dict),
                [0, conf.NUM_FRAMES_FOR_ANALYSIS],
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
        tr = tr[: conf.NUM_FRAMES_FOR_ANALYSIS]
        trajectory_object_info = get_info_from_trajectory_object(tr)

        trajectory_info.update(trajectory_object_info)

        if "params.json" in files:
            params_path = os.path.join(root, "params.json")
            if os.path.exists(params_path):
                trajectory_info.update(
                    {
                        "params_path": os.path.relpath(
                            params_path, conf.TRACKING_DATA_FOLDER_PATH
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
                trajectory_info["estimated_accuracy"] = video_object.overall_P2
        trajectories_info.append(trajectory_info)
trajectories_table = pd.DataFrame(trajectories_info)
trajectories_table.to_csv(conf.TRAJECTORIES_INDEX_FILE_NAME, index=False)

# TODO: warning if not trajectories found
# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)

import os

import cv2
import numpy as np
from idtrackerai.utils.py_utils import get_spaced_colors_util
from tqdm import tqdm


def writeIds(
    frame,
    frame_number,
    trajectories,
    centroid_trace_length,
    colors,
    resolution_reduction,
):
    ordered_centroid = trajectories[frame_number]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    font_width = 3
    font_width = 1 if font_width == 0 else font_width
    circle_size = 2

    for cur_id, centroid in enumerate(ordered_centroid):
        if sum(np.isnan(centroid)) == 0:
            if frame_number > centroid_trace_length:
                centroids_trace = trajectories[
                    frame_number - centroid_trace_length : frame_number, cur_id
                ]
            else:
                centroids_trace = trajectories[:frame_number, cur_id]
            cur_id_str = str(cur_id + 1)
            int_centroid = np.asarray(centroid).astype("int")
            cv2.circle(
                frame, tuple(int_centroid), circle_size, colors[cur_id], -1
            )
            cv2.putText(
                frame,
                cur_id_str,
                tuple(int_centroid),
                font,
                font_size,
                colors[cur_id],
                font_width,
            )
            for centroid_trace in centroids_trace:
                if sum(np.isnan(centroid_trace)) == 0:
                    int_centroid = np.asarray(centroid_trace).astype("int")
                    cv2.circle(
                        frame,
                        tuple(int_centroid),
                        circle_size,
                        colors[cur_id],
                        -1,
                    )
    return frame


def apply_func_on_frame(
    trajectories,
    frame_number,
    colors,
    cap,
    func=None,
    centroid_trace_length=10,
    resolution_reduction=1.0,
):

    cap.set(1, frame_number)
    ret, frame = cap.read()
    if ret:
        frame = func(
            frame,
            frame_number,
            trajectories,
            centroid_trace_length,
            colors,
            resolution_reduction,
        )
        if resolution_reduction != 1:
            frame = cv2.resize(
                frame,
                None,
                fx=resolution_reduction,
                fy=resolution_reduction,
                interpolation=cv2.INTER_AREA,
            )
        return frame


def generate_trajectories_video(
    video_path,
    trajectories_path,
    func=writeIds,
    centroid_trace_length=10,
    resolution_reduction=1.0,
    save_path=".",
):
    trajectories_dict = np.load(
        trajectories_path, allow_pickle=True, encoding="latin1"
    ).item()
    trajectories = trajectories_dict["trajectories"]
    number_of_animals = trajectories.shape[1]
    video_name = os.path.split(video_path)[-1].split(".")[0] + "_tracked.avi"
    colors = get_spaced_colors_util(number_of_animals, black=False)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resolution_reduction)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resolution_reduction)

    path_to_save_video = os.path.join(save_path, video_name)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(
        path_to_save_video,
        fourcc,
        trajectories_dict["frames_per_second"],
        (width, height),
    )

    for frame_number in tqdm(
        range(0, len(trajectories)),
        desc="Generating video with trajectories...",
    ):
        frame = apply_func_on_frame(
            trajectories,
            frame_number,
            colors,
            cap,
            func=writeIds,
            centroid_trace_length=centroid_trace_length,
            resolution_reduction=resolution_reduction,
        )
        video_writer.write(frame)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scans a given directory (source_dir) looking for folders "
        "with a 'params.json' file and a finished idtracker.ai session folder. "
        "It copies the 'params.json' file, the 'trajectories_wo_gaps.npy "
        "and the 'video_object.npy' file into another directory "
        "(destination_dir) with the same folder structure."
    )
    parser.add_argument(
        "-src",
        "--source_dir",
        help="Path to the folder to be scanned looking for params.json files "
        "and finished idtracker.ai sessions with a video_object.npy "
        "and trajectories_wo_gaps.npy.",
    )
    parser.add_argument(
        "-dst",
        "--destination_dir",
        help="Path to the folder "
        "where the files will be copied into. The directory "
        "structure will be the same as the source_dir.",
    )
    parser.add_argument(
        "-ns",
        "--number_of_ghost_points",
        type=int,
        default=20,
        help="Number of points used to draw the individual trajectories' traces",
    )
    parser.add_argument(
        "-rr",
        "--resolution_reduction",
        type=float,
        default=1.0,
        help="Ratio for reducing the resolution of the video. "
        "Same ratio is applied to width and height",
    )
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.source_dir):
        video_files = [file for file in files if file.endswith(".avi")]
        if video_files:
            assert len(video_files) == 1, video_files
            video_path = os.path.join(root, video_files[0])

            rel_root_path = os.path.relpath(root, args.source_dir)
            tracked_video_folder_path = os.path.join(
                args.destination_dir, rel_root_path
            )
            if not os.path.isdir(tracked_video_folder_path):
                os.makedirs(tracked_video_folder_path)

            # Check that there is session folder
            session_folders = [dir_ for dir_ in dirs if "session_" in dir_]
            if session_folders:
                assert len(session_folders) == 1
                session_path = os.path.join(root, session_folders[0])

                # Check that it has trajectories_wo_gaps.npy
                trajectories_path = os.path.join(
                    session_path,
                    "trajectories_wo_gaps",
                    "trajectories_wo_gaps.npy",
                )
                if os.path.isfile(trajectories_path):
                    print(f"Generating tracked video for {video_path}")
                    print(f"With trajectories {trajectories_path}")
                    print(f"Saving it at {tracked_video_folder_path}")
                    generate_trajectories_video(
                        video_path,
                        trajectories_path,
                        centroid_trace_length=args.number_of_ghost_points,
                        resolution_reduction=args.resolution_reduction,
                        save_path=tracked_video_folder_path,
                    )

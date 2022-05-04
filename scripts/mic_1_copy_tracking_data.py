import argparse
import os
import shutil
from pprint import pprint

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
args = parser.parse_args()

# TODO: Check condition if only video.npy exists and not trajectories_wo_gaps.npy. Otherwise Overall_P2 error is raised

videos = []
for root, dirs, files in os.walk(args.source_dir):
    params_files = [
        file for file in files if file in ["test.json", "params.json"]
    ]
    if params_files:
        print(f"*** {root}")
        assert len(params_files) == 1
        # is a video folder set for tracking
        video = {}
        video["rel_root"] = os.path.relpath(root, args.source_dir)
        # Create destination folder
        dst_subfolder = os.path.join(args.destination_dir, video["rel_root"])
        print(dst_subfolder)
        if not os.path.isdir(dst_subfolder):
            os.makedirs(dst_subfolder)
        # Copy params file
        params_src_path = os.path.join(root, params_files[0])
        params_dst_path = os.path.join(dst_subfolder, params_files[0])
        shutil.copyfile(params_src_path, params_dst_path)

        # Check that there is session folder
        session_folders = [dir_ for dir_ in dirs if "session_" in dir_]
        if session_folders:
            assert len(session_folders) == 1

            # Check that it has video_object.npy and trajectories
            video_object_src_path = os.path.join(
                root, session_folders[0], "video_object.npy"
            )
            if os.path.isfile(video_object_src_path):
                video_object_dst_path = os.path.join(
                    dst_subfolder, "video_object.npy"
                )
                shutil.copyfile(video_object_src_path, video_object_dst_path)
            else:
                print(video_object_src_path, "does not exist")

            # Check that it has trajectories_wo_gaps.npy
            trajectories_wo_gaps_src_path = os.path.join(
                root,
                session_folders[0],
                "trajectories_wo_gaps",
                "trajectories_wo_gaps.npy",
            )
            if os.path.isfile(trajectories_wo_gaps_src_path):
                trajectories_wo_gaps_dst_path = os.path.join(
                    dst_subfolder,
                    "trajectories_wo_gaps.npy",
                )
                shutil.copyfile(
                    trajectories_wo_gaps_src_path,
                    trajectories_wo_gaps_dst_path,
                )
            else:
                print(
                    trajectories_wo_gaps_src_path,
                    "does not exist",
                )

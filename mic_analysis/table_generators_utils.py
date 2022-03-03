import json
import logging
import os

import miniball
import numpy as np
import pandas as pd
from confapp import conf

from mic_analysis.utils import clean_impossible_speed_jumps

logger = logging.getLogger(__name__)


# ANIMALS TABLES
def drop_unnecessary_rows_columns(df: pd.DataFrame):
    # some tables have empty rows
    logger.info(f"Dropping unnecessary rows and columns")
    df_wo_na_rows = df.dropna(how="all")
    num_empty_rows = len(df) - len(df_wo_na_rows)
    logger.warning(f"Dropped {num_empty_rows} empty rows")
    df = df_wo_na_rows

    # Expected columns
    expected_columns = set(
        conf.EXPECTED_COLUMNS_IN_SINGLE_ANIMALS_TABLE.keys()
    )

    # Checking expected columns are in table
    missing_columns = expected_columns - set(df.columns)
    extra_columns = set(df.columns) - expected_columns
    if missing_columns or extra_columns:
        logger.info(f"Columns are: {df.columns}")
        logger.info(f"Expected columsn are: {expected_columns}")
    if missing_columns:
        logger.warning(f"Missing columns {missing_columns}")
    if extra_columns:
        df.drop(extra_columns, axis=1, inplace=True)
        logger.warning(f"Dropped extra columns {extra_columns}")

    report = {
        "num_empty_rows": num_empty_rows,
        "num_missing_columns": len(missing_columns),
        "num_extra_columns": len(extra_columns),
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
    }

    return df, report


def correct_data_types(df: pd.DataFrame):
    # Define column types
    logger.info(f"Correcting data types of columns")
    columns_with_correct_data_types = []
    columns_with_incorrect_data_types = []
    for column, dtype in conf.EXPECTED_COLUMNS_IN_SINGLE_ANIMALS_TABLE.items():
        logger.info(f"For column: {column}")
        logger.info(f"the expected data type is {dtype}")
        logger.info(f"the actual data type is {df[column].dtype}")
        if not df[column].dtype == dtype:
            columns_with_incorrect_data_types.append(
                (column, df[column].dtype, dtype)
            )
            logger.warning(f"Data type does not match")
            logger.info("Trying to correct data type")
            if column in ["size_cm", "accuracy", "px_cm"]:
                df[column] = df[column].apply(
                    lambda x: float(x.replace(",", "."))
                    if isinstance(x, str) and x[-1].isnumeric()
                    else np.nan
                )
                logger.info(
                    "Corrected by replacing . with , in numeric "
                    "strings and casting to float. None numeric "
                    "strings or missing values are casted as with NaN"
                )
            elif column == "fish_ID_track":
                df[column] = df[column].apply(
                    lambda x: float(x)
                    if (isinstance(x, str) and x[-1].isnumeric())
                    or isinstance(x, (int, float, complex))
                    else np.nan
                )
                logger.info(
                    "Corrected by casting to float numeric strings"
                    "and casting with NaN non numeric or empty values"
                )
            else:
                try:
                    df[column] = df[column].astype(dtype)
                except:
                    logger.error("Cannot")
                    raise

                logger.info("Corrected by assiging default data type")
        else:
            columns_with_correct_data_types.append(
                (column, df[column].dtype, dtype)
            )
    report = {
        "num_correct_data_type": len(columns_with_correct_data_types),
        "num_incorrect_data_type": len(columns_with_incorrect_data_types),
        "correct_data_type": columns_with_correct_data_types,
        "incorrect_data_type": columns_with_incorrect_data_types,
    }
    return df, report


def _add_genotype_and_groupsize(df):
    animals_same_trajectory = df.groupby(["folder_name_track", "trial"])
    all_animals = []
    for idx, animals in animals_same_trajectory:
        if not animals.genotype.isna().sum():
            genotype_group = "_".join(animals.genotype)
        else:
            genotype_group = "-_-"
        animals["group_size"] = [len(animals)] * len(animals)
        animals["genotype_group"] = [genotype_group] * len(animals)
        all_animals.append(animals)

        if len(animals) < 2:
            print(animals[["folder_name_track", "trial"]])
            raise Exception(
                "A group with less than 2 animals, "
                "probabliy a trial number badly set"
            )

    df = pd.concat(all_animals)
    df["genotype_group"] = [
        *map(
            lambda x: x
            if x not in conf.GENOTYPE_CONVERSION_DICT.keys()
            else conf.GENOTYPE_CONVERSION_DICT[x],
            df["genotype_group"],
        )
    ]
    df["genotype_group"] = df["genotype_group"].fillna("-_-")
    return df


def _add_trial_uid_and_folder_name_track(df, file_name):
    file_name_no_ext = os.path.splitext(file_name)[0]
    folder_name_track = file_name_no_ext[9:]
    df["folder_name_track"] = [folder_name_track] * len(df)
    df["trial_uid"] = (
        df["folder_name_track"] + "_" + df["trial"].astype(int).astype(str)
    )


def add_extra_columns(df, animals_table_path):
    file_name = os.path.split(animals_table_path)[1]
    df["file_name"] = [file_name] * len(df)
    _add_trial_uid_and_folder_name_track(df, file_name)
    df = _add_genotype_and_groupsize(df)
    return df


# TRAJECTORIES TABLES
def _info_from_video_path(
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


def _get_info_id_last_animal(
    trajectories, starting_frame=conf.NUM_FRAMES_FOR_ANALYSIS
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


def _get_info_tracked_fish(
    trajectories, last_frame=conf.NUM_FRAMES_FOR_ANALYSIS
):
    visible_fish = ~np.isnan(trajectories[:last_frame, :, 0])
    num_frames_visible_fish = np.sum(visible_fish)
    info_tracked_fish = {
        "ratio_frames_tracked": num_frames_visible_fish
        / (last_frame * trajectories.shape[1])
    }
    return info_tracked_fish


def _info_from_trajectories(trajectories):
    trajectories_info = {}
    trajectories_info["number_of_frames"] = trajectories.shape[0]
    trajectories_info["number_of_animals"] = trajectories.shape[1]
    trajectories_info.update(_get_info_tracked_fish(trajectories))
    if trajectories_info["number_of_animals"] == 2:
        trajectories_info.update(_get_info_id_last_animal(trajectories))
    return trajectories_info


def _info_from_id_probabilities(id_probabilities):
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
        _info_from_video_path(trajectory_dict["video_path"])
    )
    trajectory_dict_info.update(
        _info_from_trajectories(trajectory_dict["trajectories"])
    )
    trajectory_dict_info.update(
        _info_from_id_probabilities(trajectory_dict["id_probabilities"])
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


# VIDEOS TABLES
def _add_manually_labeled_id_last_fish(
    videos_table, animals_table, experiments_info_table
):
    # Laura manually labeled the id of the last fish by looking at the
    # trajectories. We recover this info to compare with the automatic label
    # obtained by looking at the number of NaN in the last chunck of the video.
    logger.info("Extracting manually labeled id of the last fish in the video")
    videos_table.set_index("trial_uid", inplace=True)
    videos = animals_table.groupby("trial_uid")
    experiments_with_same_animal_exp_num = []
    manual_id_last_animal = []
    for idx, video in videos:
        # The experimental animal 2 is the last animal to be extracted from the
        # arena i.e. the last animal in the video
        last_animal = video[video.fish_id_exp == 2]
        if len(last_animal) != 1:
            # There were some typing errors and in some experiments
            # the experimental id of the fish was the same for both
            # fish of the video
            experiments_with_same_animal_exp_num.append(
                video.trial_uid.values[0]
            )
        else:
            manual_id_last_animal.append(
                last_animal[["trial_uid", "fish_id_track"]]
            )
    manual_id_last_animal = pd.concat(manual_id_last_animal)
    # We rename the column to be more clear
    manual_id_last_animal.rename(
        {"fish_id_track": "manual_id_last_fish"}, axis=1, inplace=True
    )
    videos_table = pd.merge(
        videos_table, manual_id_last_animal, on="trial_uid"
    )
    videos_table["manual_id_last_fish"] = videos_table[
        "manual_id_last_fish"
    ].fillna(conf.NO_ID_LAST_FISH_FILL_VALUE)
    videos_table["automatic_id_last_fish"] = videos_table[
        "automatic_id_last_fish"
    ].fillna(conf.NO_ID_LAST_FISH_FILL_VALUE)

    videos_table["id_last_fish"] = videos_table["automatic_id_last_fish"]
    for idx, row in experiments_info_table.iterrows():
        if row.id_genotype_detection == "manual":
            videos_table.loc[
                videos_table.experiment_type == str(row.code), "id_last_fish"
            ] = videos_table.loc[
                videos_table.experiment_type == str(row.code),
                "manual_id_last_fish",
            ]
            videos_table.loc[
                videos_table.experiment_type == str(row.code),
                "certainty_id_last_fish",
            ] = 1.0

    videos_table["same_id_last_fish"] = (
        videos_table["automatic_id_last_fish"]
        == videos_table["manual_id_last_fish"]
    )
    return videos_table


def _tracking_state_code(
    row,
    keys=conf.TRACKING_STATE_COLUMNS,
):
    return " ".join(["1" if row[key] else "0" for key in keys])


def _id_last_fish_state_code(
    row,
    keys=conf.ID_LAST_FISH_STATE_COLUMNS,
):
    return " ".join(["1" if row[key] else "0" for key in keys])


def _add_video_quality_state_columns(videos_table):
    # Adding videos labels
    logger.info("Adding column `tracked`")
    videos_table["tracked"] = ~videos_table["trajectory_path"].isna()
    videos_table["valid_genotype_group"] = videos_table["genotype_group"].isin(
        conf.VALID_GENOTYPES
    )
    videos_table["valid_ratio_frames_tracked"] = (
        videos_table["ratio_frames_tracked"] > conf.THRESHOLD_RATIO_TRACKED
    )
    videos_table["valid_estimated_accuracy"] = (
        videos_table["estimated_accuracy"] > conf.THRESHOLD_ACCURACY
    )
    videos_table["valid_mean_id_probabilities"] = (
        videos_table["mean_id_probabilities"]
        > conf.THRESHOLD_MEAN_ID_PROBABILITIES
    )
    videos_table["valid_num_impossible_speed_jumps"] = (
        videos_table["num_impossible_speed_jumps"]
        < conf.THRESHOLD_NUM_IMPOSSIBLE_SPEED_JUMPS
    )
    videos_table["valid_num_unsolvable_impossible_speed_jumps"] = (
        videos_table["num_unsolvable_impossible_jumps"] == 0
    )
    videos_table["valid_id_last_fish"] = videos_table["id_last_fish"] > 0
    videos_table["valid_certainty_id_last_fish"] = (
        videos_table["certainty_id_last_fish"]
        > conf.THRESHOLD_CERTAINTY_ID_LAST_FISH
    )
    videos_table["tracking_state"] = videos_table.apply(
        _tracking_state_code, axis=1
    )
    videos_table["id_last_fish_state"] = videos_table.apply(
        _id_last_fish_state_code, axis=1
    )
    videos_table["for_analysis_state"] = videos_table[
        ["tracking_state", "id_last_fish_state"]
    ].agg("-".join, axis=1)
    videos_table["valid_tracking"] = (
        videos_table.valid_mean_id_probabilities
        & videos_table.valid_ratio_frames_tracked
        & videos_table.valid_num_unsolvable_impossible_speed_jumps
    )
    videos_table["valid_genotype_id"] = (
        videos_table.valid_id_last_fish
        & videos_table.valid_certainty_id_last_fish
    )
    videos_table["valid_for_analysis"] = (
        videos_table.tracked
        & videos_table.valid_genotype_group
        & videos_table.valid_tracking
        & videos_table.valid_genotype_id
    )
    return videos_table


def generate_videos_table(
    trajectories_table, animals_table, experiments_info_table
):
    # `animals_table` has information about each animal used in an experiment
    # regardless of whether the videos was tracked or not
    # `trajectories_table` has information about each trajectory of a video
    # tracked, but it has no information about the genotyping.
    # We create a table that has info about each experiment with genotyping
    # and trajectories data
    videos_table = pd.merge(
        animals_table[conf.PER_VIDEO_COLUMNS].drop_duplicates(),
        trajectories_table,
        left_on=["trial_uid", "folder_name_track", "trial", "gene"],
        right_on=["trial_uid", "folder_name_track", "trial", "gene"],
        how="left",
    )
    print(videos_table.columns)
    videos_table.drop(
        ["founder_x", "founder_y", "replicate"], axis=1, inplace=True
    )
    print(videos_table.columns)
    videos_table["gene"] = videos_table.trial_uid.apply(
        lambda x: x.split("_")[0]
    )
    videos_table["founder"] = videos_table.trial_uid.apply(
        lambda x: x.split("_")[1]
    )
    videos_table["line"] = videos_table.gene + "_" + videos_table.founder
    videos_table["replicate"] = videos_table.trial_uid.apply(
        lambda x: x.split("_")[2]
    )
    videos_table["line_replicate"] = (
        videos_table.line + "_" + videos_table.replicate.astype(str)
    )
    videos_table["experiment_type"] = videos_table.trial_uid.apply(
        lambda x: x.split("_")[3]
    )
    videos_table["line_experiment"] = (
        videos_table["line"] + "_" + videos_table["experiment_type"]
    )
    videos_table["line_replicate_experiment"] = (
        videos_table["line_replicate"] + "_" + videos_table["experiment_type"]
    )
    videos_table = _add_manually_labeled_id_last_fish(
        videos_table, animals_table, experiments_info_table
    )
    videos_table = _add_video_quality_state_columns(videos_table)
    return videos_table


def get_tracking_state_table(videos_table):

    videos_tracking_state = videos_table[
        ["trial_uid", "genotype_group"]
        + [
            "ratio_frames_tracked",
            "certainty_id_last_fish",
            "accuracy",
            "mean_id_probabilities",
            "ratio_impossible_speed_jumps",
            "num_impossible_speed_jumps",
            "num_unsolvable_impossible_jumps",
            "automatic_id_last_fish",
            "manual_id_last_fish",
        ]
        + conf.TRACKING_STATE_COLUMNS
        + ["valid_tracking", "valid_for_analysis"]
    ]
    return videos_tracking_state


def print_summary_tracking_state(videos_table):

    for column in (
        conf.TRACKING_STATE_COLUMNS
        + conf.ID_LAST_FISH_STATE_COLUMNS
        + ["valid_tracking", "valid_for_analysis"]
    ):

        if column != "tracked":
            logger.info(f"\n*** Videos with {column}")
            logger.info(
                (videos_table[videos_table.tracked][column])
                .value_counts()
                .to_string()
            )
        else:
            logger.info(f"\n*** Videos {column}")
            logger.info((videos_table[column]).value_counts().to_string())


def generate_videos_valid_for_analysis_table(
    videos_table,
    main_columns=[
        "folder_name_track",
        "genotype_group",
        "valid_for_analysis",
    ],
):
    logger.info("Generating videos valid for analysis table")
    columns_to_count = (
        main_columns
        + conf.TRACKING_STATE_COLUMNS
        + conf.ID_LAST_FISH_STATE_COLUMNS
    )
    videos_valid_for_analysis = (
        videos_table[columns_to_count]
        .value_counts()
        .to_frame()
        .reset_index()
        .sort_values(columns_to_count)
        .rename({0: "num_videos"}, axis=1)
        .set_index(columns_to_count)
        .unstack("valid_for_analysis")
        .fillna(0)
    )
    return videos_valid_for_analysis


def _add_identity_info(
    tr_vars_df, videos_table, animals_table, identity_column_name="identity"
):
    # Add `identity` column with numbers 0 and 1 to
    videos_table[identity_column_name] = videos_table.id_last_fish - 1
    # Add `fish_id_exp` with number 2
    # Each row in the video_table has information about the identity of the
    # last fish. Such fish is the fish_id_exp 2
    if "nb" in identity_column_name:
        suffix = "_nb"
    else:
        suffix = ""
    videos_table[f"fish_id_exp{suffix}"] = 2
    # We merge tr_indivs with video_table so that tr_indivs has columns
    # fish_id_exp that we will use to get the genotype.
    # Note that since we are merging using the identity the fish_id_exp
    # will be allocated to the correct fish, which later will allow us
    # to get the fish genotype
    tr_vars_df = pd.merge(
        tr_vars_df,
        videos_table[
            ["trial_uid", identity_column_name, f"fish_id_exp{suffix}"]
        ],
        on=["trial_uid", identity_column_name],
        how="outer",
    )
    # All fish that are not the last fish (i.e. the second fish) will
    # have NaN in the fish_id_exp column. So we fill the NaN with 1
    # to indicate that they are the first fish
    tr_vars_df[f"fish_id_exp{suffix}"].fillna(1, inplace=True)

    # We merge the tr_indivs dataframe with the animals
    # table to get information about the genotype of each fish, the age and
    # the size
    # To do so we use the fish_id_exp column
    extra_columns = ["genotype", "dpf", "size_cm"]
    if "nb" in identity_column_name:
        for extra_column in extra_columns + ["fish_id_exp"]:
            animals_table[extra_column + suffix] = animals_table[extra_column]
        extra_columns = [
            extra_column + suffix for extra_column in extra_columns
        ]
    tr_vars_df = pd.merge(
        tr_vars_df,
        animals_table[["trial_uid", f"fish_id_exp{suffix}"] + extra_columns],
        on=["trial_uid", f"fish_id_exp{suffix}"],
        how="outer",
    )
    return tr_vars_df


def add_line_and_genotype_info(tr_vars_df, videos_table, animals_table):
    if "identity" in tr_vars_df.columns:
        tr_vars_df = _add_identity_info(
            tr_vars_df,
            videos_table,
            animals_table,
            identity_column_name="identity",
        )

    if "identity_nb" in tr_vars_df.columns:
        tr_vars_df = _add_identity_info(
            tr_vars_df,
            videos_table,
            animals_table,
            identity_column_name="identity_nb",
        )

    # We merge again tr_indivs with video_table to get extra features
    # of the videos that are important for the analysis
    tr_vars_df = pd.merge(
        tr_vars_df,
        videos_table[
            [
                "trial_uid",
                "line",
                "line_replicate",
                "line_experiment",
                "line_replicate_experiment",
                "gene",
                "founder",
                "replicate",
                "experiment_type",
                "genotype_group",
            ]
        ],
        on="trial_uid",
        how="outer",
    )

    # Finally we clean the rows that have no frames or identity
    # as they are rows of animals that have not been tracked

    if "identity_nb" in tr_vars_df.columns:
        columns_to_drop_nan = [
            "frame",
            "identity",
            "identity_nb",
            "nb_position_x",
        ]
    elif "identity" in tr_vars_df.columns:
        columns_to_drop_nan = ["frame", "identity"]
    else:
        columns_to_drop_nan = ["frame"]
    tr_vars_df.dropna(subset=columns_to_drop_nan, inplace=True, how="any")
    return tr_vars_df

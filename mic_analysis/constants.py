import os

import numpy as np
from mlxtend.evaluate import permutation_test

from mic_analysis.utils import circmean, circstd, ratio_in_front

NUM_JOBS_PARALLELIZATION = 6

# TRAJECTORIES CONSTANTS
SIGMA = 1
PX_CM = 54
FRAME_RATE = 29

# LOGGERS CONSTANTS
DEFAULT_LOG_FILENAME = "log"
DEFAULT_SCREEN_FORMATTER = "%(name)-12s: %(levelname)-8s %(message)s"
DEFAULT_FILE_FORMATTER = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"

# MAIN DIRECTORY WHERE DATA IS STORED
# Absolute path to the local folder /ZFISH_MICs/_BSocial/2022_ANALYSIS_social
# that can be sync/downlowaded from Dropbox
DATA_DIR = os.environ["DATA_DIR"]
# GENERATED TABLES AND FIGURES
GENERATED_TABLES_PATH = os.path.join(DATA_DIR, "generated_tables")
GENERATED_FIGURES_PATH = os.path.join(DATA_DIR, "generated_figures")
# Conversion table from old to new names
CONVERSIONS_TABLE_PATH = os.path.join(DATA_DIR, "Conversions.csv")
# Experiments info table
EXPERIMENTS_INFO_TABLE = os.path.join(DATA_DIR, "Data_structure.csv")

# Animals information
EXPERIMENTS_DATA_FOLDER_PATH = os.path.join(DATA_DIR, "Social_Experiments")
ANIMALS_INDEX_FILE_PATH = os.path.join(
    GENERATED_TABLES_PATH, "animals_index.csv"
)
ANIMALS_INDEX_REPORT_FILE_PATH = os.path.join(
    GENERATED_TABLES_PATH, "animals_index_processing_report.csv"
)
ANIMALS_COUNT_FILE_PATH = os.path.join(
    GENERATED_TABLES_PATH, "animals_counts.csv"
)

# Trajectories information
TRACKING_DATA_FOLDER_PATH = os.path.join(DATA_DIR, "Social_DATA")
TRAJECTORIES_INDEX_FILE_NAME = os.path.join(
    GENERATED_TABLES_PATH, "trajectories_index.csv"
)

# Variables
VARIABLES_RANGES_FILE_NAME = "variables_ranges.pkl"
PER_ANIMAL_STATS_FILE_NAME = "per_animal_stats.pkl"

# Experiments information
VIDEOS_INDEX_FILE_NAME = os.path.join(
    GENERATED_TABLES_PATH, "videos_index.csv"
)
VIDEOS_VALID_FOR_ANALYSIS_FILE_PATH = os.path.join(
    GENERATED_TABLES_PATH, "videos_valid_for_analysis.csv"
)
VIDEOS_TRACKING_STATE_FILE_NAME = os.path.join(
    GENERATED_TABLES_PATH, "videos_tracking_state.csv"
)

#
EXPECTED_COLUMNS_IN_SINGLE_ANIMALS_TABLE = {
    "Date": np.object,
    "Line": np.object,
    "dpf": np.float64,
    "trail": np.int64,  # renamed to "trial" later on
    "fish_num": np.int64,
    "size_cm": np.float64,
    "px_cm": np.float64,
    "fish_ID_exp": np.int64,
    "Genotype": np.object,
    "accuracy": np.float64,
    "fish_ID_track": np.float64,
    "Observations": np.object,
}

RENAMING_SINGLE_ANIMALS_TABLE = {
    "Date": "date",
    "Line": "line",
    "trail": "trial",
    "fish_ID_exp": "fish_id_exp",
    "fish_ID_track": "fish_id_track",
    "Observations": "observations",
    "Genotype": "genotype",
}

RENAMING_CONVERSION_TABLE = {
    "Gene": "gene",
    "Experiment_type": "experiment_type",
    "Old_name_experiment": "old_name_experiment",
    "SOCIAL_DATA_old": "social_data_old",
    "Founder": "founder",
    "Replicate": "replicate",
    "Control_Genotype": "control_genotype",
    "Control_Group": "control_group",
}

GENOTYPE_CONVERSION_DICT = {
    "DEL_HET": "HET_DEL",
    "HET_WT": "WT_HET",
    "DEL_WT": "WT_DEL",
    "HET_NAN": "HET_-",
    "-_WT": "WT_-",
    "-_HET": "HET_-",
    "-_DEL": "DEL_-",
    "WT_WT_WT_WT_WT": "5WT",
    "DEL_DEL_DEL_DEL_DEL": "5DEL",
}

VALID_GENOTYPES = [
    "HET_HET",
    "HET_DEL",
    "DEL_DEL",
    "WT_WT",
    "WT_DEL",
    "WT_HET",
    "5WT",
    "5DEL",
]

GENOTYPE_GROUP_ORDER = [
    "WT_WT",
    "WT_HET",
    "HET_HET",
    "HET_DEL",
    "DEL_DEL",
    "WT_DEL",
    "5WT",
    "5DEL",
]
FOCAL_NB_GENOTYPE_ORDER = [
    "WT-WT",
    "WT-HET",
    "HET-WT",
    "WT-DEL",
    "DEL-WT",
    "HET-HET",
    "HET-DEL",
    "DEL-HET",
    "DEL-DEL",
]
GENOTYPE_ORDER = ["WT", "HET", "DEL"]
GENOTYPE_GROUP_GENOTYPE_ORDER = [
    "WT_WT-WT",
    "WT_HET-WT",
    "WT_HET-HET",
    "HET_HET-HET",
    "HET_DEL-HET",
    "HET_DEL-DEL",
    "DEL_DEL-DEL",
    "WT_DEL-DEL",
    "WT_DEL-WT",
    "5WT-WT",
    "5DEL-DEL",
]

PER_FISH_COLUMNS = [
    "fish_num",
    "size_cm",
    "fish_ID_exp",
    "Genotype",
    "fish_ID_track",
]
PER_VIDEO_COLUMNS = [
    "trial",
    "trial_uid",
    "px_cm",
    "accuracy",
    "file_name",
    "folder_name_track",
    "genotype_group",
    "group_size",
    "replicate",
    "gene",
    "experiment_type",
    "old_name_experiment",
    "social_data_old",
    "founder",
    "replicate",
    "disk_name_video",
    "control_genotype",
    "control_group",
]

TRACKING_STATE_COLUMNS = [
    "tracked",
    "valid_genotype_group",
    "valid_mean_id_probabilities",
    "valid_ratio_frames_tracked",
    "valid_num_unsolvable_impossible_speed_jumps",
]

ID_LAST_FISH_STATE_COLUMNS = [
    "valid_id_last_fish",
    "valid_certainty_id_last_fish",
    # "same_id_last_fish",
]

FOR_ANALYDID_COLUMNS = TRACKING_STATE_COLUMNS + ID_LAST_FISH_STATE_COLUMNS

NO_ID_LAST_FISH_FILL_VALUE = 0

THRESHOLD_NUM_IMPOSSIBLE_SPEED_JUMPS = 1
THRESHOLD_MEAN_ID_PROBABILITIES = 0.99
THRESHOLD_ACCURACY = 0.98
THRESHOLD_RATIO_TRACKED = 0.98
THRESHOLD_CERTAINTY_ID_LAST_FISH = 0.90

NUM_FRAMES_FOR_ANALYSIS = 18000


COLORS = {
    "WT_WT-WT": "k",
    "WT_HET-WT": "k",
    "WT_HET-HET": "k",
    "HET_HET-HET": "b",
    "HET_DEL-HET": "g",
    "HET_DEL-DEL": "y",
    "DEL_DEL-DEL": "r",
    "WT_DEL-DEL": "k",
    "WT_DEL-WT": "k",
    "5WT-WT": "k",
    "5DEL-DEL": "k",
    "HET": "b",
    "DEL": "r",
    "HET_HET": "b",
    "DEL_DEL": "r",
    "HET_DEL": "y",
    "WT_WT": "k",
    "WT_HET": "k",
    "WT_DEL": "k",
    "5WT": "k",
    "5DEL": "k",
}


# Variables
INDIVIDUAL_VARIALBES_TO_DISCARD = [
    "local_polarization",
    "distance_to_center_of_group",
]
INDIVIDUAL_NB_VARIALBES_TO_DISCARD = ["nb_cos_angle"]
GROUP_VARIABLES_TO_DISCARD = ["average_local_polarization"]


# Stats
AGGREGATION_STATS = {
    "default": ["median", "mean", "std"],
    "distance_travelled": ["median", "mean", "max"],
    "nb_angle": [ratio_in_front, circmean, circstd],
    "nb_angle_diff": [ratio_in_front, circmean, circstd],
    "nb_angle_standardized": [ratio_in_front, circmean, circstd],
}
AGGREGATION_COLUMNS = {
    "indiv": [
        "trial_uid",
        "identity",
        "genotype_group",
        "genotype",
        "line",
        "line_replicate",
        "line_experiment",
        "line_replicate_experiment",
        "experiment_type",
    ],
    "group": [
        "trial_uid",
        "genotype_group",
        "line",
        "line_replicate",
        "line_experiment",
        "line_replicate_experiment",
        "experiment_type",
    ],
    "indiv_nb": [
        "trial_uid",
        "identity",
        "identity_nb",
        "genotype_group",
        "genotype",
        "genotype_nb",
        "line",
        "line_replicate",
        "line_experiment",
        "line_replicate_experiment",
        "experiment_type",
        "focal_nb_genotype",
    ],
}
TEST_PAIRS_GROUPS = {
    "indiv": [
        {"pair": (("WT_HET-WT"), ("WT_HET-HET")), "level": 0},
        {"pair": (("HET_DEL-HET"), ("HET_DEL-DEL")), "level": 0},
        {"pair": (("WT_DEL-WT"), ("WT_DEL-DEL")), "level": 1},
        {"pair": (("WT_WT-WT"), ("WT_HET-WT")), "level": 1},
        {"pair": (("WT_HET-HET"), ("HET_HET-HET")), "level": 2},
        {"pair": (("HET_HET-HET"), ("HET_DEL-HET")), "level": 3},
        {"pair": (("HET_DEL-DEL"), ("DEL_DEL-DEL")), "level": 3},
        {"pair": (("DEL_DEL-DEL"), ("WT_DEL-DEL")), "level": 4},
        {"pair": (("WT_WT-WT"), ("HET_HET-HET")), "level": 4},
        {"pair": (("HET_HET-HET"), ("DEL_DEL-DEL")), "level": 5},
        {"pair": (("WT_WT-WT"), ("DEL_DEL-DEL")), "level": 6},
        {"pair": (("WT_WT-WT"), ("WT_DEL-WT")), "level": 7},
        {"pair": (("5WT-WT"), ("5DEL-DEL")), "level": 8},
    ],
    "group": [
        {"pair": ("HET_HET", "HET_DEL"), "level": 0},
        {"pair": ("WT_WT", "WT_HET"), "level": 0},
        {"pair": ("DEL_DEL", "WT_DEL"), "level": 0},
        {"pair": ("WT_HET", "HET_HET"), "level": 1},
        {"pair": ("HET_DEL", "DEL_DEL"), "level": 1},
        {"pair": ("WT_WT", "HET_HET"), "level": 2},
        {"pair": ("HET_HET", "DEL_DEL"), "level": 3},
        {"pair": ("WT_WT", "DEL_DEL"), "level": 3},
        {"pair": ("5WT", "5DEL"), "level": 4},
    ],
    "indiv_nb": [
        {"pair": ("WT-WT", "WT-HET"), "level": 0},
        {"pair": ("WT-HET", "WT-DEL"), "level": 1},
        {"pair": ("WT-WT", "WT-DEL"), "level": 2},
        {"pair": ("HET-WT", "HET-HET"), "level": 0},
        {"pair": ("HET-HET", "HET-DEL"), "level": 1},
        {"pair": ("HET-WT", "HET-DEL"), "level": 2},
        {"pair": ("DEL-WT", "DEL-HET"), "level": 0},
        {"pair": ("DEL-HET", "DEL-DEL"), "level": 1},
        {"pair": ("DEL-WT", "DEL-DEL"), "level": 2},
        {"pair": ("WT-HET", "HT-WT"), "level": 3},
        {"pair": ("HET-DEL", "DEL-HET"), "level": 3},
        {"pair": ("WT-DEL", "DEL-WT"), "level": 4},
        {"pair": ("WT-WT", "DEL-DEL"), "level": 5},
        {"pair": ("HET-HET", "DEL-DEL"), "level": 6},
    ],
}
PAIRS_OF_GROUPS = (
    TEST_PAIRS_GROUPS["indiv"]
    + TEST_PAIRS_GROUPS["indiv_nb"]
    + TEST_PAIRS_GROUPS["group"]
)
# This are arguments to the permutation_test function in the mlxtend python library
MEAN_STATS_KWARGS = {
    "method": "approximate",
    "num_rounds": 10000,
    "func": "mean",
    "paired": False,
}
MEDIAN_STATS_KWARGS = {
    "method": "approximate",
    "num_rounds": 10000,
    "func": "median",
    "paired": False,
}
MEAN_STATS_CONFIG = {
    "test_func": permutation_test,
    "test_func_kwargs": MEAN_STATS_KWARGS,
}
MEDIAN_STATS_CONFIG = {
    "test_func": permutation_test,
    "test_func_kwargs": MEDIAN_STATS_KWARGS,
}


# PLOTS
INDIVIDUAL_VARIABLES_TO_PLOT = [
    "normed_distance_to_origin",
    "speed",
    "normal_acceleration",
]
INDIVIDUAL_VARIABLES_STATS_TO_PLOT = [
    ("normed_distance_to_origin", "median"),
    ("normed_distance_to_origin", "mean"),
    ("normed_distance_to_origin", "std"),
    ("speed", "median"),
    ("speed", "mean"),
    ("speed", "std"),
    ("normal_acceleration", "median"),
    ("normal_acceleration", "mean"),
    ("normal_acceleration", "std"),
]

GROUP_VARIABLES_TO_PLOT = [
    "mean_distance_to_center_of_group",
    "polarization_order_parameter",
    "rotation_order_parameter",
]
GROUP_VARIABLES_STATS_TO_PLOT = [
    ("mean_distance_to_center_of_group", "median"),
    ("mean_distance_to_center_of_group", "mean"),
    ("mean_distance_to_center_of_group", "std"),
    ("polarization_order_parameter", "median"),
    ("polarization_order_parameter", "mean"),
    ("polarization_order_parameter", "std"),
    ("rotation_order_parameter", "median"),
    ("rotation_order_parameter", "mean"),
    ("rotation_order_parameter", "std"),
]

INDIVIDUAL_NB_VARIABLES_TO_PLOT = ["nb_angle", "nb_distance"]
INDIVIDUAL_NB_VARIALBES_STATS_TO_PLOT = [
    ("nb_angle", "ratio_in_front"),
    ("nb_distance", "median"),
    ("nb_distance", "mean"),
    ("nb_distance", "std"),
]

# DATASETS
TR_INDIV_BL_DIR_NAME = "tr_indiv_vars_bl"
TR_INDIV_NB_BL_DIR_NAME = "tr_indiv_nb_vars_bl"
TR_GROUP_BL_DIR_NAME = "tr_group_vars_bl"

DATA_FILTERS = {
    "experiment_1": [
        lambda x: x.experiment_type == 1,
    ],
    "experiment_2": [
        lambda x: x.experiment_type == 2,
    ],
    "experiment_3": [
        lambda x: x.experiment_type == 3,
    ],
    "experiment_4": [
        lambda x: x.experiment_type == 4,
    ],
    "experiment_5": [
        lambda x: x.experiment_type == 5,
    ],
    "no_srrm": [lambda x: ~x.line_experiment.str.contains("srrm")],
    "srrm": [lambda x: x.line_experiment.str.contains("srrm")],
}

## BOXPLOT KWARGS
INDIV_BOXPLOT_KWARGS = {
    "x": "genotype_group_genotype",
    "palette": COLORS,
    "whis": 1.5,
}
GROUP_BOXPLOT_KWARGS = {
    "x": "genotype_group",
    "palette": COLORS,
    "whis": 1.5,
}
INDIV_NB_BOXPLOT_KWARGS = {
    "x": "focal_nb_genotype",
    "whis": 1.5,
}

RATIO_Y_OFFSET = 0.1
VARIABLE_RATIO_Y_OFFSET = 0.2

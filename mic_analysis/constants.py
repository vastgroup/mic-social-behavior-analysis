import os

import numpy as np

# Constants
SIGMA = 1
PX_CM = 54
FRAME_RATE = 29

DEFAULT_LOG_FILENAME = "log"
DEFAULT_SCREEN_FORMATTER = "%(name)-12s: %(levelname)-8s %(message)s"
DEFAULT_FILE_FORMATTER = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"

# Main data dir
DATA_DIR = os.environ.get("IRIMIA_LAB_ANALYSIS_SOCIAL_DATA_DIR", None)

# GENERATED TABLES AND FIGURES
GENERATED_TABLES_PATH = os.path.join(DATA_DIR, "generated_tables")
GENERATED_FIGURES_PATH = os.path.join(DATA_DIR, "generated_figures")

# Conversion table from old to new names
CONVERSIONS_TABLE_PATH = os.path.join(DATA_DIR, "Conversions.csv")
# Experiments info table

EXPERIMENTS_INFO_TABLE = os.path.join(DATA_DIR, "Data_structure.csv")

# Animals
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

# Trajectories
TRACKING_DATA_FOLDER_PATH = os.path.join(DATA_DIR, "Social_DATA")
TRAJECTORIES_INDEX_FILE_NAME = os.path.join(
    GENERATED_TABLES_PATH, "trajectories_index.csv"
)

# Experiments
VIDEOS_INDEX_FILE_NAME = os.path.join(
    GENERATED_TABLES_PATH, "videos_index.csv"
)
VIDEOS_VALID_FOR_ANALYSIS_FILE_PATH = os.path.join(
    GENERATED_TABLES_PATH, "videos_valid_for_analysis.csv"
)
VIDEOS_TRACKING_STATE_FILE_NAME = os.path.join(
    GENERATED_TABLES_PATH, "videos_tracking_state.csv"
)


# Trajectorytools variables
TR_INDIV_VARS_BOXPLOTS_FILE_PATH = os.path.join(
    GENERATED_FIGURES_PATH, "tr_indiv_vars_boxplots.pdf"
)

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
}

VALID_GENOTYPES = [
    "HET_HET",
    "HET_DEL",
    "DEL_DEL",
    "WT_WT",
    "WT_DEL",
    "WT_HET",
    "WT_WT_WT_WT_WT",
    "DEL_DEL_DEL_DEL_DEL",
]

GENOTYPE_GROUP_ORDER = [
    "WT_WT",
    "WT_HET",
    "HET_HET",
    "HET_DEL",
    "DEL_DEL",
    "WT_DEL",
    "WT_WT_WT_WT_WT",
    "DEL_DEL_DEL_DEL_DEL",
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
    "WT_WT_WT_WT_WT-WT",
    "DEL_DEL_DEL_DEL_DEL-DEL",
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

THRESHOLD_NUM_IMPOSSIBLE_SPEED_JUMPS = 20
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
    "WT_WT_WT_WT_WT-WT": "k",
    "DEL_DEL_DEL_DEL_DEL-DEL": "k",
    "HET": "b",
    "DEL": "r",
    "HET_HET": "b",
    "DEL_DEL": "r",
    "HET_DEL": "y",
    "WT_WT": "k",
    "WT_HET": "k",
    "WT_DEL": "k",
    "WT_WT_WT_WT_WT": "k",
    "DEL_DEL_DEL_DEL_DEL": "k",
}

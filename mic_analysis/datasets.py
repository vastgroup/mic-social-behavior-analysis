import logging
import os

import numpy as np
import pandas as pd
from confapp import conf

from .stats import (
    _compute_agg_stat,
    group_agg_stasts_kwargs,
    indiv_agg_stasts_kwargs,
    indiv_nb_agg_stats_kwargs,
    standardize_replicate_data_wrt_het,
)
from .utils import data_filter
from .variables import (
    _group_variables,
    _individual_nb_variables,
    _individual_variables,
)

logger = logging.getLogger(__name__)


TRAJECTORYTOOLS_DATASETS_INFO = {
    "tr_indiv_bl": {
        "file_path": os.path.join(
            conf.GENERATED_TABLES_PATH, conf.TR_INDIV_BL_FILE_NAME
        ),
        "variables_to_compute": _individual_variables,
        "scale_to_body_length": True,
        "variables_names": [var_["name"] for var_ in _individual_variables],
        "agg_stats_kwargs": indiv_agg_stasts_kwargs,
    },
    "tr_indiv_nb_bl": {
        "file_path": os.path.join(
            conf.GENERATED_TABLES_PATH, conf.TR_INDIV_NB_BL_FILE_NAME
        ),
        "scale_to_body_length": True,
        "variables_to_compute": _individual_nb_variables,
        "agg_stats_kwargs": indiv_nb_agg_stats_kwargs,
    },
    "tr_group_bl": {
        "file_path": os.path.join(
            conf.GENERATED_TABLES_PATH, conf.TR_GROUP_BL_FILE_NAME
        ),
        "scale_to_body_length": True,
        "variables_to_compute": _group_variables,
        "agg_stats_kwargs": group_agg_stasts_kwargs,
    },
}


def get_data(path, data_filters, agg_stats_kwargs):
    logger.info("Loading data")
    data = pd.read_pickle(path)
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
    data_filtered = data_filter(data, data_filters)
    data_stats = _compute_agg_stat(data=data_filtered, **agg_stats_kwargs)
    if "indiv" in path:
        data_stats["genotype_group_genotype"] = (
            data_stats["genotype_group"] + "-" + data_stats["genotype"]
        )
    if "indiv_nb" in path:
        data_stats["genotype_group_genotype_nb"] = (
            data_stats["genotype_group"] + "-" + data_stats["genotype_nb"]
        )
    return data_filtered, data_stats


def get_datasets(data_filters):
    logger.info("Getting datasets")

    logger.info("Getting dataset data_indiv")
    data_indiv, data_indiv_stats = get_data(
        TRAJECTORYTOOLS_DATASETS_INFO["tr_indiv_bl"]["file_path"],
        data_filters=data_filters,
        agg_stats_kwargs=TRAJECTORYTOOLS_DATASETS_INFO["tr_indiv_bl"][
            "agg_stats_kwargs"
        ],
    )
    logger.info("done")

    logger.info("Getting dataset data_group")
    data_group, data_group_stats = get_data(
        TRAJECTORYTOOLS_DATASETS_INFO["tr_group_bl"]["file_path"],
        data_filters=data_filters,
        agg_stats_kwargs=TRAJECTORYTOOLS_DATASETS_INFO["tr_group_bl"][
            "agg_stats_kwargs"
        ],
    )
    logger.info("done")

    logger.info("Getting dataset data_indiv_nb")
    data_indiv_nb, data_indiv_nb_stats = get_data(
        TRAJECTORYTOOLS_DATASETS_INFO["tr_indiv_nb_bl"]["file_path"],
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

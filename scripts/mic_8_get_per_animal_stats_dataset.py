import os

import pandas as pd
from confapp import conf
from mic_analysis.datasets import TRAJECTORYTOOLS_DATASETS_INFO, load_dataset
from mic_analysis.logger import setup_logs
from mic_analysis.stats import (
    _compute_agg_stat,
    standardize_replicate_data_wrt_het,
)

logger = setup_logs("get_per_animal_stats")

videos_table = pd.read_csv(conf.VIDEOS_INDEX_FILE_NAME)


for name, dataset_info in TRAJECTORYTOOLS_DATASETS_INFO.items():
    logger.info(f"**Computing per individual stats for dataset {name}")

    all_data_stats = []

    for line_experiment in videos_table.line_experiment.unique():

        logger.info(f"*Computing stats for experiment {line_experiment}")

        logger.info("Loading data")
        data = load_dataset(
            dataset_info["dir_path"],
            video_filters=[lambda x: x["line_experiment"] == line_experiment],
        )
        logger.info("Loaded")

        if not data.empty:
            if "WT_WT" in data["genotype_group"]:
                normalizing_genotype_group = "WT_WT"
            elif "HET_HET" in data["genotype_group"]:
                normalizing_genotype_group = "HET_HET"
            else:
                # TODO: fix this
                normalizing_genotype_group = "HET_HET"
            data = standardize_replicate_data_wrt_het(
                data, normalizing_genotype_group=normalizing_genotype_group
            )
            data_stats = _compute_agg_stat(
                data=data, **dataset_info["agg_stats_kwargs"]
            )
            if "indiv" in name:
                data_stats["genotype_group_genotype"] = (
                    data_stats["genotype_group"] + "-" + data_stats["genotype"]
                )
            if "indiv_nb" in name:
                data_stats["genotype_group_genotype_nb"] = (
                    data_stats["genotype_group"]
                    + "-"
                    + data_stats["genotype_nb"]
                )
            all_data_stats.append(data_stats)
    all_data_stats = pd.concat(all_data_stats)
    save_path = os.path.join(
        dataset_info["dir_path"], conf.PER_ANIMAL_STATS_FILE_NAME
    )
    all_data_stats.to_pickle(save_path)

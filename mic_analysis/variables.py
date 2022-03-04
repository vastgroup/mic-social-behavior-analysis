import os

import numpy as np
import pandas as pd
from confapp import conf
from trajectorytools.export import (
    GROUP_VARIABLES,
    INDIVIDUAL_NEIGHBOUR_VARIABLES,
    INDIVIDUAL_VARIALBES,
)


def _filter_out_variables(variables, variables_to_discard):
    return [
        var_ for var_ in variables if not var_["name"] in variables_to_discard
    ]


def _get_variables_names(variables):
    return [var_["name"] for var_ in variables]


def _create_list_of_variables_with_suffix(variables_names, suffix):
    return [f"{v}{suffix}" for v in variables_names]


_individual_variables = _filter_out_variables(
    INDIVIDUAL_VARIALBES, conf.INDIVIDUAL_VARIALBES_TO_DISCARD
)

_individual_variables_names = _get_variables_names(_individual_variables)
_individual_variables_diff_names = _create_list_of_variables_with_suffix(
    _individual_variables_names, "_diff"
)
_individual_variables_standardized_names = (
    _create_list_of_variables_with_suffix(
        _individual_variables_names, "_standardized"
    )
)
_individual_variables_enhanced_names = (
    _individual_variables_names
    + _individual_variables_diff_names
    + _individual_variables_standardized_names
)


_individual_nb_variables = _filter_out_variables(
    INDIVIDUAL_NEIGHBOUR_VARIABLES, conf.INDIVIDUAL_NB_VARIALBES_TO_DISCARD
)
_individual_nb_variables_names = _get_variables_names(_individual_nb_variables)
_individual_nb_variables_diff_names = _create_list_of_variables_with_suffix(
    _individual_nb_variables_names, "_diff"
)
_individual_nb_variables_standardized_names = (
    _create_list_of_variables_with_suffix(
        _individual_nb_variables_names, "_standardized"
    )
)
_individual_nb_variables_enhanced_names = (
    _individual_nb_variables_names
    + _individual_nb_variables_diff_names
    + _individual_nb_variables_standardized_names
)


_group_variables = _filter_out_variables(
    GROUP_VARIABLES, conf.GROUP_VARIABLES_TO_DISCARD
)
_group_variables_names = _get_variables_names(_group_variables)
_group_variables_diff_names = _create_list_of_variables_with_suffix(
    _group_variables_names, "_diff"
)
_group_variables_standardized_names = _create_list_of_variables_with_suffix(
    _group_variables_names, "_standardized"
)
_group_varialbes_enhanced_names = (
    _group_variables_names
    + _group_variables_diff_names
    + _group_variables_standardized_names
)

all_variables_names = (
    [v for v in _individual_variables_names]
    + [v for v in _individual_nb_variables_names]
    + [v for v in _group_variables_names]
)
all_variables_names_enhanced = (
    _individual_variables_enhanced_names
    + _individual_nb_variables_enhanced_names
    + _group_varialbes_enhanced_names
)


def compute_variables_ranges(data):
    variables_ranges = []

    for col in data.columns:
        if col in all_variables_names_enhanced:
            if "accel" in col:
                min_ = np.nanpercentile(data[col], 1)
                max_ = np.nanpercentile(data[col], 99)
            else:
                min_ = np.nanmin(data[col])
                max_ = np.nanmax(data[col])
            variables_ranges.append(
                {
                    "variable": col,
                    "min": min_,
                    "max": max_,
                }
            )
    variables_ranges = pd.DataFrame(variables_ranges)
    return variables_ranges


def get_variables_ranges(datasets):
    all_variables_ranges = []
    for name, dataset_info in datasets.items():
        variables_range_path = os.path.join(
            dataset_info["dir_path"], conf.VARIABLES_RANGES_FILE_NAME
        )
        variables_range = pd.read_pickle(variables_range_path)
        all_variables_ranges.append(variables_range)
    all_variables_ranges = pd.concat(all_variables_ranges)
    return all_variables_ranges

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


def compute_variables_ranges(datasets):
    variables_ranges = []

    for dataset_name, dataset in datasets.items():
        if "stats" not in dataset:
            for col in dataset.columns:
                if col in all_variables_names_enhanced:
                    if col == "normal_acceleration":
                        min_ = np.nanpercentile(dataset[col], 1)
                        max_ = np.nanpercentile(dataset[col], 99)
                    else:
                        min_ = np.nanmin(dataset[col])
                        max_ = np.nanmax(dataset[col])
                    variables_ranges.append(
                        {
                            "variable": col,
                            "min": min_,
                            "max": max_,
                        }
                    )
    variables_ranges = pd.DataFrame(variables_ranges)
    return variables_ranges

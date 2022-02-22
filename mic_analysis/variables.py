import numpy as np
from trajectorytools.export import (
    GROUP_VARIABLES,
    INDIVIDUAL_NEIGHBOUR_VARIABLES,
    INDIVIDUAL_VARIALBES,
)

_individual_variables = [
    var_
    for var_ in INDIVIDUAL_VARIALBES
    if not var_["name"]
    in ["local_polarization", "distance_to_center_of_group"]
]
_individual_variables_names = [var_["name"] for var_ in _individual_variables]
_individual_variables_diff_names = [
    f"{v}_diff" for v in _individual_variables_names
]
_individual_variables_standardized_names = [
    f"{v}_standardized" for v in _individual_variables_names
]
_individual_variables_enhanced_names = (
    _individual_variables_names
    + _individual_variables_diff_names
    + _individual_variables_standardized_names
)


_individual_nb_variables = [
    var_
    for var_ in INDIVIDUAL_NEIGHBOUR_VARIABLES
    if not var_["name"] in ["nb_cos_angle"]
]
_individual_nb_variables_names = [
    var_["name"] for var_ in _individual_nb_variables
]
_individual_nb_variables_diff_names = [
    f"{v}_diff" for v in _individual_nb_variables_names
]
_individual_nb_variables_standardized_names = [
    f"{v}_standardized" for v in _individual_nb_variables_names
]
_individual_nb_variables_enhanced_names = (
    _individual_nb_variables_names
    + _individual_nb_variables_diff_names
    + _individual_nb_variables_standardized_names
)


_group_variables = [
    var_
    for var_ in GROUP_VARIABLES
    if var_["name"] != "average_local_polarization"
]
_group_variables_names = [var_["name"] for var_ in _group_variables]
_group_variables_diff_names = [f"{v}_diff" for v in _group_variables_names]
_group_variables_standardized_names = [
    f"{v}_standardized" for v in _group_variables_names
]
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
    variables_ranges = {}

    for dataset_name, dataset in datasets.items():
        if "stats" not in dataset:
            for col in dataset.columns:
                if col in all_variables_names_enhanced:
                    variables_ranges[col] = {
                        "min": np.nanmin(dataset[col]),
                        "max": np.nanmax(dataset[col]),
                    }
    return variables_ranges

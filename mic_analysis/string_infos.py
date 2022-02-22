animal_uid_col = "trial_uid_id"
video_uid_col = "trial_uid"

animal_info_cols = [
    "genotype",
    "fish_id_exp",
    "identity",
    "dpf",
    "size_cm",
]

video_info_cols = [
    "trial_uid",
    "experiment_type",
    "gene",
    "founder",
    "replicate",
    "genotype_group",
]


def _get_info(data, info_cols):
    info_str = ""
    for info_col in info_cols:
        assert info_col in data.columns
        infos = data[info_col].unique()
        assert len(set(infos)) == 1, set(infos)
        info = str(infos[0])
        info_str += f"{info_col}: {info} - "
    info_str = info_str[:-3]
    return info_str


def get_animal_info_str(
    animal_data, info_cols=animal_info_cols, video_info_cols=video_info_cols
):
    video_info = _get_info(animal_data, video_info_cols)
    animal_info = _get_info(animal_data, info_cols)
    return f"{video_info} \n {animal_info}"


def get_video_info_str(video_data, video_info_cols=video_info_cols):
    video_str_info = _get_info(video_data, video_info_cols)

    animals_uid_ids = video_data[animal_uid_col].unique()
    for animal_uid in animals_uid_ids:
        animal_data = video_data[video_data[animal_uid_col] == animal_uid]
        animal_str_info = _get_info(animal_data, animal_info_cols)
        video_str_info += f"\n {animal_str_info}"
    return video_str_info


def get_focal_nb_info(animal_nb_data, animal_info_cols=animal_info_cols):
    focal_info_cols = animal_info_cols
    nb_info_cols = [f"{col}_nb" for col in animal_info_cols]
    focal_info_str = _get_info(animal_nb_data, focal_info_cols)
    focal_info_str = f"focal: {focal_info_str}"
    nb_info_str = _get_info(animal_nb_data, nb_info_cols)
    nb_info_str = f"neighbour: {nb_info_str}"
    return f"{focal_info_str} \n {nb_info_str}"


def get_partition_info_str(data, partition_col):
    if partition_col == "line_replicate":
        extra_info_cols = ["replicate"]
    else:
        extra_info_cols = []
    partition_info_cols = (
        [
            "experiment_type",
            "gene",
            "founder",
        ]
        + [partition_col]
        + extra_info_cols
    )
    return _get_info(data, partition_info_cols)

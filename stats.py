# Stats
PAIRS_OF_GROUPS = [
    {"pair": (("WT_HET-WT"), ("WT_HET-HET")), "level": 0},  # individual
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
    {"pair": ("HET_HET", "HET_DEL"), "level": 0},  # group
    {"pair": ("WT_WT", "WT_HET"), "level": 0},
    {"pair": ("DEL_DEL", "WT_DEL"), "level": 0},
    {"pair": ("WT_HET", "HET_HET"), "level": 1},
    {"pair": ("HET_DEL", "DEL_DEL"), "level": 1},
    {"pair": ("WT_WT", "HET_HET"), "level": 2},
    {"pair": ("HET_HET", "DEL_DEL"), "level": 3},
    {"pair": ("WT_WT", "DEL_DEL"), "level": 3},
    {"pair": ("WT-WT", "WT-HET"), "level": 0},  # focal nb
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
]


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

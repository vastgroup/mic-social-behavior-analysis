# Stats
PAIRS_OF_GROUPS = [
    {"pair": (("WT_HET", "WT"), ("WT_HET", "HET")), "level": 0},
    {"pair": (("HET_DEL", "HET"), ("HET_DEL", "DEL")), "level": 0},
    {"pair": (("WT_DEL", "WT"), ("WT_DEL", "DEL")), "level": 1},
    {"pair": (("WT_WT", "WT"), ("WT_HET", "WT")), "level": 1},
    {"pair": (("WT_HET", "HET"), ("HET_HET", "HET")), "level": 2},
    {"pair": (("HET_HET", "HET"), ("HET_DEL", "HET")), "level": 3},
    {"pair": (("HET_DEL", "DEL"), ("DEL_DEL", "DEL")), "level": 3},
    {"pair": (("DEL_DEL", "DEL"), ("WT_DEL", "DEL")), "level": 4},
    {"pair": (("WT_WT", "WT"), ("HET_HET", "HET")), "level": 4},
    {"pair": (("HET_HET", "HET"), ("DEL_DEL", "DEL")), "level": 5},
    {"pair": (("WT_WT", "WT"), ("DEL_DEL", "DEL")), "level": 6},
    {"pair": (("WT_WT", "WT"), ("WT_DEL", "WT")), "level": 7},
]
from figures import (
    TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEAN_STAT_MEAN,
    TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEAN_STAT_MEAN_BL,
    TR_INDIVS_BOXPLOT_LINE_MEAN_STAT_MEAN,
    TR_INDIVS_BOXPLOT_LINE_MEAN_STAT_MEAN_BL,
)
from plotters import plot

data_filtered_stat, all_var_stats, all_outliers = plot(
    TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEAN_STAT_MEAN.copy()
)
data_filtered_stat, all_var_stats, all_outliers = plot(
    TR_INDIVS_BOXPLOT_LINE_REPLICATE_MEAN_STAT_MEAN_BL.copy()
)

data_filtered_stat, all_var_stats, all_outliers = plot(
    TR_INDIVS_BOXPLOT_LINE_MEAN_STAT_MEAN.copy()
)
data_filtered_stat, all_var_stats, all_outliers = plot(
    TR_INDIVS_BOXPLOT_LINE_MEAN_STAT_MEAN_BL.copy()
)

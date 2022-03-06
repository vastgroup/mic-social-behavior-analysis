from setuptools import find_packages, setup

setup(
    name="mic_analysis",
    version="0.1",
    description="mic_analysis",
    url="https://github.com/vastgroup/mic-social-behavior-analysis",
    author="Francisco Romero-Ferrero",
    author_email="paco.romero.ferrero@gmail.com",
    license="GPL",
    packages=find_packages(),
    scripts=[
        "scripts/mic_2_generate_animals_index.py",
        "scripts/mic_3_generate_trajectories_index.py",
        "scripts/mic_4_generate_master_videos_table.py",
        "scripts/mic_5_plot_tracking_sanity_check.py",
        "scripts/mic_6_generate_tr_datasets.py",
        "scripts/mic_7_get_general_variables_stats.py",
        "scripts/mic_8_get_per_animal_stats_dataset.py",
        "scripts/mic_9_plot_summary_figures.py",
        "scripts/mic_10_plot_outliers.py",
        "scripts/mic_11_plot_variable_summary.py",
    ],
    zip_safe=False,
)

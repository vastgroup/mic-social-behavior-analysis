# mic-social-behavior-analysis

## Summary

This repository includes code to:

1. 


## Repository structure


## Data folder structure and files

## 

## Install

1. Open the file `install_environment.yml` and substitute the path in the variable `DATA_DIR` to the path in your computer where the folder `2022_ANALYSIS_social` is stored.

2. Install the conda environment, the `trajectorytools` package and the `mic_analysis` package:

        conda env create -f install_environment.yml
        pip install -e trajectorytools
        pip install -e pandas_split
        pip install -e .

3. Update path to `DATA_DIR` variable:

        conda env config vars set DATA_DIR="path/to/folder/20220314_Analysis_social"

4. Check that the environment variaboes is properly set:

        echo $DATA_DIR

## Run

## Visualize figures in the jupyter notebook

## Change settings






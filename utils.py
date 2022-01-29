import pandas as pd 
from glob import glob

import logging

logger = logging.getLogger(__name__)

def read_csv(csv_path: str):
    logger.info(f"Reading {csv_path}")
    return pd.read_csv(csv_path, delimiter=";")
        
    
def get_files_with_pattern(pattern: str):
    logger.info(f"Getting files with pattern {pattern}")
    return glob(pattern)
import json
import re
import pandas as pd
from datetime import datetime

# Setup logger
import logging
from . import setup_custom_logger
logger = setup_custom_logger(__name__, level=logging.DEBUG)

# Setup current dir
from config import PROJECT_ROOT, CONFIG_DIR

def extract_numbers(text):
    """Extracts all integers and floats from a string. Returns a list"""
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    return [float(num) for num in numbers]


def load_data_paths(filepath="config/data_paths.json"):
    """Loads data paths from the JSON configuration file."""
    
    filepath = PROJECT_ROOT / filepath

    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {filepath}")
        return {}  # Return an empty dict if the file is not found
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {filepath}")
        return {}

def load_tickers_mapping(filepath="config/tickers_mapping.json"):
    """Loads tickers mapping from the JSON configuration file."""

    filepath = PROJECT_ROOT / filepath

    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {filepath}")
        return {}  # Return an empty dict if the file is not found
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {filepath}")
        return {}
    
def load_factors(filepath="jf-datashare/gabriel-cuevas/data/18_rwarh_d_1.csv")->pd.DataFrame:
    """Loads the factor dataset from the replication code."""

    filepath = PROJECT_ROOT / filepath

    try:
        return pd.read_csv(filepath, index_col=0)
    except FileNotFoundError:
        logger.error(f"Factors file not found: {filepath}")
        return {}  # Return an empty dict if the file is not found
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {filepath}")
        return {}

def date_to_num(date):
    """
    Function to convert a date to a number, takes a datetime object as input and returns an integrer
    """
    date = str(date)
    if " 00:00:00" in date:
        date = date.replace(" 00:00:00", "")
    date = date.replace("-", "")
    return int(date)

import json
import re
import logging

from . import setup_custom_logger

# Setup logger
logger = setup_custom_logger(__name__, level=logging.INFO)


def extract_numbers(text):
    """Extracts all integers and floats from a string. Returns a list"""
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    return [float(num) for num in numbers]


def load_ticker_mapping(filepath="../config/tickers_mapping.json"):
    """Loads the ticker mapping from the JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Could not find {filepath}")
        return {}  # Return an empty dict if the file is not found


def load_data_paths(filepath="../config/data_paths.json"):
    """Loads data paths from the JSON configuration file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {filepath}")
        return {}  # Return an empty dict if the file is not found
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {filepath}")
        return {}

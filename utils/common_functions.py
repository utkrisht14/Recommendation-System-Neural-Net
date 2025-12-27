import os
import pandas as pd
import sys
import yaml

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise CustomException(f"File {file_path} does not exist")

        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Successfully read the file.")
            return config

    except Exception as e:
        logger.error("Error while reading the file.")
        raise CustomException(f"Error while reading the file: {e}")


def load_data(file_path):
    try:
        logger.info(f"Loading data")
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error while loading the file: {e}")
        raise CustomException(f"Error while loading the file: {e}")





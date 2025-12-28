from src.data_processing import DataProcessor
from config.paths_config import *
from utils.common_functions import read_yaml
from src.model_training import ModelTraining


if __name__ == "__main__":
    data_processor = DataProcessor(read_yaml(CONFIG_PATH))
    data_processor.run()

    model_trainer = ModelTraining(read_yaml(CONFIG_PATH))
    model_trainer.train_model()

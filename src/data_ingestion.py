from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_names = self.config["bucket_file_names"]

        # When data ingestion is triggered it automatically creates our raw directory
        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"Data ingestion started.")

    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)

            # Get all three files
            for file_name in self.file_names:
                file_path = os.path.join(RAW_DIR, file_name)

                blob = bucket.blob(file_name)
                blob.download_to_filename(file_path)

        except Exception as e:
            logger.error("Error while downloading data from GCP.")
            raise CustomException("Failed to download data from GCP.", e)


    def run(self):
        try:
            logger.info(f"Starting Data Ingestion Process.")
            self.download_csv_from_gcp()
            logger.info("Data ingestion completed.")

        except CustomException as e:
            logger.error(f"Custom exception: {str(e)}")


        finally:
            logger.info("Data ingestion DONE...")


if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()







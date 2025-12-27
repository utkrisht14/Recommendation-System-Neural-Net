import logging
import os
from datetime import datetime
from pathlib import Path
from datetime import datetime

# Create logs directory
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True) # If already exists don't create

# We want all the logs to be stored day by day
LOG_FILE = Path(LOGS_DIR) / f"log_{datetime.now():%Y-%m-%d}.log"

logging.basicConfig(
    filename = LOG_FILE,
    format = "%(asctime)s - %(levelname)s %(message)s",
    level=logging.INFO
)

# It will create a logger with the given name whatever provided by the user
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
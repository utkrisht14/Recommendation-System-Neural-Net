import os

# Absolute project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__),"..")
)

################### DATA INGESTION #######################

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

RAW_DIR = os.path.join(ARTIFACTS_DIR, "raw")



################ CONFIG ################

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")

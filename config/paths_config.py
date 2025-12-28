import os

# Absolute project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__),"..")
)

################### DATA INGESTION #######################

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

RAW_DIR = os.path.join(ARTIFACTS_DIR, "raw")


######################## DATA PROCESSING ##########################

PROCESSED_DIR = os.path.join(ARTIFACTS_DIR, "processed")
ANIMELIST_CSV = os.path.join(ARTIFACTS_DIR, "raw", "animelist.csv")
ANIME_CSV = os.path.join(ARTIFACTS_DIR, "raw", "animelist.csv")
ANIMESYNOPSIS_CSV = os.path.join(ARTIFACTS_DIR, "raw", "anime_with_synopsis.csv")

X_TRAIN_ARRAY = os.path.join(PROCESSED_DIR, "X_train_array.pkl")
X_TEST_ARRAY = os.path.join(PROCESSED_DIR, "X_test_array.pkl")
Y_TRAIN = os.path.join(PROCESSED_DIR, "y_train.pkl")
Y_TEST = os.path.join(PROCESSED_DIR, "y_test.pkl")

RATING_DF = os.path.join(PROCESSED_DIR, "rating.csv")
DF = os.path.join(PROCESSED_DIR, "anime_df.csv")
SYNOPSIS_DF = os.path.join(PROCESSED_DIR, "synopsis_df.csv")

USER2USER_ENCODED = os.path.join(PROCESSED_DIR, "user2user_encoded.pkl")
USER2USER_DECODED = os.path.join(PROCESSED_DIR, "user2user_decoded.pkl")

ANIME2ANIME_ENCODED = os.path.join(PROCESSED_DIR, "anime2anime_encoded.pkl")
ANIME2ANIME_DECODED = os.path.join(PROCESSED_DIR, "anime2anime_decoded.pkl")


###################### MODEL TRAINING #########################

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
ANIME_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR,"anime_weights.pkl")
USER_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR,"user_weights.pkl")
CHECKPOINT_FILE_PATH = os.path.join(ARTIFACTS_DIR, "model_checkpoint", "weights.weights.h5")


################ CONFIG ################
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")


################ API KEYS ################
COMET_API_PATH = os.path.join(PROJECT_ROOT, "secret.json")

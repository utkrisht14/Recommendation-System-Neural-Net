import os
import joblib
import numpy as np
import pandas as pd

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import (
    ANIMELIST_CSV,
    ANIME_CSV,
    ANIMESYNOPSIS_CSV,
    PROCESSED_DIR,
    X_TRAIN_ARRAY,
    X_TEST_ARRAY,
    Y_TRAIN,
    Y_TEST,
    DF,
    SYNOPSIS_DF,
)

logger = get_logger(__name__)


class DataProcessor:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir

        self.rating_df = None

        self.X_train_array = None
        self.X_test_array = None
        self.y_train = None
        self.y_test = None

        self.user2user_encoded = {}
        self.user2user_decoded = {}
        self.anime2anime_encoded = {}
        self.anime2anime_decoded = {}

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("DataProcessor initialized")

    def load_data(self, usecols=None):
        """Load ratings data from CSV."""
        try:
            self.rating_df = pd.read_csv(self.input_file, low_memory=False, usecols=usecols)
            logger.info(f"Ratings data loaded: {self.rating_df.shape}")
        except Exception as e:
            raise CustomException("Failed to load data from file", e)

    def filter_users(self, min_rating: int = 400):
        """Keep only users with at least `min_rating` ratings."""
        try:
            n_ratings = self.rating_df["user_id"].value_counts()
            keep_users = n_ratings[n_ratings >= min_rating].index
            self.rating_df = self.rating_df[self.rating_df["user_id"].isin(keep_users)].copy()
            logger.info(f"Filtered users (>= {min_rating} ratings): {self.rating_df.shape}")
        except Exception as e:
            raise CustomException("Failed to filter users", e)

    def scale_ratings(self):
        """Min-max normalize the rating column to [0, 1]."""
        try:
            # Ensure numeric
            self.rating_df["rating"] = pd.to_numeric(self.rating_df["rating"], errors="coerce")
            self.rating_df = self.rating_df.dropna(subset=["rating"]).copy()

            min_r = float(self.rating_df["rating"].min())
            max_r = float(self.rating_df["rating"].max())

            if max_r == min_r:
                # Avoid divide-by-zero; all ratings identical
                self.rating_df["rating"] = 0.0
                logger.warning("All ratings identical; scaled ratings set to 0.0")
            else:
                self.rating_df["rating"] = (self.rating_df["rating"] - min_r) / (max_r - min_r)

            logger.info("Ratings scaled successfully")
        except Exception as e:
            raise CustomException("Failed to scale ratings", e)

    def encode_data(self):
        """
        Encode user_id and anime_id into contiguous integer indices for embeddings.
        Adds:
          - rating_df['user']  (encoded user index)
          - rating_df['anime'] (encoded anime index)
        """
        try:
            # Users
            user_ids = self.rating_df["user_id"].unique().tolist()
            self.user2user_encoded = {x: i for i, x in enumerate(user_ids)}
            self.user2user_decoded = {i: x for i, x in enumerate(user_ids)}
            self.rating_df["user"] = self.rating_df["user_id"].map(self.user2user_encoded)

            # Anime
            anime_ids = self.rating_df["anime_id"].unique().tolist()
            self.anime2anime_encoded = {x: i for i, x in enumerate(anime_ids)}
            self.anime2anime_decoded = {i: x for i, x in enumerate(anime_ids)}
            self.rating_df["anime"] = self.rating_df["anime_id"].map(self.anime2anime_encoded)

            # Drop any unexpected NaNs
            before = len(self.rating_df)
            self.rating_df = self.rating_df.dropna(subset=["user", "anime"]).copy()
            after = len(self.rating_df)

            logger.info(f"Encoding done for users/anime. Dropped {before - after} rows with missing encodings.")
        except Exception as e:
            raise CustomException("Failed to encode data", e)

    def split_data(self, test_size: int = 1000, random_state: int = 43):
        """
        Split into train/test sets.
        Inputs: [user_encoded, anime_encoded]
        Label : rating (scaled)
        """
        try:
            df_shuffled = self.rating_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

            # X = [user, anime], y = rating
            X = df_shuffled[["user", "anime"]].astype(int).values
            y = df_shuffled["rating"].astype(float).values

            if len(df_shuffled) < 2:
                raise ValueError("Not enough data to split.")

            test_size = min(test_size, max(1, len(df_shuffled) // 5))  # keep it safe
            train_end = len(df_shuffled) - test_size

            X_train, X_test = X[:train_end], X[train_end:]
            y_train, y_test = y[:train_end], y[train_end:]

            # Keras expects [user_array, anime_array]
            self.X_train_array = [X_train[:, 0], X_train[:, 1]]
            self.X_test_array = [X_test[:, 0], X_test[:, 1]]
            self.y_train = y_train
            self.y_test = y_test

            logger.info(f"Data split done. Train={len(y_train)}, Test={len(y_test)}")
        except Exception as e:
            raise CustomException("Failed to split data", e)

    def save_artifacts(self):
        """Save encoders, splits, and rating_df."""
        try:
            # Save mapping dicts
            artifacts = {
                "user2user_encoded": self.user2user_encoded,
                "user2user_decoded": self.user2user_decoded,
                "anime2anime_encoded": self.anime2anime_encoded,
                "anime2anime_decoded": self.anime2anime_decoded,
            }

            for name, data in artifacts.items():
                path = os.path.join(self.output_dir, f"{name}.pkl")
                joblib.dump(data, path)
                logger.info(f"Saved artifact: {path}")

            # Save train/test arrays
            joblib.dump(self.X_train_array, X_TRAIN_ARRAY)
            joblib.dump(self.X_test_array, X_TEST_ARRAY)
            joblib.dump(self.y_train, Y_TRAIN)
            joblib.dump(self.y_test, Y_TEST)

            # Save processed rating df
            rating_path = os.path.join(self.output_dir, "rating_df.csv")
            self.rating_df.to_csv(rating_path, index=False)
            logger.info(f"Saved rating_df: {rating_path}")

        except Exception as e:
            raise CustomException("Failed to save artifacts", e)

    def process_anime_data(self):
        """
        Load anime metadata and synopsis data, normalize column names,
        resolve English anime names, sort by score, and save processed CSVs.
        """
        try:
            # -----------------------------
            # Load data
            # -----------------------------
            df = pd.read_csv(ANIME_CSV)
            synopsis_df = pd.read_csv(ANIMESYNOPSIS_CSV)

            df = df.replace("Unknown", np.nan)

            # -----------------------------
            # Detect anime_id column
            # -----------------------------
            possible_id_cols = ["MAL_ID", "anime_id", "Anime_ID", "id"]
            anime_id_col = next((c for c in possible_id_cols if c in df.columns), None)

            if anime_id_col is None:
                raise ValueError(
                    f"No valid anime ID column found. Available columns: {list(df.columns)}"
                )

            df["anime_id"] = df[anime_id_col]

            # -----------------------------
            # Detect English name column
            # -----------------------------
            possible_eng_cols = ["English name", "eng_version", "english_name"]
            eng_col = next((c for c in possible_eng_cols if c in df.columns), None)

            if eng_col:
                df["eng_version"] = df[eng_col]
            else:
                df["eng_version"] = np.nan

            # Fallback to original Name column
            if "Name" in df.columns:
                df["eng_version"] = df["eng_version"].fillna(df["Name"])

            # -----------------------------
            # Detect score column
            # -----------------------------
            possible_score_cols = ["Score", "score", "rating", "mean_score", "average_rating"]
            score_col = next((c for c in possible_score_cols if c in df.columns), None)

            if score_col is None:
                raise ValueError(
                    f"No score column found. Available columns: {list(df.columns)}"
                )

            # -----------------------------
            # Sort by score
            # -----------------------------
            df.sort_values(
                by=score_col,
                ascending=False,
                inplace=True,
                kind="quicksort",
                na_position="last",
            )

            # -----------------------------
            # Ensure required optional columns exist
            # -----------------------------
            optional_cols = ["Genres", "Episodes", "Type", "Premiered", "Members"]
            for col in optional_cols:
                if col not in df.columns:
                    df[col] = np.nan

            # -----------------------------
            # Final column selection
            # -----------------------------
            df = df[
                ["anime_id", "eng_version", score_col, "Genres", "Episodes", "Type", "Premiered", "Members"]
            ].rename(columns={score_col: "Score"})

            # -----------------------------
            # Save outputs
            # -----------------------------
            df.to_csv(DF, index=False)
            synopsis_df.to_csv(SYNOPSIS_DF, index=False)

            logger.info("Processed anime metadata and synopsis data saved successfully.")

        except Exception as e:
            raise CustomException("Failed to process anime and synopsis data", e)

    def run(self):
        try:
            # Correct columns
            self.load_data(usecols=["user_id", "anime_id", "rating"])
            self.filter_users()
            self.scale_ratings()
            self.encode_data()
            self.split_data()
            self.save_artifacts()

            self.process_anime_data()
            logger.info("Data processing pipeline ran successfully.")
        except CustomException as e:
            logger.error(str(e))
            raise


if __name__ == "__main__":
    data_processor = DataProcessor(ANIMELIST_CSV, PROCESSED_DIR)
    data_processor.run()

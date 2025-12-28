import pandas as pd
import numpy as np
import joblib


############# 1. GET_ANIME_FRAME

def getAnimeFrame(anime, path_df):
    df = pd.read_csv(path_df)

    if isinstance(anime, int):
        frame = df[df["anime_id"] == anime]
    elif isinstance(anime, str):
        frame = df[df["eng_version"] == anime]
    else:
        return pd.DataFrame()

    if frame.empty:
        raise ValueError(f"Anime not found: {anime}")

    return frame


########## 2. GET_SYNOPSIS

def getSynopsis(anime, path_synopsis_df):
    synopsis_df = pd.read_csv(path_synopsis_df)

    # Robust column detection
    id_col = "MAL_ID" if "MAL_ID" in synopsis_df.columns else "anime_id"
    text_col = "sypnopsis" if "sypnopsis" in synopsis_df.columns else "synopsis"

    if isinstance(anime, int):
        row = synopsis_df[synopsis_df[id_col] == anime]
    elif isinstance(anime, str):
        row = synopsis_df[synopsis_df["Name"] == anime]
    else:
        return None

    if row.empty:
        return None

    return row[text_col].values[0]


########## 3. CONTENT RECOMMENDATION

def find_similar_animes(
    name,
    path_anime_weights,
    path_anime2anime_encoded,
    path_anime2anime_decoded,
    path_anime_df,
    n=10,
    return_dist=False,
    neg=False,
):
    anime_weights = joblib.load(path_anime_weights)
    anime2anime_encoded = joblib.load(path_anime2anime_encoded)
    anime2anime_decoded = joblib.load(path_anime2anime_decoded)

    frame = getAnimeFrame(name, path_anime_df)
    index = frame["anime_id"].values[0]

    encoded_index = anime2anime_encoded.get(index)
    if encoded_index is None:
        raise ValueError(f"Encoded index not found for anime ID: {index}")

    # FIX: ensure correct shape
    target_vec = anime_weights[encoded_index].squeeze()
    dists = np.dot(anime_weights, target_vec)
    sorted_dists = np.argsort(dists)

    n += 1
    closest = sorted_dists[:n] if neg else sorted_dists[-n:]

    if return_dist:
        return dists, closest

    results = []
    for close in closest:
        decoded_id = anime2anime_decoded.get(close)
        if decoded_id is None or decoded_id == index:
            continue

        anime_frame = getAnimeFrame(decoded_id, path_anime_df)

        results.append({
            "name": anime_frame.eng_version.values[0],
            "genre": anime_frame.Genres.values[0],
            "similarity": dists[close],
        })

    return pd.DataFrame(results).sort_values("similarity", ascending=False)


######## 4. FIND_SIMILAR_USERS

def find_similar_users(
    item_input,
    path_user_weights,
    path_user2user_encoded,
    path_user2user_decoded,
    n=10,
    return_dist=False,
    neg=False,
):
    user_weights = joblib.load(path_user_weights)
    user2user_encoded = joblib.load(path_user2user_encoded)
    user2user_decoded = joblib.load(path_user2user_decoded)

    encoded_index = user2user_encoded.get(item_input)
    if encoded_index is None:
        raise ValueError(f"User not found: {item_input}")

    target_vec = user_weights[encoded_index].squeeze()
    dists = np.dot(user_weights, target_vec)
    sorted_dists = np.argsort(dists)

    n += 1
    closest = sorted_dists[:n] if neg else sorted_dists[-n:]

    if return_dist:
        return dists, closest

    results = []
    for close in closest:
        decoded_id = user2user_decoded.get(close)
        if decoded_id is None or decoded_id == item_input:
            continue

        results.append({
            "similar_users": decoded_id,
            "similarity": dists[close],
        })

    return pd.DataFrame(results).sort_values("similarity", ascending=False)


################## 5. GET USER PREF

def get_user_preferences(user_id, path_rating_df, path_anime_df):
    rating_df = pd.read_csv(path_rating_df)
    df = pd.read_csv(path_anime_df)

    user_df = rating_df[rating_df.user_id == user_id]

    if user_df.empty:
        return pd.DataFrame(columns=["eng_version", "Genres"])

    ratings = user_df["rating"].dropna()
    if ratings.empty:
        return pd.DataFrame(columns=["eng_version", "Genres"])

    threshold = np.percentile(ratings, 75)

    top_animes = user_df[user_df.rating >= threshold].anime_id.values
    anime_df_rows = df[df["anime_id"].isin(top_animes)][["eng_version", "Genres"]]

    return anime_df_rows


######## 6. USER RECOMMENDATION

def get_user_recommendations(
    similar_users,
    user_pref,
    path_anime_df,
    path_synopsis_df,
    path_rating_df,
    n=10,
):
    recommended_animes = []
    anime_list = []

    for user_id in similar_users.similar_users.values:
        pref_list = get_user_preferences(int(user_id), path_rating_df, path_anime_df)

        pref_list = pref_list[
            ~pref_list.eng_version.isin(user_pref.eng_version.values)
        ]

        if not pref_list.empty:
            anime_list.extend(pref_list.eng_version.values)

    if not anime_list:
        return pd.DataFrame()

    counts = pd.Series(anime_list).value_counts().head(n)

    for anime_name, cnt in counts.items():
        frame = getAnimeFrame(anime_name, path_anime_df)
        anime_id = frame.anime_id.values[0]

        recommended_animes.append({
            "n": cnt,
            "anime_name": anime_name,
            "Genres": frame.Genres.values[0],
            "Synopsis": getSynopsis(anime_id, path_synopsis_df),
        })

    return pd.DataFrame(recommended_animes)

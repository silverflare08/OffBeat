from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity


data_folder = Path(__file__).resolve().parent / "data"
movies = pd.read_csv(data_folder / "movies.csv")

# Movies without listed genres cannot support genre recommendations.
movies = movies[movies["genres"] != "(no genres listed)"].copy()
movies = movies.reset_index(drop=True)

genre_lists = movies["genres"].str.split("|")

encoder = MultiLabelBinarizer()
genre_features = encoder.fit_transform(genre_lists)
ratings = pd.read_csv(data_folder / "ratings.csv")

rating_stats = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "size"))

global_average = ratings["rating"].mean()
prior_weight = 50

rating_stats["weighted_rating"] = (
    rating_stats["rating_count"] * rating_stats["average_rating"]
    + prior_weight * global_average) / (rating_stats["rating_count"] + prior_weight)

movies = movies.join(rating_stats, on="movieId")

movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)
movies["weighted_rating"] = movies["weighted_rating"].fillna( global_average)   

tags = pd.read_csv(data_folder / "tags.csv")
tags = tags.dropna(subset=["tag"]).copy()

tags["tag"] = tags["tag"].str.lower().str.strip()
tags = tags[tags["tag"] != ""]
tags = tags.drop_duplicates(subset=["movieId", "tag"])

movie_tags = tags.groupby("movieId")["tag"].agg(" ".join)
movies["tag_text"] = movies["movieId"].map(movie_tags).fillna("")

tag_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
tag_features = tag_vectorizer.fit_transform(movies["tag_text"])
has_tag_features = tag_features.getnnz(axis=1) > 0

def recommend(movie_id, n=5):
    matches = movies.index[movies["movieId"] == movie_id]

    if len(matches) == 0:
        raise ValueError("Movie ID not found or movie has no listed genres.")

    index = matches[0]

    genre_scores = cosine_similarity(
        genre_features[index:index + 1],
        genre_features
    )[0]

    tag_scores = cosine_similarity(
        tag_features[index:index + 1],
        tag_features
    )[0]

    tag_weight = 0.2

    combined_scores = (
        genre_scores
        + tag_weight * tag_scores)

    results = movies.copy()
    results["genre_similarity"] = genre_scores
    results["tag_similarity"] = tag_scores
    results["similarity"] = combined_scores

    results = results[
        (results["movieId"] != movie_id)
        & (results["similarity"] > 0)
    ]

    results = results.sort_values(
        ["similarity", "weighted_rating", "rating_count", "movieId"],
        ascending=[False, False, False, True]
    ).head(n)

    selected_genres = set(genre_lists.iloc[index])

    results["shared_genres"] = results["genres"].apply(
        lambda genres: ", ".join(
            sorted(selected_genres.intersection(genres.split("|")))
        )
    )

    return results[
        [
            "title",
            "similarity",
            "genre_similarity",
            "tag_similarity",
            "weighted_rating",
            "rating_count",
            "shared_genres",
        ]
    ].round({
        "similarity": 3,
        "genre_similarity": 3,
        "tag_similarity": 3,
        "weighted_rating": 2,
    })


if __name__ == "__main__":
    selected_id = 1

    selected_title = movies.loc[
        movies["movieId"] == selected_id, "title"
    ].iloc[0]

    print(f"Because you selected: {selected_title}\n")

    recommendations = recommend(selected_id)
    print(recommendations.to_string(index=False))
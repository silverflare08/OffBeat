from pathlib import Path

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


def recommend(movie_id, n=5):
    matches = movies.index[movies["movieId"] == movie_id]

    if len(matches) == 0:
        raise ValueError("Movie ID not found or movie has no listed genres.")

    index = matches[0]

    scores = cosine_similarity(
        genre_features[index:index + 1],
        genre_features
    )[0]

    results = movies.copy()
    results["similarity"] = scores

    results = results[
        (results["movieId"] != movie_id)
        & (results["similarity"] > 0)
    ]

    results = results.sort_values(
        ["similarity", "movieId"],
        ascending=[False, True]
    ).head(n)

    selected_genres = set(genre_lists.iloc[index])

    results["shared_genres"] = results["genres"].apply(
        lambda genres: ", ".join(
            sorted(selected_genres.intersection(genres.split("|")))
        )
    )

    return results[["title", "similarity", "shared_genres"]]


if __name__ == "__main__":
    selected_id = 2

    selected_title = movies.loc[
        movies["movieId"] == selected_id, "title"
    ].iloc[0]

    print(f"Because you selected: {selected_title}\n")

    recommendations = recommend(selected_id)
    print(recommendations.to_string(index=False))
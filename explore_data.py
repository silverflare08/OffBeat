from pathlib import Path
import pandas as pd

data_folder = Path(__file__).resolve().parent / "data"

movies = pd.read_csv(data_folder / "movies.csv")
ratings = pd.read_csv(data_folder / "ratings.csv")
tags = pd.read_csv(data_folder / "tags.csv")

print("FIRST FIVE MOVIES")
print(movies.head().to_string(index=False))

print("\nDATASET SIZES")
print("Movies:", len(movies))
print("Ratings:", len(ratings))
print("Tags:", len(tags))

print("\nMISSING VALUES IN MOVIES")
print(movies.isna().sum())

print("\nDUPLICATE MOVIE IDs")
print(movies["movieId"].duplicated().sum())
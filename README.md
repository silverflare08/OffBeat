# Offbeat

An explainable movie recommendation system exploring the balance
between familiar preferences and discovery.

## Overview

Offbeat currently recommends movies based on genre similarity.
Given a selected movie, it returns five recommendations with
similarity scores and shared genres.

The project is being developed incrementally, with planned extensions
for tag-based recommendations, personalisation, and adjustable
recommendation diversity.

## Current Features

- Load and inspect MovieLens movie, rating, and tag data
- Represent movie genres using multi-label binary encoding
- Rank recommendations using cosine similarity
- Explain recommendations through shared genres
- Exclude the selected movie and movies without listed genres
- Resolve genre-similarity ties using rating scores adjusted for rating count

## Method

Each movie is represented by a binary vector indicating its genres.
Cosine similarity measures the similarity between the selected movie
and other movies in the catalogue.

Recommendations are ranked by descending similarity. Equal genre-similarity scores are ranked using weighted ratings,then rating counts, then movie IDs.

Only genre information is used in the current recommendation method.
Genres determine similarity, while ratings provide a secondary ranking signal. Tags are reserved for future development.

## Dataset

This project uses the MovieLens Latest Small dataset provided by
GroupLens Research.

Download:
https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

Extract the archive and place these files inside a `data` directory
at the project root:

- movies.csv
- ratings.csv
- tags.csv
- README.txt

The dataset is excluded from version control. Refer to its README
for attribution and usage conditions.

## Installation

Requires Python with pip.

Install dependencies:

    python -m pip install -r requirements.txt

## Usage

Explore the dataset:

    python explore_data.py

Generate recommendations:

    python recommender.py

To select another movie, change `selected_id` in `recommender.py`
to a valid MovieLens movie ID with listed genres.

## Limitations

- Genre similarity does not capture plot, style, or individual taste.
- Movies with identical genre combinations receive identical scores.
- Similarity scores are not probabilities of user enjoyment.
- Recommendation quality has not yet been evaluated against
  held-out user ratings.

## Roadmap

- Incorporate movie tags
- Evaluate recommendations against baseline methods
- Add personalised recommendations
- Introduce a familiar-to-adventurous discovery control
- Build an interactive Streamlit application

## Acknowledgements

Movie data is provided by GroupLens Research through MovieLens:
https://grouplens.org/datasets/movielens/
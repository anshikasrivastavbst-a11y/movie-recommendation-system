# Movie Recommendation System

A content-based movie recommendation system built using Python and Natural Language Processing (NLP). It recommends movies similar to a user's selected movie by analyzing movie information such as genres, keywords, cast, director, and overview.

## Features

- Content-based movie recommendations
- Text preprocessing and stemming using NLTK
- Feature extraction using CountVectorizer
- Cosine similarity to find similar movies
- Python script and Jupyter Notebook implementation

## Tech Stack

- Python
- Pandas
- NLTK
- Scikit-learn
- Jupyter Notebook

## How It Works

1. Loads movie and credits datasets.
2. Combines relevant movie information.
3. Processes text features such as genres, keywords, cast, director, and overview.
4. Converts text into numerical vectors using CountVectorizer.
5. Calculates cosine similarity between movies.
6. Recommends movies with the highest similarity scores.

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Dataset

The project code expects these TMDB dataset files:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Place both files in the project directory before running the code.

## Run the Project

Run the Python script:

```bash
python movie_rec.py
```

Alternatively, open `movie_rec.ipynb` in Jupyter Notebook and run its cells.

## Future Improvements

- Build an interactive web interface.
- Add movie posters and ratings.
- Deploy the recommendation system online.


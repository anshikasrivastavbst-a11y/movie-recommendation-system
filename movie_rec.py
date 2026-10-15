import ast
from pathlib import Path

import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path.cwd()
MOVIES_FILE = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_FILE = DATA_DIR / "tmdb_5000_credits.csv"

movie = pd.read_csv(MOVIES_FILE)
credit = pd.read_csv(CREDITS_FILE)
print(f"Movies: {movie.shape}")
print(f"Credits: {credit.shape}")

# Merge on id (not title) to avoid duplicate rows
credit = credit.rename(columns={"movie_id": "id"}).drop(columns=["title"])
data = movie.merge(credit, on="id", how="inner")
data = data[["genres", "id", "keywords", "overview",
             "release_date", "title", "cast", "crew"]].copy()
data.dropna(subset=["genres", "keywords", "overview", "release_date",
                    "title", "cast", "crew"], inplace=True)
data.reset_index(drop=True, inplace=True)
print(f"Prepared dataset: {data.shape}")


def parse_json_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value) or str(value).strip() == "":
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def extract_names(value):
    result = []
    for item in parse_json_list(value):
        if isinstance(item, dict) and item.get("name"):
            result.append(str(item["name"]))
    return result


def extract_top_cast(value, n=3):
    result = []
    for item in parse_json_list(value)[:n]:
        if isinstance(item, dict) and item.get("name"):
            result.append(str(item["name"]))
    return result


def extract_directors(value):
    result = []
    for item in parse_json_list(value):
        if isinstance(item, dict) and item.get("job") == "Director" and item.get("name"):
            result.append(str(item["name"]))
    return result


data["genres"] = data["genres"].apply(extract_names)
data["keywords"] = data["keywords"].apply(extract_names)
data["cast"] = data["cast"].apply(extract_top_cast)
data["crew"] = data["crew"].apply(extract_directors)
data["overview"] = data["overview"].fillna("").astype(str).str.split()

for column in ["genres", "keywords", "cast", "crew"]:
    data[column] = data[column].apply(
        lambda values: [str(v).replace(" ", "") for v in values]
    )

data["tags"] = (
    data["genres"] + data["keywords"] + data["overview"]
    + data["cast"] + data["crew"]
).apply(lambda values: " ".join(values).lower())

data["title_normalized"] = data["title"].astype(str).str.strip().str.lower()
data = data[["id", "title", "title_normalized", "tags"]].reset_index(drop=True)

ps = PorterStemmer()


def stemming(text):
    return " ".join(ps.stem(word) for word in str(text).split())


data["tags"] = data["tags"].apply(stemming)

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(data["tags"])
similarity = cosine_similarity(vectors)
print(f"Similarity matrix shape: {similarity.shape}")


def recommend(movie_title, n=5):
    query = movie_title.strip().lower()
    matches = data.index[data["title_normalized"] == query].tolist()

    if not matches:
        partial = data[data["title_normalized"].str.contains(query, regex=False, na=False)]
        if partial.empty:
            print(f"Movie not found: {movie_title}")
            return pd.DataFrame(columns=["title", "id", "similarity"])
        movie_index = partial.index[0]
        print(f"Using: {data.iloc[movie_index]['title']}")
    else:
        movie_index = matches[0]

    scores = similarity[movie_index]
    ranked = [i for i in scores.argsort()[::-1] if i != movie_index][:n]

    rec = data.iloc[ranked][["title", "id"]].copy()
    rec["similarity"] = [float(scores[i]) for i in ranked]
    rec.reset_index(drop=True, inplace=True)
    return rec


print(recommend("The Dark Knight", n=5))
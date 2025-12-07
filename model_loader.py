import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------
# FIX: Load CSV files using absolute backend path
# ------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

movies_path = os.path.join(BASE_DIR, "movies.csv")
ratings_path = os.path.join(BASE_DIR, "ratings.csv")

# Load datasets
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# ------------------------------------------
# PREPARE DATA FOR CONTENT-BASED FILTERING
# ------------------------------------------
movies["title"] = movies["title"].astype(str)
movies["genres"] = movies["genres"].astype(str)
movies["combined"] = movies["title"] + " " + movies["genres"]

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["combined"]).toarray()

similarity = cosine_similarity(vectors)

# ------------------------------------------
# RECOMMENDATION FUNCTION
# ------------------------------------------
def recommend(movie_name):
    movie_name = movie_name.lower()

    matches = movies[movies["title"].str.lower().str.contains(movie_name)]

    if matches.empty:
        return []

    index = matches.index[0]

    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

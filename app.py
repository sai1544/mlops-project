from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    matrix, similarity = pickle.load(f)

def recommend(user_id: int):
    # Find user index
    user_index = list(matrix.index).index(user_id)
    scores = similarity[user_index]

    # Sort similar users
    similar_users = np.argsort(scores)[::-1][1:3]

    recommendations = set()

    for user in similar_users:
        movies = matrix.iloc[user]
        recommended_movies = movies[movies > 3].index
        recommendations.update(recommended_movies)

    return list(recommendations)[:5]

@app.get("/")
def root():
    return {"message": "Recommendation API running"}

@app.post("/recommend")
def get_recommendations(user_id: int):
    result = recommend(user_id)
    return {"user_id": user_id, "recommendations": result}


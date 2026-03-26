import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    matrix, similarity = pickle.load(f)

def recommend(user_id):
    user_index = list(matrix.index).index(user_id)
    scores = similarity[user_index]

    # Find top similar users (excluding self)
    similar_users = np.argsort(scores)[::-1][1:3]

    recommendations = set()

    for user in similar_users:
        movies = matrix.iloc[user]
        recommended_movies = movies[movies > 3].index
        recommendations.update(recommended_movies)

    return list(recommendations)[:5]

print(recommend(1))


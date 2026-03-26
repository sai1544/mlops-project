import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load data
data = pd.read_csv("data.csv")

# Create user-item matrix
matrix = data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

# Compute similarity
similarity = cosine_similarity(matrix)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump((matrix, similarity), f)

print("Model saved!")


## AI-Powered Cloud-Native Platform with MLOps + GitOps

🚀 This is a 30-day advanced project combining **DevOps + MLOps + Kubernetes** to build a production-grade AI-powered microservice platform.

---

## 📅 Day 1 — Recommendation Model (Core ML)

### 🎯 Goal
- Build a simple recommendation system (collaborative filtering).
- Save trained model as `model.pkl`.
- Implement prediction logic.

---

### 💻 Environment Setup (From Scratch)

To run this project locally, follow these steps to set up your virtual environment and install dependencies:

```bash
# 1. Clone the repository
git clone [https://github.com/your-username/mlops-project.git](https://github.com/your-username/mlops-project.git)
cd mlops-project/mlops-project

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the environment
# On Linux/WSL:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 4. Install dependencies
pip install pandas scikit-learn numpy
```
🛠 Implementation Details
1. Dataset (data.csv)
Create a file named data.csv with the following sample data:

```Code snippet
user_id,movie_id,rating
1,101,5
1,102,3
1,103,4
2,101,4
2,104,5
2,105,3
3,102,5
3,103,4
3,106,5
```
2. Training Logic (model.py)
This script processes the data into a User-Item Matrix and calculates Cosine Similarity to find users with similar tastes.

```Python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load data
data = pd.read_csv("data.csv")

# Create user-item matrix
matrix = data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

# Compute similarity
similarity = cosine_similarity(matrix)

# Save model components
with open("model.pkl", "wb") as f:
    pickle.dump((matrix, similarity), f)

print("Model trained and saved as model.pkl!")
```
3. Prediction Logic (predict.py)
This script loads the saved model and generates recommendations for a specific user by finding their "nearest neighbors."

```Python
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    matrix, similarity = pickle.load(f)

def recommend(user_id):
    # Find the row index for the given user_id
    user_index = list(matrix.index).index(user_id)
    scores = similarity[user_index]

    # Find top similar users (excluding the user themselves)
    similar_users = np.argsort(scores)[::-1][1:3]

    recommendations = set()
    for user in similar_users:
        movies = matrix.iloc[user]
        # Recommend movies rated higher than 3
        recommended_movies = movies[movies > 3].index
        recommendations.update(recommended_movies)

    return list(recommendations)[:5]

print(f"Recommendations for User 1: {recommend(1)}")
```
```✅ Outcome
Model Persistence: Trained model exported as model.pkl to allow for instant inference without retraining.

Reproducibility: Environment isolated using .venv and clear installation steps.

Foundation Ready: The logic is ready to be wrapped in a FastAPI service.
```
💬 Interview Upgrade
If asked: “What did you build for the ML component?”

“I implemented a collaborative filtering-based recommendation system. I used Pandas for matrix pivot transformations and Scikit-Learn’s Cosine Similarity to identify user clusters. I then serialized the model using Pickle to ensure it can be served efficiently via an API.”

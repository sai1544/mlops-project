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
## Run
```bash 
python module.py
```
This will save our model and we will se an output `Model Saved` in the terminal

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
## Run
```bash 
python predict.py
```
You should see movie IDs.

✅ Outcome
Model Persistence: Trained model exported as model.pkl to allow for instant inference without retraining.

Reproducibility: Environment isolated using .venv and clear installation steps.

Foundation Ready: The logic is ready to be wrapped in a FastAPI service.

💬 Interview Upgrade
If asked: “What did you build for the ML component?”

“I implemented a collaborative filtering-based recommendation system. I used Pandas for matrix pivot transformations and Scikit-Learn’s Cosine Similarity to identify user clusters. I then serialized the model using Pickle to ensure it can be served efficiently via an API.”


## 📘 Day 2 — Convert ML Model into FastAPI Service
This guide shows how to take a trained ML model (model.pkl) and expose it as a REST API using FastAPI.
By the end, you’ll have a working Model Inference Service running locally.

## 🎯 Goals
Create a FastAPI app

Add /recommend endpoint

Load ML model inside API

Test API locally with Swagger UI

🛠 Prerequisites
Python 3.9+ installed

A trained ML model saved as model.pkl (matrix + similarity objects)

Git + terminal access

🛠 Step 1 — Setup Virtual Environment
````bash
# Navigate to project folder
cd mlops-project

# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel
````
🛠 Step 2 — Install Dependencies
Create a requirements.txt file:

````Code
fastapi
uvicorn
numpy
pandas
````
Install dependencies:

````bash
pip install -r requirements.txt
````
🛠 Step 3 — Create FastAPI App
Create a file named app.py:

````python
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
````
🛠 Step 4 — Run API
Start the server:

````bash
uvicorn app:app --reload
`````
You should see logs showing Uvicorn running on 127.0.0.1:8000.

🛠 Step 5 — Test API
Open browser:

Root endpoint: http://127.0.0.1:8000  
→ Should return {"message": "Recommendation API running"}

Swagger UI: http://127.0.0.1:8000/docs  
→ Test POST /recommend with input:

````json
{
  "user_id": 1
}
`````
Expected output:

````json
{
  "user_id": 1,
  "recommendations": [
    101,
    102,
    103,
    104,
    106
  ]
}
````
✅ Success Checklist
FastAPI running

Model loaded successfully

/recommend endpoint working

Swagger UI tested

💡 Interview Upgrade
If asked “How do you deploy ML models?” you can say:

“I convert trained ML models into REST APIs using FastAPI, allowing them to serve real‑time predictions in scalable environments.”


## 📘 Day 3 — Containerize FastAPI ML App with Docker
On Day 3, we converted our FastAPI ML service into a Docker image. This makes the app portable and ready to run anywhere — local machine, cloud VM, or Kubernetes cluster.

## 🎯 Goals
Write a Dockerfile for FastAPI app

Build Docker image locally

🛠 Step 1 — Create Dockerfile
Inside your project folder (mlops-project), create a file named Dockerfile:

```dockerfile
# Use official Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```
🛠 Step 2 — Build Docker Image
Run the following command in your project folder:

```bash
docker build -t ai-recommendation-app .
```
👉 This creates a local image named ai-recommendation-app.

🛠 Step 3 — Verify Image
Check if the image was created:

```bash
docker images
```
Expected output:

```Code
REPOSITORY              TAG       IMAGE ID       CREATED          SIZE
ai-recommendation-app   latest    <image_id>     <time_created>   600MB
```
🛠 Step 4 — Run Container Locally
Start the container:

```bash
docker run -d -p 8000:8000 ai-recommendation-app
````
👉 This maps container port 8000 to your local machine.

Test in browser:
```
http://127.0.0.1:8000 → {"message": "Recommendation API running"}

http://127.0.0.1:8000/docs → Swagger UI
```


## 📘 Day 4 — Push Docker Image to Azure Container Registry (ACR)
On Day 4, we moved our containerized ML service from local Docker into Azure Container Registry (ACR). This makes the image cloud‑ready and accessible to Kubernetes (AKS), VMs, or CI/CD pipelines.

## 🎯 Goals
Create an Azure Container Registry

Enable admin access for login

Tag local Docker image for ACR

Push image into ACR

Verify image inside ACR

🛠 Step 1 — Create Azure Container Registry
Run the following command:

```bash
az acr create --resource-group ml-ops-rg --name mlacr1544 --sku Basic
```
👉 This creates a registry named mlacr1544 in resource group ml-ops-rg.

🛠 Step 2 — Enable Admin User
By default, admin login is disabled. Enable it:

```bash
az acr update -n mlacr1544 --resource-group ml-ops-rg --admin-enabled true
```
Fetch credentials:

```bash
az acr credential show -n mlacr1544 --resource-group ml-ops-rg
```
You’ll get a username and password.

🛠 Step 3 — Login to ACR
Use Docker login with the credentials:

```bash
docker login mlacr1544.azurecr.io -u mlacr1544 -p <password>
```
🛠 Step 4 — Tag Local Image
Suppose your local image is ai-recommendation-app:latest. Tag it for ACR:

```bash
docker tag ai-recommendation-app mlacr1544.azurecr.io/ai-app:v1
```
🛠 Step 5 — Push Image to ACR
Push the tagged image:

```bash
docker push mlacr1544.azurecr.io/ai-app:v1
```
🛠 Step 6 — Verify Image in ACR
List repositories:

```bash
az acr repository list --name mlacr1544 --resource-group ml-ops-rg -o table
```
List tags for ai-app:

```bash
az acr repository show-tags --name mlacr1544 --resource-group ml-ops-rg --repository ai-app -o table
```
👉 You should see v1 listed.

✅ Success Checklist
ACR created (mlacr1544)

Admin user enabled and login successful

Local image tagged (ai-app:v1)

Image pushed to ACR

Verified repository and tags

💡 Interview Upgrade
If asked “How do you deploy ML models?” you can say:

“I containerize ML services with Docker, push them into Azure Container Registry, and manage versioned images that can be pulled by AKS clusters or CI/CD pipelines.”



## 📘 Day 5 — Deploy AI Service to Kubernetes (AKS)
On Day 5, we deployed our containerized ML service into Azure Kubernetes Service (AKS). This is the real production step: running AI workloads inside a cluster.

🎯 Goals
Create Kubernetes namespace

Deploy FastAPI ML app as pods

Expose app internally via Service

Verify pods and service are running

🛠 Step 1 — Create Namespace
```bash
kubectl create namespace ai-app
```
🛠 Step 2 — Create Deployment
Create a file deployment.yaml:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-app
  namespace: ai-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ai-app
  template:
    metadata:
      labels:
        app: ai-app
    spec:
      containers:
      - name: ai-app
        image: mlacr1544.azurecr.io/ai-app:v1
        ports:
        - containerPort: 8000
```
Apply it:

```bash
kubectl apply -f deployment.yaml
```
Check pods:

```bash
kubectl get pods -n ai-app
```
🛠 Step 3 — Create Service
Create a file service.yaml:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-service
  namespace: ai-app
spec:
  type: ClusterIP
  selector:
    app: ai-app
  ports:
  - port: 80
    targetPort: 8000
```
Apply it:

```bash
kubectl apply -f service.yaml
```
Check service:

```bash
kubectl get svc -n ai-app
```
🧪 Step 4 — Test Inside Cluster
Run a temporary test pod:

```bash
kubectl run test-pod --image=busybox -it --rm -- /bin/sh
```
Inside the pod:

```bash
wget -qO- http://ai-service.ai-app.svc.cluster.local
```
👉 You should see the FastAPI root response:
{"message": "Recommendation API running"}

✅ Success Checklist
Pods running in namespace

Service created and mapped to pods

Internal API accessible via cluster DNS

No crash errors in logs

💡 Interview Upgrade
If asked “How do you deploy ML models in Kubernetes?” you can say:

“I containerize the model inference service, deploy it using Kubernetes Deployments, and expose it via a Service for internal communication. This ensures scalability, reliability, and production‑grade orchestration.”

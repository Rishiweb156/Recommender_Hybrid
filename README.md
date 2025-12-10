# Hybrid Recommender System

A **Hybrid Recommender System** that combines **Collaborative Filtering**, **Content-Based Filtering**, and **Neural Collaborative Filtering (NCF)** to provide personalized movie recommendations.  

This project demonstrates how multiple recommendation strategies can be integrated to provide more accurate and user-centric suggestions.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Example Recommendation](#example-recommendation)
- [Model Architecture](#model-architecture)

---

## Project Overview
The Hybrid Recommender System aims to improve recommendation accuracy by combining:

- **Collaborative Filtering (CF)**: Leverages user-item interactions to predict preferences.
- **Content-Based Filtering (CBF)**: Recommends items similar to what a user liked previously, based on item features like genres.
- **Neural Collaborative Filtering (NCF)**: Uses deep learning to model complex user-item interactions.

The hybrid system calculates a **weighted score** combining all three approaches to generate final recommendations.

---

## Features
- Generates personalized movie recommendations for users.
- Supports **Hybrid**, **NCF**, and **Content-Based** recommendation scores.
- Preprocessing and handling of user-item matrices and movie metadata.
- Evaluation using metrics like RMSE, MAE, Precision@K, and Recall@K.
- Scalable and modular structure for adding more recommendation strategies.

---

## Technologies
- **Python 3.9+**
- **Libraries**: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `keras`, `matplotlib`, `seaborn`
- **Environment**: Jupyter Notebook, VSCode, or any Python IDE

---

## Installation
1. Clone the repository:
```
git clone https://github.com/your-username/hybrid-recommender.git
cd hybrid-recommender
```
2.Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
3.Install dependencies:
```
pip install -r requirements.txt
```
## Usage

-->Load your dataset (user-item interactions and movie metadata).
-->Preprocess the data:
  Encode categorical features.
  Normalize ratings.
  Split into training and test sets.
-->Train models:
  Collaborative Filtering
  Content-Based Filtering
  Neural Collaborative Filtering (optional)
-->Generate hybrid recommendations for a user:
```
from hybrid import HybridRecommender

recommender = HybridRecommender(user_item_matrix, item_features)
recommendations = recommender.get_recommendations(user_id=115, top_k=5)

for i, rec in enumerate(recommendations):
    print(f"{i}\n{rec['title']} ({rec['year']})")
    print(f"Genres: {rec['genres']}")
    print(f"Score Breakdown: Hybrid: {rec['hybrid_score']:.3f} | NCF: {rec['ncf_score']:.3f} | Content: {rec['content_score']:.3f}")
    print(f"{rec['description']}\n")
```

## Project Structure

hybrid-recommender/
│
├── data/                   # Dataset files (movies.csv, ratings.csv, etc.)

├── src/                    # Source code

│   ├── ncf.py              # Neural Collaborative Filtering model

│   ├── collaborative.py    # Collaborative Filtering model

│   ├── content_based.py    # Content-Based Filtering model

│   └── hybrid.py           # Hybrid recommendation logic

├── notebooks/              # Jupyter notebooks for experiments

├── requirements.txt        # Project dependencies

└── README.md               # Project documentation

## Example Recommendation
First recommended movie for user ID 115:
```
X-Men (2000)
Genres: Action|Sci-Fi

Score Breakdown: Hybrid: 0.864 | NCF: 0.847 | Content: 0.904

Two mutants, Rogue and Wolverine, come to a private academy for their kind whose resident superhero team, the X-Men, must oppose a terrorist organization with similar powers.

```
This output shows the recommended movie, its genres, a score breakdown from the three models, and the movie description.

## Model Architecture

Collaborative Filtering: Matrix factorization of the user-item interaction matrix.
Content-Based Filtering: Cosine similarity on movie features (genres, tags, etc.).
Neural Collaborative Filtering (NCF):
  User and item embeddings
  Multi-layer Perceptron (MLP) for interaction modeling
  Output: Predicted rating or probability of user-item interaction
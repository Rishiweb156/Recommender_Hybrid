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
- [Overall Architecture](#overall-architecture)

---

## Project Overview
The Hybrid Recommender System aims to improve recommendation accuracy by combining:

- **Collaborative Filtering (CF)**: Leverages user-item interactions to predict preferences.
- **Content-Based Filtering (CBF)**: Recommends items similar to what a user liked previously, based on item features like genres.
- **Neural Collaborative Filtering (NCF)**: Uses deep learning to model complex user-item interactions.

The hybrid system calculates a **weighted score** combining all three approaches to generate final recommendations.

---

## Features


-Two-Stage Recommendation Pipeline: Efficient candidate retrieval (ANN) + accurate hybrid ranking
-Neural Collaborative Filtering: Combines Generalized Matrix Factorization (GMF) + Multi-Layer Perceptron (MLP)
-Semantic Content Embeddings: TF-IDF + Sentence-BERT for rich item representations
-FAISS-Powered ANN Search: Sub-millisecond similarity search for scalability
-Score Explainability: Transparent breakdown of hybrid, NCF, and content scores
-TMDb Integration: Enriched metadata with movie posters and overviews
-Interactive UI: Streamlit-based demo for real-time recommendations

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
3.Download MovieLens 1M dataset
```
#Download from https://grouplens.org/datasets/movielens/1m/
# Extract files to data/ml-1m/
mkdir -p data/ml-1m
# Place ratings.dat and movies.dat in data/ml-1m/
```
4.Set up TMDb API key
```
#Create .env file
echo "TMDB_API_KEY=your_api_key_here" > .env
Training the Model
bash# Train NCF model and generate all artifacts
python -m src.train
# Expected output:
# ✅ Loaded ratings: (1000209, 4), movies: (3706, 3)
# 🧠 Building content embeddings...
# 🚀 Training Neural Collaborative Filtering model...
# Epoch 9/50: val_rmse: 0.891 - val_loss: 0.873
# 💾 Saving all artifacts...
# 🎉 Training complete! Artifacts saved in 'artifacts/'
```
5.Install dependencies:
```
pip install -r requirements.txt
```
6.Running the Demo
```
streamlit run app/streamlit_app.py
#Navigate to http://localhost:8501 and:

#Select a user ID (users with existing history)
#Adjust number of recommendations (5-20)
#Click "✨ Recommend" to see personalized suggestions
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

Neural Collaborative Filtering (NCF)
```
User Input (6040)    Item Input (3706)
       ↓                    ↓
   Embedding (64)      Embedding (64)
       ↓                    ↓
       ├──────── GMF ───────┤  (Element-wise product)
       │                    │
       └──────── MLP ───────┘  (Concat → Dense[128,64,32] + BatchNorm + Dropout)
                  ↓
              Concatenate
                  ↓
              Dense(1)  → Predicted Rating
```
Key Components:

GMF Path: Captures linear user-item interactions via element-wise multiplication
MLP Path: Models complex non-linear patterns with 3 hidden layers
Regularization: L2 regularization (1e-6) + Dropout [0.2, 0.2, 0.2]
Optimization: Adam optimizer with learning rate schedule (0.001 → 0.0005)

Content Embeddings
```
Movie Metadata (Title + Genres + Overview)
              ↓
     ┌─────────────────┐
     │     TF-IDF      │  max_features=2048, ngrams=(1,2)
     │  (2048 dims)    │
     └─────────────────┘
              ↓
     ┌─────────────────┐
     │ Sentence-BERT   │  Model: all-MiniLM-L6-v2
     │   (384 dims)    │
     └─────────────────┘
              ↓
         Concatenate
              ↓
    Combined Embedding (2432 dims)
```
Design Rationale:

TF-IDF captures keyword-level similarity (genre overlap)
SBERT captures semantic similarity (plot themes)
Concatenation provides rich representation for content-based retrieval

## Overall Architecture
The Hybrid Movie Recommender is built as a two-stage pipeline that balances efficiency (via ANN retrieval) with accuracy (via neural ranking).

┌──────────────────────────────────────────────────────────────────┐
│                     OFFLINE TRAINING PHASE                        │
└──────────────────────────────────────────────────────────────────┘
                                ↓
    ┌─────────────────────────────────────────────────────┐
    │  1. Data Loading (MovieLens 1M)                     │
    │     - ratings.dat → user-item interactions          │
    │     - movies.dat → item metadata                    │
    └─────────────────────────────────────────────────────┘
                                ↓
    ┌─────────────────────────────────────────────────────┐
    │  2. Metadata Enrichment (TMDb API)                  │
    │     - Fetch movie overviews                         │
    │     - Fetch poster URLs                             │
    │     - Enrich with 200x200 poster images             │
    └─────────────────────────────────────────────────────┘
                                ↓
    ┌─────────────────────────────────────────────────────┐
    │  3. Content Embedding Generation                    │
    │     Text: title + genres + overview                 │
    │     ├─ TF-IDF (2048 dims, bigrams)                  │
    │     └─ Sentence-BERT (384 dims)                     │
    │     → Combined: 2432-dimensional vectors            │
    └─────────────────────────────────────────────────────┘
                                ↓
    ┌─────────────────────────────────────────────────────┐
    │  4. NCF Model Training                              │
    │     Architecture: GMF + MLP                         │
    │     - User embeddings (6040 users → 64 dims)        │
    │     - Item embeddings (3706 items → 64 dims)        │
    │     - Training: 80/20 split, Early Stopping         │
    │     - Output: Trained model (.keras)                │
    └─────────────────────────────────────────────────────┘
                                ↓
    ┌─────────────────────────────────────────────────────┐
    │  5. User History Extraction                         │
    │     - Filter ratings ≥ 4 (positive feedback)        │
    │     - Build user → [liked_movie_ids] mapping        │
    └─────────────────────────────────────────────────────┘
                                ↓
    ┌─────────────────────────────────────────────────────┐
    │  6. Artifact Generation                             │
    │     ├─ embeddings/content_emb.npy                   │
    │     ├─ models/ncf_model.keras                       │
    │     ├─ data/enriched_movies.csv                     │
    │     ├─ data/id_maps.json                            │
    │     └─ data/user_history.json                       │
    └─────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────┐
│                     ONLINE INFERENCE PHASE                        │
└──────────────────────────────────────────────────────────────────┘
    User Request: recommend(user_id=6, n=10)
                        ↓
    ┌─────────────────────────────────────────────────────┐
    │  STAGE 1: Candidate Retrieval (Content-Based)       │
    │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
    │  Input: user_id                                     │
    │  ├─ Lookup user history: [movie_10, movie_20, ...]  │
    │  ├─ Average liked movie embeddings → user_profile   │
    │  ├─ ANN search (FAISS/sklearn cosine similarity)    │
    │  └─ Retrieve top-500 similar items                  │
    │                                                      │
    │  Cold Start Handling:                               │
    │  If user has no history → return top-K popular      │
    │                                                      │
    │  Complexity: O(log N) with FAISS index              │
    └─────────────────────────────────────────────────────┘
                        ↓
              [500 candidate movies]
                        ↓
    ┌─────────────────────────────────────────────────────┐
    │  STAGE 2: Hybrid Ranking (NCF + Content)            │
    │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
    │  For each candidate:                                │
    │                                                      │
    │  1. NCF Score (Collaborative Signal)                │
    │     Input: (user_id, movie_id)                      │
    │     Output: Predicted rating [0-5]                  │
    │     Normalize: ncf_score = (pred - min) / (max-min) │
    │                                                      │
    │  2. Content Score (Semantic Similarity)             │
    │     Compute: cosine_sim(user_profile, candidate)    │
    │     Average similarity across all liked movies      │
    │     Normalize: content_score ∈ [0, 1]               │
    │                                                      │
    │  3. Hybrid Score (Weighted Blend)                   │
    │     hybrid = α * ncf + (1-α) * content              │
    │     Default α = 0.7 (favor collaborative)           │
    │                                                      │
    │  4. Sort by hybrid_score (descending)               │
    │  5. Return top-N with score breakdown               │
    │                                                      │
    │  Complexity: O(K) where K=500 candidates            │
    └─────────────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────────────┐
    │  Output: Top-N Recommendations                      │
    │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
    │  [                                                   │
    │    {                                                 │
    │      movie_id: 123,                                 │
    │      title: "The Matrix",                           │
    │      hybrid_score: 0.847,                           │
    │      ncf_score: 0.892,                              │
    │      content_score: 0.745                           │
    │    },                                                │
    │    ...                                               │
    │  ]                                                   │
    └─────────────────────────────────────────────────────┘

    

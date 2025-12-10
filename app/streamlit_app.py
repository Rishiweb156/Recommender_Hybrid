# app/streamlit_app.py

import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Add the project root to the Python path to allow imports from `src` and `config`
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import load_config
from src.ann import AnnIndex
from src.hybrid import HybridSystem

st.set_page_config(page_title="Movie Recommender", layout="wide")

# --- Helper Functions with Caching ---
@st.cache_resource
def load_all_artifacts(cfg):
    """
    Loads all artifacts using the central config file to ensure paths are always correct.
    """
    exp_cfg = cfg['experiment']
    paths_cfg = cfg['paths']

    # Dynamically format paths based on the dataset from the config
    for key, path in paths_cfg.items():
        paths_cfg[key] = path.format(dataset=exp_cfg['dataset'])
    
    artifacts_dir = paths_cfg['artifacts_dir']
    if not os.path.exists(artifacts_dir):
        st.error(f"Artifacts directory not found: '{artifacts_dir}'. Please run the training pipeline first.")
        return None

    try:
        # Load all artifacts using the correct paths from the config file
        embeddings = np.load(paths_cfg['embeddings_cache'])
        movies_df = pd.read_csv(paths_cfg['enriched_data_cache'])
        with open(paths_cfg['id_maps_path'], 'r') as f:
            id_maps = json.load(f)
            id_maps['user_map'] = {int(k): v for k, v in id_maps['user_map'].items()}
            id_maps['item_map'] = {int(k): v for k, v in id_maps['item_map'].items()}

        # Load NCF model
        ncf_model_path = os.path.join(paths_cfg['models_dir'], "ncf_model.keras")
        if os.path.exists(ncf_model_path):
            import tensorflow as tf
            model = tf.keras.models.load_model(ncf_model_path)
            
            class NCFAdapter:
                def __init__(self, model, user_map, item_map):
                    self.model = model; self.user_map = user_map; self.item_map = item_map
                def predict_batch(self, users_orig, items_orig):
                    users_new = np.array([self.user_map.get(u, 0) for u in users_orig])
                    items_new = np.array([self.item_map.get(i, 0) for i in items_orig])
                    return self.model.predict([users_new, items_new], verbose=0).flatten()
            ncf_adapter = NCFAdapter(model, id_maps['user_map'], id_maps['item_map'])
        else:
            st.warning("NCF model not found. Running in content-only mode.")
            class DummyNCF:
                def predict_batch(self, users, items): return np.zeros(len(items))
            ncf_adapter = DummyNCF()

        # Initialize components
        ann_index = AnnIndex(embeddings, index_type='auto', metric='cosine')
        hybrid = HybridSystem(ncf_adapter, embeddings, movies_df['movie_id'].tolist(), ann_index, exp_cfg['alpha_blend'])

        with open(paths_cfg['user_history_path'], 'r') as f:
            history_data = json.load(f)
        for user_id_str, liked_movies in history_data.items():
            hybrid.user_history[int(user_id_str)] = liked_movies

        return hybrid, movies_df, embeddings

    except FileNotFoundError as e:
        st.error(f"Artifact file not found: {e}. Please ensure the training pipeline has run successfully.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Main Application UI ---
st.title("🎬 Hybrid Movie Recommender")
st.markdown("Combines **Neural Collaborative Filtering** with **Semantic Content Analysis** for personalized recommendations.")

try:
    config = load_config()
    loaded_data = load_all_artifacts(config)
except FileNotFoundError:
    st.error("`config/config.yaml` not found. Please ensure the file exists in the `config` directory.")
    loaded_data = None

if loaded_data:
    hybrid, movies_df, embeddings = loaded_data
    st.sidebar.success("✅ Artifacts loaded successfully!")

    # --- UI for getting recommendations ---
    st.sidebar.header("Get Recommendations")
    users_with_history = sorted(list(hybrid.user_history.keys()))
    user_id = st.sidebar.selectbox("Select a User ID", options=users_with_history, index=20)
    top_k = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

    if st.sidebar.button("✨ Recommend"):
        recs = hybrid.recommend(user_id, n_recommendations=top_k)
        
        st.header(f"Top {top_k} Recommendations for User {user_id}")
        if recs:
            recs_df = pd.DataFrame(recs, columns=['movie_id', 'hybrid_score', 'ncf_score', 'content_score'])
            recs_df = recs_df.merge(movies_df, on='movie_id', how='left')

            for _, row in recs_df.iterrows():
                st.divider()
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(row.get('poster_path', "https://i.imgur.com/v9s1d4g.png"))
                with col2:
                    st.subheader(row['title'])
                    st.caption(f"**Genres:** {row.get('genres', 'N/A')}")
                    st.markdown(f"**Score Breakdown:** Hybrid: **{row['hybrid_score']:.3f}** | NCF: `{row['ncf_score']:.3f}` | Content: `{row['content_score']:.3f}`")
                    st.write(row.get('overview', 'No overview available.'))
else:
    st.warning("Could not load artifacts. Please run the training pipeline via `make train` and then refresh this page.")